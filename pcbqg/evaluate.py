"""
evaluate.py — 评测与三档对照实验

指标（对齐答辩要求）：
  缺陷级：Recall(漏检率)、Precision(误报率)、F1、按类别召回
  板级  ：判定准确率、漏检板(FN board)、误报板(FP board)
  对照  ：baseline / improved / improved+decision 三档
IoU 命中阈值默认 0.15（小目标宽松匹配，三档一致，保证对照公平）。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .synth import DEFECT_TYPES, synth_dataset
from .detector import detect_baseline, detect_improved, decide, Detection

IOU_THR = 0.15


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    board_correct: int = 0
    board_total: int = 0
    fn_board: int = 0
    fp_board: int = 0
    sev_correct: int = 0
    sev_total: int = 0


def _match(dets, board):
    """返回 (matched_pairs, unmatched_dets)。"""
    used = set(); pairs = []
    for d in dets:
        best, bi = 0.0, -1
        for i, gt in enumerate(board.defects):
            if i in used:
                continue
            v = iou(d.bbox, gt.bbox)
            if v > best:
                best, bi = v, i
        if best >= IOU_THR:
            used.add(bi); pairs.append((d, board.defects[bi], best))
    unmatched = [d for i, d in enumerate(dets) if all(p[0] is not d for p in pairs)]
    return pairs, unmatched


def run_dataset(boards, mode="baseline", policy=None):
    """mode: baseline / improved / improved_decision"""
    c = Confusion()
    per_class = {k: [0, 0] for k in DEFECT_TYPES}  # [hit, total]
    for b in boards:
        if mode == "baseline":
            dets = detect_baseline(b)
            chosen = dets
            board_verdict = "NG" if dets else "OK"
        elif mode == "improved":
            dets = detect_improved(b)
            chosen = dets
            board_verdict = "NG" if dets else "OK"
        elif mode == "improved_decision":
            dets = detect_improved(b)
            dec = decide(dets, b, policy)
            chosen = dec["accepted"]
            board_verdict = dec["verdict"]
        else:
            raise ValueError(mode)

        pairs, unmatched = _match(chosen, b)
        c.tp += len(pairs)
        c.fn += max(0, len(b.defects) - len(pairs))
        c.fp += len(unmatched)
        c.sev_total += len(pairs)
        c.sev_correct += sum(1 for d, gt, _ in pairs if d.severity == gt.severity)

        for d, gt, _ in pairs:
            if gt.kind in per_class:
                per_class[gt.kind][0] += 1
        for gt in b.defects:
            if gt.kind in per_class:
                per_class[gt.kind][1] += 1

        # 板级真值按 IPC 严重度规则：critical/major 判 NG，minor 可放行
        sev_order = {"critical": 3, "major": 2, "minor": 1}
        top_sev = max((g.severity for g in b.defects),
                      key=lambda s: sev_order.get(s, 0), default="none")
        truth_verdict = "NG" if sev_order.get(top_sev, 0) >= 2 else "OK"
        c.board_total += 1
        if board_verdict == truth_verdict:
            c.board_correct += 1
        elif truth_verdict == "NG" and board_verdict == "OK":
            c.fn_board += 1
        else:
            c.fp_board += 1

    prec = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0
    rec = c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "mode": mode,
        "tp": c.tp, "fp": c.fp, "fn": c.fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "miss_rate": round(1 - rec, 4),
        "false_alarm": round(1 - prec, 4) if (c.tp + c.fp) else 0.0,
        "f1": round(f1, 4),
        "board_acc": round(c.board_correct / c.board_total, 4),
        "fn_board": c.fn_board,
        "fp_board": c.fp_board,
        "sev_acc": round(c.sev_correct / c.sev_total, 4) if c.sev_total else 0.0,
        "per_class_recall": {k: round(v[0] / v[1], 3) if v[1] else 0.0
                             for k, v in per_class.items()},
    }


def run_experiment(n_per_type=20, n_clean=20, difficulty="normal", seed0=1000,
                   policy=None):
    boards = synth_dataset(n_per_type, n_clean, difficulty=difficulty, seed0=seed0)
    results = [run_dataset(boards, m, policy)
               for m in ("baseline", "improved", "improved_decision")]
    return {"config": {"n_per_type": n_per_type, "n_clean": n_clean,
                       "difficulty": difficulty, "seed0": seed0,
                       "total_boards": len(boards), "iou_thr": IOU_THR},
            "results": results}
