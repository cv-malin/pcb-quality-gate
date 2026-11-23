"""
detector.py — 三档检测器（对照实验的核心）

本作品不把"用了 YOLO/AI"当创新，技术增量在下面两处，均可量化对照：

  基线 baseline : 单色阈值法 —— 只能抓"颜色突变"型缺陷（短路/毛刺/杂铜），
                  对"结构缺失"型（漏孔/鼠咬/开路）完全漏检。
  改进 improved : 标准板参考差分法 (golden board comparison) —— 对理想板与
                  观测板做像素差分，6 类缺陷全部可见；再用连通域形状特征
                  分类缺陷类型与严重度。
  判定 decision : 在改进检测之上加"误报过滤 + 致命度分级 + 可解释判定"，
                  这是本作品的核心创新层。

接口约定：detector 输出 Detection 列表，decision 消费它产出板级判定。
真实产线中 improved 的前端可替换为 YOLO bbox，判定层无需改动。
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

from .synth import _severity_for


@dataclass
class Detection:
    bbox: tuple
    kind: str
    severity: str
    score: float
    stage: str = ""


def _singlecolor_mask(img: np.ndarray) -> np.ndarray:
    """基线法：仅按缺陷色阈值分割（无法发现结构缺失型缺陷）。"""
    b, g, r = cv2.split(img)
    return ((r.astype(int) - g.astype(int)) > 80).astype(np.uint8) * 255


def _reference_diff_mask(board) -> np.ndarray:
    """改进法：标准板参考差分。干净理想板由合成器提供，等价产线首件标准板。"""
    d = cv2.absdiff(board.clean, board.img)
    g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return (g > 25).astype(np.uint8) * 255


def detect_baseline(board, min_area=45) -> list:
    """基线：单色阈值 + 形态学 + 轮廓面积过滤，不做类型判断。"""
    m = _singlecolor_mask(board.img)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append(Detection((int(x), int(y), int(w), int(h)), "unknown",
                             "unknown", float(area), "baseline"))
    return out


def _classify(w, h, area) -> tuple:
    """形状先验分类：基于连通域的宽高比/尺寸/面积（对齐 6 类缺陷形态）。"""
    ar = max(w, h) / max(1, min(w, h))
    if ar > 3.0 and max(w, h) > 120:
        return "short", 1.5
    if ar > 2.0 and min(w, h) <= 12 and 100 < area < 400:
        return "open_circuit", 1.5
    if area >= 300 and ar < 1.8:
        return "spurious_copper", 0.6
    if ar > 1.4 and area < 250:
        return "spur", 0.7
    if area < 100 and min(w, h) <= 11:
        return "missing_hole", 1.2
    return "mouse_bite", 1.1


def detect_improved(board, min_area=25) -> list:
    """改进：参考差分 + 连通域 + 形状特征分类 + 严重度映射。"""
    m = _reference_diff_mask(board)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        kind, scale = _classify(int(w), int(h), int(area))
        sev = _severity_for(kind, scale)
        out.append(Detection((int(x), int(y), int(w), int(h)), kind, sev,
                             float(area), "improved"))
    return out


# ---------------- 判定层（核心创新） ----------------
DEFAULT_POLICY = {
    "min_area": 45,          # 误报过滤：面积下限
    "ignore_border": 14,     # 误报过滤：贴边伪影
    "reject_unknown": True,  # 拒绝无法分类的候选
    "critical_is_ng": True,  # 致命缺陷直接判 NG
}


def decide(dets: list, board, policy: dict | None = None) -> dict:
    """判定层：误报过滤 -> 致命度分级 -> 板级 OK/NG + 可解释理由。"""
    p = dict(DEFAULT_POLICY)
    if policy:
        p.update(policy)
    H, W = board.img.shape[:2]
    accepted, rejected = [], []
    for d in dets:
        x, y, w, h = d.bbox
        reason = None
        if p["reject_unknown"] and d.kind == "unknown":
            reason = "未分类候选（基线输出）被判定层拒绝"
        elif d.score < p["min_area"]:
            reason = f"面积 {d.score:.0f}px < 阈值 {p['min_area']}，判为噪声误报"
        elif x <= p["ignore_border"] or y <= p["ignore_border"] or \
             x + w >= W - p["ignore_border"] or y + h >= H - p["ignore_border"]:
            reason = "贴边伪影，判为误报"
        if reason:
            rejected.append({"bbox": d.bbox, "reason": reason})
        else:
            accepted.append(d)

    order = {"critical": 3, "major": 2, "minor": 1, "unknown": 0}
    top = max((d.severity for d in accepted), key=lambda s: order.get(s, 0), default="none")
    verdict = "NG" if order.get(top, 0) >= 2 else "OK"
    reasons = []
    for d in accepted:
        reasons.append(f"{d.kind}（{d.severity}，面积 {d.score:.0f}px）→ 计入判定")
    for r in rejected:
        reasons.append(r["reason"])
    if not accepted:
        reasons.append("无有效缺陷 → 板级判定 OK")
    return {
        "verdict": verdict,
        "severity": top,
        "accepted": accepted,
        "rejected": rejected,
        "reasons": reasons,
    }
