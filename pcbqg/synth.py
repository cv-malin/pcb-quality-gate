"""
synth.py — 可复现的 PCB 基准样本合成器

答辩导向说明：
真实 PKU-Market-PCB 数据集需向北大申请，且原始缺陷本身是 photoshop 合成。
本模块用确定性算法合成"干净的布线板 + 6 类缺陷"，参数化控制难度，
使基线对照实验可复现、有真值标注、可现场演示。所有随机性由 seed 控制。

6 类缺陷（对齐 PKU-Market-PCB）：
  missing_hole 漏孔 / mouse_bite 鼠咬 / open_circuit 开路
  short 短路 / spur 毛刺 / spurious_copper 杂铜
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import cv2

# 颜色（BGR）
C_BG = (26, 92, 48)        # 绿色阻焊层
C_TRACE = (36, 168, 112)   # 铜绿色走线
C_PAD = (32, 190, 196)     # 焊盘（黄铜）
C_HOLE = (18, 54, 30)      # 孔（暗）
C_DEFECT = (26, 26, 178)   # 缺陷着色（蓝，便于观察，与真实铜色区分）

DEFECT_TYPES = ["missing_hole", "mouse_bite", "open_circuit",
                "short", "spur", "spurious_copper"]
DEFECT_CN = {
    "missing_hole": "漏孔", "mouse_bite": "鼠咬", "open_circuit": "开路",
    "short": "短路", "spur": "毛刺", "spurious_copper": "杂铜",
}


@dataclass
class Defect:
    kind: str
    bbox: tuple  # (x, y, w, h)
    severity: str  # critical/major/minor
    note: str = ""


@dataclass
class Board:
    img: np.ndarray            # BGR, 合成观测图
    truth_mask: np.ndarray     # uint8, 1=缺陷像素
    clean: np.ndarray          # BGR, 无缺陷的理想板
    defects: list = field(default_factory=list)
    seed: int = 0
    difficulty: str = "normal"


def _base_board(rng: np.random.Generator, size: int) -> tuple:
    """生成干净 PCB：阻焊层 + 网格走线 + 焊盘阵列 + 过孔。"""
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = C_BG
    # 轻微阻焊纹理噪声
    noise = rng.normal(0, 3, (size, size, 1)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    mask = np.zeros((size, size), np.uint8)  # "铜/应有结构"参考掩膜

    # 走线：横竖若干条
    n_line = 5
    lines = []
    for i in range(1, n_line + 1):
        p = int(size * i / (n_line + 1)) + rng.integers(-6, 7)
        orient = 'h' if i % 2 else 'v'
        lines.append((orient, int(p)))
        if orient == 'h':
            cv2.line(img, (int(size * .06), p), (int(size * .94), p), C_TRACE, 5, cv2.LINE_AA)
            cv2.line(mask, (int(size * .06), p), (int(size * .94), p), 255, 5)
        else:
            cv2.line(img, (p, int(size * .06)), (p, int(size * .94)), C_TRACE, 5, cv2.LINE_AA)
            cv2.line(mask, (p, int(size * .06)), (p, int(size * .94)), 255, 5)

    # 焊盘阵列（网格交点处）
    pads = []
    for (o1, p1), (o2, p2) in zip(lines, lines[1:]):
        if o1 == o2:
            continue
        x, y = (p2, p1) if o1 == 'h' else (p1, p2)
        if size * .1 < x < size * .9 and size * .1 < y < size * .9:
            r = 11
            cv2.circle(img, (x, y), r, C_PAD, -1, cv2.LINE_AA)
            cv2.circle(mask, (x, y), r, 255, -1)
            cv2.circle(img, (x, y), 4, C_HOLE, -1, cv2.LINE_AA)
            cv2.circle(mask, (x, y), 4, 255, -1)
            pads.append((x, y, r))
    return img, mask, lines, pads


def _severity_for(kind: str, scale: float) -> str:
    """严重度规则（简化 IPC-A-610 思路），供真值与判定层共用。"""
    base = {
        "short": "critical", "open_circuit": "critical",
        "missing_hole": "major", "mouse_bite": "major",
        "spur": "minor", "spurious_copper": "minor",
    }[kind]
    if scale > 1.6:
        return {"minor": "major", "major": "critical", "critical": "critical"}[base]
    return base


def synth_board(seed: int, kind: str | None = None, size: int = 640,
                difficulty: str = "normal") -> Board:
    """合成一块板。kind=None 表示良板（无缺陷）。"""
    rng = np.random.default_rng(seed)
    clean, mask, lines, pads = _base_board(rng, size)
    img = clean.copy()
    truth = np.zeros((size, size), np.uint8)
    defects = []
    if kind is None or kind == "clean":
        return Board(img, truth, clean, [], seed, difficulty)

    diff = {"easy": 1.4, "normal": 1.0, "hard": 0.65}[difficulty]
    x, y, w, h = 0, 0, 0, 0

    if not pads:
        pads = [(size // 2, size // 2, 11)]

    if kind == "missing_hole":
        px, py, r = pads[int(rng.integers(len(pads)))]
        rr = max(3, int(4 * diff))
        cv2.circle(img, (px, py), rr, C_PAD, -1, cv2.LINE_AA)
        cv2.circle(truth, (px, py), rr, 1, -1)
        x, y, w, h = px - rr, py - rr, rr * 2, rr * 2
        scale = 1.2

    elif kind == "mouse_bite":
        px, py, r = pads[int(rng.integers(len(pads)))]
        ang = rng.uniform(0, 2 * math.pi)
        bx = int(px + (r + 2) * math.cos(ang))
        by = int(py + (r + 2) * math.sin(ang))
        br = max(4, int(8 * diff))
        cv2.circle(img, (bx, by), br, C_BG, -1, cv2.LINE_AA)
        cv2.circle(img, (bx, by), br - 2, C_BG, -1, cv2.LINE_AA)
        cv2.circle(truth, (bx, by), br, 1, -1)
        x, y, w, h = bx - br, by - br, br * 2, br * 2
        scale = 1.1

    elif kind == "open_circuit":
        o, p = lines[int(rng.integers(len(lines)))]
        along = int(rng.integers(int(size * .25), int(size * .75)))
        gap = max(8, int(22 * diff))
        if o == 'h':
            cv2.line(img, (along, p), (along + gap, p), C_BG, 9, cv2.LINE_AA)
            cv2.line(truth, (along, p), (along + gap, p), 1, 9)
            x, y, w, h = along, p - 5, gap, 10
        else:
            cv2.line(img, (p, along), (p, along + gap), C_BG, 9, cv2.LINE_AA)
            cv2.line(truth, (p, along), (p, along + gap), 1, 9)
            x, y, w, h = p - 5, along, 10, gap
        scale = 1.5

    elif kind == "short":
        hs = [(o, p) for o, p in lines if o == 'h']
        vs = [(o, p) for o, p in lines if o == 'v']
        if len(hs) >= 2 and len(vs) >= 1:
            y1 = hs[0][1]; y2 = hs[1][1]; vx = vs[0][1]
            cv2.line(img, (vx, y1), (vx, y2), C_DEFECT, 7, cv2.LINE_AA)
            cv2.line(truth, (vx, y1), (vx, y2), 1, 7)
            x, y, w, h = vx - 5, min(y1, y2), 10, abs(y2 - y1)
        else:
            x, y, w, h = size // 2, size // 2, 10, 40
        scale = 1.4

    elif kind == "spur":
        o, p = lines[int(rng.integers(len(lines)))]
        along = int(rng.integers(int(size * .2), int(size * .8)))
        L = max(8, int(20 * diff))
        if o == 'h':
            cv2.line(img, (along, p), (along + L // 2, p - L), C_DEFECT, 3, cv2.LINE_AA)
            cv2.line(truth, (along, p), (along + L // 2, p - L), 1, 3)
            x, y, w, h = along, p - L, L // 2 + 3, L
        else:
            cv2.line(img, (p, along), (p + L, along + L // 2), C_DEFECT, 3, cv2.LINE_AA)
            cv2.line(truth, (p, along), (p + L, along + L // 2), 1, 3)
            x, y, w, h = p, along, L, L // 2 + 3
        scale = 0.7

    elif kind == "spurious_copper":
        cx = int(rng.integers(int(size * .15), int(size * .85)))
        cy = int(rng.integers(int(size * .15), int(size * .85)))
        rw = max(5, int(rng.integers(8, 18) * diff))
        rh = max(5, int(rng.integers(8, 18) * diff))
        cv2.ellipse(img, (cx, cy), (rw, rh), rng.uniform(0, 180), 0, 360, C_DEFECT, -1, cv2.LINE_AA)
        cv2.ellipse(truth, (cx, cy), (rw, rh), 0, 0, 360, 1, -1)
        x, y, w, h = cx - rw, cy - rh, rw * 2, rh * 2
        scale = 0.6
    else:
        raise ValueError(f"unknown kind {kind}")

    sev = _severity_for(kind, scale)
    defects.append(Defect(kind, (int(x), int(y), int(w), int(h)), sev))
    return Board(img, truth, clean, defects, seed, difficulty)


def synth_dataset(n_per_type: int = 20, n_clean: int = 20,
                  size: int = 640, difficulty: str = "normal",
                  seed0: int = 1000) -> list:
    boards = []
    s = seed0
    for k in DEFECT_TYPES:
        for _ in range(n_per_type):
            boards.append(synth_board(s, k, size, difficulty))
            s += 1
    for _ in range(n_clean):
        boards.append(synth_board(s, None, size, difficulty))
        s += 1
    return boards
