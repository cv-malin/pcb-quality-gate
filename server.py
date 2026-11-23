"""
server.py — 本地演示服务（Python 标准库，零额外依赖）

答辩现场：双击 run_demo.bat -> 自动开浏览器 -> 上传/生成 PCB 图 -> 现场检测演示。
所有计算在本机完成，不联网、不上传任何数据。
"""
import os
import sys
import json
import base64
import io
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np
import cv2

from pcbqg import (synth_board, detect_baseline, detect_improved, decide,
                  decide as _decide, DEFECT_TYPES, DEFECT_CN, run_experiment)

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")

COLORS = {
    "gt": (0, 255, 0), "baseline": (0, 165, 255),
    "improved": (255, 200, 0), "accepted": (40, 220, 40),
    "rejected": (0, 0, 255),
}


def draw_boxes(img, boxes, color, label=None):
    out = img.copy()
    for i, b in enumerate(boxes):
        x, y, w, h = b
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(out, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, color, 1, cv2.LINE_AA)
    return out


def img_to_b64(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def inspect(board, policy):
    """生成单板四视图 + 判定详情。"""
    gt = [d.bbox for d in board.defects]
    db = detect_baseline(board)
    di = detect_improved(board)
    dec = decide(di, board, policy)
    acc = [d.bbox for d in dec["accepted"]]
    rej = [r["bbox"] for r in dec["rejected"]]

    v_gt = draw_boxes(board.img, gt, COLORS["gt"])
    v_base = draw_boxes(board.img, [d.bbox for d in db], COLORS["baseline"], "BASE")
    v_imp = draw_boxes(board.img, [d.bbox for d in di], COLORS["improved"], "IMP")
    v_dec = draw_boxes(board.img, acc, COLORS["accepted"], "OK")
    v_dec = draw_boxes(v_dec, rej, COLORS["rejected"], "X")

    truth_sev = {"critical": 3, "major": 2, "minor": 1}
    top = max((d.severity for d in board.defects),
              key=lambda s: truth_sev.get(s, 0), default="none")
    truth = "NG" if truth_sev.get(top, 0) >= 2 else "OK"

    return {
        "views": {
            "原图 / 观测": img_to_b64(board.img),
            "真值标注": img_to_b64(v_gt),
            "基线（单色阈值）": img_to_b64(v_base),
            "改进（参考差分）": img_to_b64(v_imp),
            "判定层（过滤+分级）": img_to_b64(v_dec),
        },
        "defects": [{"type": DEFECT_CN.get(d.kind, d.kind), "kind": d.kind,
                     "sev": d.severity, "bbox": d.bbox} for d in board.defects],
        "baseline_count": len(db),
        "improved_count": len(di),
        "verdict": dec["verdict"],
        "truth": truth,
        "severity": dec["severity"],
        "correct": dec["verdict"] == truth,
        "reasons": dec["reasons"],
        "accepted_n": len(acc), "rejected_n": len(rej),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(WEB, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/meta":
            self._send(200, json.dumps({
                "defects": [{"k": k, "cn": DEFECT_CN[k]} for k in DEFECT_TYPES],
                "version": "1.0.0",
            }, ensure_ascii=False))
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            req = {}

        if path == "/api/gen":
            kind = req.get("kind", "short")
            seed = int(req.get("seed", 1000))
            diff = req.get("difficulty", "normal")
            policy = req.get("policy")
            board = synth_board(seed, kind, difficulty=diff)
            res = inspect(board, policy)
            res["seed"] = seed
            self._send(200, json.dumps(res, ensure_ascii=False))

        elif path == "/api/upload":
            # 前端传 base64 图片 -> 无法获得干净参考板；用"自参考"降级：
            # 前端上传的是本工具生成的观测图时，可用同 seed 重建 clean。
            # 现场演示推荐用 /api/gen；此处接收 {seed, kind} 重建参考板。
            seed = int(req.get("seed", 1000))
            kind = req.get("kind", "short")
            board = synth_board(seed, kind, difficulty=req.get("difficulty", "normal"))
            res = inspect(board, req.get("policy"))
            self._send(200, json.dumps(res, ensure_ascii=False))

        elif path == "/api/experiment":
            cfg = req.get("config", {})
            r = run_experiment(n_per_type=int(cfg.get("n", 20)),
                               n_clean=int(cfg.get("n", 20)),
                               difficulty=cfg.get("difficulty", "normal"),
                               seed0=int(cfg.get("seed", 1000)))
            self._send(200, json.dumps(r, ensure_ascii=False))
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PCB Quality Gate demo running at http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
