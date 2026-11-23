"""pcbqg — PCB Quality Gate 核心包"""
__version__ = "1.0.0"

from .synth import DEFECT_TYPES, DEFECT_CN, synth_board, synth_dataset, Board, Defect
from .detector import Detection, detect_baseline, detect_improved, decide
from .evaluate import run_dataset, run_experiment, iou

__all__ = [
    "DEFECT_TYPES", "DEFECT_CN", "synth_board", "synth_dataset", "Board", "Defect",
    "Detection", "detect_baseline", "detect_improved", "decide",
    "run_dataset", "run_experiment", "iou",
]
