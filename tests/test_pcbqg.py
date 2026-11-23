"""PCB Quality Gate 单元测试（标准库 unittest，零额外依赖）"""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcbqg import (synth_board, synth_dataset, detect_baseline, detect_improved,
                   decide, run_dataset, run_experiment, DEFECT_TYPES, iou)
from pcbqg.evaluate import iou as iou_fn


class TestSynth(unittest.TestCase):
    def test_all_types_have_truth(self):
        for k in DEFECT_TYPES:
            b = synth_board(100, k)
            self.assertEqual(len(b.defects), 1, k)
            self.assertEqual(b.defects[0].kind, k)
            self.assertGreater(b.truth_mask.sum(), 0)

    def test_clean_board_has_no_defect(self):
        b = synth_board(100, None)
        self.assertEqual(len(b.defects), 0)
        self.assertEqual(b.truth_mask.sum(), 0)

    def test_reproducible(self):
        a = synth_board(777, "short")
        b = synth_board(777, "short")
        self.assertTrue((a.img == b.img).all())


class TestIOU(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(iou_fn((10, 10, 20, 20), (10, 10, 20, 20)), 1.0)

    def test_no_overlap(self):
        self.assertEqual(iou_fn((0, 0, 5, 5), (50, 50, 5, 5)), 0.0)


class TestDetector(unittest.TestCase):
    def test_baseline_misses_gap_types(self):
        # 基线（单色阈值）对结构缺失型缺陷漏检
        for k in ["missing_hole", "mouse_bite", "open_circuit"]:
            b = synth_board(1000, k)
            self.assertEqual(len(detect_baseline(b)), 0, k)

    def test_improved_detects_all(self):
        for k in DEFECT_TYPES:
            b = synth_board(1000, k)
            self.assertGreaterEqual(len(detect_improved(b)), 1, k)

    def test_decision_filters_and_grades(self):
        b = synth_board(1000, "short")
        d = decide(detect_improved(b), b)
        self.assertEqual(d["verdict"], "NG")
        self.assertEqual(d["severity"], "critical")

    def test_clean_board_passes(self):
        b = synth_board(1000, None)
        d = decide(detect_improved(b), b)
        self.assertEqual(d["verdict"], "OK")


class TestExperiment(unittest.TestCase):
    def test_three_modes_improve(self):
        r = run_experiment(n_per_type=6, n_clean=6, seed0=3000)
        base, imp, dec = r["results"]
        self.assertLess(base["recall"], imp["recall"])
        self.assertLessEqual(imp["recall"], dec["recall"])
        self.assertLess(base["board_acc"], dec["board_acc"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
