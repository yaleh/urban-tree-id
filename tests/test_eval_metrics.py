"""TDD tests for eval_metrics.py — written before extraction."""
import sys
import json
import tempfile
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestReadGtLabel:
    def test_returns_label_from_json(self, tmp_path):
        from eval_metrics import read_gt_label
        ann_dir = tmp_path / "ann"
        ann_dir.mkdir()
        img = tmp_path / "img" / "foo.jpg"
        ann = ann_dir / "foo.jpg.json"
        ann.write_text(json.dumps({"tags": [{"value": "Acer_platanoides"}]}))
        assert read_gt_label(ann_dir, img) == "Acer_platanoides"

    def test_returns_none_when_missing(self, tmp_path):
        from eval_metrics import read_gt_label
        img = tmp_path / "nope.jpg"
        assert read_gt_label(tmp_path, img) is None

    def test_returns_none_when_malformed(self, tmp_path):
        from eval_metrics import read_gt_label
        ann = tmp_path / "bad.jpg.json"
        ann.write_text("{}")
        img = tmp_path / "bad.jpg"
        assert read_gt_label(tmp_path, img) is None


class TestScoreBatch:
    def _fresh_state(self):
        return 0, 0, 0, defaultdict(lambda: {"correct": 0, "total": 0})

    def test_throughput_only_counts_images(self):
        from eval_metrics import score_batch
        n_total, n_correct, n_no_det, per_class = self._fresh_state()
        paths = ["a.jpg", "b.jpg"]
        predictions = [("Oak", 0.9), ("Maple", 0.8)]
        n_boxes = [1, 2]
        result = score_batch(
            paths, predictions, n_boxes, {},
            n_total, n_correct, n_no_det, per_class,
            log_fn=lambda _: None, throughput_only=True,
        )
        assert result[0] == 2  # n_total

    def test_correct_prediction_increments_correct(self):
        from eval_metrics import score_batch
        n_total, n_correct, n_no_det, per_class = self._fresh_state()
        gt = {"a.jpg": "Oak"}
        paths = ["a.jpg"]
        preds = [("Oak", 0.9)]
        n_boxes = [1]
        n_total, n_correct, n_no_det, per_class = score_batch(
            paths, preds, n_boxes, gt,
            n_total, n_correct, n_no_det, per_class, lambda _: None,
        )
        assert n_total == 1 and n_correct == 1

    def test_wrong_prediction_does_not_increment_correct(self):
        from eval_metrics import score_batch
        n_total, n_correct, n_no_det, per_class = self._fresh_state()
        gt = {"a.jpg": "Oak"}
        paths = ["a.jpg"]
        preds = [("Maple", 0.5)]
        n_boxes = [1]
        n_total, n_correct, n_no_det, per_class = score_batch(
            paths, preds, n_boxes, gt,
            n_total, n_correct, n_no_det, per_class, lambda _: None,
        )
        assert n_total == 1 and n_correct == 0

    def test_zero_boxes_increments_no_detection(self):
        from eval_metrics import score_batch
        n_total, n_correct, n_no_det, per_class = self._fresh_state()
        gt = {"a.jpg": "Oak"}
        paths = ["a.jpg"]
        preds = [("", 0.0)]
        n_boxes = [0]
        _, _, n_no_det, _ = score_batch(
            paths, preds, n_boxes, gt,
            n_total, n_correct, n_no_det, per_class, lambda _: None,
        )
        assert n_no_det == 1

    def test_per_class_accumulation(self):
        from eval_metrics import score_batch
        n_total, n_correct, n_no_det, per_class = self._fresh_state()
        gt = {"a.jpg": "Oak", "b.jpg": "Oak"}
        paths = ["a.jpg", "b.jpg"]
        preds = [("Oak", 0.9), ("Maple", 0.4)]
        n_boxes = [1, 1]
        n_total, n_correct, n_no_det, per_class = score_batch(
            paths, preds, n_boxes, gt,
            n_total, n_correct, n_no_det, per_class, lambda _: None,
        )
        assert per_class["Oak"]["total"] == 2
        assert per_class["Oak"]["correct"] == 1

    def test_log_fn_called(self):
        from eval_metrics import score_batch
        n_total, n_correct, n_no_det, per_class = self._fresh_state()
        gt = {"a.jpg": "Oak"}
        calls = []
        score_batch(
            ["a.jpg"], [("Oak", 1.0)], [1], gt,
            n_total, n_correct, n_no_det, per_class, calls.append,
        )
        assert len(calls) == 1
