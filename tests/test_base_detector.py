"""TDD tests for BaseDetector contract — written before implementation."""
import pytest
from PIL import Image


def _make_concrete(batch_return=None):
    from base_detector import BaseDetector

    class _Concrete(BaseDetector):
        def detect_batch(self, pil_imgs):
            return batch_return if batch_return is not None else [[] for _ in pil_imgs]

    return _Concrete()


class TestBaseDetectorContract:
    def test_cannot_instantiate_abstract(self):
        from base_detector import BaseDetector
        with pytest.raises(TypeError):
            BaseDetector()

    def test_detect_delegates_to_detect_batch(self):
        det = _make_concrete(batch_return=[[[1.0, 2.0, 3.0, 4.0]]])
        result = det.detect(Image.new("RGB", (100, 100)))
        assert result == [[1.0, 2.0, 3.0, 4.0]]

    def test_detect_returns_first_image_only(self):
        det = _make_concrete(batch_return=[[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]])
        result = det.detect(Image.new("RGB", (100, 100)))
        assert result == [[1.0, 2.0, 3.0, 4.0]]

    def test_supports_pipeline_defaults_false(self):
        assert _make_concrete().supports_pipeline is False

    def test_preprocess_cpu_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _make_concrete().preprocess_cpu([Image.new("RGB", (10, 10))])

    def test_forward_preprocessed_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _make_concrete().forward_preprocessed(None)

    def test_detect_empty_returns_empty_list(self):
        det = _make_concrete(batch_return=[[]])
        assert det.detect(Image.new("RGB", (10, 10))) == []
