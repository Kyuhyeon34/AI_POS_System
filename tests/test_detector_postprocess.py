"""ai/detector.py 의 후처리 — TensorRT 없이 가짜 YOLO 출력으로 확인한다."""
import numpy as np
import pytest

from ai.detector import JetsonTensorRTDetector


def make_raw(rows, n_classes=2, n_anchors=12):
    """Ultralytics detect 출력 [1, 4+클래스수, 앵커수] 를 만든다.

    rows: [(cx, cy, w, h, class_id, score), ...]  나머지 앵커는 점수 0.
    """
    out = np.zeros((1, 4 + n_classes, n_anchors), dtype=np.float32)
    for i, (cx, cy, w, h, cid, score) in enumerate(rows):
        out[0, 0:4, i] = (cx, cy, w, h)
        out[0, 4 + cid, i] = score
    return out


def test_decode_filters_by_confidence_and_converts_xywh():
    d = JetsonTensorRTDetector(conf=0.5)
    raw = make_raw([(320, 320, 100, 50, 1, 0.9), (100, 100, 40, 40, 0, 0.3)])
    dets = d._decode([raw], ratio=1.0, pad_x=0, pad_y=0, original_shape=(640, 640, 3))
    assert len(dets) == 1
    assert dets[0]["class_id"] == 1
    assert dets[0]["confidence"] == pytest.approx(0.9)
    assert dets[0]["xyxy"] == [270, 295, 370, 345]


def test_decode_undoes_letterbox():
    """640x480 프레임은 640x640 에 위아래 80px 여백으로 들어간다."""
    d = JetsonTensorRTDetector(conf=0.5)
    raw = make_raw([(320, 320, 100, 100, 0, 0.8)])
    dets = d._decode([raw], ratio=1.0, pad_x=0, pad_y=80, original_shape=(480, 640, 3))
    assert dets[0]["xyxy"] == [270, 190, 370, 290]


def test_decode_supports_nms_export_format():
    d = JetsonTensorRTDetector(conf=0.5)
    rows = np.array([[10, 20, 110, 120, 0.9, 2], [0, 0, 5, 5, 0.1, 1]], dtype=np.float32)
    dets = d._decode([rows], 1.0, 0, 0, (480, 640, 3))
    assert [x["class_id"] for x in dets] == [2]


def test_decode_empty_output():
    d = JetsonTensorRTDetector(conf=0.5)
    assert d._decode([], 1.0, 0, 0, (480, 640, 3)) == []
    assert d._decode([make_raw([])], 1.0, 0, 0, (480, 640, 3)) == []


def test_drop_nested_removes_small_box_inside_big_one():
    """와플 하나를 작은 waffle + 큰 pizza 로 동시에 잡는 경우 (IoU 로는 안 걸린다)."""
    xyxy = np.array([[0, 0, 400, 400], [100, 100, 160, 160], [500, 500, 560, 560]], float)
    scores = np.array([0.6, 0.9, 0.7])
    keep = JetsonTensorRTDetector._drop_nested(xyxy, scores, np.array([0, 1, 2]))
    assert sorted(keep.tolist()) == [1, 2]


def test_scale_detections_clips_to_frame():
    dets = JetsonTensorRTDetector._scale_detections(
        [[-50, -50, 900, 900]], [0.9], [0], 1.0, 0, 0, (480, 640, 3))
    assert dets[0]["xyxy"] == [0, 0, 639, 479]


def test_infer_without_engine_raises():
    with pytest.raises(RuntimeError):
        JetsonTensorRTDetector().infer(np.zeros((10, 10, 3), np.uint8))
