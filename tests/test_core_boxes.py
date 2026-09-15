"""core/boxes.py — 수집 탭 박스 계산."""
import pytest

from conftest import det
from core.boxes import SteadyBox, center_guide_box, iou, pick_label_box, yolo_line

SHAPE = (480, 640, 3)


def test_iou_and_yolo_line():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert yolo_line(2, (64, 48, 320, 240), 640, 480) == "2 0.300000 0.300000 0.400000 0.400000"


def test_center_guide_box():
    assert center_guide_box(640, 480) == (176, 132, 464, 348)


def test_steady_box_follows_small_moves_slowly():
    s = SteadyBox()
    assert s.update((100, 100, 200, 200)) == (100, 100, 200, 200)
    assert s.update((110, 100, 210, 200)) == (104, 100, 204, 200)     # 35% 만 따라간다


def test_steady_box_jumps_only_after_repeated_far_boxes():
    s = SteadyBox()
    s.update((0, 0, 100, 100))
    far = (400, 300, 500, 400)
    assert s.update(far) == (0, 0, 100, 100)
    assert s.update(far) == (0, 0, 100, 100)
    assert s.update(far) == far


def test_steady_box_keeps_box_briefly_when_lost():
    s = SteadyBox()
    s.update((0, 0, 100, 100))
    for _ in range(SteadyBox.KEEP_FRAMES):
        assert s.update(None) == (0, 0, 100, 100)
    assert s.update(None) is None


def test_pick_label_box_prefers_center():
    dets = [det(0, 0.9, (0, 0, 100, 100)), det(1, 0.5, (270, 190, 370, 290))]
    best, why = pick_label_box(dets, SHAPE, min_conf=0.35, min_area=0.01, max_area=0.85)
    assert best["class_id"] == 1
    assert why == "conf 0.50"


@pytest.mark.parametrize("dets,reason", [
    ([], "no bread found"),
    ([det(0, 0.2, (270, 190, 370, 290))], "low conf 0.20"),
    ([det(0, 0.9, (300, 200, 310, 210))], "too small"),
    ([det(0, 0.9, (0, 0, 640, 480))], "too big"),
])
def test_pick_label_box_rejections(dets, reason):
    best, why = pick_label_box(dets, SHAPE, min_conf=0.35, min_area=0.01, max_area=0.85)
    assert best is None
    assert why.startswith(reason)
