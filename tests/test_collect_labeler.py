"""core/collect_labeler.py — 「여러 개」 수집에서 라벨 후보·지정·무시·고정."""
import numpy as np
import pytest

import core.collect_labeler as cl
from conftest import det
from core.collect_labeler import CollectLabeler, patch_sig, sig_match

FRAME = np.zeros((480, 640, 3), np.uint8)


@pytest.fixture(autouse=True)
def no_tracker(monkeypatch):
    """opencv 빌드마다 KCF 유무가 달라 결과가 흔들리지 않게, 검출만으로 따라가는 경로를 쓴다."""
    monkeypatch.setattr(cl, "make_tracker", lambda frame, box: None)


def labeler(blobs=()):
    return CollectLabeler(segmenter=lambda frame, min_area, max_area: list(blobs))


def test_model_boxes_filtered_and_unknown_blobs_added():
    lab = labeler(blobs=[(100, 100, 200, 200), (400, 100, 500, 200)])
    dets = [det(0, 0.9, (102, 98, 202, 198)),     # 배경 분리 덩어리와 같은 물체
            det(1, 0.10, (0, 300, 100, 400)),     # 신뢰도가 너무 낮다
            det(2, 0.9, (0, 0, 640, 480))]        # 화면 전체 = 배경
    boxes = lab.refresh(FRAME, dets)
    assert [(b["source"], b["class_id"]) for b in boxes] == [("model", 0), ("new", None)]


def test_assign_follows_moving_box():
    lab = labeler(blobs=[(100, 100, 200, 200)])
    lab.refresh(FRAME, [])
    assert lab.assign(5) is None
    lab.segmenter = lambda frame, min_area, max_area: [(110, 100, 210, 200)]    # 빵을 조금 옮김
    boxes = lab.refresh(FRAME, [])
    assert boxes[0]["class_id"] == 5 and boxes[0]["fixed"]
    assert lab.manual == [((110, 100, 210, 200), 5)]


def test_ignore_removes_box_and_keeps_ignoring_after_move():
    lab = labeler(blobs=[(100, 100, 200, 200), (300, 100, 400, 200)])
    lab.refresh(FRAME, [])
    lab.ignore()
    assert [b["xyxy"] for b in lab.refresh(FRAME, [])] == [(300, 100, 400, 200)]
    lab.segmenter = lambda frame, min_area, max_area: [(105, 100, 205, 200), (300, 100, 400, 200)]
    assert [b["xyxy"] for b in lab.refresh(FRAME, [])] == [(300, 100, 400, 200)]


def test_lock_keeps_label_and_hides_unnamed_boxes():
    lab = labeler(blobs=[(400, 300, 480, 380)])                 # 이름 없는 다른 물체
    lab.refresh(FRAME, [det(1, 0.9, (100, 100, 200, 200))])
    locked, msg = lab.toggle_lock(FRAME, class_id=7)            # 모델은 1번이라 했지만 7번으로 찍는다
    assert locked and msg is None

    boxes = lab.refresh(FRAME, [det(1, 0.6, (106, 100, 206, 200))])
    assert [(b["class_id"], b["locked"]) for b in boxes] == [(7, True)]
    assert lab.cycle().startswith("현재 상태: 촬영 대상이 고정")

    boxes = lab.refresh(FRAME, [])                               # 검출이 끊겨도 자리를 지킨다
    assert [b["source"] for b in boxes] == ["lock"]

    assert lab.toggle_lock(FRAME, 7) == (False, None)


def test_actions_without_boxes_return_messages():
    lab = labeler()
    lab.refresh(FRAME, [])
    assert lab.cycle() and lab.assign(0) and lab.ignore()
    assert lab.toggle_lock(FRAME, 0)[1]
    assert lab.hint({}).startswith("현재 상태: 잡힌 물체가 없습니다")


def test_hint_counts_unnamed():
    lab = labeler(blobs=[(100, 100, 200, 200), (300, 100, 400, 200)])
    lab.refresh(FRAME, [])
    assert "이름 없는 것 2개" in lab.hint({})


def test_signature_matching():
    a = np.zeros((40, 40, 3), np.uint8)
    b = np.full((40, 40, 3), 255, np.uint8)
    sa, sb = patch_sig(a, (0, 0, 40, 40)), patch_sig(b, (0, 0, 40, 40))
    assert sig_match(sa, sa) == 1.0
    assert sig_match(sa, None) == 1.0
    assert sig_match(sa, sb) == 0.0
