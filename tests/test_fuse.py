"""ai/fuse.py — 앙상블 결합 규칙.

POS 는 신뢰도 0.55 로 자르므로, 합칠 때 신뢰도를 어떻게 정하느냐가 곧
"그 빵이 계산대에 담기느냐" 다. 여기가 조용히 바뀌면 매장에서 빵이 빠진다.
"""
import pytest

from ai.fuse import fuse, gate_specialists, iou
from conftest import det

BOX = (100, 100, 200, 200)
NEAR = (104, 102, 204, 198)          # BOX 와 IoU 약 0.9
FAR = (400, 400, 500, 500)


def test_iou_basic():
    assert iou(BOX, BOX) == pytest.approx(1.0)
    assert iou(BOX, FAR) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_agree_does_not_penalize_single_model():
    """한 모델만 본 0.90 이 0.45 로 깎이면 POS 문턱 0.55 를 못 넘어 빵이 빠진다."""
    out = fuse([[det(3, 0.90, BOX)], []], mode="agree")
    assert len(out) == 1
    assert out[0]["confidence"] == pytest.approx(0.90)
    assert out[0]["votes"] == 1


def test_agree_bonus_when_both_models_agree():
    out = fuse([[det(3, 0.50, BOX)], [det(3, 0.48, NEAR)]], mode="agree")
    assert len(out) == 1
    assert out[0]["confidence"] == pytest.approx(0.50 * 1.25)
    assert out[0]["votes"] == 2


def test_penalty_mode_halves_single_model_score():
    """원논문 방식은 비교용으로 남아 있다. agree 와 차이를 고정해 둔다."""
    out = fuse([[det(3, 0.90, BOX)], []], mode="penalty")
    assert out[0]["confidence"] == pytest.approx(0.45)


def test_confidence_is_capped_at_one():
    out = fuse([[det(1, 0.95, BOX)], [det(1, 0.95, NEAR)]], mode="agree")
    assert out[0]["confidence"] == 1.0


def test_votes_count_models_not_boxes():
    """한 모델이 같은 빵에 박스를 두 번 내도 1표다."""
    lists = [[det(2, 0.8, BOX), det(2, 0.7, NEAR)], []]
    assert fuse(lists)[0]["votes"] == 1
    assert fuse(lists, min_votes=2) == []


def test_fused_box_is_confidence_weighted():
    out = fuse([[det(0, 0.75, (0, 0, 100, 100))], [det(0, 0.25, (8, 0, 108, 100))]])
    assert out[0]["xyxy"] == [2, 0, 102, 100]


def test_different_places_stay_separate():
    out = fuse([[det(0, 0.9, BOX)], [det(0, 0.8, FAR)]])
    assert len(out) == 2


def test_cross_class_keeps_only_most_confident_at_same_place():
    """TTA·앙상블에서 같은 빵을 다른 이름으로 부르면 박스가 겹쳐 뜬다."""
    lists = [[det(7, 0.22, BOX)], [det(6, 0.76, NEAR)]]
    assert len(fuse(lists)) == 2
    out = fuse(lists, cross_class=True)
    assert [d["class_id"] for d in out] == [6]


def test_cross_class_drops_box_nested_inside_bigger_one():
    small = (120, 120, 160, 160)
    out = fuse([[det(0, 0.9, BOX)], [det(1, 0.5, small)]], cross_class=True)
    assert [d["class_id"] for d in out] == [0]


def test_output_sorted_by_confidence():
    out = fuse([[det(0, 0.3, BOX), det(1, 0.9, FAR)]])
    assert [d["confidence"] for d in out] == sorted((d["confidence"] for d in out), reverse=True)


def test_gate_specialists_without_scopes_is_noop():
    lists = [[det(0, 0.9, BOX)], [det(1, 0.9, BOX)]]
    assert gate_specialists(lists, [None, None]) is lists


def test_gate_specialists_removes_specialist_box_on_unknown_bread():
    """전문가(0번만 앎)가 모르는 빵(1번) 자리에서 0번이라고 우기면 뺀다."""
    general = [det(1, 0.8, BOX)]
    specialist = [det(0, 0.95, NEAR), det(0, 0.9, FAR)]
    out = gate_specialists([general, specialist], [None, {0}])
    assert out[0] == general
    assert [tuple(d["xyxy"]) for d in out[1]] == [FAR]


def test_gate_specialists_ignores_low_confidence_other_model():
    general = [det(1, 0.3, BOX)]                  # gate_conf 0.5 미만
    specialist = [det(0, 0.95, NEAR)]
    out = gate_specialists([general, specialist], [None, {0}])
    assert len(out[1]) == 1
