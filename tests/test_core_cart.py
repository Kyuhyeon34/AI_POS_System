"""core/cart.py — POS 금액 계산."""
from conftest import det
from core.cart import POS_CONF, Cart, count_by_class, pos_detections

PRODUCTS = {
    "0": {"class_name": "salt", "display_name": "소금빵", "price": 3000},
    "1": {"class_name": "pie", "display_name": "파이", "price": 4500},
}


def test_add_change_remove():
    cart = Cart()
    cart.add(0)
    cart.add("0")
    cart.add(1, 3)
    assert cart.items == {0: 2, 1: 3}
    cart.change(1, -3)
    assert cart.items == {0: 2}
    cart.remove(0)
    assert not cart


def test_total_uses_current_price_and_skips_unknown():
    cart = Cart()
    cart.add(0, 2)
    cart.add(1)
    cart.add(9)                          # 상품 목록에서 지워진 것
    assert cart.total(PRODUCTS) == 3000 * 2 + 4500
    assert [line[0] for line in cart.lines(PRODUCTS)] == [0, 1]
    assert cart.total_qty() == 4
    assert cart.kinds() == 3


def test_add_counts_accumulates_and_reports_unknown():
    """같은 트레이를 두 번 인식하면 수량이 더해져야 한다 (덮어쓰면 두 번째 누름이 무시된다)."""
    cart = Cart()
    added, unknown = cart.add_counts({0: 2, 7: 1}, PRODUCTS)
    cart.add_counts({0: 2}, PRODUCTS)
    assert added == {0: 2}
    assert unknown == [7]
    assert cart.items == {0: 4}


def test_pos_detections_threshold_is_inclusive():
    dets = [det(0, POS_CONF, (0, 0, 1, 1)), det(0, 0.549, (0, 0, 1, 1)), det(1, 0.9, (0, 0, 1, 1))]
    kept = pos_detections(dets)
    assert [d["confidence"] for d in kept] == [POS_CONF, 0.9]
    assert count_by_class(kept) == {0: 1, 1: 1}
    assert pos_detections(None) == []
