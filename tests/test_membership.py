"""membership/supabase_api.py — 네트워크를 타기 전에 걸러지는 부분만 확인한다."""
import pytest

from membership import supabase_api
from membership.config import POINT_RATE, QR_PREFIX


def test_non_member_qr_is_rejected_without_network(monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("회원 QR 이 아닌데 Supabase 를 호출했습니다")

    monkeypatch.setattr(supabase_api.requests, "post", no_network)
    api = supabase_api.SupabaseMembershipAPI()
    with pytest.raises(RuntimeError):
        api.find_member_by_qr_payload("https://example.com")


def test_member_qr_sends_token_without_prefix(monkeypatch):
    sent = {}

    class Resp:
        ok = True
        content = b"[]"

        def json(self):
            return [{"id": "u1", "name": "홍길동", "point": 0}]

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, json=json)
        return Resp()

    monkeypatch.setattr(supabase_api.requests, "post", fake_post)
    member = supabase_api.SupabaseMembershipAPI().find_member_by_qr_payload(QR_PREFIX + " tok123 ")
    assert sent["json"] == {"p_qr_token": "tok123"}
    assert sent["url"].endswith("/rpc/pos_find_member")
    assert member["id"] == "u1"


def test_point_rate_is_three_percent():
    assert POINT_RATE == pytest.approx(0.03)
