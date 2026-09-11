
import requests
from .config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, QR_PREFIX

class SupabaseMembershipAPI:
    def __init__(self, url=SUPABASE_URL, key=SUPABASE_PUBLISHABLE_KEY):
        self.url = url.rstrip("/")
        self.key = key
        self.base_headers = {
            "apikey": self.key,
            "Content-Type": "application/json",
        }

    def _headers(self, access_token=None):
        h = dict(self.base_headers)
        if access_token:
            h["Authorization"] = f"Bearer {access_token}"
        else:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def signup(self, email, password, name, phone):
        payload = {
            "email": email.strip(),
            "password": password,
            "data": {
                "name": name.strip(),
                "phone": phone.replace("-", "").strip(),
            },
        }
        r = requests.post(
            f"{self.url}/auth/v1/signup",
            headers=self._headers(),
            json=payload,
            timeout=15,
        )
        data = r.json() if r.content else {}
        if not r.ok:
            raise RuntimeError(data.get("msg") or data.get("error_description") or data.get("message") or str(data))
        return data

    def login(self, email, password):
        payload = {"email": email.strip(), "password": password}
        r = requests.post(
            f"{self.url}/auth/v1/token?grant_type=password",
            headers=self._headers(),
            json=payload,
            timeout=15,
        )
        data = r.json() if r.content else {}
        if not r.ok:
            raise RuntimeError(data.get("error_description") or data.get("msg") or data.get("message") or str(data))
        return data

    def get_my_profile(self, access_token, user_id):
        params = {
            "id": f"eq.{user_id}",
            "select": "id,name,phone,point,qr_token,created_at",
            "limit": "1",
        }
        r = requests.get(
            f"{self.url}/rest/v1/profiles",
            headers=self._headers(access_token),
            params=params,
            timeout=15,
        )
        data = r.json() if r.content else []
        if not r.ok:
            raise RuntimeError(str(data))
        return data[0] if data else None

    def get_my_history(self, access_token, user_id, limit=30):
        params = {
            "member_id": f"eq.{user_id}",
            "select": "delta,type,memo,created_at",
            "order": "id.desc",
            "limit": str(int(limit)),
        }
        r = requests.get(
            f"{self.url}/rest/v1/point_history",
            headers=self._headers(access_token),
            params=params,
            timeout=15,
        )
        data = r.json() if r.content else []
        if not r.ok:
            raise RuntimeError(str(data))
        return data

    def find_member_by_qr_payload(self, qr_payload):
        if not qr_payload.startswith(QR_PREFIX):
            raise RuntimeError("BreadVision 멤버십 QR이 아닙니다.")
        token = qr_payload[len(QR_PREFIX):].strip()
        r = requests.post(
            f"{self.url}/rest/v1/rpc/pos_find_member",
            headers=self._headers(),
            json={"p_qr_token": token},
            timeout=15,
        )
        data = r.json() if r.content else []
        if not r.ok:
            raise RuntimeError(str(data))
        return data[0] if isinstance(data, list) and data else None

    def complete_payment(self, member_id, total_price):
        r = requests.post(
            f"{self.url}/rest/v1/rpc/pos_complete_payment",
            headers=self._headers(),
            json={
                "p_member_id": member_id,
                "p_total_price": int(total_price),
            },
            timeout=15,
        )
        data = r.json() if r.content else []
        if not r.ok:
            raise RuntimeError(str(data))
        return data[0] if isinstance(data, list) and data else None
