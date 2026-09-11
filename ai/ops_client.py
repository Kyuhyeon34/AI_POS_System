"""PC 에서 젯슨에 작업을 시키는 클라이언트.

업체는 PC 한 대만 만진다. 데이터셋 분할·전이학습·ENGINE 변환·배포 버튼을 누르면
이 클래스가 젯슨의 inference_server.py 에 요청을 보내고, 진행 로그를 받아 온다.
젯슨에 SSH 로 들어가는 일이 없다.

urllib 만 쓴다 (PC 에 requests 를 따로 깔 필요가 없게).
"""
import json
import urllib.error
import urllib.request

TIMEOUT = 15.0
UPLOAD_TIMEOUT = 600.0        # 수집 ZIP 은 수십 MB 라 넉넉히


class OpsClient:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.available = False

    # ---- 기본 ---------------------------------------------------------
    def _get(self, path, timeout=TIMEOUT):
        with urllib.request.urlopen(f"{self.url}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path, data, content_type="application/json", timeout=TIMEOUT):
        req = urllib.request.Request(f"{self.url}{path}", data=data,
                                     headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 서버가 이유를 JSON 으로 알려준다. 그걸 그대로 보여줘야 원인을 안다.
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error", "")
            except Exception:
                msg = ""
            raise RuntimeError(msg or f"HTTP {e.code}")

    def _json_post(self, path, obj, timeout=TIMEOUT):
        return self._post(path, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                          timeout=timeout)

    # ---- 상태 ---------------------------------------------------------
    def refresh(self):
        """젯슨 서버가 작업 실행까지 지원하는 버전인지 확인한다."""
        try:
            info = self._get("/health")
            self.available = bool(info.get("ops"))
            return info
        except Exception:
            self.available = False
            return None

    def files(self, kind):
        """젯슨에 있는 파일 목록. PC 에서 젯슨 폴더를 뒤질 필요가 없다."""
        return self._get(f"/files?kind={kind}").get("files", [])

    # ---- 작업 ---------------------------------------------------------
    def upload_zip(self, path, replace_prefix=None):
        from pathlib import Path
        p = Path(path)
        q = f"/upload?name={urllib.request.quote(p.name)}"
        if replace_prefix:
            q += f"&replace={urllib.request.quote(replace_prefix)}"
        return self._post(q, p.read_bytes(), "application/zip", timeout=UPLOAD_TIMEOUT)

    def start(self, task, **args):
        """작업을 걸고 job id 를 받는다. 실제 진행은 poll() 로 따라간다."""
        return self._json_post("/task", {"task": task, **args})["job"]

    def poll(self, job_id, start=0):
        return self._get(f"/job?id={job_id}&from={start}")

    def deploy(self, engine_path):
        """새 엔진을 POS 추론에 적용하고 settings.json 에 기록한다."""
        return self._json_post("/deploy", {"engine": engine_path}, timeout=180.0)
