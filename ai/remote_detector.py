"""원격 추론 클라이언트.

PC 에서 GUI 를 돌리고, 추론만 젯슨에 맡길 때 쓴다.
JetsonTensorRTDetector 와 완전히 같은 인터페이스라 GUI 코드는 바뀌지 않는다.

  PC (이 클래스)  ──JPEG 한 장──▶  젯슨 (inference_server.py)
                 ◀──검출 결과────

「빵 인식」 버튼을 누를 때만 통신한다. 카메라 미리보기는 PC 로컬이라 네트워크를 타지 않는다.
"""
import json
import urllib.error
import urllib.request

import cv2

from ai.render import draw_detections

JPEG_QUALITY = 80          # 57KB 정도. 100Mbps 에서 전송 4.5ms
TIMEOUT = 10.0


class RemoteDetector:
    def __init__(self, url, model_path=None, conf=0.45, imgsz=640, iou=0.45):
        self.url = url.rstrip("/")
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.iou = float(iou)
        self.model_path = None
        self._ready = False
        self.refresh()
        if model_path:
            self.load(model_path)

    # ---- 서버 상태 ----------------------------------------------------
    def _get(self, path):
        req = urllib.request.Request(f"{self.url}{path}")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path, data, content_type):
        req = urllib.request.Request(
            f"{self.url}{path}", data=data,
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))

    def refresh(self):
        """서버가 어떤 모델을 물고 있는지 확인한다."""
        try:
            info = self._get("/health")
            self._ready = bool(info.get("ready"))
            self.model_path = info.get("model_path")
            self.conf = float(info.get("conf", self.conf))
            return True
        except Exception:
            self._ready = False
            self.model_path = None
            return False

    def ready(self):
        return self._ready

    # ---- 모델 지정 ----------------------------------------------------
    def load(self, model_path):
        """젯슨에 있는 .engine 경로를 서버에 알려준다. (경로는 젯슨 기준)"""
        try:
            info = self._post(
                "/load",
                json.dumps({"model_path": str(model_path)}).encode("utf-8"),
                "application/json",
            )
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"추론 서버가 모델을 열지 못했습니다: {e.read().decode('utf-8', 'replace')}"
            )
        except Exception as e:
            raise RuntimeError(f"추론 서버({self.url})에 연결할 수 없습니다: {e}")

        if not info.get("ready"):
            raise RuntimeError(f"추론 서버가 모델을 열지 못했습니다: {info}")
        self._ready = True
        self.model_path = info.get("model_path")
        return True

    # ---- 추론 ---------------------------------------------------------
    def infer(self, frame, draw=True, tta=False):
        """tta=True 면 젯슨이 좌우반전으로 한 번 더 보고 합친다 (「빵 인식」 버튼용)."""
        if not self._ready and not self.refresh():
            raise RuntimeError(
                f"추론 서버({self.url})에 연결할 수 없습니다.\n"
                "젯슨에서 inference_server.py 가 켜져 있는지 확인하세요."
            )

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError("프레임을 JPEG 로 변환하지 못했습니다.")

        try:
            res = self._post("/infer?tta=1" if tta else "/infer",
                             buf.tobytes(), "image/jpeg")
        except Exception as e:
            self._ready = False
            raise RuntimeError(f"추론 요청 실패: {e}")

        detections = res.get("detections", [])
        if not draw:
            counts = {}
            for d in detections:
                counts[d["class_id"]] = counts.get(d["class_id"], 0) + 1
            return frame, detections, counts

        # 서버는 좌표만 보낸다. 그리기는 로컬에서, 로컬 추론과 같은 코드로.
        plotted, counts = draw_detections(frame, detections)
        return plotted, detections, counts
