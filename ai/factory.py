"""추론기를 하나 만들어 준다. GUI 는 어느 쪽인지 몰라도 된다.

  BREADVISION_INFER_URL 이 있으면 → 원격 추론 (PC 에서 GUI 를 돌릴 때)
  없으면                        → 젯슨 로컬 TensorRT

PC 와 젯슨이 같은 소스를 그대로 쓰고, 환경변수로만 갈린다.
"""
import os


def make_detector(conf=0.45, imgsz=640, iou=0.45):
    url = os.environ.get("BREADVISION_INFER_URL", "").strip()
    if url:
        from ai.remote_detector import RemoteDetector
        print(f"[추론] 원격 서버 사용: {url}")
        return RemoteDetector(url, conf=conf, imgsz=imgsz, iou=iou)

    from ai.detector import JetsonTensorRTDetector
    return JetsonTensorRTDetector(conf=conf, imgsz=imgsz, iou=iou)
