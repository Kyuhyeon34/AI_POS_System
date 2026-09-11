"""GStreamer 로 카메라를 연다. (젯슨 전용, 선택 사항)

왜 필요한가
    OpenCV 의 V4L2 캡처는 MJPG 압축을 CPU 로 푼다.
    실측(Orin Nano, 1280x720 30fps): 코어 1개의 44.7% 를 디코딩에만 쓴다.
    젯슨에는 JPEG 디코딩 전용 하드웨어가 있으므로 그쪽에 넘기면 CPU 가 남는다.

    파이프라인 뒤쪽의 appsink 에 drop=true max-buffers=1 을 준다.
    처리하는 동안 들어온 프레임은 쌓지 않고 버린다. 밀린 프레임을 뒤늦게
    처리해봐야 결과가 과거의 것이라 쓸모가 없고 부하만 는다.

주의
    pip 로 설치한 opencv-python 에는 GStreamer 지원이 없다(빌드에서 빠져 있다).
    그래서 cv2.VideoCapture(..., CAP_GSTREAMER) 를 쓰지 않고
    PyGObject 로 파이프라인을 직접 돌린 뒤 넘파이 배열로 받는다.

쓰는 법
    BREADVISION_CAM_BACKEND=GST 로 실행하면 CameraManager 가 이걸 쓴다.
"""
import threading

import numpy as np


class GstCapture:
    """cv2.VideoCapture 와 같은 방식으로 쓸 수 있는 최소 구현."""

    def __init__(self, device="/dev/video0", width=1280, height=720, fps=30):
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp     # noqa: F401  (GstApp 등록용)

        self._gst = Gst
        if not Gst.is_initialized():
            Gst.init(None)

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._lock = threading.Lock()
        self._opened = False

        desc = (
            f"v4l2src device={device} io-mode=2 ! "
            f"image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1 ! "
            # 하드웨어 JPEG 디코더. nvjpegdec 는 NVMM 밖으로 내보낼 때 실패하는
            # 경우가 있어 nvv4l2decoder 를 쓴다.
            "nvv4l2decoder mjpeg=1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
        )
        self._pipeline = Gst.parse_launch(desc)
        self._sink = self._pipeline.get_by_name("sink")
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer 파이프라인을 시작하지 못했습니다.")

        # 첫 프레임이 나올 때까지 기다려 본다. 안 나오면 실패로 본다.
        ok, _ = self.read(timeout_sec=5.0)
        if not ok:
            self.release()
            raise RuntimeError("GStreamer 로 프레임을 받지 못했습니다.")
        self._opened = True

    def isOpened(self):                      # noqa: N802  (cv2 이름 그대로)
        return self._pipeline is not None

    def read(self, timeout_sec=2.0):
        if self._pipeline is None:
            return False, None
        Gst = self._gst
        sample = self._sink.try_pull_sample(int(timeout_sec * Gst.SECOND))
        if sample is None:
            return False, None

        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w = caps.get_value("width")
        h = caps.get_value("height")
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            # 버퍼는 곧 반납되므로 복사해서 넘긴다.
            frame = np.frombuffer(info.data, dtype=np.uint8).reshape(h, w, 3).copy()
        finally:
            buf.unmap(info)
        return True, frame

    def get(self, prop):
        """cv2.CAP_PROP_* 중 화면에 쓰는 것만 흉내낸다."""
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop == cv2.CAP_PROP_FOURCC:
            return float(int.from_bytes(b"MJPG", "little"))
        return 0.0

    def set(self, prop, value):
        return False                         # 파이프라인을 만들 때 이미 정했다

    def release(self):
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None
            self._sink = None
