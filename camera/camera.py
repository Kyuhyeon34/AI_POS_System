import os
import sys
import threading

import cv2


GST = "GST"        # cv2 백엔드가 아니라 우리 파이프라인이라는 표시


def _is_jetson():
    try:
        with open("/proc/device-tree/model", "rb") as f:
            return b"NVIDIA" in f.read() or b"Jetson" in f.read()
    except OSError:
        return False


def _backends():
    """OS 별로 시도할 캡처 백엔드 순서. (첫 번째가 그 OS 에서 가장 잘 맞는 것)"""
    if sys.platform.startswith("win"):
        # Windows: DirectShow 가 USB 웹캠에서 가장 안정적. MSMF 는 카메라에 따라 실패한다.
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("기본", cv2.CAP_ANY)]
    if sys.platform == "darwin":
        return [("AVFoundation", cv2.CAP_AVFOUNDATION), ("기본", cv2.CAP_ANY)]
    if _is_jetson():
        # 젯슨에서는 JPEG 디코딩을 전용 하드웨어에 넘긴다.
        # 실측(1280x720 30fps): CPU 44.7% → 26.5%
        # 실패하면 아래 V4L2 로 자동으로 넘어간다.
        return [("GST", GST), ("V4L2", cv2.CAP_V4L2), ("기본", cv2.CAP_ANY)]
    return [("V4L2", cv2.CAP_V4L2), ("기본", cv2.CAP_ANY)]


class CameraManager:
    """
    Jetson/Linux 전용 Logitech/USB Camera manager.
    기본 장치: /dev/video0
    우선 backend: V4L2

    캡처는 별도 스레드에서 돈다.
    cap.read() 는 카메라 속도만큼(15fps 카메라면 약 67ms) 블로킹되는데,
    이걸 GUI 타이머 안에서 부르면 그 시간 동안 화면과 버튼이 모두 멈춘다.
    스레드가 최신 프레임만 계속 갱신해 두고, read() 는 그걸 즉시 돌려준다.
    """

    # 기본 640x480.
    #   모델 입력이 640x640 이라 720p 로 받아도 인식 정확도는 그대로다.
    #   반면 720p 는 픽셀이 2.25배라 디코딩·화면변환·전송이 모두 그만큼 비싸다.
    #   실측(젯슨, 미리보기+실시간인식):
    #     1280x720 → 화면 26.1 fps, 타이머 지연 44 ms
    #      640x480 → 화면 29.9 fps, 타이머 지연 33 ms (= 지연 없음)
    #   더 선명한 미리보기가 필요하면 BREADVISION_CAM_SIZE=1280x720 으로 되돌린다.
    def __init__(self, index=0, width=640, height=480, fps=30, use_env=True):
        # index 에 문자열 URL 을 주면 네트워크 카메라로 동작한다.
        #   예) CameraManager("http://192.168.0.5:8090/stream")   ← PC 웹캠 중계
        #       CameraManager("rtsp://...")                        ← IP 카메라
        # 환경변수 BREADVISION_CAMERA 가 있으면 그 값을 우선 쓴다.
        # use_env=False 면 무시한다 (GUI 에서 사람이 카메라를 직접 고른 경우).
        env = os.environ.get("BREADVISION_CAMERA", "").strip() if use_env else ""
        if env:
            index = int(env) if env.isdigit() else env
        self.index = index

        # 해상도/코덱도 환경변수로 바꿀 수 있다. 카메라가 MJPG 를 안 받아주면
        # 무압축으로 받게 되어 720p 에서 7fps 까지 떨어지는 일이 흔하다.
        #   BREADVISION_CAM_SIZE=640x480      해상도 지정
        #   BREADVISION_CAM_FOURCC=none       MJPG 요청 끄기 (기본은 MJPG)
        size = os.environ.get("BREADVISION_CAM_SIZE", "").strip().lower()
        if "x" in size:
            try:
                w, h = size.split("x")
                width, height = int(w), int(h)
            except ValueError:
                print(f"[카메라] BREADVISION_CAM_SIZE 형식이 잘못됨: {size} (예: 640x480)")
        self.fourcc = os.environ.get("BREADVISION_CAM_FOURCC", "MJPG").strip().upper()
        #   BREADVISION_CAM_BACKEND=MSMF    백엔드 강제 (DSHOW / MSMF / V4L2 / ANY)
        self.force_backend = os.environ.get("BREADVISION_CAM_BACKEND", "").strip().upper()
        self.is_network = isinstance(index, str)
        self.backend = None
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._frame = None
        self._frame_id = 0          # 새 프레임인지 판별용
        self._read_fail = 0

    def start(self):
        self.stop()

        if self.is_network:
            # 네트워크 스트림(MJPEG/RTSP). 해상도/포맷은 보내는 쪽이 정한다.
            self.cap = cv2.VideoCapture(self.index)
            if not self.cap.isOpened():
                self.cap = None
                raise RuntimeError(
                    f"스트림을 열 수 없습니다: {self.index}\n"
                    "보내는 쪽 프로그램이 켜져 있는지, 주소와 포트가 맞는지 확인하세요."
                )
        else:
            # OS 에 맞는 백엔드를 차례로 시도한다.
            # "열리기는 했는데 프레임이 안 나오는" 경우가 있어서, 실제로 한 장 받아봐야 안다.
            last = None
            candidates = _backends()
            if self.force_backend:
                picked = [(n, a) for n, a in candidates if n.upper() == self.force_backend]
                if picked:
                    candidates = picked
                else:
                    print(f"[카메라] 알 수 없는 백엔드: {self.force_backend} "
                          f"(가능: {', '.join(n for n, _ in candidates)})")
            for name, api in candidates:
                if api is GST:
                    if isinstance(self.index, str):
                        continue
                    try:
                        from camera.gst_capture import GstCapture
                        cap = GstCapture(f"/dev/video{self.index}",
                                         self.width, self.height, self.fps)
                    except Exception as e:
                        print(f"[카메라] GStreamer 실패 → 다음 방식으로 넘어갑니다: {e}")
                        last = name
                        continue
                else:
                    cap = cv2.VideoCapture(self.index, api)
                    if not cap.isOpened():
                        cap.release()
                        continue
                    self._apply_settings(cap)
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.cap = cap
                    self.backend = name
                    self._frame = frame
                    self._frame_id = 1
                    break

                cap.release()
                last = name

            if self.cap is None:
                hint = (
                    "카메라 번호를 확인하려면:  python list_cameras.py"
                    if not sys.platform.startswith("linux")
                    else "ls /dev/video* 와 v4l2-ctl --list-devices 를 확인하세요."
                )
                raise RuntimeError(
                    f"{self.index} 번 카메라를 열 수 없습니다."
                    + (f" ({last} 까지 시도)" if last else "")
                    + f"\n다른 프로그램이 카메라를 쓰고 있지 않은지 확인하세요.\n{hint}"
                )

        # CAP_PROP_BUFFERSIZE 는 건드리지 않는다.
        # 1로 줄이면 드라이버가 다음 프레임을 미리 받아둘 자리가 없어져서,
        # 디코딩하는 동안 들어오는 프레임을 통째로 놓친다 → 30fps 카메라가 15fps 가 된다.
        # 기본 버퍼로 두면 캡처와 디코딩이 겹쳐 돌아 30fps 가 그대로 나오고,
        # 이 캡처 스레드가 쉬지 않고 빼가므로 프레임이 밀려 쌓이지도 않는다.
        #   실측(1280x720 MJPG): BUFFERSIZE=1 → 15.2fps / 기본 → 30.2fps

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _apply_settings(self, cap):
        """해상도/포맷 지정. 카메라가 거부해도 그냥 진행한다 (기본값으로 동작)."""
        try:
            if self.fourcc and self.fourcc != "NONE":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc[:4]))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception:
            pass

    @staticmethod
    def describe(cap):
        """카메라가 실제로 어떤 설정으로 열렸는지. (요청과 다를 수 있다)"""
        cc = int(cap.get(cv2.CAP_PROP_FOURCC))
        chars = [chr((cc >> 8 * i) & 0xFF) for i in range(4)]
        if all(32 <= ord(c) < 127 for c in chars):
            name = "".join(chars).strip() or "?"
        else:
            name = f"0x{cc:08X}"          # 출력할 수 없는 값이면 숫자로 보여준다
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), name)

    def _loop(self):
        while self._running:
            cap = self.cap
            if cap is None:
                break
            ok, frame = cap.read()
            if not ok:
                self._read_fail += 1
                if self._read_fail > 60:      # 카메라가 빠진 경우
                    break
                continue
            self._read_fail = 0
            with self._lock:
                self._frame = frame           # cap.read() 는 매번 새 배열을 준다
                self._frame_id += 1

    def read(self):
        """블로킹하지 않는다. 가장 최근 프레임을 즉시 돌려준다."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame

    def read_latest(self):
        """(ok, frame, frame_id) — 같은 프레임을 두 번 그리지 않으려고 id 를 같이 준다."""
        with self._lock:
            if self._frame is None:
                return False, None, self._frame_id
            return True, self._frame, self._frame_id

    def stop(self):
        self._running = False
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        with self._lock:
            self._frame = None

    # aliases for convenience
    open = start
    release = stop

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()
