"""카메라 영상 서버 — 젯슨에 꽂힌 USB 카메라 영상을 네트워크(MJPEG)로 내보낸다.

PC 의 POS 프로그램이 이 영상을 받아 화면에 그린다. 보통은 ./jetson/start.sh 가 켠다.
    python3 pc_camera_stream.py --camera 0 --width 640 --height 480 --port 8090
카메라 번호가 바뀌어도(다시 꽂음) /dev/video* 를 훑어 알아서 찾고, 빠지면 다시 연결한다.
브라우저로 http://<젯슨IP>:8090/stream 을 열면 영상이 나가는지 바로 확인된다.
"""
import argparse
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import glob

import cv2

BOUNDARY = "breadvision"
latest = {"jpg": None}
args = None


def _try_open(index):
    """그 번호를 열어 실제로 프레임이 나오는지까지 본다.

    "열리기는 하는데 프레임이 안 나오는" 장치가 있다.
    USB 카메라 하나가 /dev/video1(영상) 과 /dev/video2(메타데이터) 처럼
    두 개로 잡히는데, 뒤엣것은 열려도 그림이 안 나온다.
    isOpened() 만 보면 그걸 골라 잡아 정지 화면이 나간다.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    for _ in range(6):                       # 첫 몇 장은 비어 나오기도 한다
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return cap
    cap.release()
    return None


def open_camera():
    """쓸 수 있는 카메라를 찾는다.

    카메라를 뽑았다 꽂으면 번호가 바뀐다 (video0 → video1).
    번호를 박아 두면 그때마다 정지 화면이 나가고, 포트는 열려 있어
    겉으로는 정상으로 보여서 알아채기 어렵다.
    그래서 지정한 번호를 먼저 시도하고, 안 되면 실제로 있는 장치를 훑는다.
    """
    tried = []
    order = []
    if isinstance(args.camera, int) or str(args.camera).isdigit():
        order.append(int(args.camera))
    for path in sorted(glob.glob("/dev/video*")):
        try:
            n = int(path.replace("/dev/video", ""))
        except ValueError:
            continue
        if n not in order:
            order.append(n)

    for n in order:
        cap = _try_open(n)
        tried.append(n)
        if cap is not None:
            if tried[0] != n:
                print(f"카메라 {tried[0]} 번이 안 되어 {n} 번으로 바꿨습니다 "
                      f"(다시 꽂으면 번호가 바뀝니다)")
            print(f"카메라 {n} 번 열림 "
                  f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
            return cap

    return None


def _open_any():
    if isinstance(args.camera, str) and not str(args.camera).isdigit():
        cap = cv2.VideoCapture(args.camera)
        return cap if cap.isOpened() else None
    return open_camera()


def grabber():
    """카메라를 붙잡고 최신 프레임을 JPEG 로 만들어 둔다.

    카메라가 없거나 도중에 빠져도 죽지 않는다.
    전에는 못 찾으면 SystemExit 을 던졌는데, 이 함수는 스레드라서
    스레드만 죽고 HTTP 서버는 계속 살아 있었다. 포트가 열려 있어
    겉으로는 정상인데 프레임을 안 줘서 PC 가 시간 초과로 끊겼다.
    지금은 카메라가 나타날 때까지 기다렸다가 붙고, 빠지면 다시 찾는다.
    """
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    waited = 0
    while True:
        cap = _open_any()
        if cap is None:
            if waited % 10 == 0:            # 10초마다 한 번만 알린다
                print("카메라를 기다리는 중… (USB 를 확인하세요)", flush=True)
            waited += 1
            latest["jpg"] = None
            time.sleep(1.0)
            continue
        waited = 0

        fails = 0
        n, t0 = 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails >= 40:             # 약 2초 연속 실패 → 빠진 것으로 본다
                    print("카메라 신호가 끊겼습니다. 다시 찾습니다.", flush=True)
                    latest["jpg"] = None
                    cap.release()
                    break
                time.sleep(0.05)
                continue
            fails = 0
            ok, buf = cv2.imencode(".jpg", frame, enc)
            if ok:
                latest["jpg"] = buf.tobytes()
                n += 1
                if n % 150 == 0:
                    d = time.time() - t0
                    kb = len(latest["jpg"]) / 1024
                    print(f"  보내는 중: {n/d:.1f} fps, 프레임당 {kb:.0f} KB "
                          f"(약 {kb*8*n/d/1024:.1f} Mbps)")
                    n, t0 = 0, time.time()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path not in ("/stream", "/"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        print(f"접속됨: {self.client_address[0]}")
        try:
            while True:
                jpg = latest["jpg"]
                if jpg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / args.fps)
        except (BrokenPipeError, ConnectionResetError):
            print(f"연결 끊김: {self.client_address[0]}")


def my_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80, help="JPEG 화질 1~100")
    args = ap.parse_args()

    threading.Thread(target=grabber, daemon=True).start()
    ip = my_ip()
    print(f"\n스트림 주소: http://{ip}:{args.port}/stream")
    print("PC 의 POS 프로그램(run_pc.bat)이 이 영상을 자동으로 받습니다.\n")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
