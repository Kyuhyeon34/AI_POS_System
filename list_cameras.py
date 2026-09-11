"""연결된 카메라 번호를 찾아준다.

PC 든 젯슨이든 그냥 실행하면 된다:
    py -3.11 list_cameras.py          (Windows)
    python3 list_cameras.py           (젯슨/Linux)

출력에서 쓰고 싶은 카메라의 번호를 골라:
    $env:BREADVISION_CAMERA="1"       (Windows PowerShell)
    BREADVISION_CAMERA=1 python3 main.py   (Linux)

--show 를 붙이면 각 카메라의 첫 화면을 창으로 띄워 확인할 수 있다.
"""
import argparse
import sys

import cv2

from camera.camera import _backends

MAX_INDEX = 8


def probe(index, api_name, api, want_frame=True):
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    info = None
    if ok and frame is not None:
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        info = {"index": index, "backend": api_name, "w": w, "h": h,
                "fps": fps, "frame": frame if want_frame else None}
    cap.release()
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="찾은 카메라의 화면을 창으로 띄운다")
    ap.add_argument("--max", type=int, default=MAX_INDEX)
    args = ap.parse_args()

    print(f"카메라를 찾는 중... (0~{args.max - 1} 번, OS: {sys.platform})\n")
    found = []
    for i in range(args.max):
        for name, api in _backends():
            info = probe(i, name, api, want_frame=args.show)
            if info:
                found.append(info)
                print(f"  [{i}] {name:12s} {info['w']}x{info['h']} "
                      f"@ {info['fps']:.0f}fps  ← 사용 가능")
                break                      # 이 번호는 확인됐으니 다음 번호로
        else:
            pass

    if not found:
        print("사용 가능한 카메라를 찾지 못했습니다.")
        print("  - 다른 프로그램(줌, 팀즈, 카메라 앱)이 쓰고 있지 않은지 확인하세요.")
        print("  - USB 웹캠이면 다시 꽂아보세요.")
        sys.exit(1)

    print(f"\n{len(found)}개 찾음. 보통 [0] 이 노트북 내장캠, 뒤 번호가 USB 웹캠입니다.")
    print("어느 것인지 확실하지 않으면 --show 로 화면을 보고 고르세요:")
    print("    py -3.11 list_cameras.py --show")

    if args.show:
        print("\n창에서 아무 키나 누르면 다음 카메라로 넘어갑니다.")
        for info in found:
            title = f"[{info['index']}] {info['backend']} {info['w']}x{info['h']}"
            cv2.imshow(title, info["frame"])
            cv2.waitKey(0)
            cv2.destroyWindow(title)
