"""배경에서 물체를 분리해 박스를 잡는다. (모델이 모르는 신메뉴용)

학습된 모델은 아는 빵만 찾는다. 신메뉴는 모델이 못 볼 수 있으므로,
트레이(흰 종이/밝은 배경) 위의 물체를 색·밝기로 직접 분리한다.
클래스는 판단하지 않는다 — 사용자가 고른 품목을 쓴다.
"""
import cv2
import numpy as np


def _blobs(frame):
    """배경에서 갈라낸 덩어리들을 (x, y, w, h, area) 로 돌려준다. 큰 것부터.

    흰 유산지·밝은 트레이 위에서는 채도와 어두움으로 물체가 갈린다.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)

    # 배경(흰 종이)은 채도가 낮고 밝다. 물체는 그 반대.
    score = np.clip(sat + (255 - val) // 2, 0, 255).astype(np.uint8)
    score = cv2.GaussianBlur(score, (0, 0), 3)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(5, (min(h, w) // 100) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    # 화면 가장자리를 통째로 감싸는 덩어리(책상·그림자)는 물체가 아니다
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x <= 2 and y <= 2 and x + bw >= w - 2 and y + bh >= h - 2:
            continue
        out.append((int(x), int(y), int(bw), int(bh), int(area)))
    out.sort(key=lambda b: -b[4])
    return out


def _pad(x, y, bw, bh, w, h, margin):
    """경계가 잘리지 않게 살짝 여유를 준다."""
    mx, my = int(bw * margin), int(bh * margin)
    return (max(0, x - mx), max(0, y - my),
            min(w - 1, x + bw + mx), min(h - 1, y + bh + my))


def object_box(frame, min_area=0.01, max_area=0.85, margin=0.02):
    """가장 큰 물체 하나. (박스, 사유, 후보박스) 를 돌려준다.

    박스가 None 이면 사유를 보고, 후보박스가 있으면 "무엇을 잡았는데 왜 버렸는지" 를
    화면에 보여줄 수 있다.
    """
    h, w = frame.shape[:2]
    blobs = _blobs(frame)
    if not blobs:
        return None, "no object", None

    x, y, bw, bh, _area = blobs[0]
    candidate = (x, y, x + bw, y + bh)
    ratio = (bw * bh) / float(w * h)
    if ratio < min_area:
        return None, f"too small {ratio * 100:.1f}%", candidate
    if ratio > max_area:
        return None, f"too big {ratio * 100:.0f}% (background caught)", candidate
    return _pad(x, y, bw, bh, w, h, margin), "ok", candidate


def object_boxes(frame, min_area=0.01, max_area=0.85, margin=0.02, limit=12):
    """화면에 있는 물체 후보를 **전부** 돌려준다. 왼쪽에 있는 것부터.

    한 장에 여러 개를 라벨링할 때 쓴다. 순서를 왼쪽부터로 고정하는 이유는,
    사용자가 「다음 물체」로 포커스를 옮길 때 번호가 프레임마다 뒤바뀌면
    무엇을 고르고 있는지 알 수 없기 때문이다.
    """
    h, w = frame.shape[:2]
    boxes = []
    for x, y, bw, bh, _area in _blobs(frame):
        ratio = (bw * bh) / float(w * h)
        if ratio < min_area or ratio > max_area:
            continue
        boxes.append(_pad(x, y, bw, bh, w, h, margin))
        if len(boxes) >= limit:
            break
    boxes.sort(key=lambda b: (b[0] + b[2]) / 2)      # 왼쪽부터
    return boxes
