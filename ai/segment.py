"""배경에서 물체를 분리해 박스를 잡는다. (모델이 모르는 신메뉴용)

학습된 모델은 아는 빵만 찾는다. 신메뉴는 모델이 못 볼 수 있으므로,
트레이(흰 종이/밝은 배경) 위의 물체를 색·밝기로 직접 분리한다.
클래스는 판단하지 않는다 — 사용자가 고른 품목을 쓴다.
"""
import cv2
import numpy as np


def _blobs(frame, roi=0.0):
    """배경에서 갈라낸 덩어리들을 (x, y, w, h, area) 로 돌려준다. 큰 것부터.

    흰 유산지·밝은 트레이 위에서는 채도와 어두움으로 물체가 갈린다.

    roi 를 주면 "무엇이 물체인가" 하는 기준값을 **화면 가운데 그 범위 안에서만**
    정한다. 촬영대 밖의 어두운 책상·케이블까지 함께 보면 기준이 그쪽에 끌려가서,
    밝은 색 물체(연분홍·크림색 빵 등)가 통째로 배경으로 처리된다.
    (2026-09-13 실측: 전체 기준 130 → 분홍 물체(87)를 놓침 / 가운데 기준 → 잡음)
    기준만 가운데에서 정하고, 찾기는 화면 전체에서 한다 (물체가 잘리지 않게).
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)

    # 배경(흰 종이)은 채도가 낮고 밝다. 물체는 그 반대.
    score = np.clip(sat + (255 - val) // 2, 0, 255).astype(np.uint8)
    score = cv2.GaussianBlur(score, (0, 0), 3)

    area = score
    if roi > 0:
        rx, ry = int(w * (1 - roi) / 2), int(h * (1 - roi) / 2)
        area = score[ry:h - ry, rx:w - rx]
        if area.size < 100:
            area = score
    thr, _ = cv2.threshold(area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(score, thr, 255, cv2.THRESH_BINARY)

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


def object_box(frame, min_area=0.01, max_area=0.85, margin=0.02, roi=0.0):
    """가운데에 있는 물체 하나. (박스, 사유, 후보박스) 를 돌려준다.

    "가장 큰 덩어리" 를 고르면 그림자·케이블·트레이 모서리가 커 보이는 순간마다
    박스가 그쪽으로 튄다. 촬영대 가운데에 물체를 놓고 찍으므로,
    화면 중심에 가장 가까운 덩어리를 고르는 편이 훨씬 안정적이다.

    roi: 0 보다 크면 화면 가운데 그 비율(예: 0.6) 안에 중심이 있는 덩어리만 본다.
    """
    h, w = frame.shape[:2]
    cx0, cy0 = w / 2.0, h / 2.0
    blobs = _blobs(frame, roi=roi)
    if not blobs:
        return None, "no object", None

    inside = blobs
    if roi > 0:
        rw, rh = w * roi / 2.0, h * roi / 2.0
        inside = [b for b in blobs
                  if abs(b[0] + b[2] / 2.0 - cx0) <= rw and abs(b[1] + b[3] / 2.0 - cy0) <= rh]
        if not inside:
            x, y, bw, bh, _ = blobs[0]
            return None, "not in center", (x, y, x + bw, y + bh)

    # 크기가 맞는 것 중 화면 중심에 가장 가까운 것
    def dist(b):
        return ((b[0] + b[2] / 2.0 - cx0) ** 2 + (b[1] + b[3] / 2.0 - cy0) ** 2) ** 0.5
    ok = [b for b in inside if min_area <= (b[2] * b[3]) / float(w * h) <= max_area]
    if not ok:
        x, y, bw, bh, _ = min(inside, key=dist)
        candidate = (x, y, x + bw, y + bh)
        ratio = (bw * bh) / float(w * h)
        if ratio < min_area:
            return None, f"too small {ratio * 100:.1f}%", candidate
        return None, f"too big {ratio * 100:.0f}% (background caught)", candidate

    x, y, bw, bh, _area = min(ok, key=dist)
    return _pad(x, y, bw, bh, w, h, margin), "ok", (x, y, x + bw, y + bh)


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
