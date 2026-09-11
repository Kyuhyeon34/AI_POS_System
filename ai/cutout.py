"""사진에서 빵만 오려낸다. 합성 학습 데이터의 재료.

왜 필요한가
    신상품이 나오면 업체가 빵 하나만 턴테이블에 올리고 버튼을 누른다.
    한 개뿐이라 라벨이 명확하다 — 고르고 지정할 일이 없다.

    그런데 그 사진만으로 학습하면 매대에서 겹친 빵을 못 잡는다.
    (실측: 한 개짜리 329장을 넣었더니 겹친 빵 검출률 87.5% → 54.2%)

    그래서 오려내서 여러 개로 합성한다. 찍는 건 한 개, 학습하는 건 여러 개.

박스가 아니라 마스크로 오리는 이유
    박스는 배경을 절반까지 물고 온다 (파이는 박스의 49% 만 실제 빵이었다).
    그대로 겹쳐 붙이면 뒤 빵이 네모나게 잘려 보인다.
"""
import cv2
import numpy as np


def object_mask(patch):
    """잘라온 조각에서 빵 부분만 남긴 마스크. 못 찾으면 None.

    흰 유산지는 채도가 낮고 밝다. 빵은 그 반대다. ai/segment.py 와 같은 기준.
    """
    if patch.size == 0 or min(patch.shape[:2]) < 8:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    score = np.clip(sat + (255 - val) // 2, 0, 255).astype(np.uint8)
    score = cv2.GaussianBlur(score, (0, 0), 2)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    # 가장 큰 덩어리 하나만. 부스러기와 그림자를 버린다.
    i = 1 + int(np.argmax(stats[1:, 4]))
    mask = (lab == i).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # 속에 뚫린 구멍을 메운다.
    # 빵 속의 밝고 흐린 부분(바게트 속살, 소금빵 결)은 배경과 색이 비슷해서
    # 구멍으로 남는다. 그대로 붙이면 빵 한가운데로 배경이 비쳐 보이고,
    # 모델은 빵이 아니라 그 구멍 무늬를 배운다.
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return filled


def cut(image, xyxy, pad=0.06, min_fill=0.25, max_fill=0.97):
    """라벨 박스에서 빵을 오려낸다. (그림, 마스크) 또는 None.

    min_fill 아래면 마스크가 빵을 놓친 것이고,
    max_fill 위면 배경까지 통째로 잡은 것이다. 둘 다 버린다 —
    나쁜 재료로 합성하면 나쁜 학습 데이터가 그대로 나온다.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1 = max(0, int(x1 - px)); y1 = max(0, int(y1 - py))
    x2 = min(w, int(x2 + px)); y2 = min(h, int(y2 + py))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    patch = image[y1:y2, x1:x2]
    mask = object_mask(patch)
    if mask is None:
        return None

    fill = float(mask.mean()) / 255.0
    if not (min_fill <= fill <= max_fill):
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    a, b, c, d = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return patch[c:d, a:b].copy(), mask[c:d, a:b].copy()
