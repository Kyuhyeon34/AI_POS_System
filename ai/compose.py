"""오려낸 빵들을 트레이 배경에 흩어 놓아 학습 사진을 만든다.

라벨은 사람이 안 그린다. 어디에 무엇을 붙였는지 우리가 알고 있으므로
붙인 마스크에서 박스가 바로 나온다. 그래서 라벨이 항상 정확하다.

"붙여넣은 티" 를 지우는 게 핵심이다
    그냥 얹으면 모델이 빵이 아니라 '칼로 자른 듯한 경계' 를 배운다.
    합성본에서만 잘 맞고 실제 매대에서는 못 잡는 모델이 된다. 그래서
      · 가장자리를 1.5px 흐린다 (제일 큰 단서)
      · 바닥에 그림자를 깐다
      · 밝기·색을 흔든다 (조명이 매번 같지 않다)
      · 회전·크기를 무작위로 준다
"""
import random

import cv2
import numpy as np


def _place(canvas, obj, mask, cx, cy, angle, scale, gain,
           placed=(), max_iou=0.20):
    """한 개를 붙이고 (x1,y1,x2,y2) 를 돌려준다.

    화면 밖이거나, 이미 놓인 것과 max_iou 보다 많이 겹치면 붙이지 않고 None.
    실측(실제 val)에서 겹침 중앙값이 0.043 인데 제한 없이 뿌리면 0.137 까지
    올라간다. 실제보다 심하게 겹쳐 놓고 배우면 매대에서 오히려 못 잡는다.
    """
    h, w = obj.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    if nw < 8 or nh < 8:
        return None
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    o = cv2.warpAffine(obj, M, (nw, nh), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST)
    if m.max() == 0:
        return None

    o = np.clip(o.astype(np.float32) * gain, 0, 255)

    x1, y1 = int(cx - nw / 2), int(cy - nh / 2)
    x2, y2 = x1 + nw, y1 + nh
    H, W = canvas.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None

    ys, xs = np.where(m > 0)
    box = (x1 + int(xs.min()), y1 + int(ys.min()),
           x1 + int(xs.max()), y1 + int(ys.max()))

    for q in placed:
        if _iou(box, q) > max_iou:
            return None

    _blend(canvas, o, m, x1, y1, x2, y2)
    return box


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def _blend(canvas, o, m, x1, y1, x2, y2):
    """가장자리를 흐리고 그림자를 깔아 붙인다."""
    alpha = (cv2.GaussianBlur(m, (0, 0), 1.5).astype(np.float32) / 255.0)[..., None]
    shadow = cv2.GaussianBlur(m, (0, 0), 9).astype(np.float32) / 255.0 * 0.21
    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    roi *= (1.0 - shadow[..., None])
    canvas[y1:y2, x1:x2] = (roi * (1 - alpha) + o * alpha).astype(np.uint8)


def compose(background, library, n_min=1, n_max=4, tries=60,
            scale=(0.85, 1.55), max_iou=0.10, weights=None, rng=None):
    """(사진, [(class_id, x1,y1,x2,y2), ...]) 를 만든다.

    library: {class_id: [(그림, 마스크), ...]}
    """
    rng = rng or random
    canvas = background.copy()
    H, W = canvas.shape[:2]
    labels = []

    ids = list(library)
    if not ids:
        return canvas, labels
    n = rng.randint(n_min, n_max)
    # 같은 빵이 여러 개 놓이는 경우도 실제로 흔하다.
    #
    # weights 를 주면 그 비율대로 고른다. 무작위로만 고르면 품목마다 등장 횟수가
    # 운에 좌우되는데, 신상품이 적게 등장하면 그 품목만 재현율이 떨어진다.
    # (실측: 등장 115회 → 재현율 0.82, 231회 → 0.98)
    if weights:
        w = [max(1e-6, float(weights.get(c, 1.0))) for c in ids]
        picks = rng.choices(ids, weights=w, k=n)
    else:
        picks = [rng.choice(ids) for _ in range(n)]

    boxes = []
    for cid in picks:
        obj, mask = rng.choice(library[cid])
        for _ in range(tries):
            box = _place(canvas, obj, mask,
                         rng.randint(int(W * 0.15), int(W * 0.85)),
                         rng.randint(int(H * 0.15), int(H * 0.85)),
                         rng.uniform(0, 360), rng.uniform(*scale),
                         rng.uniform(0.85, 1.12),
                         placed=boxes, max_iou=max_iou)
            if box is not None:
                labels.append((cid,) + box)
                boxes.append(box)
                break
    return canvas, labels


def to_yolo(labels, width, height):
    """(cid,x1,y1,x2,y2) 목록을 YOLO 라벨 줄로."""
    out = []
    for cid, x1, y1, x2, y2 in labels:
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        bw = (x2 - x1) / float(width)
        bh = (y2 - y1) / float(height)
        if bw <= 0 or bh <= 0:
            continue
        out.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out
