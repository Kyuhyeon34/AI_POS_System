"""박스 계산 — 겹침, 흔들림 다듬기, 라벨로 쓸 박스 고르기, YOLO 좌표 변환."""


def iou(a, b):
    """두 박스 (x1, y1, x2, y2) 가 얼마나 겹치는가 (0~1)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def yolo_line(class_id, xyxy, width, height):
    """픽셀 박스 → YOLO 라벨 한 줄 "cid cx cy w h" (0~1)."""
    x1, y1, x2, y2 = xyxy
    return (f"{class_id} {((x1 + x2) / 2) / width:.6f} {((y1 + y2) / 2) / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def center_guide_box(width, height, ratio=0.45):
    """AI 자동 라벨을 끈 경우의 고정 박스 — 화면 가운데 45%."""
    bw, bh = int(width * ratio), int(height * ratio)
    x1, y1 = (width - bw) // 2, (height - bh) // 2
    return x1, y1, x1 + bw, y1 + bh


class SteadyBox:
    """프레임마다 박스가 튀지 않게 다듬는다 (수집 탭 「한 개」 방식).

    · 가까운 자리면 조금씩 따라간다 (떨림 제거)
    · 엉뚱한 곳이면 몇 프레임 계속 나올 때만 옮긴다 (튐 방지)
    · 잠깐 놓치면 직전 박스를 유지한다 (깜빡임 방지)
    """

    FOLLOW = 0.35        # 새 박스를 얼마나 따라갈지 (작을수록 덜 흔들린다)
    JUMP_IOU = 0.30      # 이만큼도 안 겹치면 "다른 물체" 로 본다
    JUMP_FRAMES = 3      # 다른 물체라도 이만큼 계속 나오면 그때 옮긴다
    KEEP_FRAMES = 5      # 잠깐 놓쳐도 이만큼은 이전 박스를 유지한다

    def __init__(self):
        self.reset()

    def reset(self):
        self._box = None
        self._miss = 0
        self._pending = 0

    def update(self, box):
        prev = self._box
        if box is None:
            self._miss += 1
            if prev is not None and self._miss <= self.KEEP_FRAMES:
                return tuple(int(round(v)) for v in prev)
            self._box = None
            return None
        self._miss = 0
        box = tuple(float(v) for v in box)
        if prev is None:
            self._box, self._pending = box, 0
        elif iou(prev, box) >= self.JUMP_IOU:
            a = self.FOLLOW
            self._box = tuple(prev[i] * (1 - a) + box[i] * a for i in range(4))
            self._pending = 0
        else:
            self._pending += 1
            if self._pending >= self.JUMP_FRAMES:
                self._box, self._pending = box, 0
        return tuple(int(round(v)) for v in self._box)


def pick_label_box(detections, frame_shape, min_conf, min_area, max_area):
    """검출 결과 중 라벨로 쓸 만한 박스 하나. 쓸 수 없으면 (None, 이유).

    클래스는 모델 판정이 아니라 사용자가 고른 품목을 쓸 것이므로 여기서는 위치만 본다.
    가운데에 놓고 찍으므로 화면 중심에 가장 가까운 것을 고른다.
    """
    if not detections:
        return None, "no bread found"

    h, w = frame_shape[:2]
    cx0, cy0 = w / 2.0, h / 2.0
    ok = [d for d in detections if float(d.get("confidence", 0)) >= min_conf]
    if ok:
        best = min(ok, key=lambda d: ((d["xyxy"][0] + d["xyxy"][2]) / 2 - cx0) ** 2
                                     + ((d["xyxy"][1] + d["xyxy"][3]) / 2 - cy0) ** 2)
    else:
        best = max(detections, key=lambda d: d["confidence"])
    if best["confidence"] < min_conf:
        return None, f"low conf {best['confidence']:.2f}"

    x1, y1, x2, y2 = best["xyxy"]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None, "bad box"

    area = (bw * bh) / float(w * h)
    if area < min_area:
        return None, f"too small {area * 100:.1f}%"
    if area > max_area:
        return None, f"too big {area * 100:.0f}%"
    return best, f"conf {best['confidence']:.2f}"
