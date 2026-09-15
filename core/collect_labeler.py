"""수집 탭 「여러 개」 방식 — 화면의 빵 전부에 라벨을 붙이고, 사람이 이름을 바로잡는다.

한 장에 빵이 여러 개면 전부 라벨을 달아야 한다. 하나라도 빠지면
YOLO 가 그것을 배경으로 학습해 앞으로 못 잡게 된다.

박스는 두 경로에서 온다.
    source="model"  모델이 빵이라고 잡은 것. 클래스가 이미 있다.
                    데이터에 없는 새 빵은 여기서 "비슷한 기존 빵" 으로 잘못 불린다.
    source="new"    모델이 못 본 덩어리 (배경 분리). 품목을 주거나 무시해야 저장된다.
    source="lock"   고정한 빵을 추적기가 따라가는 자리 (검출이 끊긴 프레임)

화면(Qt)과 떨어뜨려 두어 추적 규칙을 테스트할 수 있게 했다.
"""
import cv2
import numpy as np

from ai.segment import object_boxes
from core.boxes import iou
from core.log import get_logger
from core.products import display_name

log = get_logger("collect")

AUTOLABEL_CONF = 0.15       # 이보다 낮은 검출은 화면에도 안 그린다
LOW_CONF = 0.55             # 15~55% 노랑, 55% 이상 초록 (사이에 빈 구간 없음)
AUTOLABEL_MIN_AREA = 0.01   # 화면의 1% 미만이면 오검출로 본다
AUTOLABEL_MAX_AREA = 0.85   # 85% 초과면 배경을 잡은 것으로 본다


def make_tracker(frame, box):
    """고정한 빵을 그림으로 따라가는 추적기. 못 만들면 None.

    검출만으로는 안 된다. 신뢰도가 55% 근처에서 흔들리면 초록↔노랑을
    오가다 아예 안 잡히는 프레임이 생기고, 그때 옆 빵이 그 자리를 차지하면
    빨강이 옮겨 붙는다. 추적기는 모델을 거치지 않고 그림만 보고 따라가므로
    인식이 끊긴 구간에서도 그 빵을 놓치지 않는다.

    KCF 를 쓴다. CSRT 가 더 정확하지만 젯슨에서 37ms/프레임이라 미리보기가
    느려진다 (KCF 24ms). 검출과 맞아떨어질 때마다 자리를 다시 잡아 주므로
    조금씩 밀리는 것은 따라잡힌다.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 8:
        return None
    try:
        t = cv2.TrackerKCF_create()
        t.init(frame, (x1, y1, w, h))
        return t
    except (AttributeError, cv2.error) as e:
        # AttributeError: opencv 빌드에 KCF 가 없는 경우 (contrib 미포함)
        log.info("추적기를 만들지 못해 검출만으로 따라갑니다: %s", e)
        return None


def track(tracker, frame):
    """추적기가 알려주는 이번 프레임의 자리. 놓쳤으면 None."""
    if tracker is None:
        return None
    try:
        ok, box = tracker.update(frame)
    except cv2.error as e:
        log.debug("추적 실패: %s", e)
        return None
    if not ok:
        return None
    x, y, w, h = box
    if w < 4 or h < 4:
        return None
    return (float(x), float(y), float(x + w), float(y + h))


def patch_sig(frame, box):
    """박스 안 그림을 8x8 로 줄인 값. 같은 빵인지 겉모습으로 확인하는 용도.

    위치와 크기만 보면 "내 빵이 옆으로 갔다" 와 "옆 빵이 내 자리로 왔다" 를
    구분할 수 없다. 턴테이블이 돌면 두 상황이 실제로 번갈아 일어난다.
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(box[0])))
    y1 = max(0, min(h - 1, int(box[1])))
    x2 = max(x1 + 1, min(w, int(box[2])))
    y2 = max(y1 + 1, min(h, int(box[3])))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    small = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32).ravel()


def sig_match(a, b):
    """두 겉모습이 얼마나 닮았나 (0~1). 비교할 수 없으면 막지 않는다(1.0)."""
    if a is None or b is None:
        return 1.0
    diff = float(np.abs(a - b).mean())
    return max(0.0, 1.0 - diff / 40.0)  # 채널당 평균 40 차이면 완전히 다른 것


class CollectLabeler:
    MATCH_IOU = 0.35        # 이 이상 겹치면 "같은 물체" 로 본다
    # 같은 빵도 움직이면 0.45 까지 떨어진다. 그래서 문턱은 낮게 두고,
    # 명백히 다른 것(초코머핀 vs 바게트 = 0.00)만 잘라낸다.
    LOCK_SIG_MIN = 0.35
    # 추적기와 이만큼 겹치는 검출만 믿는다. 추적기는 한 프레임 뒤처지므로
    # 진짜 같은 빵이라도 겹침이 0.75 언저리까지 떨어지는데, 옆 빵은 0.5 근처였다.
    LOCK_SNAP_IOU = 0.60
    LOCK_MAX_MISS = 12      # 이만큼 연달아 놓치면 따라가기를 멈춘다

    def __init__(self, segmenter=object_boxes):
        self.segmenter = segmenter      # 테스트에서 배경 분리를 바꿔 끼울 수 있게
        self.focus = 0                  # 「다음 ▶」 로 고르고 있는 박스 번호
        self.manual = []                # 사용자가 품목을 지정한 박스들 [(box, class_id)]
        self.ignored = []               # 빵이 아니라고 표시한 박스들 (책상·손 등)
        self.lock = None                # 촬영 대상으로 고정한 것 {xyxy, class_id, vel, sig, miss}
        self.tracker = None
        self.boxes = []                 # 이번 프레임의 라벨 후보

    # ── 한 프레임의 라벨 ────────────────────────────────────────────────
    def match_locked(self, lock_box, boxes, frame=None, sig=None):
        """고정한 물체가 이번 프레임의 어느 박스인지 찾는다. 없으면 None.

        중심 거리(자기 크기의 60% 안), 크기 비슷함(2.5배 이내), 겉모습(결정적)을 함께 본다.
        """
        lx1, ly1, lx2, ly2 = lock_box
        lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
        lw, lh = lx2 - lx1, ly2 - ly1
        larea = max(1.0, float(lw * lh))
        limit = 0.6 * max(lw, lh)
        if limit <= 0:
            return None

        best, best_score = None, 0.0
        for bx in boxes:
            x1, y1, x2, y2 = bx["xyxy"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
            if dist > limit:
                continue
            area = max(1.0, float((x2 - x1) * (y2 - y1)))
            ratio = min(area, larea) / max(area, larea)
            if ratio < 0.4:
                continue
            look = 1.0
            if sig is not None and frame is not None:
                look = sig_match(sig, patch_sig(frame, bx["xyxy"]))
                if look < self.LOCK_SIG_MIN:
                    continue                # 겉모습이 다르면 아무리 가까워도 아니다
            score = (1.0 - dist / limit) * ratio * look
            if score > best_score:
                best, best_score = bx, score
        return best

    def label_boxes(self, frame, detections):
        """이 프레임에 저장될 라벨들. [{xyxy, class_id, model_id, fixed, locked, source, conf}]"""
        h, w = frame.shape[:2]
        area_all = float(w * h)

        boxes = []
        for d in (detections or []):
            if d.get("confidence", 0) < AUTOLABEL_CONF:
                continue
            x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
            ratio = ((x2 - x1) * (y2 - y1)) / area_all
            if ratio < AUTOLABEL_MIN_AREA or ratio > AUTOLABEL_MAX_AREA:
                continue
            cid = int(d["class_id"])
            boxes.append({"xyxy": (x1, y1, x2, y2), "class_id": cid, "model_id": cid,
                          "fixed": False, "locked": False, "source": "model",
                          "conf": float(d.get("confidence", 0))})

        # 모델이 못 본 덩어리 = 아직 이름이 없는 물체
        for box in self.segmenter(frame, min_area=AUTOLABEL_MIN_AREA, max_area=AUTOLABEL_MAX_AREA):
            if any(iou(box, b["xyxy"]) >= self.MATCH_IOU for b in boxes):
                continue
            boxes.append({"xyxy": box, "class_id": None, "model_id": None,
                          "fixed": False, "locked": False, "source": "new", "conf": 0.0})

        # 빵이 아니라고 표시한 것은 뺀다. 움직였으면 자리도 갱신한다.
        if self.ignored:
            alive, kept_ignore = [], []
            for b in boxes:
                match = next((b["xyxy"] for ig in self.ignored
                              if iou(ig, b["xyxy"]) >= self.MATCH_IOU), None)
                if match is None:
                    alive.append(b)
                else:
                    kept_ignore.append(match)
            boxes, self.ignored = alive, kept_ignore

        # 사용자가 바꿔 준 품목을 따라붙인다. 빵을 움직이며 찍으므로
        # 겹침으로 같은 물체를 찾아 지정을 이어주고, 그 자리도 갱신한다.
        kept = []
        for old_box, cid in self.manual:
            best, best_iou = None, self.MATCH_IOU
            for b in boxes:
                v = iou(old_box, b["xyxy"])
                if v > best_iou:
                    best, best_iou = b, v
            if best is not None:
                best["class_id"] = cid
                best["fixed"] = True
                kept.append((best["xyxy"], cid))
        self.manual = kept

        if self.lock:
            boxes = self._follow_lock(frame, boxes)

        boxes.sort(key=lambda b: (b["xyxy"][0] + b["xyxy"][2]) / 2)   # 왼쪽부터
        return boxes

    def _follow_lock(self, frame, boxes):
        """촬영 대상 고정.

        1) 그림을 직접 따라가는 추적기가 기준이다 (검출과 무관하게 돈다)
        2) 검출은 자리를 다듬는 데만 쓴다 — 추적기와 충분히 겹치고 겉모습도 닮은 것 하나만
        3) 둘 다 놓치면 직전 속도만큼만 흘려보내고 자리를 지킨다
        4) 이름 없는 것(주황)은 화면에서도 라벨에서도 뺀다
        """
        lock = self.lock
        vx, vy = lock.get("vel", (0.0, 0.0))
        x1, y1, x2, y2 = lock["xyxy"]

        tracked = track(self.tracker, frame)

        if tracked is not None:
            # 추적기는 한 프레임 뒤처지므로 직전 속도만큼 앞으로 밀어 두고 잰다.
            ax1, ay1, ax2, ay2 = tracked
            aim = (ax1 + vx, ay1 + vy, ax2 + vx, ay2 + vy)
            sig = lock.get("sig")
            best, best_score = None, 0.0
            for cand in boxes:
                v = iou(aim, cand["xyxy"])
                if v < self.LOCK_SNAP_IOU:
                    continue
                look = sig_match(sig, patch_sig(frame, cand["xyxy"]))
                if look < self.LOCK_SIG_MIN:
                    continue
                score = v * look
                if score > best_score:
                    best, best_score = cand, score
        else:
            predicted = (x1 + vx, y1 + vy, x2 + vx, y2 + vy)
            best = self.match_locked(predicted, boxes, frame, lock.get("sig"))

        if best is not None:
            ox, oy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            nx1, ny1, nx2, ny2 = best["xyxy"]
            dx = (nx1 + nx2) / 2.0 - ox
            dy = (ny1 + ny2) / 2.0 - oy
            lock["vel"] = (0.5 * vx + 0.5 * dx, 0.5 * vy + 0.5 * dy)
            lock["xyxy"] = best["xyxy"]
            lock["miss"] = 0
            new_sig = patch_sig(frame, best["xyxy"])
            if new_sig is not None:
                old_sig = lock.get("sig")
                lock["sig"] = new_sig if old_sig is None else 0.8 * old_sig + 0.2 * new_sig
            if tracked is None or iou(tracked, best["xyxy"]) < 0.75:
                self.tracker = make_tracker(frame, best["xyxy"])     # 밀렸으면 다시 잡는다
            best["class_id"] = lock["class_id"]
            best["fixed"] = True
            best["locked"] = True

        elif tracked is not None:
            # 검출은 없지만 추적기는 보고 있다. 이 경우가 핵심이다 —
            # 신뢰도가 오르내려 인식이 끊긴 구간에서도 빨강이 그 빵에 붙어 있게 된다.
            lock["vel"] = ((tracked[0] - x1) * 0.5 + vx * 0.5,
                           (tracked[1] - y1) * 0.5 + vy * 0.5)
            lock["xyxy"] = tracked
            lock["miss"] = 0
            boxes.append({"xyxy": tracked, "class_id": lock["class_id"], "model_id": None,
                          "fixed": True, "locked": True, "source": "lock", "conf": 0.0})

        else:
            miss = lock.get("miss", 0) + 1
            lock["miss"] = miss
            if miss <= self.LOCK_MAX_MISS and (abs(vx) > 0.3 or abs(vy) > 0.3):
                h_, w_ = frame.shape[:2]
                nx1 = max(-20.0, min(w_ + 20.0, x1 + vx))
                ny1 = max(-20.0, min(h_ + 20.0, y1 + vy))
                lock["xyxy"] = (nx1, ny1, nx1 + (x2 - x1), ny1 + (y2 - y1))
            lock["vel"] = (vx * 0.7, vy * 0.7)
            self.tracker = make_tracker(frame, lock["xyxy"])
            boxes.append({"xyxy": lock["xyxy"], "class_id": lock["class_id"], "model_id": None,
                          "fixed": True, "locked": True, "source": "lock", "conf": 0.0})

        return [b for b in boxes if b["class_id"] is not None]

    def refresh(self, frame, detections):
        """미리보기 한 프레임. 후보를 새로 잡고, 고정했으면 포커스를 그것에 맞춘다."""
        self.boxes = self.label_boxes(frame, detections)
        if self.boxes:
            locked_at = next((i for i, b in enumerate(self.boxes) if b.get("locked")), None)
            if locked_at is not None:
                self.focus = locked_at
            else:
                self.focus %= len(self.boxes)
        return self.boxes

    # ── 사용자 조작 (돌려주는 문자열이 있으면 상태창에 보여준다) ─────────────
    def cycle(self):
        if self.lock:
            return ("현재 상태: 촬영 대상이 고정되어 있습니다.\n"
                    "다른 것을 고르려면 「고정 해제」 를 먼저 누르세요.")
        if not self.boxes:
            return "현재 상태: 인식된 빵이 없습니다.\n빵을 화면에 올려주세요."
        self.focus = (self.focus + 1) % len(self.boxes)
        return None

    def _forget_manual_at(self, box):
        self.manual = [(b, c) for b, c in self.manual if iou(b, box) < self.MATCH_IOU]

    def assign(self, class_id):
        """포커스한 박스에 품목을 지정한다."""
        if not self.boxes:
            return "현재 상태: 바꿀 빵이 없습니다."
        target = self.boxes[min(self.focus, len(self.boxes) - 1)]
        self._forget_manual_at(target["xyxy"])
        self.manual.append((target["xyxy"], int(class_id)))
        target["class_id"] = int(class_id)
        target["fixed"] = True        # 다음 프레임을 기다리지 않고 바로 반영
        return None

    def ignore(self):
        """포커스한 것이 빵이 아니라고 표시한다 (책상·손·그림자)."""
        if not self.boxes:
            return "현재 상태: 무시할 빵이 없습니다."
        box = self.boxes[min(self.focus, len(self.boxes) - 1)]["xyxy"]
        self._forget_manual_at(box)
        self.ignored.append(box)
        self.focus = 0
        return None

    def toggle_lock(self, frame, class_id):
        """고른 박스를 촬영 대상으로 고정한다. 이미 고정했으면 푼다. (잠겼나, 안내문)"""
        if self.lock:
            self.lock = None
            self.tracker = None
            return False, None
        if not self.boxes:
            return False, "현재 상태: 고정할 것이 없습니다."
        bx = self.boxes[min(self.focus, len(self.boxes) - 1)]
        # 고정 = "지금 이걸 찍는다" 이므로 라벨은 「수집할 상품」 을 따른다.
        # 새 빵은 모델이 비슷한 기존 빵으로 부르는데, 그 이름을 쓰면 틀린 라벨이 쌓인다.
        cid = int(class_id)
        sig = patch_sig(frame, bx["xyxy"]) if frame is not None else None
        self.lock = {"xyxy": bx["xyxy"], "class_id": cid, "vel": (0.0, 0.0), "sig": sig, "miss": 0}
        self.tracker = make_tracker(frame, bx["xyxy"]) if frame is not None else None
        self._forget_manual_at(bx["xyxy"])
        self.manual.append((bx["xyxy"], cid))
        bx["class_id"] = cid
        bx["fixed"] = True
        bx["locked"] = True
        return True, None

    def clear(self):
        """지정·무시·고정을 전부 되돌린다."""
        self.manual = []
        self.ignored = []
        self.lock = None
        self.tracker = None
        self.focus = 0

    def hint(self, products):
        """무엇이 어떤 이름으로 저장될지 보여주는 안내문."""
        if not self.boxes:
            return "현재 상태: 잡힌 물체가 없습니다.\n빵을 화면에 올려주세요."
        lines, todo = [], 0
        if self.lock:
            lines.append(f"🔒 고정: {display_name(products, self.lock['class_id'])} — 이것만 찍습니다")
        for i, b in enumerate(self.boxes):
            mark = "🔒" if b.get("locked") else ("▶" if i == self.focus else "  ")
            if b["class_id"] is None:
                todo += 1
                lines.append(f"{mark} [{i + 1}] 이름 없음 — 지정하거나 무시")
                continue
            now = display_name(products, b["class_id"])
            if b["fixed"] and b["model_id"] is not None and b["model_id"] != b["class_id"]:
                lines.append(f"{mark} [{i + 1}] {now}   ← 모델은 {display_name(products, b['model_id'])} 라고 함")
            elif b["fixed"]:
                lines.append(f"{mark} [{i + 1}] {now} (내가 지정)")
            elif b["conf"] < LOW_CONF:
                lines.append(f"{mark} [{i + 1}] {now} ⚠ {b['conf'] * 100:.0f}% — 확인 필요")
            else:
                lines.append(f"{mark} [{i + 1}] {now} (모델 그대로)")
        if todo:
            lines.append(f"※ 이름 없는 것 {todo}개 — 저장되지 않습니다")
        return "\n".join(lines)
