"""수집 탭 미리보기 그리기. 무엇이 어떤 이름으로 저장될지 색으로 보여준다."""
import cv2

# 박스 색 (BGR)
BOX_FOCUS = (0, 0, 235)       # 빨강 — 지금 고른 것
BOX_MODEL = (0, 220, 0)       # 초록 — 모델이 부른 그대로 저장
BOX_FIXED = (255, 170, 0)     # 파랑 — 내가 품목을 바꾼 것
BOX_TODO = (0, 165, 255)      # 주황 — 이름이 없다. 지정하거나 무시해야 저장된다
BOX_LOW = (0, 255, 255)       # 노랑 — 모델이 확신하지 못한다


def draw_single(frame, roi, box, why, candidate, name):
    """「한 개」 방식: 지금 찍으면 저장될 박스 하나만 초록으로 그린다."""
    view = frame.copy()
    if roi > 0:                                   # 가운데 영역 안내선
        h, w = view.shape[:2]
        rx, ry = int(w * (1 - roi) / 2), int(h * (1 - roi) / 2)
        cv2.rectangle(view, (rx, ry), (w - rx, h - ry), (150, 150, 150), 1)
        cv2.putText(view, "put ONE bread here", (rx + 6, ry + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    if box is None:
        if candidate:                             # 무엇을 잡았는데 왜 버렸는지 (주황)
            cv2.rectangle(view, (int(candidate[0]), int(candidate[1])),
                          (int(candidate[2]), int(candidate[3])), BOX_TODO, 2)
        cv2.putText(view, f"NO BOX - {why}", (18, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 235), 2, cv2.LINE_AA)
        cv2.putText(view, "put ONE bread on a plain background", (18, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 235), 2, cv2.LINE_AA)
        return view

    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(view, (x1, y1), (x2, y2), BOX_MODEL, 3)
    cv2.putText(view, f"{name}  ({why})", (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_MODEL, 2, cv2.LINE_AA)
    return view


def draw_multi(frame, boxes, focus, names, locked, low_conf):
    """「여러 개」 방식: 잡힌 박스를 전부 그린다. 고른 것은 빨강."""
    view = frame.copy()
    for i, bx in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in bx["xyxy"]]
        focused = i == focus
        if bx["class_id"] is None:
            base, tag = BOX_TODO, "SELECT ME"
        elif bx["fixed"]:
            base = BOX_FIXED
            tag = names.get(bx["class_id"], "?")
            if bx["model_id"] is not None and bx["model_id"] != bx["class_id"]:
                tag += f" (was {names.get(bx['model_id'], '?')})"
        elif bx["conf"] < low_conf:
            base = BOX_LOW
            tag = f"{names.get(bx['class_id'], '?')} {bx['conf']:.2f} ?"
        else:
            base = BOX_MODEL
            tag = f"{names.get(bx['class_id'], '?')} {bx['conf']:.2f}"
        color = BOX_FOCUS if focused else base
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 4 if focused else 2)
        cv2.putText(view, f"[{i + 1}] {tag}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # 안내는 아래쪽에 둔다. 위에 두면 박스 라벨과 겹쳐 둘 다 안 읽힌다.
    msg = color = None
    if not boxes:
        msg, color = "NO OBJECT - put bread in view", (40, 40, 235)
    elif not locked and any(b["class_id"] is None for b in boxes):
        msg, color = "name or ignore the orange box before capturing", BOX_TODO
    if msg:
        y = view.shape[0] - 16
        cv2.putText(view, msg, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, msg, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return view
