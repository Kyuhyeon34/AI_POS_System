"""검출 결과를 프레임에 그리는 공용 코드.

로컬 추론(JetsonTensorRTDetector)과 원격 추론(RemoteDetector)이 같은 함수를 쓴다.
그려지는 모양이 두 경로에서 절대 달라지지 않게 하려고 여기 한 곳에만 둔다.
"""
from collections import Counter

import cv2

BOX_COLOR = (0, 220, 0)          # 초록 (BGR) — 믿을 만한 검출
LOW_COLOR = (0, 255, 255)        # 노랑 — 신뢰도가 낮아 확인이 필요한 검출
LOW_CONF = 0.55                  # 이보다 낮으면 노랑으로 그린다


def draw_detections(frame, detections, color=BOX_COLOR, labels=None, inplace=False,
                    low_conf=LOW_CONF, low_color=LOW_COLOR):
    """(그려진 프레임, 클래스별 개수) 를 돌려준다.

    labels: {class_id: "이름"} 을 주면 ID 대신 그 이름을 쓴다.
            (cv2 는 한글을 못 그리므로 영문 class_name 을 넘길 것)
    inplace=True 면 넘어온 frame 에 바로 그린다 (복사 비용 절약).

    신뢰도가 low_conf 미만이면 노랑으로 그린다. 모델이 확신하지 못한 것을
    한눈에 알아보라는 뜻이다 — 이름이 틀렸을 수 있다.
    low_conf=0 을 주면 이 구분을 끈다.
    """
    plotted = frame if inplace else frame.copy()
    counts = Counter()

    for d in detections:
        cid = int(d["class_id"])
        conf = float(d["confidence"])
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        counts[cid] += 1
        name = (labels or {}).get(cid, f"ID {cid}")
        low = low_conf > 0 and conf < low_conf
        c = low_color if low else color
        cv2.rectangle(plotted, (x1, y1), (x2, y2), c, 2)
        cv2.putText(
            plotted,
            f"{name} {conf:.2f}" + (" ?" if low else ""),
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            c,
            2,
            cv2.LINE_AA
        )

    return plotted, dict(counts)
