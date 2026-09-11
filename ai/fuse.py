"""여러 모델의 검출 결과를 하나로 합친다 (Weighted Box Fusion).

왜 앙상블인가
    모델 하나는 어떤 빵을 놓치고, 다른 하나는 다른 빵을 놓친다.
    합치면 서로의 구멍을 메운다. 실측(실제 매대 사진 186장, conf 0.50):

        A 단독        놓침 73건   재현율 0.819
        C 단독        놓침 79건   재현율 0.804
        A+C           놓침 20건   재현율 0.950
        A+C+D3        놓침  8건   재현율 0.980

    없는 걸 잡는 오검출은 늘지 않았다 (11 → 7건).

왜 NMS 가 아니라 WBF 인가
    NMS 는 겹친 박스 중 하나만 남기고 버린다. 두 모델이 같은 빵을 조금씩
    다르게 잡았을 때 한쪽을 버리면 그 정보가 사라진다.
    WBF 는 신뢰도로 가중평균해 하나로 녹인다 — 위치가 더 정확해지고,
    두 모델이 동의한 박스는 신뢰도가 올라간다.
"""


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def fuse(detection_lists, iou_thr=0.55, n_models=None, min_votes=1,
         mode="agree", agree_bonus=1.25, cross_class=False):
    """[[det, ...], [det, ...]] → 합친 [det, ...]

    det 는 {"class_id", "confidence", "xyxy"} 형태 (ai/detector.py 와 같다).

    min_votes 를 2 이상으로 주면 '그만큼의 모델이 동의한 것' 만 남긴다.
    오검출을 줄이고 싶을 때 쓴다. 대신 놓침이 늘어난다.
    """
    # 어느 모델이 낸 박스인지 함께 들고 다닌다.
    # 한 모델이 같은 빵에 박스를 두 개 내는 일이 있어서, 박스 수로 표를 세면
    # "모델 2개인데 3표" 같은 값이 나온다. min_votes 를 쓸 때 위험하다.
    flat = []
    for mi, dets in enumerate(detection_lists):
        for d in (dets or []):
            flat.append((mi, d))
    flat.sort(key=lambda t: -float(t[1]["confidence"]))

    groups = []
    for mi, d in flat:
        box = [float(v) for v in d["xyxy"]]
        cid = int(d["class_id"])
        conf = float(d["confidence"])
        hit = None
        for g in groups:
            if g["cid"] == cid and iou(g["box"], box) > iou_thr:
                hit = g
                break
        if hit is None:
            groups.append({"box": box, "cid": cid, "best": conf,
                           "wsum": conf, "models": {mi}})
        else:
            w = hit["wsum"] + conf
            hit["box"] = [(hit["box"][i] * hit["wsum"] + box[i] * conf) / w
                          for i in range(4)]
            hit["wsum"] = w
            hit["models"].add(mi)
            hit["best"] = max(hit["best"], conf)

    total = n_models or len(detection_lists) or 1
    out = []
    for g in groups:
        votes = len(g["models"])          # 몇 개의 '모델' 이 동의했는가
        if votes < min_votes:
            continue

        # 신뢰도를 어떻게 정할 것인가.
        #
        # 원논문(WBF)은 한 모델만 본 박스의 점수를 votes/total 로 깎는다.
        # mAP 를 재는 데는 맞지만 POS 에는 해롭다 — POS 는 0.55 같은 고정
        # 문턱으로 자르기 때문에, 한 모델만 본 0.90 짜리가 0.45 로 깎여
        # 통째로 버려진다. 실측에서 이것 때문에 앙상블(놓침 90건)이
        # 단일 모델(32건)보다 나빴다.
        #
        #   penalty  원논문 방식 (mAP 용)
        #   max      깎지 않는다. 둘 중 하나라도 확신하면 채택
        #   agree    max + 둘 다 본 것은 올려 준다 (기본)
        #            → 합의한 검출은 문턱을 낮춰 주는 효과
        #   mean     신뢰도 평균
        if mode == "penalty":
            conf = g["best"] * votes / total
        elif mode == "mean":
            conf = g["wsum"] / max(1, len(g["models"]))
        elif mode == "agree":
            conf = g["best"] * (agree_bonus if votes >= 2 else 1.0)
        else:                                     # "max"
            conf = g["best"]
        out.append({
            "class_id": g["cid"],
            "confidence": round(min(1.0, conf), 4),
            "xyxy": [int(round(v)) for v in g["box"]],
            "votes": votes,
        })
    out.sort(key=lambda d: -d["confidence"])

    # 같은 자리에 이름이 다른 박스가 둘 남는 것을 막는다 (선택).
    #
    # WBF 는 같은 클래스끼리만 합친다. 그런데 TTA(좌우반전)나 앙상블에서
    # 한쪽은 "rotus 0.22", 다른 쪽은 "fruit 0.76" 처럼 같은 빵을 다른 이름으로
    # 부르면 둘 다 살아남아 박스가 겹쳐 뜬다 (2026-09-11 휘낭시에에서 실측).
    # 빵끼리 같은 자리에 겹쳐 있을 수는 없으므로 확신 높은 쪽 하나만 남긴다.
    if cross_class and len(out) > 1:
        kept = []
        for d in out:                         # 신뢰도 높은 순
            a = d["xyxy"]
            aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
            clash = False
            for k in kept:
                b = k["xyxy"]
                bb = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = iw * ih
                if inter <= 0:
                    continue
                if inter / (aa + bb - inter) > iou_thr or inter / min(aa, bb) >= 0.85:
                    clash = True
                    break
            if not clash:
                kept.append(d)
        out = kept
    return out


def gate_specialists(detection_lists, scopes, gate_conf=0.5, iou_thr=0.3, ios_thr=0.6):
    """일부 클래스만 배운 모델(전문가)의 박스를 다른 모델의 판단으로 걸러낸다.

    scopes[i] 는 i 번째 모델이 배운 class_id 집합 (None = 전부 안다).
    전문가는 모르는 빵을 자기가 아는 비슷한 빵 이름으로, 그것도 높은 신뢰도로 부른다.
    그래서 다른 모델이 '전문가가 모르는 클래스' 를 gate_conf 이상으로 본 자리에서는
    전문가의 박스를 뺀다. 전문가가 아는 빵 자리에서는 그대로 투표한다.

    실측 (2026-09-11, bread_0911 = 휘낭시에·와플 6종만 학습, 짝 m12_B, 빵 인식 버튼):
        그냥 합침       M12 기존 빵 오류 200/315
        역할 분담       M12 기존 빵 오류   3/315 · 트레이 사진 오류 10/1559 (0911 단독과 같음)
    """
    if not any(s is not None for s in scopes):
        return detection_lists
    out = []
    for i, dets in enumerate(detection_lists):
        scope = scopes[i]
        if scope is None:
            out.append(dets)
            continue
        others = [o for j, lst in enumerate(detection_lists) if j != i for o in (lst or [])
                  if int(o["class_id"]) not in scope and float(o["confidence"]) >= gate_conf]
        keep = []
        for d in dets or []:
            a = [float(v) for v in d["xyxy"]]
            clash = False
            for o in others:
                b = [float(v) for v in o["xyxy"]]
                iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = iw * ih
                if inter <= 0:
                    continue
                aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
                bb = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                if inter / (aa + bb - inter) > iou_thr or inter / min(aa, bb) >= ios_thr:
                    clash = True
                    break
            if not clash:
                keep.append(d)
        out.append(keep)
    return out
