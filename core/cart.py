"""POS 주문(장바구니) 계산. 화면과 떨어뜨려 두어 금액 계산을 테스트할 수 있게 한다."""

# 주문에 담을 때 요구하는 신뢰도.
# 추론 서버는 낮은 것까지 다 보낸다 (수집 탭에서 "신뢰도 낮음" 을 노랑으로 보여줘야 하므로).
# 돈이 오가는 POS 쪽은 여기서 확실한 것만 남긴다.
POS_CONF = 0.55


def pos_detections(detections, min_conf=POS_CONF):
    return [d for d in (detections or []) if float(d.get("confidence", 0)) >= min_conf]


def count_by_class(detections):
    counts = {}
    for d in detections:
        cid = int(d["class_id"])
        counts[cid] = counts.get(cid, 0) + 1
    return counts


class Cart:
    """{class_id(int): 수량}. 가격은 products 에서 그때그때 읽는다 (가격 변경이 바로 반영)."""

    def __init__(self):
        self.items = {}

    def __bool__(self):
        return bool(self.items)

    def add(self, class_id, qty=1):
        cid = int(class_id)
        self.items[cid] = self.items.get(cid, 0) + qty

    def change(self, class_id, delta):
        cid = int(class_id)
        self.items[cid] = self.items.get(cid, 0) + delta
        if self.items[cid] <= 0:
            self.items.pop(cid, None)

    def remove(self, class_id):
        self.items.pop(int(class_id), None)

    def clear(self):
        self.items.clear()

    def add_counts(self, counts, products):
        """인식 결과를 주문에 "더한다". 등록된 상품만 담고, (담은 것, 모르는 class_id) 를 돌려준다.

        덮어쓰면 같은 빵을 다시 인식했을 때 수량이 그대로여서 두 번째 누름이 무시된다.
        products.json 의 키는 문자열("0"), 검출 결과의 class_id 는 정수(0) 이므로 str() 로 맞춘다.
        """
        added, unknown = {}, []
        for cid, count in counts.items():
            if str(cid) in products:
                self.add(cid, count)
                added[int(cid)] = count
            else:
                unknown.append(cid)
        return added, unknown

    def lines(self, products):
        """[(class_id, 상품, 수량, 금액)] — class_id 순서. 목록에서 사라진 상품은 뺀다."""
        out = []
        for cid, qty in sorted(self.items.items()):
            p = products.get(str(cid))
            if p:
                out.append((cid, p, qty, int(p["price"]) * qty))
        return out

    def total(self, products):
        return sum(line[3] for line in self.lines(products))

    def total_qty(self):
        return sum(self.items.values())

    def kinds(self):
        return len([q for q in self.items.values() if q > 0])
