"""상품 목록 (database/products.json).

형식: {"0": {"class_name": "salt", "display_name": "소금빵", "price": 3000}, ...}
키는 YOLO class_id 를 문자열로 쓴 것이다. 줄 순서가 곧 모델의 클래스 번호다.
"""
from pathlib import Path

from core.jsonfile import read_json, write_json
from core.log import get_logger

log = get_logger("products")


def convert_legacy(data):
    """옛 형식 {"salt": {"class_id": 0, "name": "소금빵", "price": 3000}} → 지금 형식."""
    converted = {}
    for key, value in data.items():
        cid = int(value.get("class_id", len(converted)))
        converted[str(cid)] = {
            "class_name": key,
            "display_name": value.get("name", key),
            "price": int(value.get("price", 0)),
        }
    return converted


def load_products(path):
    path = Path(path)
    if not path.exists():
        # 처음 설치한 PC/젯슨: 기본 12종 목록으로 시작한다 (가격은 「상품 관리」에서)
        default = path.with_name("products.default.json")
        if not default.exists():
            return {}
        path.write_text(default.read_text(encoding="utf-8"), encoding="utf-8")

    data = read_json(path)
    if not isinstance(data, dict):
        log.error("%s 형식이 잘못되어 빈 목록으로 시작합니다", path.name)
        return {}
    if data and not all(str(k).isdigit() for k in data):
        data = convert_legacy(data)
        write_json(path, data)
    return data


def save_products(path, products):
    write_json(path, products)


def ordered(products):
    """[(class_id 문자열, 상품)] — class_id 순서."""
    return sorted(products.items(), key=lambda x: int(x[0]))


def next_class_id(products):
    return max((int(k) for k in products), default=-1) + 1


def class_name_map(products):
    """{class_id: 영문 class_name} — cv2 로 그릴 라벨용 (cv2 는 한글을 못 그린다)."""
    return {int(cid): p["class_name"] for cid, p in products.items()}


def display_name(products, class_id, default="?"):
    return products.get(str(class_id), {}).get("display_name", default)


def normalize_class_name(name):
    return name.strip().lower().replace(" ", "_")


def add_product(products, display, class_name, price):
    """새 상품을 넣고 배정한 class_id 를 돌려준다. 입력이 잘못되면 ValueError."""
    display = display.strip()
    class_name = normalize_class_name(class_name)
    if not display or not class_name:
        raise ValueError("상품명과 YOLO Class Name을 입력해주세요.")
    if any(p["class_name"] == class_name for p in products.values()):
        raise ValueError("이미 존재하는 YOLO Class Name입니다.")
    cid = next_class_id(products)
    products[str(cid)] = {"class_name": class_name, "display_name": display, "price": int(price)}
    return cid


def data_yaml_text(products, master_dir):
    """YOLO 학습용 data.yaml 내용. 경로는 윈도우에서도 슬래시로 쓴다."""
    lines = [
        f"path: {str(Path(master_dir).resolve()).replace(chr(92), '/')}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for cid, p in ordered(products):
        lines.append(f"  {cid}: {p['class_name']}")
    return "\n".join(lines) + "\n"
