"""탭들이 함께 쓰는 것 — 상품 목록, 설정, 추론기, 젯슨 작업 창구, 회원 API.

예전에는 MainWindow 한 클래스가 이것들을 전부 self.xxx 로 들고 있었고,
탭끼리는 서로의 위젯을 직접 건드렸다 (예: 상품 등록 → 수집 탭 콤보박스를 직접 갱신).
그래서 한 곳을 고치면 어디가 같이 깨지는지 알기 어려웠다.

지금은 공용 상태를 여기 한 곳에 두고, 바뀌었다는 사실만 신호로 알린다.
    products_changed  상품이 추가되거나 가격이 바뀌었다 → 각 탭이 자기 화면만 다시 그린다
    model_changed     POS 모델이 바뀌었다 → POS 머리의 모델 배지 등
    status            아래 상태바에 잠깐 보여줄 문구
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal

from core import products as product_db
from core.products import class_name_map, data_yaml_text
from core.settings import load_settings, save_settings

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    base: Path = BASE_DIR
    products: Path = field(default=None)
    settings: Path = field(default=None)
    collect_img: Path = field(default=None)
    collect_lbl: Path = field(default=None)
    master: Path = field(default=None)
    train_img: Path = field(default=None)
    data_yaml: Path = field(default=None)
    runs: Path = field(default=None)
    imports: Path = field(default=None)
    models: Path = field(default=None)

    def __post_init__(self):
        b = Path(self.base)
        self.products = self.products or b / "database" / "products.json"
        self.settings = self.settings or b / "database" / "settings.json"
        self.collect_img = self.collect_img or b / "dataset" / "collected" / "images"
        self.collect_lbl = self.collect_lbl or b / "dataset" / "collected" / "labels"
        self.master = self.master or b / "dataset" / "master"
        self.train_img = self.train_img or self.master / "images" / "train"
        self.data_yaml = self.data_yaml or b / "dataset" / "data.yaml"
        self.runs = self.runs or b / "runs"
        self.imports = self.imports or b / "dataset" / "import"
        self.models = self.models or b / "models"


class AppContext(QObject):
    products_changed = pyqtSignal()
    model_changed = pyqtSignal()
    status = pyqtSignal(str, int)       # 문구, 보여줄 시간(ms). 0 = 지울 때까지, "" = 지우기

    def __init__(self, detector, ops=None, membership_api=None, paths=None, parent=None):
        super().__init__(parent)
        self.paths = paths or Paths()
        self.products = product_db.load_products(self.paths.products)
        self.settings = load_settings(self.paths.settings)
        self.detector = detector
        self.ops = ops                  # 젯슨 작업 창구 (PC 에서 젯슨에 연결됐을 때만)
        self.membership_api = membership_api

    # ── 저장 ────────────────────────────────────────────────────────────
    def save_products(self):
        product_db.save_products(self.paths.products, self.products)

    def save_settings(self):
        save_settings(self.paths.settings, self.settings)

    def write_data_yaml(self):
        self.paths.data_yaml.parent.mkdir(parents=True, exist_ok=True)
        self.paths.data_yaml.write_text(data_yaml_text(self.products, self.paths.master),
                                        encoding="utf-8")
        return self.paths.data_yaml

    # ── 알림 ────────────────────────────────────────────────────────────
    def show_status(self, text, msec=0):
        self.status.emit(text, int(msec))

    def clear_status(self):
        self.status.emit("", 0)

    # ── 조회 ────────────────────────────────────────────────────────────
    def class_names(self):
        return class_name_map(self.products)

    def active_engine(self):
        return self.settings.get("active_engine", "")

    def can_run_on_jetson(self):
        """PC 에서 눌러도 젯슨이 대신 해 줄 수 있는 상태인가."""
        return self.ops is not None

    @staticmethod
    def is_jetson():
        """젯슨에서 직접 돌고 있는가. (TensorRT 변환·긴급 전이학습은 젯슨에서만 가능)"""
        return sys.platform.startswith("linux") and Path("/etc/nv_tegra_release").exists()
