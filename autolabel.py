"""자동 라벨링 + 검수 도구.

사진 폴더를 고르면 젯슨 추론 서버가 먼저 박스를 쳐 준다.
사람은 틀린 것만 고치고 Space 로 「확인 완료」 → 다음 사진으로 넘어간다.

    PC:    run_autolabel.bat 더블클릭 (젯슨 주소는 main.py 와 같은 방식으로 찾는다)
    젯슨:  python3 autolabel.py [폴더]
    창 없이 라벨만 한꺼번에:  python3 autolabel.py 폴더 --batch

저장 형식은 지금 쓰는 LabelImg YOLO 방식 그대로다.
    폴더/classes.txt     클래스 이름 (줄 순서 = 번호)
    폴더/사진.jpg
    폴더/사진.txt        "번호 cx cy w h" (0~1)
그래서 LabelImg 로 열어도 되고, 그대로 zip 해서 Colab 학습에 써도 된다.
검수 상태(자동/확인 완료)와 신뢰도는 폴더/.autolabel.json 에만 적는다.

클래스는 번호가 아니라 이름으로 맞춘다. 서버 모델의 클래스 순서가
폴더의 classes.txt 와 달라도 번호가 뒤섞이지 않는다.
"""
import argparse
import copy
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# main.py 와 같은 이유: pip opencv 가 심어 둔 Qt 플러그인 경로를 지워야 PyQt5 가 뜬다.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from core.log import get_logger

log = get_logger("autolabel")

BASE_DIR = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STATE_NAME = ".autolabel.json"
LOW_CONF = 0.55        # 이보다 낮으면 「확인 필요」 (POS 노랑 기준과 같다)
MIN_CONF = 0.30        # 이보다 낮은 검출은 라벨로 쓰지 않는다
JPEG_QUALITY = 95

SEL_COLOR = QColor(255, 40, 40)      # 선택한 박스
LOW_COLOR = QColor(255, 214, 0)      # 확인 필요
PALETTE = [(0, 200, 83), (41, 121, 255), (255, 145, 0), (170, 0, 255), (0, 184, 212),
           (255, 64, 129), (118, 200, 3), (141, 110, 99), (0, 137, 123), (92, 107, 192),
           (129, 212, 250), (96, 125, 139)]
KEY_HINTS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="]
CLASS_KEYS = [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6, Qt.Key_7,
              Qt.Key_8, Qt.Key_9, Qt.Key_0, Qt.Key_Minus, Qt.Key_Equal]
STATUS_STYLE = {           # 아이콘, 글자색, 설명
    "none":     ("○", "#9e9e9e", "라벨 없음"),
    "auto":     ("●", "#1565c0", "자동 라벨 (미검수)"),
    "check":    ("⚠", "#e65100", "확인 필요 (신뢰도 낮음 / 검출 없음)"),
    "verified": ("✔", "#2e7d32", "확인 완료"),
    "existing": ("■", "#6a1b9a", "기존 라벨 (직접 만든 것)"),
}


# ---- 파일 ---------------------------------------------------------------
def list_images(folder):
    return sorted((p for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS),
                  key=lambda p: p.name.lower())


def imread(path):
    """한글 경로도 읽는다 (윈도우에서 cv2.imread 는 한글 경로를 못 연다)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def label_path(img):
    return img.with_suffix(".txt")


def write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_classes(folder):
    f = Path(folder) / "classes.txt"
    if not f.exists():
        return None
    names = [l.strip() for l in f.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    return names or None


def products():
    """database/products.json → [(class_name, display_name)] (번호 순서)."""
    try:
        data = json.loads((BASE_DIR / "database" / "products.json").read_text(encoding="utf-8"))
        return [(data[k]["class_name"], data[k].get("display_name", ""))
                for k in sorted(data, key=int)]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, KeyError) as e:
        log.warning("database/products.json 을 읽지 못했습니다: %s", e)
        return []


def folder_classes(folder, server_names):
    """(클래스 목록, classes.txt 를 새로 써야 하는가)."""
    names = read_classes(folder)
    if names:
        return names, False
    return list(server_names or [n for n, _ in products()]), True


def read_yolo(txt, W, H):
    boxes = []
    if not txt.exists():
        return boxes
    for line in txt.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            c = int(float(p[0]))
            cx, cy, w, h = (float(v) for v in p[1:5])
        except ValueError:
            continue
        boxes.append({"c": c, "x1": (cx - w / 2) * W, "y1": (cy - h / 2) * H,
                      "x2": (cx + w / 2) * W, "y2": (cy + h / 2) * H, "conf": None})
    return boxes


def write_yolo(txt, boxes, W, H):
    """저장하고, 실제로 저장된 박스들을 돌려준다 (너무 작은 박스는 버린다)."""
    kept, lines = [], []
    for b in boxes:
        x1, x2 = sorted((min(max(b["x1"], 0.0), W), min(max(b["x2"], 0.0), W)))
        y1, y2 = sorted((min(max(b["y1"], 0.0), H), min(max(b["y2"], 0.0), H)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        kept.append({**b, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        lines.append(f"{b['c']} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    write_text(txt, "\n".join(lines) + ("\n" if lines else ""))
    return kept


class FolderState:
    """검수 상태. txt 에는 신뢰도·상태를 넣을 수 없어서 따로 둔다."""

    def __init__(self, folder):
        self.path = Path(folder) / STATE_NAME
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.data = {}
        except (OSError, ValueError) as e:
            log.warning("검수 상태 파일이 깨져 새로 시작합니다: %s (%s)", self.path, e)
            self.data = {}
        self.files = self.data.setdefault("files", {})

    def save(self):
        write_text(self.path, json.dumps(self.data, ensure_ascii=False, indent=1))

    def status(self, img):
        if not label_path(img).exists():
            return "none"
        st = self.files.get(img.name) or {}
        if st.get("status") == "verified":
            return "verified"
        if st.get("status") != "auto":
            return "existing"
        confs = st.get("conf", [])
        if not confs or any(c is not None and c < LOW_CONF for c in confs):
            return "check"
        return "auto"


def save_labels(state, img, boxes, W, H, status):
    kept = write_yolo(label_path(img), boxes, W, H)
    state.files[img.name] = {"status": status, "conf": [b["conf"] for b in kept]}
    state.save()
    return kept


# ---- 서버 ---------------------------------------------------------------
class Server:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.info = None

    def health(self, timeout=3.0):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=timeout) as r:
                self.info = json.loads(r.read().decode("utf-8"))
        except Exception:
            self.info = None
        return self.info

    @property
    def names(self):
        return (self.info or {}).get("names")

    def infer(self, img, tta=True, timeout=30.0):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError("JPEG 변환 실패")
        req = urllib.request.Request(self.url + ("/infer?tta=1" if tta else "/infer"),
                                     data=buf.tobytes(), headers={"Content-Type": "image/jpeg"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                res = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(e.read().decode("utf-8", "replace")[:200])
        if res.get("error"):
            raise RuntimeError(res["error"])
        return res.get("detections", [])


SERVER_CONFIG = BASE_DIR / "autolabel_server.json"   # 찾았거나 직접 넣은 주소를 기억한다
DEFAULT_HOST = ""   # 현장마다 다르다. 비워 두면 기억한 주소 → 네트워크 훑기 순서로 찾는다
PORT = 8092          # 젯슨 AI 추론 서버 포트


def probe(url, timeout=2.5):
    """추론 서버가 맞으면 /health 응답, 아니면 None."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            info = json.loads(r.read().decode("utf-8"))
        return info if "ready" in info else None
    except Exception:
        return None


def url_of(entry):
    """'192.168.0.23' → http://192.168.0.23:8092. 포트나 http:// 를 붙여 넣어도 된다."""
    e = str(entry).strip().rstrip("/")
    if not e:
        return None
    if "://" not in e:
        e = "http://" + e
    return e if ":" in e.split("://", 1)[1] else f"{e}:{PORT}"


def save_server(url):
    try:
        write_text(SERVER_CONFIG, json.dumps({"url": url}) + "\n")
    except OSError as e:
        log.warning("젯슨 주소를 저장하지 못했습니다 (다음 실행 때 다시 찾습니다): %s", e)


def _saved_hosts():
    """기억해 둔 주소 → POS 의 jetson.json (이 폴더, 한 칸 위 폴더) 순서."""
    out = []
    for f in (SERVER_CONFIG, BASE_DIR / "jetson.json", BASE_DIR.parent / "jetson.json"):
        try:
            cfg = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append(cfg.get("url") or cfg.get("host") or "")
    return out


def _local_prefixes():
    import socket
    ips = []
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))            # 패킷은 보내지 않는다. 나가는 주소만 알아낸다
        ips.append(s.getsockname()[0])
    except OSError:
        pass
    finally:
        s.close()
    try:
        ips += [i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    out = []
    for ip in ips:
        p = ip.rsplit(".", 1)[0]
        if not ip.startswith("127.") and p not in out:
            out.append(p)
    return out


def scan_network():
    """같은 네트워크(/24)에서 8092 가 열린 젯슨을 찾는다."""
    import socket
    from concurrent.futures import ThreadPoolExecutor

    def is_open(h):
        try:
            with socket.create_connection((h, PORT), timeout=0.4):
                return True
        except OSError:
            return False

    for prefix in _local_prefixes():
        hosts = [f"{prefix}.{i}" for i in range(1, 255)]
        with ThreadPoolExecutor(max_workers=64) as pool:
            opened = [h for h, ok in zip(hosts, pool.map(is_open, hosts)) if ok]
        for h in opened:
            if probe(f"http://{h}:{PORT}"):
                return f"http://{h}:{PORT}"
    return None


def find_server_url(log=print):
    """젯슨 추론 서버 주소. 못 찾으면 None.

    환경변수 → (젯슨 자신이면 localhost) → 기억한 주소 → POS 의 jetson.json → 기본 주소
    → 같은 네트워크 훑기. POS 폴더가 아닌 곳(하위 폴더 등)에 풀어도 찾는다.
    """
    global PORT
    # POS 와 같은 jetson.json 의 infer_port 를 따른다 (포트를 바꾼 현장)
    for f in (BASE_DIR / "jetson.json", BASE_DIR.parent / "jetson.json"):
        try:
            PORT = int(json.loads(f.read_text(encoding="utf-8")).get("infer_port", PORT))
            break
        except Exception:
            continue
    cands = [os.environ.get("BREADVISION_AUTOLABEL_URL", "")]
    if Path("/etc/nv_tegra_release").exists():
        cands.append("127.0.0.1")
    cands += _saved_hosts() + [DEFAULT_HOST]
    tried = []
    for c in cands:
        u = url_of(c)
        if not u or u in tried:
            continue
        tried.append(u)
        if probe(u):
            save_server(u)
            return u
    log("[자동 라벨링] 저장된 주소에 젯슨이 없어 같은 네트워크를 훑습니다… (몇 초)")
    u = scan_network()
    if u:
        save_server(u)
    return u


def class_map(server_names, classes):
    """서버 번호 → 폴더 번호 (이름으로 맞춘다). 서버가 이름을 안 주면 None = 번호 그대로."""
    if not server_names:
        return None
    idx = {n: i for i, n in enumerate(classes)}
    return {i: idx.get(n) for i, n in enumerate(server_names)}


def to_boxes(dets, cmap, n_classes, min_conf):
    boxes, unknown = [], set()
    for d in sorted(dets, key=lambda d: -float(d["confidence"])):
        conf = float(d["confidence"])
        if conf < min_conf:
            continue
        sid = int(d["class_id"])
        cid = sid if cmap is None else cmap.get(sid)
        if cid is None or not 0 <= cid < n_classes:
            unknown.add(sid)
            continue
        x1, y1, x2, y2 = (float(v) for v in d["xyxy"])
        boxes.append({"c": cid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "conf": round(min(conf, 1.0), 3)})
    return boxes, unknown


# ---- 창 없이 한꺼번에 -----------------------------------------------------
def run_batch(folder, url, min_conf, tta, redo):
    folder = Path(folder)
    server = Server(url)
    if not server.health():
        sys.exit(f"추론 서버에 연결할 수 없습니다: {url}")
    classes, new = folder_classes(folder, server.names)
    if not classes:
        sys.exit("클래스 목록을 정할 수 없습니다 (classes.txt 없음, 서버도 이름을 모름)")
    if new:
        write_text(folder / "classes.txt", "\n".join(classes) + "\n")
    cmap = class_map(server.names, classes)
    state = FolderState(folder)
    redo_set = {"none", "auto", "check"} if redo else {"none"}
    targets = [p for p in list_images(folder) if state.status(p) in redo_set]
    print(f"서버 {url} · 모델 {Path(server.info.get('model_path') or '').name} · 대상 {len(targets)}장")
    for i, p in enumerate(targets, 1):
        img = imread(p)
        if img is None:
            print(f"[{i}/{len(targets)}] {p.name}: 읽기 실패")
            continue
        try:
            dets = server.infer(img, tta)
        except Exception as e:
            print(f"[{i}/{len(targets)}] {p.name}: 추론 실패 {e}")
            continue
        boxes, _ = to_boxes(dets, cmap, len(classes), min_conf)
        save_labels(state, p, boxes, img.shape[1], img.shape[0], "auto")
        low = sum(b["conf"] < LOW_CONF for b in boxes)
        print(f"[{i}/{len(targets)}] {p.name}: {len(boxes)}개" + (f"  ⚠ 낮음 {low}" if low else ""))
    counts = {}
    for p in list_images(folder):
        s = state.status(p)
        counts[s] = counts.get(s, 0) + 1
    print("완료:", ", ".join(f"{STATUS_STYLE[k][2]} {v}" for k, v in counts.items()))


# ---- 창 -----------------------------------------------------------------
def class_color(c):
    if 0 <= c < len(PALETTE):
        return QColor(*PALETTE[c])
    return QColor.fromHsv((c * 47) % 360, 200, 230)


def text_color(bg):
    return QColor(0, 0, 0) if 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue() > 140 \
        else QColor(255, 255, 255)


class AutoWorker(QThread):
    result = pyqtSignal(str, object, int, int)     # 경로, 검출, W, H
    failed = pyqtSignal(str)
    step = pyqtSignal(int)

    def __init__(self, url, paths, tta):
        super().__init__()
        self.server = Server(url)
        self.paths = paths
        self.tta = tta
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        fails = 0
        for i, p in enumerate(self.paths, 1):
            if self._stop:
                break
            img = imread(p)
            if img is None:
                self.failed.emit(f"{p.name}: 이미지를 읽지 못했습니다")
            else:
                try:
                    dets = self.server.infer(img, self.tta)
                    fails = 0
                    self.result.emit(str(p), dets, img.shape[1], img.shape[0])
                except Exception as e:
                    fails += 1
                    self.failed.emit(f"{p.name}: {e}")
                    if fails >= 3:
                        self.failed.emit("서버 오류가 계속되어 중단합니다.")
                        break
            self.step.emit(i)


class FindWorker(QThread):
    """창을 먼저 띄우고 뒤에서 젯슨을 찾는다 (네트워크 훑기는 몇 초 걸린다)."""
    found = pyqtSignal(str)          # 못 찾으면 ""

    def run(self):
        try:
            url = find_server_url(log=lambda m: None) or ""
        except Exception:
            url = ""
        self.found.emit(url)


class Canvas(QWidget):
    """사진 + 박스. 드래그로 새 박스, 박스 드래그로 이동, 모서리로 크기 조절."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(640, 400)
        self.pix = None
        self.W = self.H = 1
        self.boxes = []
        self.sel = -1
        self.verified = False
        self.drag = None          # ("new", 시작점) | ("move", 시작점, 원래 박스) | ("resize", 모서리)
        self.temp = None
        self.mouse = None
        self.press = None
        self._pushed = False

    def set_image(self, img, boxes, verified):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.pix = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
        self.W, self.H = w, h
        self.boxes = boxes
        self.sel = -1
        self.verified = verified
        self.drag = self.temp = None
        self.update()

    # 좌표 변환 -------------------------------------------------------------
    def _geom(self):
        s = min(self.width() / self.W, self.height() / self.H)
        return s, (self.width() - self.W * s) / 2, (self.height() - self.H * s) / 2

    def to_img(self, pos):
        s, ox, oy = self._geom()
        return (pos.x() - ox) / s, (pos.y() - oy) / s

    def _clamp(self, x, y):
        return min(max(x, 0.0), float(self.W)), min(max(y, 0.0), float(self.H))

    def _corners(self, b):
        s, ox, oy = self._geom()
        return [(ox + b["x1"] * s, oy + b["y1"] * s), (ox + b["x2"] * s, oy + b["y1"] * s),
                (ox + b["x1"] * s, oy + b["y2"] * s), (ox + b["x2"] * s, oy + b["y2"] * s)]

    def _corner_at(self, pos):
        if not 0 <= self.sel < len(self.boxes):
            return None
        for k, (cx, cy) in enumerate(self._corners(self.boxes[self.sel])):
            if abs(pos.x() - cx) <= 8 and abs(pos.y() - cy) <= 8:
                return k
        return None

    def _box_at(self, pos):
        """겹친 박스는 작은 것을 고른다 (안쪽 빵을 집을 수 있게)."""
        x, y = self.to_img(pos)
        best, area = -1, float("inf")
        for i, b in enumerate(self.boxes):
            if b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"]:
                a = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
                if a < area:
                    best, area = i, a
        return best

    def select(self, i):
        self.sel = i
        self.win.on_select(i)
        self.update()

    # 마우스 ----------------------------------------------------------------
    def mousePressEvent(self, e):
        if self.pix is None:
            return
        self.setFocus()
        if e.button() == Qt.RightButton:
            i = self._box_at(e.pos())
            if i >= 0:
                self.select(i)
                self.win.box_menu(e.globalPos())
            return
        if e.button() != Qt.LeftButton:
            return
        self.press = e.pos()
        self._pushed = False
        k = self._corner_at(e.pos())
        if k is not None:
            self.drag = ("resize", k)
            return
        i = self._box_at(e.pos())
        if i >= 0:
            self.select(i)
            self.drag = ("move", self._clamp(*self.to_img(e.pos())), dict(self.boxes[i]))
            return
        self.select(-1)
        x, y = self._clamp(*self.to_img(e.pos()))
        self.drag = ("new", (x, y))
        self.temp = (x, y, x, y)

    def mouseMoveEvent(self, e):
        self.mouse = e.pos()
        if self.drag is not None:
            kind = self.drag[0]
            x, y = self._clamp(*self.to_img(e.pos()))
            if kind == "new":
                sx, sy = self.drag[1]
                self.temp = (sx, sy, x, y)
            elif 0 <= self.sel < len(self.boxes):
                if not self._pushed:
                    if (e.pos() - self.press).manhattanLength() < 3:
                        return               # 클릭할 때 손떨림은 무시
                    self.win.push_undo()
                    self._pushed = True
                b = self.boxes[self.sel]
                if kind == "move":
                    (sx, sy), ob = self.drag[1], self.drag[2]
                    w, h = ob["x2"] - ob["x1"], ob["y2"] - ob["y1"]
                    nx = min(max(ob["x1"] + x - sx, 0.0), self.W - w)
                    ny = min(max(ob["y1"] + y - sy, 0.0), self.H - h)
                    b.update(x1=nx, y1=ny, x2=nx + w, y2=ny + h)
                else:
                    k = self.drag[1]
                    b["x1" if k in (0, 2) else "x2"] = x
                    b["y1" if k in (0, 1) else "y2"] = y
                b["conf"] = None             # 사람이 손댄 박스는 확인된 것
                self.win.mark_dirty()
        elif self.pix is not None:
            k = self._corner_at(e.pos())
            if k is not None:
                self.setCursor(Qt.SizeFDiagCursor if k in (0, 3) else Qt.SizeBDiagCursor)
            elif self._box_at(e.pos()) >= 0:
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.CrossCursor)
        self.update()

    def mouseReleaseEvent(self, e):
        if self.drag is None:
            return
        kind = self.drag[0]
        self.drag = None
        if kind == "new" and self.temp:
            x1, x2 = sorted((self.temp[0], self.temp[2]))
            y1, y2 = sorted((self.temp[1], self.temp[3]))
            self.temp = None
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                self.win.push_undo()
                self.boxes.append({"c": self.win.current_class, "x1": x1, "y1": y1,
                                   "x2": x2, "y2": y2, "conf": None})
                self.win.mark_dirty()
                self.select(len(self.boxes) - 1)
                self.win.refresh_boxes()
        elif self._pushed and 0 <= self.sel < len(self.boxes):
            b = self.boxes[self.sel]
            b["x1"], b["x2"] = sorted((b["x1"], b["x2"]))
            b["y1"], b["y2"] = sorted((b["y1"], b["y2"]))
            self.win.refresh_boxes()
        self.temp = None
        self._pushed = False
        self.update()

    def leaveEvent(self, e):
        self.mouse = None
        self.update()

    # 그리기 ----------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(45, 45, 48))
        if self.pix is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignCenter, "폴더를 열면 사진이 여기에 나옵니다")
            return
        s, ox, oy = self._geom()
        target = QRectF(ox, oy, self.W * s, self.H * s)
        p.drawPixmap(target, self.pix, QRectF(0, 0, self.pix.width(), self.pix.height()))
        font = p.font()
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)
        fm = p.fontMetrics()

        order = [i for i in range(len(self.boxes)) if i != self.sel]
        if 0 <= self.sel < len(self.boxes):
            order.append(self.sel)             # 선택한 박스를 맨 위에
        for i in order:
            b = self.boxes[i]
            low = not self.verified and b.get("conf") is not None and b["conf"] < LOW_CONF
            color = SEL_COLOR if i == self.sel else LOW_COLOR if low else class_color(b["c"])
            pen = QPen(color, 3 if i == self.sel else 2)
            if low and i != self.sel:
                pen.setStyle(Qt.DashLine)
            x1, x2 = sorted((b["x1"], b["x2"]))
            y1, y2 = sorted((b["y1"], b["y2"]))
            r = QRectF(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            text = self.win.class_name(b["c"])
            if b.get("conf") is not None:
                text += f" {b['conf']:.2f}"
            if low:
                text += " ?"
            tw, th = fm.horizontalAdvance(text) + 8, fm.height() + 2
            ty = r.top() - th if r.top() - th >= oy else r.top()
            lab = QRectF(r.left(), ty, tw, th)
            p.fillRect(lab, color)
            p.setPen(text_color(color))
            p.drawText(lab, Qt.AlignCenter, text)
            if i == self.sel:
                for cx, cy in self._corners({"x1": x1, "y1": y1, "x2": x2, "y2": y2}):
                    p.fillRect(QRectF(cx - 4, cy - 4, 8, 8), color)

        if self.temp:
            x1, x2 = sorted((self.temp[0], self.temp[2]))
            y1, y2 = sorted((self.temp[1], self.temp[3]))
            r = QRectF(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s)
            color = class_color(self.win.current_class)
            p.setPen(QPen(color, 2, Qt.DashLine))
            p.drawRect(r)
            p.setPen(color)
            p.drawText(QPointF(r.left() + 2, r.top() - 4), self.win.class_name(self.win.current_class))
        elif self.mouse is not None and self.drag is None and target.contains(QPointF(self.mouse)):
            p.setPen(QPen(QColor(255, 255, 255, 110), 1, Qt.DashLine))
            p.drawLine(QPointF(self.mouse.x(), target.top()), QPointF(self.mouse.x(), target.bottom()))
            p.drawLine(QPointF(target.left(), self.mouse.y()), QPointF(target.right(), self.mouse.y()))


class LabelWindow(QMainWindow):
    FILTERS = [("전체", None), ("⚠ 확인 필요", {"check"}), ("● 미검수 (자동 전부)", {"auto", "check"}),
               ("○ 라벨 없음", {"none"}), ("✔ 확인 완료", {"verified"}), ("■ 기존 라벨", {"existing"})]

    def __init__(self, url=None, folder=None):
        super().__init__()
        self.server = Server(url or "")
        self.finder = None
        self.folder = None
        self.images, self.visible = [], []
        self.row = -1
        self.classes, self.classes_new = [], False
        self.display = dict(products())
        self.state = None
        self.cur = None                 # {"path", "img", "W", "H", "dirty"}
        self.undo = []
        self.current_class = 0
        self.worker = None
        self.cmap = None
        self._build()
        self.update_title()
        if url:
            self.check_server()
        else:
            self.lbl_server.setText("🟡 젯슨 찾는 중…")
            self.btn_auto.setEnabled(False)
            self.btn_relabel.setEnabled(False)
            self.finder = FindWorker()
            self.finder.found.connect(self.on_found)
            self.finder.start()
        if folder:
            self.open_folder(Path(folder))

    def on_found(self, url):
        self.finder = None
        if url:
            self.set_server(url)
        else:
            self.check_server()
            self.statusBar().showMessage(
                "⚠ 젯슨을 자동으로 찾지 못했습니다. 위쪽 「🔌 젯슨 주소」를 눌러 IP 를 넣어 주세요.")

    def set_server(self, url):
        self.server = Server(url)
        save_server(url)
        if self.check_server() and self.folder:
            self.warn_names()

    def ask_address(self):
        if self.finder:
            self.statusBar().showMessage("아직 젯슨을 찾는 중입니다. 잠시만 기다려 주세요.", 3000)
            return
        now = self.server.url.split("://")[-1].split(":")[0] if self.server.url else DEFAULT_HOST
        text, ok = QInputDialog.getText(
            self, "젯슨 주소",
            "젯슨 IP 주소를 넣어 주세요.\n"
            "(젯슨 터미널에서  hostname -I  를 치면 맨 앞에 나오는 숫자)\n\n예: 192.168.0.23",
            text=now)
        if not ok or not text.strip():
            return
        url = url_of(text)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            info = probe(url)
        finally:
            QApplication.restoreOverrideCursor()
        if info:
            self.set_server(url)
            self.statusBar().showMessage(f"🟢 젯슨 연결됨: {url} (다음부터는 자동으로 붙습니다)", 5000)
        else:
            QMessageBox.warning(
                self, "젯슨 주소",
                f"{url} 에서 응답이 없습니다.\n\n"
                "1) 젯슨이 켜져 있나요?\n"
                "2) PC 와 젯슨이 같은 인터넷(공유기)에 연결돼 있나요?\n"
                "   (PC 가 와이파이, 젯슨이 다른 랜선 망이면 서로 못 봅니다)\n"
                "3) IP 를 정확히 넣었나요?\n"
                "4) POS 프로그램(run_pc.bat)은 연결되나요? 그것도 안 되면 네트워크 문제입니다.")

    # 화면 구성 ------------------------------------------------------------
    def _build(self):
        self.resize(1440, 880)
        top = QHBoxLayout()
        self.btn_open = QPushButton("📂 폴더 열기")
        self.btn_open.clicked.connect(self.choose_folder)
        self.lbl_folder = QLabel("사진 폴더를 여세요")
        self.lbl_server = QLabel()
        self.btn_addr = QPushButton("🔌 젯슨 주소")
        self.btn_addr.setToolTip("젯슨을 못 찾을 때 IP 를 직접 넣습니다")
        self.btn_addr.clicked.connect(self.ask_address)
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(0.05, 0.95)
        self.spin_min.setSingleStep(0.05)
        self.spin_min.setValue(MIN_CONF)
        self.spin_min.setPrefix("저장 기준 ")
        self.spin_min.setToolTip("이보다 신뢰도가 낮은 검출은 라벨로 만들지 않습니다")
        self.spin_min.editingFinished.connect(lambda: self.canvas.setFocus())
        self.chk_tta = QCheckBox("좌우반전 보강")
        self.chk_tta.setChecked(True)
        self.chk_tta.setToolTip("사진을 좌우로 뒤집어 한 번 더 보고 합칩니다 (놓침 감소, 2배 느림)")
        self.chk_redo = QCheckBox("미검수 자동 라벨도 다시")
        self.btn_auto = QPushButton("🤖 자동 라벨링")
        self.btn_auto.clicked.connect(self.start_auto)
        self.btn_stop = QPushButton("중지")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.worker and self.worker.stop())
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setFormat("%v / %m")
        self.progress.hide()
        for w in (self.btn_open, self.lbl_folder, self.lbl_server, self.btn_addr):
            top.addWidget(w)
        top.addStretch(1)
        for w in (self.spin_min, self.chk_tta, self.chk_redo, self.btn_auto, self.btn_stop, self.progress):
            top.addWidget(w)

        # 왼쪽: 사진 목록
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems([n for n, _ in self.FILTERS])
        self.cmb_filter.currentIndexChanged.connect(self.on_filter)
        self.lbl_summary = QLabel()
        self.lbl_summary.setWordWrap(True)
        self.lst_files = QListWidget()
        self.lst_files.setTextElideMode(Qt.ElideMiddle)      # 긴 파일 이름은 가운데를 줄인다
        self.lst_files.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lst_files.itemClicked.connect(lambda it: self.show_row(self.lst_files.row(it)))
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.cmb_filter)
        lv.addWidget(self.lbl_summary)
        lv.addWidget(self.lst_files)

        self.canvas = Canvas(self)

        # 오른쪽: 클래스 / 박스 / 도움말
        self.lst_classes = QListWidget()
        self.lst_classes.itemClicked.connect(lambda it: self.set_class(self.lst_classes.row(it)))
        self.lst_boxes = QListWidget()
        self.lst_boxes.itemClicked.connect(lambda it: self.canvas.select(self.lst_boxes.row(it)))
        self.btn_verify = QPushButton("✔ 확인 완료 → 다음  (Space)")
        self.btn_verify.setStyleSheet("font-weight:bold; padding:8px;")
        self.btn_verify.clicked.connect(self.verify_next)
        self.btn_delete = QPushButton("🗑 선택 박스 삭제  (Del)")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_relabel = QPushButton("🤖 이 사진만 다시 자동  (R)")
        self.btn_relabel.clicked.connect(self.relabel_current)
        help_lbl = QLabel(
            "<b>마우스</b><br>"
            "빈 곳 드래그: 새 박스<br>"
            "박스 드래그: 이동 · 모서리: 크기<br>"
            "우클릭: 클래스 변경 / 삭제<br><br>"
            "<b>키보드</b><br>"
            "Space: 확인 완료 → 다음<br>"
            "A / D: 이전 / 다음 사진<br>"
            "F: 다음 미검수 사진<br>"
            "1~9, 0, -, =: 클래스<br>"
            "&nbsp;&nbsp;(박스 선택 중이면 그 박스를 바꿈,<br>"
            "&nbsp;&nbsp;&nbsp;선택 없으면 새 박스 클래스)<br>"
            "Del: 삭제 · Esc: 선택 해제<br>"
            "Ctrl+Z: 되돌리기 · Ctrl+S: 저장<br><br>"
            "<span style='color:#b8860b'><b>노랑 점선</b></span> = 신뢰도 55% 미만<br>"
            "사진을 넘기면 자동 저장됩니다.")
        help_lbl.setWordWrap(True)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel("<b>클래스</b>"))
        rv.addWidget(self.lst_classes, 3)
        rv.addWidget(QLabel("<b>이 사진의 박스</b>"))
        rv.addWidget(self.lst_boxes, 2)
        rv.addWidget(self.btn_verify)
        rv.addWidget(self.btn_delete)
        rv.addWidget(self.btn_relabel)
        rv.addWidget(help_lbl)

        # 키보드가 목록·버튼으로 새지 않게 (Space 가 버튼을 누르거나 A 가 목록 검색이 되는 것 방지)
        for w in (self.btn_open, self.btn_addr, self.chk_tta, self.chk_redo, self.btn_auto, self.btn_stop,
                  self.cmb_filter, self.lst_files, self.lst_classes, self.lst_boxes,
                  self.btn_verify, self.btn_delete, self.btn_relabel):
            w.setFocusPolicy(Qt.NoFocus)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.canvas)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([270, 900, 270])

        central = QWidget()
        v = QVBoxLayout(central)
        v.addLayout(top)
        v.addWidget(split, 1)
        self.setCentralWidget(central)

    # 서버 / 폴더 -----------------------------------------------------------
    def check_server(self):
        info = self.server.health()
        if info and info.get("ready"):
            model = Path(info.get("model_path") or "").name
            ens = " + 앙상블" if info.get("ensemble") else ""
            names = info.get("names")
            self.lbl_server.setText(
                f"🟢 {self.server.url} · {model}{ens} · "
                f"{len(names) if names else '?'}클래스")
            self.btn_auto.setEnabled(self.worker is None)
            self.btn_relabel.setEnabled(True)
        else:
            where = f" ({self.server.url})" if self.server.url else ""
            self.lbl_server.setText(f"🔴 젯슨 연결 안 됨{where} — 「🔌 젯슨 주소」를 눌러 IP 를 넣어 주세요")
            self.btn_auto.setEnabled(False)
            self.btn_relabel.setEnabled(False)
        return bool(info and info.get("ready"))

    def choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "사진 폴더 선택", str(self.folder or Path.home()))
        if d:
            self.open_folder(Path(d))

    def open_folder(self, folder):
        if self.worker:
            QMessageBox.information(self, "자동 라벨링", "자동 라벨링이 끝난 뒤에 다른 폴더를 여세요.")
            return
        if not folder.is_dir():
            QMessageBox.warning(self, "폴더", f"폴더가 없습니다: {folder}")
            return
        self.save_current()
        imgs = list_images(folder)
        if not imgs:
            QMessageBox.warning(self, "폴더", "사진(jpg/png)이 없는 폴더입니다.")
            return
        self.folder, self.images = folder, imgs
        self.state = FolderState(folder)
        self.classes, self.classes_new = folder_classes(folder, self.server.names)
        self.cur = None
        self.lbl_folder.setText(f"📁 {folder.name}  ({len(imgs)}장)")
        self.lbl_folder.setToolTip(str(folder))
        self.refresh_classes()
        self.warn_names()
        self.refresh_list()
        start = next((i for i, p in enumerate(self.visible)
                      if self.state.status(p) != "verified"), 0)
        self.show_row(start)

    def warn_names(self):
        names = self.server.names
        if not self.classes:
            self.statusBar().showMessage("⚠ classes.txt 가 없고 서버에서도 클래스 이름을 못 받았습니다.")
        elif names is None and self.server.info:
            self.statusBar().showMessage("⚠ 서버가 클래스 이름을 알려주지 않아 번호를 그대로 씁니다 "
                                         "(모델과 classes.txt 순서가 같아야 합니다).")
        elif names:
            missing = [n for n in names if n not in self.classes]
            if missing:
                self.statusBar().showMessage(
                    f"⚠ 이 폴더 classes.txt 에 없는 클래스는 자동 라벨에서 빠집니다: {', '.join(missing)}")

    def ensure_classes_file(self):
        if self.classes_new and self.folder and self.classes:
            write_text(self.folder / "classes.txt", "\n".join(self.classes) + "\n")
            self.classes_new = False

    # 목록 -----------------------------------------------------------------
    def class_name(self, c):
        return self.classes[c] if 0 <= c < len(self.classes) else f"ID {c}"

    def class_label(self, c):
        name = self.class_name(c)
        disp = self.display.get(name)
        return f"{name} ({disp})" if disp else name

    def refresh_classes(self):
        self.lst_classes.clear()
        for i, _ in enumerate(self.classes):
            pm = QPixmap(14, 14)
            pm.fill(class_color(i))
            key = KEY_HINTS[i] if i < len(KEY_HINTS) else " "
            self.lst_classes.addItem(QListWidgetItem(QIcon(pm), f"[{key}] {self.class_label(i)}"))
        if self.classes:
            self.current_class = min(self.current_class, len(self.classes) - 1)
            self.lst_classes.setCurrentRow(self.current_class)

    def _paint_item(self, item, p):
        icon, color, desc = STATUS_STYLE[self.state.status(p)]
        item.setText(f"{icon} {p.name}")
        item.setForeground(QColor(color))
        item.setToolTip(f"{p.name}\n{desc}")

    def refresh_list(self):
        flt = self.FILTERS[self.cmb_filter.currentIndex()][1]
        cur = self.cur["path"] if self.cur else None
        self.visible = [p for p in self.images if flt is None or self.state.status(p) in flt]
        self.lst_files.clear()
        for p in self.visible:
            item = QListWidgetItem()
            self._paint_item(item, p)
            self.lst_files.addItem(item)
        self.row = self.visible.index(cur) if cur in self.visible else -1
        if self.row >= 0:
            self.lst_files.setCurrentRow(self.row)
        self.refresh_summary()

    def update_item(self, p):
        if p in self.visible:
            self._paint_item(self.lst_files.item(self.visible.index(p)), p)

    def refresh_summary(self):
        if not self.state:
            return
        n = {k: 0 for k in STATUS_STYLE}
        for p in self.images:
            n[self.state.status(p)] += 1
        text = (f"전체 {len(self.images)}장<br>"
                f"<span style='color:#2e7d32'>✔ 확인 {n['verified']}</span> · "
                f"<span style='color:#e65100'>⚠ 확인필요 {n['check']}</span> · "
                f"<span style='color:#1565c0'>● 자동 {n['auto']}</span> · "
                f"<span style='color:#9e9e9e'>○ 없음 {n['none']}</span>")
        if n["existing"]:
            text += f" · <span style='color:#6a1b9a'>■ 기존 {n['existing']}</span>"
        self.lbl_summary.setText(text)

    def on_filter(self):
        self.save_current()
        self.refresh_list()
        if self.row < 0 and self.visible:
            self.show_row(0)

    # 사진 이동 / 저장 -------------------------------------------------------
    def show_row(self, row):
        if not 0 <= row < len(self.visible):
            return
        self.save_current()
        p = self.visible[row]
        img = imread(p)
        if img is None:
            self.statusBar().showMessage(f"⚠ 사진을 읽지 못했습니다: {p.name}")
            return
        H, W = img.shape[:2]
        boxes = read_yolo(label_path(p), W, H)
        confs = (self.state.files.get(p.name) or {}).get("conf", [])
        if len(confs) == len(boxes):
            for b, c in zip(boxes, confs):
                b["conf"] = c
        self.cur = {"path": p, "img": img, "W": W, "H": H, "dirty": False}
        self.row = row
        self.undo = []
        self.canvas.set_image(img, boxes, self.state.status(p) == "verified")
        self.lst_files.setCurrentRow(row)
        self.lst_files.scrollToItem(self.lst_files.item(row))
        self.refresh_boxes()
        self.update_title()
        self.canvas.setFocus()

    def save_current(self, status=None):
        """사진을 넘기거나 닫을 때 부른다. status 를 주면 바뀐 게 없어도 저장한다."""
        if not self.cur or (not self.cur["dirty"] and status is None):
            return
        p = self.cur["path"]
        if status is None:
            prev = (self.state.files.get(p.name) or {}).get("status")
            status = prev if prev in ("auto", "verified") else "manual"
        self.ensure_classes_file()
        kept = save_labels(self.state, p, self.canvas.boxes, self.cur["W"], self.cur["H"], status)
        if len(kept) != len(self.canvas.boxes):
            self.canvas.boxes[:] = kept
            self.canvas.sel = -1
        self.cur["dirty"] = False
        self.update_item(p)
        self.refresh_summary()
        self.update_title()

    def verify_next(self):
        if not self.cur:
            return
        self.save_current(status="verified")
        self.canvas.verified = True
        self.canvas.update()
        if self.row + 1 < len(self.visible):
            self.show_row(self.row + 1)
        else:
            self.statusBar().showMessage("✔ 마지막 사진까지 확인했습니다.", 5000)

    def next_unverified(self):
        n = len(self.visible)
        for k in range(1, n + 1):
            r = (self.row + k) % n
            if self.state.status(self.visible[r]) != "verified":
                self.show_row(r)
                return
        self.statusBar().showMessage("✔ 이 목록의 사진은 모두 확인 완료입니다.", 5000)

    def update_title(self):
        name = self.cur["path"].name if self.cur else ""
        dirty = " *" if self.cur and self.cur["dirty"] else ""
        self.setWindowTitle(f"BreadVision 자동 라벨링 — {name}{dirty}" if name
                            else "BreadVision 자동 라벨링")

    # 편집 -----------------------------------------------------------------
    def mark_dirty(self):
        if self.cur and not self.cur["dirty"]:
            self.cur["dirty"] = True
            self.update_title()

    def push_undo(self):
        self.undo.append(copy.deepcopy(self.canvas.boxes))
        del self.undo[:-50]

    def do_undo(self):
        if not self.undo:
            return
        self.canvas.boxes = self.undo.pop()
        self.canvas.sel = -1
        self.mark_dirty()
        self.refresh_boxes()
        self.canvas.update()

    def refresh_boxes(self):
        self.lst_boxes.clear()
        for i, b in enumerate(self.canvas.boxes):
            conf = "" if b.get("conf") is None else f"  {b['conf']:.2f}"
            item = QListWidgetItem(f"{i + 1}. {self.class_label(b['c'])}{conf}")
            if not self.canvas.verified and b.get("conf") is not None and b["conf"] < LOW_CONF:
                item.setForeground(QColor("#b8860b"))
            self.lst_boxes.addItem(item)
        if 0 <= self.canvas.sel < len(self.canvas.boxes):
            self.lst_boxes.setCurrentRow(self.canvas.sel)

    def on_select(self, i):
        if 0 <= i < len(self.canvas.boxes):
            self.lst_boxes.setCurrentRow(i)
            self.current_class = self.canvas.boxes[i]["c"]
            if self.current_class < len(self.classes):
                self.lst_classes.setCurrentRow(self.current_class)
        else:
            self.lst_boxes.clearSelection()

    def set_class(self, c):
        if not 0 <= c < len(self.classes):
            return
        self.current_class = c
        self.lst_classes.setCurrentRow(c)
        sel = self.canvas.sel
        if 0 <= sel < len(self.canvas.boxes):
            self.push_undo()
            self.canvas.boxes[sel]["c"] = c
            self.canvas.boxes[sel]["conf"] = None
            self.mark_dirty()
            self.refresh_boxes()
            self.canvas.update()
        else:
            self.statusBar().showMessage(f"새 박스 클래스: {self.class_label(c)}", 3000)

    def delete_selected(self):
        sel = self.canvas.sel
        if 0 <= sel < len(self.canvas.boxes):
            self.push_undo()
            del self.canvas.boxes[sel]
            self.canvas.sel = -1
            self.mark_dirty()
            self.refresh_boxes()
            self.canvas.update()

    def box_menu(self, pos):
        menu = QMenu(self)
        acts = {}
        for i in range(len(self.classes)):
            key = KEY_HINTS[i] if i < len(KEY_HINTS) else ""
            acts[menu.addAction(f"{key}  {self.class_label(i)}")] = i
        menu.addSeparator()
        act_del = menu.addAction("🗑 삭제")
        chosen = menu.exec_(pos)
        if chosen is act_del:
            self.delete_selected()
        elif chosen in acts:
            self.set_class(acts[chosen])

    def relabel_current(self):
        if not self.cur or self.worker:
            return
        if not self.check_server():
            return
        try:
            dets = self.server.infer(self.cur["img"], self.chk_tta.isChecked())
        except Exception as e:
            QMessageBox.warning(self, "자동 라벨", f"추론 실패: {e}")
            return
        boxes, _ = to_boxes(dets, class_map(self.server.names, self.classes),
                            len(self.classes), self.spin_min.value())
        self.push_undo()
        self.canvas.boxes = boxes
        self.canvas.sel = -1
        self.canvas.verified = False
        self.cur["dirty"] = True
        self.save_current(status="auto")
        self.refresh_boxes()
        self.canvas.update()

    # 자동 라벨링 ----------------------------------------------------------
    def start_auto(self):
        if not self.folder or self.worker:
            return
        if self.finder:
            self.statusBar().showMessage("아직 젯슨을 찾는 중입니다. 잠시만 기다려 주세요.", 3000)
            return
        if not self.check_server():
            QMessageBox.warning(self, "자동 라벨링", "추론 서버에 연결할 수 없습니다.")
            return
        if not self.classes:
            self.classes, self.classes_new = folder_classes(self.folder, self.server.names)
            self.refresh_classes()
        if not self.classes:
            QMessageBox.warning(self, "자동 라벨링", "클래스 목록(classes.txt)을 정할 수 없습니다.")
            return
        self.save_current()
        redo = {"none", "auto", "check"} if self.chk_redo.isChecked() else {"none"}
        cur = self.cur["path"] if self.cur else None
        targets = [p for p in self.images if self.state.status(p) in redo]
        if not targets:
            QMessageBox.information(self, "자동 라벨링",
                                    "라벨이 없는 사진이 없습니다.\n"
                                    "(미검수 자동 라벨도 새로 하려면 「미검수 자동 라벨도 다시」를 켜세요)")
            return
        if cur in targets:                        # 보고 있는 사진을 먼저
            targets.remove(cur)
            targets.insert(0, cur)
        self.warn_names()
        self.cmap = class_map(self.server.names, self.classes)
        self.ensure_classes_file()
        self.auto_fail = []
        self.progress.setRange(0, len(targets))
        self.progress.setValue(0)
        self.progress.show()
        self.btn_auto.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker = AutoWorker(self.server.url, targets, self.chk_tta.isChecked())
        self.worker.result.connect(self.on_auto_result)
        self.worker.failed.connect(self.on_auto_failed)
        self.worker.step.connect(self.on_auto_step)
        self.worker.finished.connect(self.on_auto_finished)
        self.auto_targets = targets
        self.worker.start()

    def on_auto_result(self, path, dets, W, H):
        p = Path(path)
        boxes, _ = to_boxes(dets, self.cmap, len(self.classes), self.spin_min.value())
        showing = self.cur is not None and self.cur["path"] == p
        if showing and self.cur["dirty"]:
            return                                # 사람이 고치는 중인 사진은 건드리지 않는다
        kept = save_labels(self.state, p, boxes, W, H, "auto")
        if showing:
            self.canvas.boxes = kept
            self.canvas.sel = -1
            self.canvas.verified = False
            self.undo = []
            self.refresh_boxes()
            self.canvas.update()
        self.update_item(p)

    def on_auto_step(self, i):
        self.progress.setValue(i)
        if i % 10 == 0:
            self.refresh_summary()

    def on_auto_failed(self, msg):
        self.auto_fail.append(msg)
        self.statusBar().showMessage(f"⚠ {msg}", 5000)

    def on_auto_finished(self):
        self.worker = None
        self.btn_stop.setEnabled(False)
        self.progress.hide()
        self.check_server()
        self.refresh_summary()
        n_check = sum(self.state.status(p) == "check" for p in self.auto_targets)
        done = sum(self.state.status(p) != "none" for p in self.auto_targets)
        msg = (f"{done}장 자동 라벨링을 마쳤습니다.\n"
               f"⚠ 확인 필요 {n_check}장 — 왼쪽 위 목록에서 「⚠ 확인 필요」를 고르면 그것만 볼 수 있습니다.\n"
               f"나머지도 한 번씩 훑어보고 Space 로 확인 완료를 눌러 주세요.")
        if self.auto_fail:
            msg += f"\n\n실패 {len(self.auto_fail)}건:\n" + "\n".join(self.auto_fail[:5])
        QMessageBox.information(self, "자동 라벨링", msg)

    # 키보드 / 종료 ---------------------------------------------------------
    def keyPressEvent(self, e):
        k = e.key()
        if e.modifiers() & Qt.ControlModifier:
            if k == Qt.Key_Z:
                self.do_undo()
            elif k == Qt.Key_S:
                self.save_current()
                self.statusBar().showMessage("저장했습니다.", 2000)
            return
        if k in CLASS_KEYS:
            self.set_class(CLASS_KEYS.index(k))
        elif k in (Qt.Key_A, Qt.Key_Left):
            self.show_row(self.row - 1)
        elif k in (Qt.Key_D, Qt.Key_Right):
            self.show_row(self.row + 1)
        elif k == Qt.Key_Space:
            self.verify_next()
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif k == Qt.Key_F:
            self.next_unverified()
        elif k == Qt.Key_R:
            self.relabel_current()
        elif k == Qt.Key_Escape:
            self.canvas.select(-1)
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.save_current()
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
        e.accept()


def main():
    ap = argparse.ArgumentParser(description="자동 라벨링 + 검수")
    ap.add_argument("folder", nargs="?", help="사진 폴더")
    ap.add_argument("--url", default="", help="추론 서버 주소 (비우면 자동으로 찾는다)")
    ap.add_argument("--batch", action="store_true", help="창 없이 라벨만 만든다")
    ap.add_argument("--min-conf", type=float, default=MIN_CONF)
    ap.add_argument("--no-tta", action="store_true", help="좌우반전 보강을 끈다")
    ap.add_argument("--redo", action="store_true", help="미검수 자동 라벨도 다시 만든다")
    args = ap.parse_args()

    if args.batch:
        if not args.folder:
            ap.error("--batch 에는 폴더가 필요합니다")
        url = url_of(args.url) if args.url else find_server_url()
        if not url:
            sys.exit("젯슨을 찾지 못했습니다.  --url 192.168.0.23  처럼 주소를 지정하세요.")
        run_batch(args.folder, url, args.min_conf, not args.no_tta, args.redo)
        return

    app = QApplication(sys.argv)
    win = LabelWindow(url_of(args.url) if args.url else None, args.folder)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
