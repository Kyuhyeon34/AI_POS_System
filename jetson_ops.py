"""PC 에서 누른 버튼을 젯슨에서 대신 실행해 주는 부분.

왜 필요한가
    업체는 PC 한 대만 만진다. 젯슨은 매대 밑에 두고 전원만 넣는다.
    그런데 데이터셋 분할·전이학습·ENGINE 변환은 젯슨 GPU 와 Docker 가 있어야 한다.
    (특히 TensorRT ENGINE 은 그 기기·그 버전 전용 바이너리라 PC 에서 만들 수 없다)
    그래서 PC 는 "무엇을 해달라"만 보내고, 실제 실행은 여기서 한다.

안전
    아무 명령이나 실행하지 않는다. 아래 TASKS 에 적힌 것만 돈다.
    경로는 전부 프로젝트 폴더 안으로 제한한다 (상위로 빠져나가는 경로는 거부).

동시 실행
    한 번에 한 작업만 돈다. 학습 중에 또 학습을 걸면 거절한다.
"""
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

from core.jsonfile import read_json, write_json

BASE_DIR = Path(__file__).resolve().parent
IMPORT_DIR = BASE_DIR / "dataset" / "import"
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
PRODUCTS = BASE_DIR / "database" / "products.json"
SETTINGS = BASE_DIR / "database" / "settings.json"

# 파일 목록을 보여줄 때 쓰는 규칙
FILE_KINDS = {
    "pt":      ["models/*.pt", "runs/*/weights/*.pt"],
    "engine":  ["models/*.engine", "runs/*/weights/*.engine"],
    "onnx":    ["models/*.onnx", "runs/*/weights/*.onnx"],
    "import":  ["dataset/import/*.zip"],
}

_lock = threading.Lock()
_jobs = {}          # id -> {"lines":[], "running":bool, "exit":int|None, "result":{}}
_seq = 0


# ── 경로 안전 ────────────────────────────────────────────────────────────
def safe_path(rel):
    """프로젝트 폴더 안의 경로만 허용한다.

    PC(윈도우)가 보낸 경로에는 역슬래시가 섞여 온다.
    (예: Path("models/x.onnx").with_suffix(".engine") 가 윈도우에서는 "models\\x.engine")
    리눅스에서는 그게 폴더 구분이 아니라 파일명의 일부로 읽히므로 여기서 바꿔 준다.
    """
    if not rel:
        raise ValueError("경로가 비어 있습니다.")
    rel = str(rel).replace("\\", "/")
    p = (BASE_DIR / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
    if BASE_DIR not in p.parents and p != BASE_DIR:
        raise ValueError(f"프로젝트 밖의 경로는 쓸 수 없습니다: {rel}")
    return p


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(p)


# ── 파일 목록 · 업로드 ───────────────────────────────────────────────────
def list_files(kind):
    pats = FILE_KINDS.get(kind)
    if pats is None:
        raise ValueError(f"알 수 없는 종류: {kind} (가능: {', '.join(FILE_KINDS)})")
    out = []
    for pat in pats:
        for f in BASE_DIR.glob(pat):
            if f.is_file():
                st = f.stat()
                out.append({"path": rel(f), "size": st.st_size, "mtime": int(st.st_mtime)})
    out.sort(key=lambda x: -x["mtime"])
    return out


# 받을 수 있는 파일과 저장 위치. 그 외 확장자는 거부한다.
UPLOAD_DEST = {
    ".zip":    IMPORT_DIR,                       # 수집 데이터
    ".onnx":   BASE_DIR / "models",              # Colab 에서 받은 모델
    ".pt":     BASE_DIR / "models",              # Colab 에서 받은 가중치
}


def save_upload(name, data, replace_prefix=None):
    """PC 가 보낸 파일을 알맞은 폴더에 저장한다.

    replace_prefix 를 주면 같은 폴더에서 그 접두사로 시작하는 옛 파일을 지운다.
    collected_*.zip 은 수집 폴더 '전체'를 담으므로, 옛것이 남아 있으면
    같은 사진이 두 번 학습에 들어간다. 그래서 새로 올릴 때 옛것을 치운다.
    """
    name = Path(name).name                       # 경로 조작 방지
    ext = Path(name).suffix.lower()
    dest_dir = UPLOAD_DEST.get(ext)
    if dest_dir is None:
        raise ValueError(f"받을 수 없는 형식입니다: {ext} (가능: {', '.join(UPLOAD_DEST)})")
    dest_dir.mkdir(parents=True, exist_ok=True)
    removed = []
    if replace_prefix:
        for old in dest_dir.glob(f"{replace_prefix}*{ext}"):
            removed.append(old.name)
            old.unlink()
    dest = dest_dir / name
    dest.write_bytes(data)
    return {"saved": rel(dest), "bytes": len(data), "removed": removed}


# ── data.yaml ────────────────────────────────────────────────────────────
def write_products(products):
    """PC 의 상품 목록을 젯슨에도 반영한다. (클래스 이름의 기준은 PC)"""
    PRODUCTS.parent.mkdir(parents=True, exist_ok=True)
    PRODUCTS.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_data_yaml():
    products = json.loads(PRODUCTS.read_text(encoding="utf-8"))
    lines = [
        f"path: {(BASE_DIR / 'dataset' / 'master').resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for cid, p in sorted(products.items(), key=lambda x: int(x[0])):
        lines.append(f"  {cid}: {p['class_name']}")
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rel(DATA_YAML)


# ── 작업 실행 ────────────────────────────────────────────────────────────
def _build_command(task, args):
    """실행할 명령을 만든다. 여기에 없는 작업은 실행되지 않는다."""
    if task == "split":
        return ["python3", str(BASE_DIR / "import_master_dataset.py"),
                "--project", str(BASE_DIR)]

    if task == "synth":
        # split 이 만든 dataset/master 를 재료로 여러 개짜리 사진을 합성해 넣는다.
        # 신상품은 혼자 있는 모습만 찍히므로, 그것만으로 학습하면 겹친 빵을 못 잡는다.
        count = int(args.get("count", 400))
        if not (1 <= count <= 5000):
            raise ValueError("합성 장수는 1~5000 사이여야 합니다.")
        return ["python3", str(BASE_DIR / "make_synth.py"),
                "--project", str(BASE_DIR), "--count", str(count)]

    if task == "train":
        model = safe_path(args["model"])
        if not model.exists():
            raise ValueError(f"모델 파일이 없습니다: {args['model']}")
        epochs = int(args.get("epochs", 30))
        imgsz = int(args.get("imgsz", 640))
        name = str(args.get("name") or ("bread_" + time.strftime("%Y%m%d_%H%M%S")))
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("실행 이름에는 영문·숫자·_·- 만 쓸 수 있습니다.")
        return ["bash", str(BASE_DIR / "training" / "run_training_container.sh"),
                str(model), str(DATA_YAML), str(epochs), str(imgsz), name]

    if task == "engine":                     # best.pt → onnx → engine
        pt = safe_path(args["pt"])
        if not pt.exists():
            raise ValueError(f"파일이 없습니다: {args['pt']}")
        return ["bash", str(BASE_DIR / "training" / "export_and_build_engine.sh"),
                str(pt), str(int(args.get("imgsz", 640)))]

    if task == "onnx2engine":                # Colab 에서 받은 onnx → engine
        onnx = safe_path(args["onnx"])
        if not onnx.exists():
            raise ValueError(f"파일이 없습니다: {args['onnx']}")
        out = onnx.with_suffix(".engine")
        return ["/usr/src/tensorrt/bin/trtexec", f"--onnx={onnx}",
                f"--saveEngine={out}", "--fp16"]

    raise ValueError(f"알 수 없는 작업: {task}")


def _run(job_id, cmd, on_done=None):
    job = _jobs[job_id]
    try:
        p = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        job["pid"] = p.pid
        for line in p.stdout:
            line = line.rstrip("\n")
            job["lines"].append(line)
            if line.startswith("BEST_MODEL="):
                job["result"]["best_pt"] = rel(line.split("=", 1)[1].strip())
            elif line.startswith("ENGINE="):
                job["result"]["engine"] = rel(line.split("=", 1)[1].strip())
            elif line.startswith("ONNX="):
                job["result"]["onnx"] = rel(line.split("=", 1)[1].strip())
        p.wait()
        job["exit"] = p.returncode
    except Exception as e:                    # noqa: BLE001 - 로그로 남겨 PC 에 보여준다
        job["lines"].append(f"[오류] {e}")
        job["exit"] = -1
    finally:
        job["running"] = False
        if on_done:
            try:
                on_done(job)
            except Exception as e:            # noqa: BLE001
                job["lines"].append(f"[후처리 오류] {e}")


def start_task(task, args, on_done=None):
    global _seq
    with _lock:
        for j in _jobs.values():
            if j["running"]:
                raise RuntimeError("이미 다른 작업이 진행 중입니다. 끝난 뒤에 다시 실행하세요.")
        cmd = _build_command(task, args or {})
        _seq += 1
        job_id = str(_seq)
        _jobs[job_id] = {"task": task, "lines": [f"$ {' '.join(cmd)}"],
                         "running": True, "exit": None, "result": {}, "pid": None}
    threading.Thread(target=_run, args=(job_id, cmd, on_done), daemon=True).start()
    return job_id


def job_status(job_id, start=0):
    job = _jobs.get(str(job_id))
    if job is None:
        raise ValueError(f"그런 작업이 없습니다: {job_id}")
    start = max(0, int(start))
    lines = job["lines"][start:]
    return {"task": job["task"], "running": job["running"], "exit": job["exit"],
            "lines": lines, "next": start + len(lines), "result": job["result"]}


def running_job():
    for jid, j in _jobs.items():
        if j["running"]:
            return jid
    return None


# ── 배포 ─────────────────────────────────────────────────────────────────
def remember_engine(engine_path):
    """settings.json 에 현재 POS 모델을 기록한다. (서버가 다시 떠도 유지되도록)"""
    p = safe_path(engine_path)
    s = read_json(SETTINGS)          # 깨졌으면 백업해 두고 새로 쓴다 (core/jsonfile.py)
    s["active_engine"] = str(p)
    write_json(SETTINGS, s)
    return str(p)


def remember_ensemble(engines):
    """앙상블 짝(대표 모델 제외)을 기록한다. 재부팅해도 jetson/start.sh 가 같은 짝으로 띄운다."""
    s = read_json(SETTINGS)
    s["ensemble_engines"] = [str(e) for e in engines]
    write_json(SETTINGS, s)


def disk_free_mb():
    return shutil.disk_usage(BASE_DIR).free // (1024 * 1024)
