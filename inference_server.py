"""젯슨 AI 추론 서버 (TensorRT) + PC 원격 작업 창구.

PC 의 POS 프로그램이 사진(JPEG)을 보내면 빵 위치와 이름을 JSON 으로 돌려준다.
모델을 여러 개 주면 앙상블(소프트 보팅 + 역할 분담)로 합친다.

보통은 직접 실행하지 않고 ./jetson/start.sh 가 켠다 (database/settings.json 의 모델 사용).
직접 켤 때:
    python3 inference_server.py --engine models/bread_0911.engine --ensemble models/m12_B.engine --port 8092
확인:
    curl http://localhost:8092/health
"""
import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import jetson_ops
import turntable_ops

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "database" / "settings.json"

jobs = queue.Queue()
state = {"ready": False, "model_path": None, "conf": 0.45, "imgsz": 640,
         "n": 0, "ms": 0.0, "error": None}


def saved_engine():
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8")).get("active_engine", "")
    except Exception:
        return ""


def _full(path):
    """settings.json·명령줄의 상대 경로(models/x.engine)를 프로젝트 기준 절대 경로로."""
    p = Path(str(path))
    return str(p if p.is_absolute() else BASE_DIR / p) if str(path) else ""


def _names_of(engine_path):
    """엔진이 학습한 클래스 목록. 변환할 때 옆에 둔 <engine>.names.json 을 읽는다."""
    try:
        f = Path(str(engine_path) + ".names.json")
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


_SCOPES = {}


def _scope_of(engine_path):
    """엔진이 배운 클래스 범위 (<engine>.scope.json 의 classes). 없으면 None = 전부 안다.

    일부 빵만 학습한 모델(예: bread_0911 은 휘낭시에·와플 6종)을 앙상블에 넣을 때 쓴다.
    그 모델은 모르는 빵 자리에서 투표하지 못한다 (ai/fuse.py gate_specialists).
    """
    key = str(engine_path)
    if key not in _SCOPES:
        try:
            f = Path(key + ".scope.json")
            _SCOPES[key] = (set(json.loads(f.read_text(encoding="utf-8"))["classes"])
                            if f.exists() else None)
        except Exception:
            _SCOPES[key] = None
    return _SCOPES[key]


def _same_classes(primary, mate):
    """짝으로 붙여도 되는가. 모르면 None(경고만), 다르면 False(거부)."""
    a, b = _names_of(primary), _names_of(mate)
    if a is None or b is None:
        return None
    return a == b


def worker(engine_path, conf, extra_engines=(), min_votes=1):
    """CUDA 컨텍스트를 소유하는 유일한 스레드.

    엔진을 여러 개 주면 앙상블로 돈다. 한 모델이 놓친 빵을 다른 모델이 잡아
    놓침이 크게 줄어든다 (실측: 놓침 73 → 8건, ai/fuse.py 주석 참고).
    첫 번째 엔진이 '대표' 다 — 모델 교체(/deploy)는 그것만 바꾼다.
    """
    from ai.detector import JetsonTensorRTDetector      # pycuda 초기화가 이 스레드에서 일어나야 한다
    from ai.fuse import fuse, gate_specialists

    detector = JetsonTensorRTDetector(conf=conf)
    mates = []                                   # 앙상블 동료들
    try:
        detector.load(engine_path)
        state.update(ready=True, model_path=detector.model_path, conf=detector.conf,
                     imgsz=detector.imgsz)
        print(f"모델 로드 완료: {detector.model_path}  (conf={detector.conf})")
        for ep in extra_engines:
            # 클래스 목록이 다른 모델은 짝으로 붙이지 않는다.
            # 번호가 어긋나 같은 빵을 서로 다른 이름으로 부르고, WBF 는 같은
            # 클래스끼리만 합치므로 둘 다 살아남아 박스가 겹쳐 뜬다.
            # (2026-09-11: 12클래스 + 옛 7클래스 짝 → 와플 위에 pizza)
            ok = _same_classes(engine_path, ep)
            if ok is False:
                print(f"[앙상블] 짝 거부: {ep}\n"
                      f"        클래스 목록이 대표 모델과 다릅니다. 단일로 돕니다.")
                state.setdefault("ensemble_rejected", []).append(str(ep))
                continue
            if ok is None:
                print(f"[앙상블] 경고: {ep} 의 클래스 목록 파일(.names.json)이 없어 "
                      f"같은 모델 계열인지 확인하지 못했습니다.")
            d = JetsonTensorRTDetector(conf=conf)
            d.load(ep)
            mates.append(d)
            print(f"모델 로드 완료(앙상블): {d.model_path}")
        if mates:
            state["ensemble"] = [detector.model_path] + [d.model_path for d in mates]
            state["min_votes"] = min_votes
            print(f"[앙상블] 모델 {len(mates)+1}개, 동의 최소 {min_votes}표")
    except Exception as e:
        state["error"] = str(e)
        print(f"[오류] 모델 로드 실패: {e}")

    while True:
        kind, payload, reply = jobs.get()
        try:
            if kind == "load":
                if mates and Path(str(payload)).resolve() == Path(str(detector.model_path)).resolve():
                    # 지금 대표 모델을 다시 불러 달라는 것뿐이면 앙상블을 풀지 않는다
                    # (PC 의 「엔진 다시 불러오기」 버튼 등)
                    reply.put(("ok", {"ready": True, "model_path": detector.model_path,
                                      "ensemble_cleared": False}))
                    continue
                detector.load(payload)
                # 새로 학습한 모델로 갈아끼울 때는 앙상블 짝을 반드시 놓아 준다.
                #
                # 짝을 그대로 두면 이런 일이 난다 — 새 상품은 새 모델만 알고
                # 옛 짝은 모른다. WBF 는 한 모델만 본 검출의 신뢰도를 절반으로
                # 깎으므로(0.90 → 0.45), POS 문턱 0.55 를 못 넘어 새로 배운 빵이
                # 계산대에서 조용히 빠진다. 새 빵을 팔려고 학습했는데 그 빵만
                # 안 팔리는 셈이다.
                #
                # 짝도 같이 쓰려면 두 모델을 다 학습한 뒤 /ensemble 로 지정한다.
                if mates:
                    old = [d.model_path for d in mates]
                    for d in mates:
                        try:
                            d.close()
                        except Exception:
                            pass
                    mates.clear()
                    state["ensemble"] = None
                    try:
                        jetson_ops.remember_ensemble([])     # 재부팅 때 옛 짝이 다시 붙지 않게
                    except Exception:
                        pass
                    print(f"[앙상블] 모델을 교체했으므로 짝을 내렸습니다: {old}\n"
                          f"        새 모델 하나로 돕니다. 앙상블을 다시 쓰려면 "
                          f"POST /ensemble 로 짝을 지정하세요.")
                state.update(ready=True, model_path=detector.model_path, error=None)
                print(f"[모델 교체] {detector.model_path}")
                reply.put(("ok", {"ready": True, "model_path": detector.model_path,
                                  "ensemble_cleared": bool(state.get("ensemble") is None)}))

            elif kind == "ensemble":
                # 앙상블 짝을 통째로 다시 지정한다. 두 모델을 새로 학습한 뒤 쓴다.
                for d in mates:
                    try:
                        d.close()
                    except Exception:
                        pass
                mates.clear()
                loaded = []
                for ep in payload:
                    if _same_classes(detector.model_path, ep) is False:
                        print(f"[앙상블] 짝 거부: {ep} (클래스 목록이 다름)")
                        continue
                    d = JetsonTensorRTDetector(conf=conf)
                    d.load(ep)
                    mates.append(d)
                    loaded.append(d.model_path)
                state["ensemble"] = ([detector.model_path] + loaded) if loaded else None
                print(f"[앙상블] 짝 재지정: {loaded or '없음(단일)'}")
                reply.put(("ok", {"ensemble": state["ensemble"]}))
            elif kind == "infer":
                t0 = time.perf_counter()
                # payload = (프레임, tta). tta 는 「빵 인식」 버튼에서만 켠다.
                frame_in, tta = payload if isinstance(payload, tuple) else (payload, False)
                _, detections, counts = detector.infer(frame_in, draw=False, tta=tta)
                if mates:
                    lists = [detections]
                    for d in mates:
                        _, dets2, _ = d.infer(frame_in, draw=False, tta=tta)
                        lists.append(dets2)
                    # 일부 빵만 배운 모델(전문가)은 모르는 빵 자리에서 빠진다 (역할 분담)
                    scopes = [_scope_of(detector.model_path)] + [_scope_of(d.model_path) for d in mates]
                    lists = gate_specialists(lists, scopes)
                    # 모델끼리 같은 빵을 다른 이름으로 부르면 둘 다 남는다 → 교차억제
                    detections = fuse(lists, min_votes=min_votes, cross_class=True)
                    counts = {}
                    for d in detections:
                        counts[d["class_id"]] = counts.get(d["class_id"], 0) + 1
                ms = (time.perf_counter() - t0) * 1000
                state["n"] += 1
                state["ms"] += ms
                reply.put(("ok", {
                    "tta": bool(tta),
                    "detections": detections,
                    "counts": {str(k): v for k, v in counts.items()},
                    "infer_ms": round(ms, 1),
                }))
        except Exception as e:
            reply.put(("error", str(e)))


def submit(kind, payload, timeout=30.0):
    reply = queue.Queue(maxsize=1)
    jobs.put((kind, payload, reply))
    status, data = reply.get(timeout=timeout)
    if status != "ok":
        raise RuntimeError(data)
    return data


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        return u.path, {k: v[0] for k, v in parse_qs(u.query).items()}

    def do_GET(self):
        path, q = self._query()

        if path == "/health":
            avg = state["ms"] / state["n"] if state["n"] else 0.0
            self._json({
                "ready": state["ready"],
                "model_path": state["model_path"],
                # 클래스 이름 목록 (자동 라벨링이 번호를 이름으로 맞출 때 쓴다)
                "names": _names_of(state["model_path"]) if state["model_path"] else None,
                "conf": state["conf"],
                "imgsz": state["imgsz"],
                "served": state["n"],
                "avg_ms": round(avg, 1),
                "error": state["error"],
                "ops": True,                     # 작업 실행 창구가 있는 서버인지
                "busy": jetson_ops.running_job(),
                "free_mb": jetson_ops.disk_free_mb(),
                "ensemble": state.get("ensemble"),        # 앙상블이면 엔진 목록
                # 일부 빵만 배운 모델과 그 범위 (역할 분담 앙상블)
                "specialists": {Path(e).name: sorted(_scope_of(e))
                                for e in (state.get("ensemble") or []) if _scope_of(e)},
                "min_votes": state.get("min_votes", 1),
                "turntable": turntable_ops.available(),   # 모터 보드가 젯슨에 꽂혀 있나
                "turntable_state": turntable_ops.status(),
            })
            return

        if path == "/files":
            try:
                self._json({"files": jetson_ops.list_files(q.get("kind", "engine"))})
            except Exception as e:
                self._json({"error": str(e)}, 400)
            return

        if path == "/job":
            try:
                self._json(jetson_ops.job_status(q.get("id", ""), q.get("from", 0)))
            except Exception as e:
                self._json({"error": str(e)}, 404)
            return

        self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        path, q = self._query()

        # ── 턴테이블 ──────────────────────────────────────────────
        # 모터 보드는 젯슨 USB 에 꽂혀 있다. PC 는 시리얼을 열 수 없으므로
        # 여기로 "몇 도 돌려라" 만 보내고, 실제 회전은 젯슨이 한다.
        if path.startswith("/turn"):
            try:
                req = json.loads(body.decode("utf-8")) if body else {}
                if path == "/turn":
                    self._json(turntable_ops.rotate(req.get("deg", 0)))
                elif path == "/turn/speed":
                    self._json(turntable_ops.set_speed(req.get("step_us", 2500)))
                elif path == "/turn/release":
                    self._json(turntable_ops.release())
                elif path == "/turn/reset":
                    self._json(turntable_ops.reset_angle())
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": str(e)}, 500)
            return

        if path == "/upload":
            try:
                self._json(jetson_ops.save_upload(
                    q.get("name", "upload.zip"), body, q.get("replace")))
            except Exception as e:
                self._json({"error": str(e)}, 400)
            return

        if path == "/task":
            try:
                req = json.loads(body.decode("utf-8")) if body else {}
                task = req.pop("task", "")
                # 분할 전에 PC 의 상품 목록을 젯슨에 반영하고 data.yaml 을 새로 쓴다.
                # 클래스 이름의 기준은 PC 이고, 젯슨은 따라간다.
                if task == "split":
                    if req.get("products"):
                        jetson_ops.write_products(req.pop("products"))
                    jetson_ops.generate_data_yaml()
                elif task == "train":
                    jetson_ops.generate_data_yaml()
                # synth 는 data.yaml 을 새로 쓰지 않는다.
                # 젯슨의 products.json 이 데이터셋보다 적으면(신상품 등록 전 등)
                # 이름 목록이 짧아져, 라벨에 있는 클래스가 사라진 yaml 이 만들어진다.
                # 합성은 이름을 읽기만 하면 된다.
                self._json({"job": jetson_ops.start_task(task, req)})
            except Exception as e:
                self._json({"error": str(e)}, 400)
            return

        if path == "/ensemble":
            try:
                req = json.loads(body.decode("utf-8")) if body else {}
                engines = [jetson_ops.remember_engine(e) if e else e
                           for e in req.get("engines", [])]
                res = submit("ensemble", engines, timeout=180.0)
                jetson_ops.remember_ensemble((res.get("ensemble") or [])[1:])
                self._json(res)
            except Exception as e:
                self._json({"error": str(e)}, 400)
            return

        if path == "/deploy":
            try:
                engine = json.loads(body.decode("utf-8"))["engine"]
                full = jetson_ops.remember_engine(engine)
                res = submit("load", full, timeout=180.0)
                self._json({"deployed": full, **res})
            except Exception as e:
                self._json({"error": str(e)}, 400)
            return

        if path == "/load":
            try:
                model_path = json.loads(body.decode("utf-8"))["model_path"]
                self._json(submit("load", model_path))
            except Exception as e:
                self._json({"error": str(e)}, 400)
            return

        if path != "/infer":
            self._json({"error": "not found"}, 404)
            return

        if not state["ready"]:
            self._json({"error": "서버에 로드된 .engine 모델이 없습니다."}, 503)
            return

        try:
            frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("JPEG 를 디코딩하지 못했습니다.")
            tta = str(q.get("tta", "")).lower() in ("1", "true", "yes")
            self._json(submit("infer", (frame, tta)))
        except Exception as e:
            self._json({"error": str(e)}, 500)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="", help="비우면 database/settings.json 의 active_engine")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--ensemble", default="",
                    help="같이 돌릴 엔진들 (쉼표로 구분). 놓침을 줄인다.")
    ap.add_argument("--min-votes", type=int, default=1,
                    help="이만큼의 모델이 동의한 것만 남긴다 (오검출을 줄이되 놓침은 늘어난다)")
    args = ap.parse_args()

    engine = _full(args.engine or saved_engine())
    if not engine:
        raise SystemExit(
            "사용할 .engine 이 없습니다.\n"
            "  --engine models/bakery_add.engine 으로 지정하거나\n"
            "  GUI 에서 모델을 한 번 선택해 database/settings.json 에 저장하세요."
        )

    extra = [_full(e.strip()) for e in args.ensemble.split(",") if e.strip()]
    threading.Thread(target=worker,
                     args=(engine, args.conf, extra, max(1, args.min_votes)),
                     daemon=True).start()
    for _ in range(200):                      # 모델 로드 대기
        if state["ready"] or state["error"]:
            break
        time.sleep(0.1)

    print(f"추론 서버 시작: http://{args.host}:{args.port}")
    print(f"  PC 에서:  BREADVISION_INFER_URL=\"http://<젯슨IP>:{args.port}\" python main.py")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
