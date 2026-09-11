"""PC 에서 켤 때 젯슨을 알아서 찾아 붙는다.

새 PC 는 폴더를 복사하고 main.py 를 실행하기만 하면 된다.
PowerShell 에 환경변수를 매번 칠 필요가 없다.

    1) BREADVISION_INFER_URL 을 이미 지정했으면 손대지 않는다 (개발용 우회)
    2) jetson.json 에 적힌 주소로 /health 를 물어본다
    3) 응답이 없으면 같은 대역을 훑어 8092 가 열린 젯슨을 찾는다.
       찾으면 jetson.json 에 다시 적어 둔다
       (DHCP 로 주소가 바뀌어도 사람이 손댈 일이 없다)

젯슨 자신에서 돌 때는 아무것도 하지 않는다.
거기서는 환경변수가 비어 있어야 로컬 TensorRT 를 쓴다 (ai/factory.py).
"""
import json
import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "jetson.json"

# 포트는 jetson.json 에서 바꿀 수 있다 (없으면 아래 기본값).
#   {"host": "192.168.0.23", "infer_port": 8092, "stream_port": 8090}
# 젯슨 주소는 처음 찾았을 때 jetson.json 에 자동으로 적힌다.
INFER_PORT = 8092
STREAM_PORT = 8090

PROBE_TIMEOUT = 2.0         # 저장된 주소에 물어볼 때
SCAN_CONNECT_TIMEOUT = 0.4  # 대역을 훑을 때 한 대당
SCAN_WORKERS = 64


def _is_jetson():
    """여기가 젯슨인가. 젯슨에서는 자동 연결을 하면 안 된다."""
    return Path("/etc/nv_tegra_release").exists()


# ---- 설정 파일 --------------------------------------------------------
def load_config():
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[젯슨 연결] {CONFIG_PATH.name} 을 읽지 못했습니다: {e}")
        return {}


def save_host(host):
    """찾은 주소를 기억해 둔다. 다음 실행은 훑지 않고 바로 붙는다."""
    cfg = load_config()
    if cfg.get("host") == host:
        return
    cfg["host"] = host
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"[젯슨 연결] 주소를 {CONFIG_PATH.name} 에 저장했습니다: {host}")
    except Exception as e:
        print(f"[젯슨 연결] 주소를 저장하지 못했습니다: {e}")


# ---- 탐색 -------------------------------------------------------------
def _port_open(host, port, timeout=SCAN_CONNECT_TIMEOUT):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe(host, timeout=PROBE_TIMEOUT):
    """추론·작업 서버가 맞는지 확인한다. 맞으면 /health 응답을 돌려준다.

    포트만 열린 다른 장비를 젯슨으로 오인하지 않도록 ops 플래그까지 본다.
    """
    import urllib.request
    try:
        url = f"http://{host}:{INFER_PORT}/health"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            info = json.loads(r.read().decode("utf-8"))
        return info if info.get("ops") else None
    except Exception:
        return None


def local_ipv4s():
    """이 PC 가 쓰는 IPv4 주소들. 어느 대역을 훑을지 정하는 데 쓴다."""
    found = []
    # 기본 경로로 나가는 주소 (패킷은 보내지 않는다)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        found.append(s.getsockname()[0])
    except OSError:
        pass
    finally:
        s.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass

    out = []
    for ip in found:
        if ip.startswith("127.") or ip in out:
            continue
        out.append(ip)
    return out


def _prefixes(saved_host):
    """훑을 /24 대역들. 이 PC 의 대역이 먼저, 저장된 주소의 대역이 그 다음."""
    prefixes = []
    for ip in local_ipv4s():
        p = ip.rsplit(".", 1)[0]
        if p not in prefixes:
            prefixes.append(p)
    if saved_host and saved_host.count(".") == 3:
        p = saved_host.rsplit(".", 1)[0]
        if p not in prefixes:
            prefixes.append(p)
    return prefixes


def scan(saved_host=None, verbose=True):
    """같은 대역에서 8092 가 열린 젯슨을 찾는다. 없으면 None."""
    prefixes = _prefixes(saved_host)
    if not prefixes:
        return None

    for prefix in prefixes:
        if verbose:
            print(f"[젯슨 연결] {prefix}.0/24 를 훑는 중… (몇 초 걸립니다)")
        hosts = [f"{prefix}.{i}" for i in range(1, 255)]
        with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as pool:
            opened = [h for h, ok in zip(hosts, pool.map(
                lambda h: _port_open(h, INFER_PORT), hosts)) if ok]

        # 포트가 열린 것 중 진짜 우리 서버를 고른다
        for host in opened:
            if probe(host, timeout=PROBE_TIMEOUT):
                return host

    return None


# ---- 진입점 -----------------------------------------------------------
def autoconfigure(verbose=True):
    """환경변수를 채운다. 젯슨 주소를 돌려주고, 못 찾으면 None."""
    global INFER_PORT, STREAM_PORT
    if _is_jetson():
        return None

    # 사람이 직접 지정했으면 그게 우선이다 (빈 문자열로 준 것도 뜻이 있다)
    if "BREADVISION_INFER_URL" in os.environ:
        # 추론 주소만 주고 카메라 주소를 빠뜨리면 PC 내장 카메라가 열린다.
        # 같은 젯슨의 영상 주소를 채워 둔다.
        from urllib.parse import urlparse
        host = urlparse(os.environ["BREADVISION_INFER_URL"].strip()).hostname
        if host:
            port = int(load_config().get("stream_port", STREAM_PORT))
            os.environ["BREADVISION_JETSON_STREAM"] = f"http://{host}:{port}/stream"
            os.environ.setdefault("BREADVISION_CAMERA", os.environ["BREADVISION_JETSON_STREAM"])
        return None

    cfg = load_config()
    saved = str(cfg.get("host", "")).strip()

    INFER_PORT = int(cfg.get("infer_port", INFER_PORT))
    STREAM_PORT = int(cfg.get("stream_port", STREAM_PORT))

    host = None
    if saved:
        if probe(saved):
            host = saved
        elif verbose:
            print(f"[젯슨 연결] {saved} 이 응답하지 않습니다.")

    if host is None and os.environ.get("BREADVISION_NO_SCAN", "").strip() != "1":
        host = scan(saved, verbose=verbose)
        if host:
            save_host(host)

    if host is None:
        print("[젯슨 연결] 젯슨을 찾지 못했습니다.\n"
              "        젯슨에서 ./jetson/status.sh 로 켜져 있는지 확인하세요.\n"
              f"        주소를 아신다면 {CONFIG_PATH.name} 의 host 를 고치면 됩니다.")
        return None

    os.environ["BREADVISION_INFER_URL"] = f"http://{host}:{INFER_PORT}"
    os.environ["BREADVISION_JETSON_STREAM"] = f"http://{host}:{STREAM_PORT}/stream"
    os.environ.setdefault("BREADVISION_CAMERA", os.environ["BREADVISION_JETSON_STREAM"])
    if verbose:
        print(f"[젯슨 연결] {host} — 추론 {INFER_PORT}, 카메라 {STREAM_PORT}")
    return host


if __name__ == "__main__":
    # 연결만 확인해 보고 싶을 때:  py -3.11 pc_link.py
    host = autoconfigure()
    if not host:
        sys.exit(1)
    info = probe(host)
    print(json.dumps(info, ensure_ascii=False, indent=2))
