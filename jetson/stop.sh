#!/bin/bash
# jetson/start.sh 로 켠 카메라 영상 서버와 AI 추론 서버를 끈다.
# (다른 폴더에서 켠 프로그램은 건드리지 않는다 — 켤 때 적어 둔 번호만 끈다)
RUN_DIR="${BREADVISION_RUN_DIR:-$HOME/.breadvision}"
QUIET=0; [ "$1" = "--quiet" ] && QUIET=1

for name in infer stream; do
    f="$RUN_DIR/$name.pid"
    [ -f "$f" ] || continue
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null && grep -qaE "inference_server.py|pc_camera_stream.py" "/proc/$pid/cmdline" 2>/dev/null; then
        kill "$pid" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
        kill -9 "$pid" 2>/dev/null
        [ $QUIET = 1 ] || echo "끔: $name (PID $pid)"
    fi
    rm -f "$f"
done
[ $QUIET = 1 ] || echo "종료했습니다."
exit 0
