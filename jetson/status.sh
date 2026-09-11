#!/bin/bash
# 지금 BreadVision 이 제대로 켜져 있는지 한눈에 보여준다.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STREAM_PORT="${BREADVISION_STREAM_PORT:-8090}"
INFER_PORT="${BREADVISION_INFER_PORT:-8092}"
RUN_DIR="${BREADVISION_RUN_DIR:-$HOME/.breadvision}"
IP=$(hostname -I | awk '{print $1}')

echo
echo "================ BreadVision 상태 ================"
echo "젯슨 IP 주소 : $IP"

if curl -s --max-time 3 -o /dev/null "http://localhost:$STREAM_PORT/stream" --max-filesize 1 2>/dev/null \
   || ss -ltn 2>/dev/null | grep -q ":$STREAM_PORT "; then
    echo "카메라 영상  : ✅ 켜짐   (브라우저에서 http://$IP:$STREAM_PORT/stream)"
else
    echo "카메라 영상  : ❌ 꺼짐   (로그: $RUN_DIR/logs/stream.log)"
fi

H=$(curl -s --max-time 3 "http://localhost:$INFER_PORT/health")
if echo "$H" | grep -q '"ready": true'; then
    echo "$H" | python3 -c "
import json, sys
d = json.load(sys.stdin)
name = lambda p: p.rsplit('/', 1)[-1] if p else '-'
ens = d.get('ensemble') or []
print('AI 추론 서버 : ✅ 켜짐   (포트 $INFER_PORT)')
print('  모델       :', ' + '.join(name(e) for e in ens) + '  (앙상블)' if ens else name(d.get('model_path')))
if d.get('specialists'):
    print('  역할 분담  :', ', '.join(f\"{k} → 클래스 {v}\" for k, v in d['specialists'].items()))
print('  클래스     :', len(d.get('names') or []), '종')
print('  턴테이블   :', '연결됨' if d.get('turntable') else '없음 (선택 사항)')
"
else
    echo "AI 추론 서버 : ❌ 꺼짐   (로그: $RUN_DIR/logs/infer.log)"
fi

if crontab -l 2>/dev/null | grep -q "breadvision:$ROOT"; then
    echo "자동 시작    : ✅ 부팅하면 자동으로 켜짐"
else
    echo "자동 시작    : ⚪ 등록 안 됨   (./jetson/install.sh 로 등록)"
fi
echo "=================================================="
echo "PC 에서는 run_pc.bat 을 더블클릭하면 이 젯슨을 자동으로 찾습니다."
echo
