#!/bin/bash
# 좌우(2번) 모터가 안 움직일 때 원인을 찾는 점검.
#   ./jetson/turntable_check.sh
# 모터를 눈으로 보면서 각 단계에서 무엇이 보이는지 확인하세요.
URL="http://localhost:${BREADVISION_INFER_PORT:-8092}"
post() { curl -s -m 120 -X POST "$URL$1" -H 'Content-Type: application/json' -d "${2:-{\}}"; echo; }
ask()  { echo; read -rp "   ▶ 준비되면 Enter 를 누르세요 "; }

echo "=== 좌우 모터 점검 ==="
curl -s -m 5 "$URL/health" | grep -q '"turntable": true' || { echo "젯슨 서버에 모터 보드가 안 보입니다. ls /dev/ttyACM* 를 확인하세요."; exit 1; }

echo
echo "[1/4] 배선 점검 — 드라이버(ULN2003) 판의 LED 4개를 보세요."
echo "      LED 가 1→2→3→4 순서로 하나씩 켜지면 배선 정상입니다."
ask; post /turn2/test
echo "      → LED 4개가 차례로 켜졌나요? 하나도 안 켜지면 전원/배선 문제입니다."

echo
echo "[2/4] 가장 힘이 센 설정으로 20도 (풀스텝 + 아주 느리게)"
ask; post /turn2/mode '{"half": false}'; post /turn2/speed '{"step_us": 6000}'; post /turn2 '{"deg": 20}'
sleep 5; echo "      → 움직였나요?"; post /turn2 '{"deg": -20}'; sleep 5

echo
echo "[3/4] 중간 설정으로 20도 (반스텝 + 느리게)"
ask; post /turn2/mode '{"half": true}'; post /turn2/speed '{"step_us": 4000}'; post /turn2 '{"deg": 20}'
sleep 4; echo "      → 움직였나요?"; post /turn2 '{"deg": -20}'; sleep 4

echo
echo "[4/4] 기본 설정으로 20도 (반스텝 2500us)"
ask; post /turn2/speed '{"step_us": 2500}'; post /turn2 '{"deg": 20}'
sleep 3; echo "      → 움직였나요?"; post /turn2 '{"deg": -20}'; sleep 3
post /turn2/stop

echo
echo "=== 결과 보는 법 ==="
echo " · 1단계 LED 가 안 켜짐        → 드라이버 전원(5V)이나 IN1~IN4 배선 문제"
echo " · LED 는 켜지는데 안 돌아감   → 힘 부족. 별도 5V 어댑터를 쓰거나 기구 마찰을 줄이세요"
echo " · 2단계만 돌아감              → 풀스텝 + 느린 속도로 쓰면 됩니다 (설정 알려주세요)"
echo " · 2·3단계 돌고 4단계만 안 됨  → 기본 속도가 너무 빠릅니다 (느리게 설정하겠습니다)"
echo " · 전부 잘 돌아감              → 촬영 중 각도가 너무 작아 못 느낀 것일 수 있습니다"
