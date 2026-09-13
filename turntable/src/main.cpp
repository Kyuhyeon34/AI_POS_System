/*
 * 빵 촬영용 턴테이블 — STM32 Nucleo + ULN2003 + 28BYJ-48
 *
 * 젯슨이 USB 시리얼로 한 줄씩 명령을 보내면, 다 돌린 뒤 OK 를 돌려준다.
 * 젯슨은 그 OK 를 기다렸다가 사진을 찍으면 된다.
 *
 * 명령 (끝에 개행)
 *   P          연결 확인          → "OK TURNTABLE 1"
 *   R<각도>    각도만큼 회전       → "OK"      예) R90  R-30  R7.5
 *   S<스텝>    하프스텝만큼 회전   → "OK"      예) S1019
 *   V<마이크로초>  스텝 간격 설정  → "OK <값>"  예) V1500
 *   Z          코일 끄기          → "OK"
 *   T          코일 하나씩 켜보기   → "OK"      (LED 4개가 차례로 켜지는지 확인)
 *   C1~C4      그 코일만 켠 채 유지  → "OK C1"   (축을 손으로 만져 확인) / C0 = 끄기
 *   M0 / M1    구동 방식           → "OK FULL"  M0=힘 센 2상, M1=부드러운 하프스텝
 *   O<순서>    핀 순서 바꾸기      → "OK 1324"  예) O1324
 *
 *   28BYJ-48 은 IN 순서가 1-2-3-4 가 아니라 1-3-2-4 인 경우가 흔하다.
 *   순서가 어긋나면 모터가 웅웅대기만 하고 돌지 않는다.
 *   다시 굽지 않고 O 명령으로 바꿔 볼 수 있게 해 두었다.
 *   ?          현재 설정 보기      → "OK <간격> <한바퀴스텝> <누적스텝>"
 *
 * 배선
 *   ULN2003 IN1~IN4 → D2, D3, D4, D5   (Nucleo CN9)
 *   버튼 (선택)      → D6 = 왼쪽, D7 = 오른쪽
 *       버튼 한쪽은 D6/D7, 다른 쪽은 GND 에 연결한다.
 *       보드 내부 풀업을 쓰므로 저항을 따로 달 필요가 없다.
 *       누르고 있는 동안 계속 돌고, 떼면 멈춘다.
 *   ULN2003 +       → 젯슨 헤더 5V (핀 2 또는 4)   ← 보드 USB 에서 끌지 않는다
 *   ULN2003 −       → 젯슨 GND(핀 6) 와 Nucleo GND 를 함께 묶는다
 *
 * 2번 모터 (버튼 전용, 시리얼 명령 없음)
 *   ULN2003 #2 IN1~IN4 → D8, D9, D10, D11   (Nucleo CN5)
 *   버튼               → A0 = 왼쪽, A1 = 오른쪽   (CN8, 반대쪽 다리는 GND)
 *   ULN2003 #2 +/−     → 1번과 같은 5V / 공통 GND
 *
 *   D0·D1 은 ST-Link 가상 COM 이라 쓰면 시리얼이 깨진다.
 *   D13 은 보드 LED(작업 표시용)라 비워 둔다.
 *   버튼을 A0/A1 로 뺀 이유는 모터 배선(CN5·CN9)과 커넥터를 갈라
 *   코일 노이즈가 버튼 입력에 타지 않게 하려는 것이다.
 *
 *   전원을 젯슨에서 받는 이유: 28BYJ-48 은 300mA 쯤 먹는데
 *   Nucleo 의 USB 5V 로 끌어오면 ST-Link 쪽이 불안정해질 수 있다.
 *   GND 는 셋 다 공통이어야 신호 기준이 생긴다.
 */
#include <Arduino.h>

// CN9 커넥터의 D2~D5 → IN1, IN2, IN3, IN4.
// 여기가 틀리면 코일이 한 칸씩 밀려서 3상만 돌고, 모터가 웅웅대며 힘을 못 낸다.
// ULN2003 의 LED 로 확인할 수 있다: 코일 1 을 켰을 때 LED A 가 켜져야 한다.
//   python3 motor_serial.py --coils
static const uint8_t PINS[4] = {D2, D3, D4, D5};

// 손으로 위치를 맞출 때 쓰는 버튼. GND 로 떨어뜨리는 방식(내부 풀업).
static const uint8_t BTN_LEFT  = D6;
static const uint8_t BTN_RIGHT = D7;

// ── 2번 모터 — 버튼으로만 돈다 ───────────────────────────────────
//
// 1번과 같은 ULN2003 + 28BYJ-48 을 하나 더 붙인 것이다.
// 시리얼 명령(R/S/V/M/O…)은 전부 1번 모터에만 간다. 2번은 자동 회전이 없고
// 버튼을 누르고 있는 동안만 돈다.
//
// 핀을 이렇게 고른 이유
//   D0·D1  ST-Link 가상 COM (시리얼) — 쓰면 통신이 깨진다
//   D2~D7  1번 모터가 이미 씀
//   D13    보드 LED (작업 표시용)
//   → 남는 것 중 CN5 에 나란히 붙어 있는 D8~D11 을 코일로 쓰고,
//     버튼은 커넥터가 다른 A0/A1 로 빼서 코일 노이즈와 갈라 놓는다.
static const uint8_t PINS2[4] = {D8, D9, D10, D11};
static const uint8_t BTN2_LEFT  = A0;
static const uint8_t BTN2_RIGHT = A1;

// 2번 모터의 핀 순서. 1번과 같은 배선이면 그대로 두면 된다.
// 웅웅대기만 하고 안 돌면 여기를 {0,2,1,3} 으로 바꿔 다시 굽는다.
// (1번은 O 명령으로 바꿀 수 있지만 2번은 시리얼을 안 받으므로 코드로 정한다)
static uint8_t order2[4] = {0, 1, 2, 3};

// 2번은 하프스텝 고정. 1번의 M0/M1 에 영향받지 않는다.
static bool half2 = false;         // 2번 모터 방식 (N 명령). 기본 FULL — 코일 2개를 동시에
                                   // 켜서 힘이 세다 (LED 2개가 늘 켜져 보인다).
                                   // HALF 는 부드럽지만 코일 1개만 켜지는 구간이 있어 약하다.
static bool hold2 = false;         // 멈춘 뒤에도 코일을 켜 둘까 (H 명령).
                                   // 켜 두면 자리를 붙잡지만 모터가 뜨거워진다.
static const bool MOTOR2_HALF = true;
static uint32_t delay2Us = 2500;   // 2번 모터 스텝 간격 (U 명령). 느릴수록 힘이 세다
static const uint32_t MOTOR2_DELAY_US = 1500;   // (옛 기본값)

static int8_t phase2 = 0;

// 시퀀스의 각 자리가 어느 핀으로 갈지. O 명령으로 바꿀 수 있다.
//
// 이 장비에서는 1234 가 맞았다 (IN1,IN2,IN3,IN4 순서 그대로).
// 순서가 어긋나면 코일이 앞뒤로 당기기만 해서 모터가 웅웅대고 돌지 않는다.
// 배선을 다시 꽂았거나 다른 모터를 쓰면 O 명령으로 다시 찾으면 된다:
//   python3 motor_serial.py --try-orders
static uint8_t order[4] = {0, 1, 2, 3};

// 두 가지 구동 방식.
//
// FULL(기본) — 항상 코일 2개를 켠다. 힘이 제일 세다.
//   한 바퀴 2038 스텝.
// HALF       — 코일 1개와 2개를 번갈아 켠다. 더 부드럽고 분해능이 2배지만,
//   코일 1개만 켜지는 자리에서 힘이 뚝 떨어져 무거운 판을 못 돌리는 일이 있다.
//   한 바퀴 4076 스텝.
static const uint8_t SEQ_FULL[4][4] = {
    {1, 1, 0, 0}, {0, 1, 1, 0}, {0, 0, 1, 1}, {1, 0, 0, 1},
};
static const uint8_t SEQ_HALF[8][4] = {
    {1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 1, 0},
    {0, 0, 1, 0}, {0, 0, 1, 1}, {0, 0, 0, 1}, {1, 0, 0, 1},
};

// 28BYJ-48 의 실제 기어비는 63.684:1 이다. 흔히 쓰는 4096 은 어림값.
static const float REV_HALF = 4076.0f;
static const float REV_FULL = 2038.0f;

// 이 장비에서 실제로 돌았던 설정이 하프스텝 + 1500us 였다. 그걸 기본으로 둔다.
// 무거운 판을 올려 안 돌면 M0(FULL) 로 바꾸면 힘이 세진다.
static bool halfStep = true;
static float stepsPerRev() { return halfStep ? REV_HALF : REV_FULL; }

// 모드를 인자로 받는 쪽을 따로 둔다.
// 2번 모터는 시리얼 명령이 없으므로 1번의 M0/M1 에 따라 방식이 바뀌면 안 된다.
static uint8_t seqLenOf(bool half) { return half ? 8 : 4; }
static uint8_t seqAtOf(bool half, int8_t p, uint8_t i) {
    return half ? SEQ_HALF[p][i] : SEQ_FULL[p][i];
}
static uint8_t seqLen()    { return seqLenOf(halfStep); }
static uint8_t seqAt(int8_t p, uint8_t i) { return seqAtOf(halfStep, p, i); }

// 5V 기준. 하프스텝 1500us 면 약 9.8 RPM.
static uint32_t stepDelayUs = 1500;

// 멈춰 있다가 갑자기 목표 속도로 출발하면 관성 때문에 미끄러진다(탈조).
// 처음 몇 스텝을 느리게 시작해 서서히 붙인다. 무거운 판일수록 효과가 크다.
static const uint32_t RAMP_START_US = 9000;    // 출발 간격
static const uint16_t RAMP_STEPS = 60;         // 몇 스텝에 걸쳐 붙일지

static int8_t phase = 0;        // 코일을 껐다 켜도 이어지도록 위상을 기억
static long totalSteps = 0;     // 1번 누적 (밀림 확인용)
static long totalSteps2 = 0;    // 2번 누적 — 촬영이 끝나고 정확히 제자리로 돌아가려면 필요하다

// 어느 모터인지 인자로 받는다. 1번과 2번이 같은 코드를 쓴다.
static void applyPhaseOn(const uint8_t *pins, const uint8_t *ord,
                         bool half, int8_t p) {
    for (uint8_t i = 0; i < 4; i++) {
        digitalWrite(pins[ord[i]], seqAtOf(half, p, i) ? HIGH : LOW);
    }
}

static void releaseOn(const uint8_t *pins) {
    for (uint8_t i = 0; i < 4; i++) digitalWrite(pins[i], LOW);
}

static void applyPhase(int8_t p) {
    applyPhaseOn(PINS, order, halfStep, p);
}



static void release() { releaseOn(PINS); }

// 코일을 하나씩 켜 본다. ULN2003 의 LED 4개가 차례로 켜져야 한다.
// 안 켜지는 게 있으면 그 선이 안 물린 것 — 순서 문제가 아니라 배선 문제다.
static void testCoils() {
    for (uint8_t i = 0; i < 4; i++) {
        for (uint8_t j = 0; j < 4; j++) digitalWrite(PINS[j], (i == j) ? HIGH : LOW);
        delay(700);
    }
    release();
}

// 한 스텝만. 코일은 끄지 않는다 (버튼으로 계속 돌릴 때 필요).
static void stepOnce(int8_t dir) {
    uint8_t len = seqLen();
    phase = (int8_t)((phase + dir + len) % len);
    applyPhase(phase);
    totalSteps += dir;
}

static void serviceMotor2();      // 아래에 정의 (2번 모터를 한 스텝 돌본다)

// 1번 모터가 도는 동안 2번 모터가 멈추지 않게, 기다리는 시간을 잘게 쪼개
// 그 틈에 2번 모터(버튼·W 명령)를 돌본다. 두 모터가 동시에 움직인다.
static void waitUs(uint32_t us) {
    uint32_t t0 = micros();
    while ((uint32_t)(micros() - t0) < us) {
        serviceMotor2();
        delayMicroseconds(20);
    }
}

static void stepBy(long n) {
    int8_t dir = (n >= 0) ? 1 : -1;
    long count = labs(n);
    uint8_t len = seqLen();

    for (long i = 0; i < count; i++) {
        phase = (int8_t)((phase + dir + len) % len);
        applyPhase(phase);

        // 출발할 때와 멈출 때를 느리게 (양끝에서 미끄러지는 걸 막는다)
        long fromEnd = count - 1 - i;
        long edge = (i < fromEnd) ? i : fromEnd;
        uint32_t d = stepDelayUs;
        if (edge < RAMP_STEPS && RAMP_START_US > stepDelayUs) {
            uint32_t extra = RAMP_START_US - stepDelayUs;
            d += extra - (extra * (uint32_t)edge) / RAMP_STEPS;
        }
        waitUs(d);
    }
    totalSteps += n;
    // 멈춰 있는 동안 코일을 켜 두면 모터가 뜨거워진다. 턴테이블은 유지 토크가 필요 없다.
    release();
}

void setup() {
    for (uint8_t i = 0; i < 4; i++) {
        pinMode(PINS[i], OUTPUT);
        digitalWrite(PINS[i], LOW);
    }
    pinMode(BTN_LEFT, INPUT_PULLUP);
    pinMode(BTN_RIGHT, INPUT_PULLUP);

    for (uint8_t i = 0; i < 4; i++) {
        pinMode(PINS2[i], OUTPUT);
        digitalWrite(PINS2[i], LOW);
    }
    pinMode(BTN2_LEFT, INPUT_PULLUP);
    pinMode(BTN2_RIGHT, INPUT_PULLUP);

    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(115200);
}

// 버튼을 누르고 있는 동안 한 스텝씩 돌린다.
//
// stepBy 처럼 통째로 막아 버리면 버튼을 떼도 멈추지 않으므로,
// loop 를 돌 때마다 한 스텝씩만 보낸다.
//
// 기다리는 방식이 delayMicroseconds 가 아니라 micros() 비교인 이유:
// 모터가 둘이라 한쪽이 1.5ms 를 붙잡고 있으면 다른 쪽도 그만큼 늦어져
// 두 버튼을 같이 누를 때 둘 다 반쯤 느려진다. 시각만 확인하고 바로 빠져
// 나오면 서로 방해하지 않는다.
struct ButtonMotor {
    const uint8_t *pins;
    uint8_t *ord;
    bool half;
    uint32_t delayUs;
    uint8_t btnL, btnR;
    int8_t *phase;
    bool moving;
    uint32_t lastUs;
    bool countSteps;          // 1번만 누적 스텝을 센다 (? 명령이 읽는 값)
    long remain;              // 시리얼(W 명령)로 받은 남은 스텝. loop 가 조금씩 소화한다
    int8_t autoDir;           // 그 방향
    long total;               // 이번 이동의 전체 스텝 수 (출발·도착을 느리게 하려고)
};

static ButtonMotor bm1 = {PINS,  order,  true, 0, BTN_LEFT,  BTN_RIGHT,
                          &phase,  false, 0, true, 0, 0, 0};
static ButtonMotor bm2 = {PINS2, order2, false, 2500,
                          BTN2_LEFT, BTN2_RIGHT, &phase2, false, 0, false, 0, 0, 0};

static bool handleButtonMotor(ButtonMotor &m) {
    bool left  = (digitalRead(m.btnL) == LOW);
    bool right = (digitalRead(m.btnR) == LOW);

    // 자동 이동(W·D 명령)은 "버튼을 대신 눌러 주는" 것으로 처리한다.
    // 사람이 버튼으로 돌릴 때와 완전히 같은 코드·같은 속도로 돌게 하려는 것이다.
    // 버튼을 실제로 누르면 그쪽이 우선이고 자동 이동은 취소된다.
    bool autoMove = false;
    if (left == right) {
        if (m.remain > 0) {
            autoMove = true;
            left  = (m.autoDir < 0);
            right = (m.autoDir > 0);
        } else {
            if (m.moving) {
                bool keepOn = (!m.countSteps) && hold2;   // 2번 모터만 붙잡아 둘 수 있다
                if (!keepOn) releaseOn(m.pins);           // 켜 둔 채로 두면 뜨거워진다
                m.moving = false;
            }
            return false;
        }
    } else {
        m.remain = 0;                 // 사람이 버튼을 누르면 자동 이동은 그만둔다
    }

    uint32_t want = m.delayUs ? m.delayUs : stepDelayUs;
    uint32_t now = micros();
    if (m.moving && (uint32_t)(now - m.lastUs) < want) {
        return true;                  // 아직 다음 스텝 시각이 아니다
    }
    m.lastUs = now;
    m.moving = true;

    int8_t dir = left ? -1 : 1;
    uint8_t len = seqLenOf(m.half);
    *m.phase = (int8_t)((*m.phase + dir + len) % len);
    applyPhaseOn(m.pins, m.ord, m.half, *m.phase);
    if (m.countSteps) totalSteps += dir;
    else totalSteps2 += dir;

    if (autoMove && --m.remain <= 0) {
        bool keepOn = (!m.countSteps) && hold2;
        if (!keepOn) releaseOn(m.pins);
        m.moving = false;
    }
    return true;
}

// 2번 모터 한 번 돌보기 (1번 모터가 도는 동안에도 불린다)
static void serviceMotor2() {
    handleButtonMotor(bm2);
}

static void handleButtons() {
    bool a = handleButtonMotor(bm1);
    bool b = handleButtonMotor(bm2);
    digitalWrite(LED_BUILTIN, (a || b) ? HIGH : LOW);
}

void loop() {
    handleButtons();

    static String line;
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            line.trim();
            if (line.length() == 0) { line = ""; continue; }

            digitalWrite(LED_BUILTIN, HIGH);   // 처리 중 표시
            char cmd = (char)toupper(line[0]);
            String arg = line.substring(1);

            if (cmd == 'P') {
                Serial.println("OK TURNTABLE 1");
            } else if (cmd == 'R') {
                long n = lroundf(arg.toFloat() / 360.0f * stepsPerRev());
                stepBy(n);
                Serial.println("OK");
            } else if (cmd == 'S') {
                stepBy(arg.toInt());
                Serial.println("OK");
            } else if (cmd == 'V') {
                long v = arg.toInt();
                if (v < 800 || v > 300000) {
                    // 800us 아래는 탈조가 난다. 위쪽은 "촬영 내내 아주 천천히"
                    // 연속으로 돌릴 때 필요해서 0.3초까지 열어 둔다.
                    Serial.println("ERR 간격은 800~300000us");
                } else {
                    stepDelayUs = (uint32_t)v;
                    Serial.print("OK "); Serial.println(stepDelayUs);
                }
            } else if (cmd == 'M') {
                // M0 = FULL(힘 셈, 2038), M1 = HALF(부드러움, 4076)
                halfStep = (arg.toInt() != 0);
                release();
                phase = 0;
                Serial.print("OK "); Serial.println(halfStep ? "HALF" : "FULL");
            } else if (cmd == 'C') {
                // C1~C4 = 그 코일만 켠 채로 유지, C0 = 끄기
                int n = arg.toInt();
                if (n < 0 || n > 4) {
                    Serial.println("ERR C0~C4");
                } else {
                    for (uint8_t j = 0; j < 4; j++)
                        digitalWrite(PINS[j], (n && (int)j == n - 1) ? HIGH : LOW);
                    Serial.print("OK C"); Serial.println(n);
                }
            } else if (cmd == 'T') {
                testCoils();
                Serial.println("OK");
            } else if (cmd == 'O') {
                // "1324" 처럼 1~4 를 한 번씩 쓴 네 자리
                bool good = (arg.length() == 4);
                bool seen[4] = {false, false, false, false};
                uint8_t tmp[4];
                for (uint8_t i = 0; good && i < 4; i++) {
                    int v = arg[i] - '1';
                    if (v < 0 || v > 3 || seen[v]) { good = false; break; }
                    seen[v] = true;
                    tmp[i] = (uint8_t)v;
                }
                if (!good) {
                    Serial.println("ERR 1~4 를 한 번씩 쓴 네 자리로 (예: O1324)");
                } else {
                    for (uint8_t i = 0; i < 4; i++) order[i] = tmp[i];
                    release();
                    phase = 0;
                    Serial.print("OK "); Serial.println(arg);
                }
            } else if (cmd == 'Z') {
                release();
                Serial.println("OK");
            } else if (cmd == '?') {
                Serial.print("OK ");
                Serial.print(stepDelayUs); Serial.print(' ');
                Serial.print((long)stepsPerRev()); Serial.print(' ');
                Serial.print(totalSteps); Serial.print(' ');
                for (uint8_t i = 0; i < 4; i++) Serial.print(order[i] + 1);
                Serial.print(' ');
                Serial.println(halfStep ? "HALF" : "FULL");
            } else if (cmd == 'D') {
                // 회전판(1번)을 몇 도 — 바로 응답하고 뒤에서 조금씩 돈다.
                // R 은 다 돌 때까지 기다리지만, D 는 기다리지 않으므로
                // 그동안 좌우 모터 명령도 따로 받을 수 있다 (완전 독립).
                long n = lroundf(arg.toFloat() / 360.0f * stepsPerRev());
                bm1.autoDir = (n >= 0) ? 1 : -1;
                bm1.remain = labs(n);
                bm1.total = bm1.remain;
                Serial.print("OK "); Serial.println(bm1.remain);
            } else if (cmd == 'W') {
                // 2번(좌우) 모터를 몇 도. 바로 응답하고 뒤에서 조금씩 돈다
                // → 1번 모터(R/S)와 동시에 움직일 수 있다.
                float rev2 = half2 ? REV_HALF : REV_FULL;
                long n = lroundf(arg.toFloat() / 360.0f * rev2);
                bm2.autoDir = (n >= 0) ? 1 : -1;
                bm2.remain = labs(n);
                bm2.total = bm2.remain;
                Serial.print("OK "); Serial.println(bm2.remain);
            } else if (cmd == 'U') {
                // 2번 모터 스텝 간격(us). 느릴수록 힘이 세다 (기본 2500)
                long v = arg.toInt();
                if (v < 800 || v > 300000) {
                    Serial.println("ERR 간격은 800~300000us");
                } else {
                    delay2Us = (uint32_t)v; bm2.delayUs = delay2Us;
                    Serial.print("OK "); Serial.println(delay2Us);
                }
            } else if (cmd == 'N') {
                // N0 = FULL(힘 셈), N1 = HALF(부드러움). 2번 모터만
                half2 = (arg.toInt() != 0); bm2.half = half2;
                releaseOn(PINS2); phase2 = 0;
                Serial.print("OK "); Serial.println(half2 ? "HALF" : "FULL");
            } else if (cmd == 'Q') {
                // 2번 모터 코일 순서 (예: Q1324).
                // 배선 순서가 다르면 제자리에서 떨기만 하고 돌지 않는다.
                bool good = (arg.length() == 4);
                bool seen[4] = {false, false, false, false};
                uint8_t tmp[4];
                for (uint8_t i = 0; good && i < 4; i++) {
                    int v = arg[i] - '1';
                    if (v < 0 || v > 3 || seen[v]) { good = false; break; }
                    seen[v] = true;
                    tmp[i] = (uint8_t)v;
                }
                if (!good) {
                    Serial.println("ERR 1~4 를 한 번씩 쓴 네 자리로 (예: Q1324)");
                } else {
                    for (uint8_t i = 0; i < 4; i++) order2[i] = tmp[i];
                    releaseOn(PINS2);
                    phase2 = 0;
                    Serial.print("OK "); Serial.println(arg);
                }
            } else if (cmd == 'H') {
                // H1 = 멈춘 뒤에도 2번 모터 코일을 켜 둔다 (자리를 붙잡음, 뜨거워짐)
                hold2 = (arg.toInt() != 0);
                if (!hold2) releaseOn(PINS2);
                Serial.print("OK "); Serial.println(hold2 ? "HOLD" : "FREE");
            } else if (cmd == 'G') {
                // 2번 모터 코일 점검: 드라이버 LED 4개가 차례로 켜져야 한다
                for (uint8_t i = 0; i < 4; i++) {
                    for (uint8_t j = 0; j < 4; j++)
                        digitalWrite(PINS2[j], (i == j) ? HIGH : LOW);
                    delay(700);
                }
                releaseOn(PINS2);
                Serial.println("OK");
            } else if (cmd == 'X') {
                // X1 = 회전판 멈춤, X(또는 X2) = 좌우 모터 멈춤
                if (arg.toInt() == 1) {
                    bm1.remain = 0; bm1.moving = false; release();
                } else {
                    bm2.remain = 0; bm2.moving = false; releaseOn(PINS2);
                }
                Serial.println("OK");
            } else if (cmd == 'K') {
                // 두 모터가 출발점에서 몇 스텝 가 있는지 (제자리로 돌릴 때 쓴다)
                // K0 을 보내면 지금 자리를 0 으로 삼는다 (버튼으로 얼마나 돌렸는지 잴 때)
                if (arg.length() && arg[0] == '0') { totalSteps = 0; totalSteps2 = 0; }
                Serial.print("OK ");
                Serial.print(totalSteps); Serial.print(' ');
                Serial.print(totalSteps2); Serial.print(' ');
                Serial.print((long)stepsPerRev()); Serial.print(' ');
                Serial.println((long)(half2 ? REV_HALF : REV_FULL));
            } else if (cmd == 'Y') {
                // Y1 = 회전판에 남은 스텝, Y(또는 Y2) = 좌우 모터
                long v = (arg.toInt() == 1) ? bm1.remain : bm2.remain;
                Serial.print("OK "); Serial.println(v);
            } else {
                Serial.print("ERR 알 수 없는 명령: "); Serial.println(cmd);
            }
            digitalWrite(LED_BUILTIN, LOW);
            line = "";
        } else if (line.length() < 32) {
            line += c;
        }
    }
}
