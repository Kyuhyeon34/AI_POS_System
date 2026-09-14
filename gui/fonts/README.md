# 폰트

| 파일 | 폰트 | 쓰는 곳 | 라이선스 |
|---|---|---|---|
| `Jua-Regular.ttf` | 배민 주아체 (Jua) | POS 탭·제목·버튼·결제 금액 | SIL Open Font License 1.1 — Copyright 2018 The BM JUA Project Authors |

- 회원 앱(BreadVision Member)과 같은 폰트다. `gui/main_window.py` 의 `load_fonts()` 가 시작할 때 등록한다.
- 표·입력칸·긴 안내문은 읽기 쉽게 맑은 고딕(젯슨은 Noto Sans CJK KR)을 그대로 쓴다.
- Jua 에는 `→ · — ● ✔` 같은 기호가 없어서, 그런 글자는 운영체제가 다른 폰트로 대신 그린다.
- 라이선스 전문: https://openfontlicense.org
