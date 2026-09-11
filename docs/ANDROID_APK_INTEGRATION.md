
# Android APK 연동 계약

Supabase 테이블:
- profiles
- transactions
- point_history

회원가입:
Supabase Auth email/password 사용
회원가입 metadata:
- name
- phone

회원 프로필:
profiles.id = auth.users.id
profiles.qr_token = 개인 QR 토큰
profiles.point = 현재 포인트

APK QR payload:
BV1:<profiles.qr_token>

APK는 로그인한 사용자의 profiles만 조회합니다.
Jetson POS는 QR 토큰을 직접 테이블에서 조회하지 않고
Supabase RPC `pos_find_member()`를 호출합니다.

Jetson POS 결제 적립:
Supabase RPC `pos_complete_payment(member_id, total_price)`

포인트 적립률:
3%
