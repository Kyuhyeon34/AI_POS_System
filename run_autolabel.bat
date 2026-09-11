@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 자동 라벨링 + 검수 도구 - 더블클릭으로 실행한다.
rem 젯슨 주소는 main.py 와 같은 방식(jetson.json → 자동 탐색)으로 찾는다.

set "PY=py -3.11"
py -3.11 -c "pass" >nul 2>&1
if errorlevel 1 set "PY=python"

%PY% -c "import PyQt5, cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo [설치] 처음 실행이라 필요한 패키지를 받습니다. 몇 분 걸립니다...
    %PY% -m pip install -r requirements-pc.txt
    if errorlevel 1 goto fail
)

%PY% autolabel.py %*
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
echo 오류로 끝났습니다. 위 메시지를 확인하세요.
pause
exit /b 1
