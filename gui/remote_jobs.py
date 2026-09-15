"""젯슨에 건 작업(분할·합성·학습·변환)의 진행을 1초마다 받아 온다.

    runner.start("train", "전이학습", on_success, model=..., epochs=...)
    → log_line 신호로 로그가 한 줄씩 오고, 끝나면 on_success(result) 가 불린다.
    → 실패하면 failed(title, message) 신호.
"""
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from core.log import get_logger

log = get_logger("jobs")

# 젯슨 작업 창구(ai/ops_client.py)가 던지는 오류:
#   연결 실패·시간 초과(OSError), 서버가 거절한 이유(RuntimeError), 응답 형식(ValueError, KeyError)
OPS_ERRORS = (OSError, RuntimeError, ValueError, KeyError)


class RemoteJobRunner(QObject):
    log_line = pyqtSignal(str)
    started = pyqtSignal(str)
    finished = pyqtSignal()
    failed = pyqtSignal(str, str)       # 작업 이름, 사유

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.job = None
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.poll)

    @property
    def busy(self):
        return self.job is not None

    def start(self, task, title, on_success, **args):
        """작업을 건다. 걸지 못하면 OPS_ERRORS 중 하나를 그대로 올린다."""
        job_id = self.ctx.ops.start(task, **args)
        self.job = {"id": job_id, "next": 0, "title": title, "on_success": on_success}
        log.info("젯슨 작업 시작: %s (%s) %s", title, task, job_id)
        self.log_line.emit("")
        self.log_line.emit(f"[젯슨] {title} 시작")
        self.started.emit(title)
        self.timer.start()

    def poll(self):
        job = self.job
        if not job:
            self.timer.stop()
            return
        try:
            st = self.ctx.ops.poll(job["id"], job["next"])
        except OPS_ERRORS as e:
            log.warning("젯슨 작업 진행을 받지 못했습니다: %s", e)
            self.log_line.emit(f"[젯슨] 연결 끊김: {e}")
            self._finish(None)
            self.failed.emit(job["title"], f"젯슨과 연결이 끊겼습니다: {e}")
            return

        for line in st.get("lines", []):
            self.log_line.emit(line)
        job["next"] = st.get("next", job["next"])
        if st.get("running"):
            return

        code = st.get("exit")
        if code == 0:
            self.log_line.emit(f"[젯슨] {job['title']} 완료")
            self._finish(st.get("result") or {})
        else:
            self.log_line.emit(f"[젯슨] {job['title']} 실패 (exit={code})")
            log.warning("젯슨 작업 실패: %s exit=%s", job["title"], code)
            self._finish(None)
            self.failed.emit(job["title"], "젯슨에서 작업이 정상적으로 끝나지 않았습니다.\n"
                                           "아래 로그를 확인해주세요.")

    def _finish(self, result):
        self.timer.stop()
        job, self.job = self.job, None
        self.finished.emit()
        if result is not None and job and job.get("on_success"):
            job["on_success"](result)
