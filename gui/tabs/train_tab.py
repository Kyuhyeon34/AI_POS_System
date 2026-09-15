"""모델 배포 및 모델 학습 탭.

두 가지 실행 위치
    PC + 젯슨 연결: 버튼을 누르면 젯슨이 대신 실행한다 (gui/remote_jobs.py)
    젯슨에서 직접 GUI 실행: 여기서 QProcess 로 스크립트를 돌린다
TensorRT ENGINE 은 그 기기·그 버전 전용 바이너리라 PC 에서는 만들 수 없다.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import (QApplication, QCheckBox, QFileDialog, QGridLayout, QGroupBox,
                             QInputDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                             QSpinBox, QVBoxLayout, QWidget)

from core.log import get_logger
from gui import style
from gui.remote_jobs import OPS_ERRORS, RemoteJobRunner
from gui.style import button, page_header

log = get_logger("train")


class TrainTab(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.process = None             # 젯슨에서 직접 실행할 때의 학습/변환 프로세스
        self.last_best_model = ""
        self.pending_engine = ""

        self.jobs = RemoteJobRunner(ctx, self)
        self.jobs.log_line.connect(lambda line: self.log.appendPlainText(line))
        self.jobs.started.connect(lambda title: ctx.show_status(f"젯슨에서 {title} 진행 중…"))
        self.jobs.finished.connect(ctx.clear_status)
        self.jobs.failed.connect(self._on_job_failed)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)
        page_header(root, "모델 배포 및 모델 학습",
                    "정식 업데이트는 Colab ONNX 배포를 그대로 사용하고, 긴급 신상품은 전이학습할 수 있습니다.")

        self.ops_status = QLabel()
        self.ops_status.setWordWrap(True)
        root.addWidget(self.ops_status)
        root.addWidget(self._build_deploy())
        root.addWidget(self._build_training())

        self.phase_label = QLabel("")
        self.phase_label.setWordWrap(True)
        self.phase_label.setVisible(False)
        root.addWidget(self.phase_label)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(220)
        self.log.setPlainText(
            "[운영 원칙] Colab = 정식 학습 / Jetson = 긴급 전이학습 / TensorRT ENGINE = 실제 POS 추론\n"
            "dataset/import 에 원본 ZIP을 넣고 import_master_dataset.py를 실행하면 master 데이터셋을 만들 수 있습니다.")
        root.addWidget(self.log)

        note = QLabel(
            "학습은 젯슨의 NVIDIA PyTorch Docker 컨테이너에서 돌기 때문에 젯슨의 파이썬 환경을 건드리지 않습니다. "
            "TensorRT ENGINE 은 그 기기·그 버전 전용 바이너리라 PC 에서는 만들 수 없어, 변환도 젯슨이 맡습니다. "
            "PC 에서는 파일을 보내고 버튼만 누르면 됩니다.")
        note.setWordWrap(True)
        note.setStyleSheet(style.NOTE_BOX)
        root.addWidget(note)
        root.addStretch(1)

        self.refresh_ops_status()
        self.refresh_model_labels()

    # ── 화면 구성 ────────────────────────────────────────────────────────
    def _build_deploy(self):
        box = QGroupBox("① 외부 모델 배포 — Colab → Jetson")
        lay = QGridLayout(box)
        self.onnx_edit = QLineEdit()
        self.onnx_edit.setPlaceholderText("Colab에서 만든 best.onnx 선택")
        self.engine_label = QLabel("활성 ENGINE: 미설정")
        self.engine_label.setWordWrap(True)
        lay.addWidget(QLabel("ONNX"), 0, 0)
        lay.addWidget(self.onnx_edit, 0, 1)
        lay.addWidget(button("ONNX 선택", self.select_onnx), 0, 2)
        lay.addWidget(button("ONNX → TensorRT ENGINE", self.convert_onnx, "Primary"), 1, 1)
        lay.addWidget(QLabel("현재 POS 모델"), 2, 0)
        lay.addWidget(self.engine_label, 2, 1)
        lay.addWidget(button("기존 .engine 선택", self.select_engine), 3, 1)
        lay.addWidget(button("ENGINE 다시 로드", self.reload_engine), 3, 2)
        return box

    def _build_training(self):
        box = QGroupBox("② Jetson 긴급 전이학습")
        lay = QGridLayout(box)
        self.base_model_label = QLabel("기본 .pt: 미설정")
        self.base_model_label.setWordWrap(True)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 300)
        self.epochs_spin.setValue(30)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)

        # 신상품은 턴테이블로 혼자 있는 모습만 찍힌다. 그것만 학습하면
        # 매대에서 겹쳐 놓인 빵을 못 잡으므로, 오려내 여러 개짜리로 합성한다.
        self.synth_check = QCheckBox("합성 데이터 만들기 (신상품 보강)")
        self.synth_check.setChecked(True)
        self.synth_check.setToolTip("찍은 사진에서 빵만 오려내 여러 개 놓인 장면으로 합성합니다.\n"
                                    "라벨은 자동으로 붙습니다 — 사람이 그리지 않습니다.\n"
                                    "한 개씩만 찍어도 겹친 빵을 인식하게 하려는 것입니다.")
        self.synth_spin = QSpinBox()
        self.synth_spin.setRange(50, 3000)
        self.synth_spin.setValue(400)
        self.synth_spin.setSingleStep(50)
        self.synth_spin.setSuffix(" 장")
        self.synth_spin.setToolTip("최소 장수입니다. 신상품이 합성 안에 250번은 나오도록\n"
                                   "젯슨이 필요하면 알아서 더 만듭니다.\n\n"
                                   "실측 — 신상품 등장 횟수와 인식률(놓치지 않는 비율):\n"
                                   "  115회 82%  ·  177회 92%  ·  231회 98%\n"
                                   "231회면 실제 사진으로 배운 것과 같습니다.")

        lay.addWidget(self.synth_check, 0, 0, 1, 2)
        lay.addWidget(self.synth_spin, 0, 2)
        lay.addWidget(QLabel("전이학습 시작 모델"), 1, 0)
        lay.addWidget(self.base_model_label, 1, 1)
        lay.addWidget(button("기존 best.pt 선택", self.select_base_model), 1, 2)
        lay.addWidget(QLabel("Epoch"), 2, 0)
        lay.addWidget(self.epochs_spin, 2, 1)
        lay.addWidget(QLabel("Image size"), 3, 0)
        lay.addWidget(self.imgsz_spin, 3, 1)
        lay.addWidget(button("Master 데이터셋 준비/갱신", self.prepare_dataset,
                             tooltip="찍은 사진을 Train/Val 로 나눕니다.\n"
                                     "아래 합성이 켜져 있으면 이어서 합성까지 합니다."), 4, 0)
        lay.addWidget(button("전이학습 시작 (학습 → 변환 → 적용 자동)", self.start_training, "Success",
                             "학습이 끝나면 ENGINE 변환과 POS 적용까지 이어서 자동으로 합니다."), 4, 1)
        lay.addWidget(button("변환·적용만 다시 하기", self.apply_last_best_model,
                             tooltip="이미 학습된 best.pt 를 ENGINE 으로 만들어 적용합니다. "
                                     "학습 중 창을 닫았거나 적용만 다시 하고 싶을 때 씁니다."), 4, 2)
        return box

    # ── 상태 표시 ────────────────────────────────────────────────────────
    def set_phase(self, text=None):
        """학습 → 변환 → 적용 중 지금 어디인지 보여준다. (한 시간짜리 작업이라 필요하다)"""
        if not text:
            self.phase_label.setVisible(False)
            return
        self.phase_label.setText(text)
        self.phase_label.setStyleSheet(style.PHASE_BOX)
        self.phase_label.setVisible(True)

    def refresh_ops_status(self):
        if self.ctx.can_run_on_jetson():
            self.ops_status.setText(f"● 젯슨 연결됨 ({self.ctx.ops.url}) — 아래 버튼을 누르면 젯슨에서 실행됩니다. "
                                    "젯슨에 따로 접속할 필요가 없습니다.")
            self.ops_status.setStyleSheet(style.OK_BOX)
        elif self.ctx.is_jetson():
            self.ops_status.setText("● 젯슨에서 직접 실행 중 — 아래 기능을 모두 쓸 수 있습니다.")
            self.ops_status.setStyleSheet(style.OK_BOX)
        else:
            self.ops_status.setText(
                "● 젯슨에 연결되지 않았습니다 — 학습·변환은 젯슨이 해야 하는 일이라 여기서는 실행할 수 없습니다.\n"
                "   젯슨에서 ./jetson/start.sh 를 실행하고, BREADVISION_INFER_URL 을 젯슨 주소로 지정한 뒤 다시 켜주세요.")
            self.ops_status.setStyleSheet(style.ERROR_BOX)

    def refresh_model_labels(self):
        engine = self.ctx.active_engine()
        self.engine_label.setText(f"활성 ENGINE: {Path(engine).name}" if engine else "활성 ENGINE: 미설정")
        pt = self.ctx.settings.get("training_base_model", "")
        self.base_model_label.setText(f"기본 .pt: {Path(pt).name}" if pt else "기본 .pt: 미설정")
        self.ctx.model_changed.emit()

    def _on_job_failed(self, title, message):
        self.set_phase(None)
        QMessageBox.warning(self, f"{title} 실패", message)

    def _set_active_engine(self, path):
        self.ctx.settings["active_engine"] = path
        self.ctx.save_settings()
        self.refresh_model_labels()
        log.info("POS 모델 적용: %s", path)

    def _sync_remote_model(self):
        """젯슨이 실제로 물고 있는 모델을 다시 확인한다 (원격 추론일 때)."""
        try:
            self.ctx.detector.refresh()
        except Exception as e:          # noqa: BLE001 - 네트워크·서버 오류 종류가 다양하다
            # 예전에는 조용히 넘겨서, 실패하면 화면에 옛 모델명이 남은 채로 아무도 몰랐다.
            log.warning("젯슨이 쓰는 모델을 확인하지 못했습니다: %s", e)
            self.ctx.show_status(f"젯슨 모델 정보를 확인하지 못했습니다 — 화면의 모델명이 실제와 다를 수 있습니다: {e}",
                                 8000)

    # ── 시작할 때 ────────────────────────────────────────────────────────
    def try_load_saved_engine(self):
        """시작할 때 POS 모델을 맞춘다.

        원격(PC + 젯슨)일 때는 젯슨이 물고 있는 모델을 그대로 따른다.
        PC 의 settings.json 경로를 젯슨에 밀어 넣으면 젯슨에서 의도적으로 띄운 모델을 조용히 갈아치운다.
        모델을 바꾸는 것은 사용자가 「기존 .engine 선택」 을 눌렀을 때만 일어나야 한다.
        """
        import os
        if self.ctx.ops is not None or os.environ.get("BREADVISION_INFER_URL", "").strip():
            self._sync_remote_model()
            path = getattr(self.ctx.detector, "model_path", None)
            if path:
                self.ctx.settings["active_engine"] = path
            self.refresh_model_labels()
            return

        engine = self.ctx.active_engine()
        if not engine or not Path(engine).exists():
            return
        try:
            self.ctx.detector.load(engine)
            self.refresh_model_labels()
        except Exception as e:          # noqa: BLE001 - 시작은 계속하고, 모델 없이 뜬다
            log.warning("저장된 ENGINE 을 불러오지 못했습니다: %s (%s)", engine, e)

    # ── ① 배포 ──────────────────────────────────────────────────────────
    def select_onnx(self):
        path, _ = QFileDialog.getOpenFileName(self, "Colab ONNX 선택", str(self.ctx.paths.models),
                                              "ONNX Model (*.onnx)")
        if path:
            self.onnx_edit.setText(path)

    def _selected_onnx(self):
        path = self.onnx_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "ONNX 필요", "변환할 .onnx 파일을 먼저 선택해주세요.")
            return None
        return path

    def convert_onnx(self):
        if self.ctx.can_run_on_jetson():
            self._convert_onnx_on_jetson()
            return
        if not self.require_jetson("ONNX → TensorRT ENGINE 변환"):
            return
        path = self._selected_onnx()
        if not path:
            return
        out = str(Path(path).with_suffix(".engine"))
        cmd = ["/usr/src/tensorrt/bin/trtexec", f"--onnx={path}", f"--saveEngine={out}", "--fp16"]
        self.log.appendPlainText("[TRT] " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            self.ctx.detector.load(out)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
            log.error("TensorRT 변환 실패: %s", e)
            QMessageBox.critical(self, "TensorRT 변환 실패", str(e))
            return
        self._set_active_engine(out)
        QMessageBox.information(self, "배포 완료", f"ENGINE 생성 및 적용 완료\n{out}")

    def _convert_onnx_on_jetson(self):
        """Colab 에서 받은 .onnx 를 젯슨에 올려 ENGINE 으로 만들고 바로 적용한다."""
        if self.busy_with_job():
            return
        path = self._selected_onnx()
        if not path:
            return
        size_mb = Path(path).stat().st_size / 1024 / 1024
        self.log.appendPlainText(f"[젯슨] ONNX 전송 중… {Path(path).name} ({size_mb:.0f} MB)")
        QApplication.processEvents()
        try:
            remote = self.ctx.ops.upload_zip(path)["saved"]     # .onnx 도 같은 창구로 올라간다
        except OPS_ERRORS as e:
            QMessageBox.critical(self, "전송 실패", str(e))
            return
        self.log.appendPlainText(f"[젯슨] 전송 완료 → {remote}")
        # 젯슨(리눅스)에 보낼 경로는 항상 슬래시로. 윈도우에서 Path 를 쓰면 역슬래시가 된다.
        engine_path = Path(remote).with_suffix(".engine").as_posix()
        self._start_job("onnx2engine", "ONNX → ENGINE 변환",
                        lambda r: self.deploy_on_jetson({"engine": engine_path}), onnx=remote)

    def select_engine(self):
        if self.ctx.can_run_on_jetson():
            picked = self.pick_jetson_file("engine", "젯슨의 .engine 선택")
            if not picked:
                return
            try:
                res = self.ctx.ops.deploy(picked)
            except OPS_ERRORS as e:
                QMessageBox.critical(self, "적용 실패", str(e))
                return
            self.ctx.settings["active_engine"] = res.get("deployed", picked)
            self.ctx.save_settings()
            self._sync_remote_model()
            self.refresh_model_labels()
            QMessageBox.information(self, "적용 완료", res.get("deployed", picked))
            return

        path, _ = QFileDialog.getOpenFileName(self, "Jetson POS TensorRT ENGINE 선택",
                                              str(self.ctx.paths.models), "TensorRT Engine (*.engine)")
        if not path:
            return
        try:
            self.ctx.detector.load(path)
        except Exception as e:          # noqa: BLE001
            log.warning("ENGINE 로드 실패: %s (%s)", path, e)
            QMessageBox.critical(self, "ENGINE 로드 실패", str(e))
            return
        self.ctx.settings["active_model_classes"] = {str(cid): p["class_name"]
                                                     for cid, p in self.ctx.products.items()}
        self._set_active_engine(path)
        QMessageBox.information(self, "ENGINE 적용 완료", f"POS 추론 모델을 적용했습니다.\n\n{Path(path).name}")

    def reload_engine(self):
        path = self.ctx.active_engine()
        if not path:
            QMessageBox.information(self, "ENGINE 미설정", "먼저 .engine 모델을 선택해주세요.")
            return
        try:
            self.ctx.detector.load(path)
        except Exception as e:          # noqa: BLE001
            log.warning("ENGINE 다시 로드 실패: %s (%s)", path, e)
            QMessageBox.critical(self, "ENGINE 로드 실패", str(e))
            return
        self.refresh_model_labels()
        QMessageBox.information(self, "ENGINE 로드 완료", Path(path).name)

    def select_base_model(self):
        if self.ctx.can_run_on_jetson():
            picked = self.pick_jetson_file("pt", "젯슨의 .pt 선택 (전이학습 기준 모델)")
        else:
            picked, _ = QFileDialog.getOpenFileName(self, "전이학습 기본 YOLO 모델 선택",
                                                    str(self.ctx.paths.models), "PyTorch Model (*.pt)")
        if not picked:
            return
        self.ctx.settings["training_base_model"] = picked
        self.ctx.save_settings()
        self.refresh_model_labels()

    # ── 젯슨 원격 작업 도우미 ────────────────────────────────────────────
    def busy_with_job(self):
        if self.jobs.busy:
            QMessageBox.warning(self, "작업 진행 중", "젯슨에서 작업이 진행 중입니다. 끝난 뒤에 다시 실행하세요.")
            return True
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "작업 진행 중", "이미 진행 중인 작업이 있습니다.")
            return True
        return False

    def _start_job(self, task, title, on_success, **args):
        try:
            self.jobs.start(task, title, on_success, **args)
        except OPS_ERRORS as e:
            log.warning("젯슨 작업을 걸지 못했습니다: %s (%s)", title, e)
            self.set_phase(None)
            QMessageBox.critical(self, f"{title} 실패", str(e))

    def pick_jetson_file(self, kind, title):
        """젯슨에 있는 파일을 목록에서 고른다. (PC 에서 젯슨 폴더를 뒤질 수 없으므로)"""
        try:
            files = self.ctx.ops.files(kind)
        except OPS_ERRORS as e:
            QMessageBox.critical(self, "목록을 받지 못했습니다", str(e))
            return None
        if not files:
            QMessageBox.information(self, "파일 없음", f"젯슨에 {kind} 파일이 없습니다.")
            return None
        labels = [f"{f['path']}   ({f['size'] // 1024 // 1024} MB)" for f in files]
        choice, ok = QInputDialog.getItem(self, title, "젯슨에 있는 파일:", labels, 0, False)
        return files[labels.index(choice)]["path"] if ok else None

    def require_jetson(self, feature):
        """젯슨 전용 기능을 PC 에서 눌렀을 때 원인을 알려준다."""
        if self.ctx.is_jetson():
            return True
        QMessageBox.information(
            self, f"{feature} — 젯슨에서만 가능",
            f"{feature} 은(는) 젯슨에서만 동작합니다.\n\n"
            "지금 이 GUI 는 PC 에서 돌고 있고, 추론만 젯슨에 맡기는 구성입니다.\n"
            "이 기능은 젯슨 GPU(TensorRT · Docker) 를 직접 써야 해서 PC 에서는 실행할 수 없습니다.\n\n"
            "[운영 원칙]\n"
            "  · 정식 학습     → Colab (BreadVision_Colab_Train.ipynb)\n"
            "  · 긴급 전이학습 → 젯슨에서 이 GUI 를 직접 실행\n"
            "  · ENGINE 변환   → 젯슨에서 ./onnx_to_engine.sh models/best.onnx")
        return False

    # ── ② 데이터셋 ──────────────────────────────────────────────────────
    def prepare_dataset(self):
        if self.ctx.can_run_on_jetson():
            # PC 의 상품 목록을 같이 보내서 젯슨의 data.yaml 이 PC 기준으로 만들어지게 한다
            if not self.busy_with_job():
                self._start_job("split", "데이터셋 분할", self._after_split, products=self.ctx.products)
            return

        script = self.ctx.paths.base / "import_master_dataset.py"
        try:
            r = subprocess.run([sys.executable, str(script), "--project", str(self.ctx.paths.base)],
                               capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            msg = (e.stdout or "") + "\n" + (e.stderr or "")
            self.log.appendPlainText(msg)
            QMessageBox.critical(self, "데이터셋 준비 실패", msg[-2000:])
            return
        self.log.appendPlainText(r.stdout)
        QMessageBox.information(self, "Master 데이터셋 준비 완료", r.stdout[-1200:] if r.stdout else "완료")

    def _after_split(self, _result):
        """분할이 끝났다. 합성을 켜 두었으면 이어서 합성까지 한다. (val 은 실제 사진 그대로)"""
        if not self.synth_check.isChecked():
            QMessageBox.information(self, "데이터셋 준비 완료",
                                    "젯슨에서 Train/Val 분할이 끝났습니다.\n아래 로그에서 장수와 클래스를 확인하세요.")
            return
        count = self.synth_spin.value()
        self.log.appendPlainText(f"[자동] 분할 완료 → 합성 {count}장 만들기로 이어갑니다.")

        def done(_r):
            self.set_phase(None)
            QMessageBox.information(
                self, "데이터셋 준비 완료",
                "Train/Val 분할과 합성이 모두 끝났습니다.\n\n"
                "합성 사진은 실제 사진에서 빵만 오려내 여러 개 놓은 것이고,\n"
                "라벨은 자동으로 붙었습니다. 아래 로그에서 품목별 개수를 확인하세요.\n\n"
                "이제 「전이학습 시작」 을 누르면 됩니다.")

        self.set_phase(f"합성 중…   {count}장")
        self._start_job("synth", "합성 데이터 생성", done, count=count)

    # ── ② 학습 ──────────────────────────────────────────────────────────
    def start_training(self):
        if self.ctx.can_run_on_jetson():
            self._start_training_on_jetson()
            return
        if not self.require_jetson("긴급 전이학습"):
            return

        model_path = self.ctx.settings.get("training_base_model", "").strip()
        if not model_path:
            QMessageBox.warning(self, "기존 모델 필요", "STEP 1에서 기존 .pt 모델을 먼저 선택해주세요.")
            return
        if not Path(model_path).exists():
            QMessageBox.warning(self, "모델 파일 없음", "선택한 .pt 파일을 찾을 수 없습니다.")
            return
        if not list(self.ctx.paths.train_img.glob("*.jpg")):
            QMessageBox.warning(self, "학습 데이터 없음", "STEP 2의 「Train / Val 80:20 자동 분할」을 먼저 실행해주세요.")
            return
        if not self.ctx.paths.data_yaml.exists():
            self.ctx.write_data_yaml()
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "학습 진행 중", "현재 YOLO 학습이 이미 진행 중입니다.")
            return

        epochs, imgsz = self.epochs_spin.value(), self.imgsz_spin.value()
        run_name = "bread_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_process(self._training_finished, [
            str(self.ctx.paths.base / "training" / "run_training_container.sh"),
            model_path, str(self.ctx.paths.data_yaml), str(epochs), str(imgsz), run_name])
        self.set_phase(f"1 / 3  전이학습 중…   Epoch {epochs} · 약 {max(1, epochs * 100 // 60)}분 예상")
        self.log.appendPlainText(f"\n[TRAIN START]\n기존 모델: {model_path}\nEpoch={epochs}, imgsz={imgsz}")
        log.info("로컬 전이학습 시작: %s epochs=%s", model_path, epochs)

    def _start_training_on_jetson(self):
        if self.busy_with_job():
            return
        model = self.ctx.settings.get("training_base_model", "").strip()
        if not model:
            QMessageBox.warning(self, "기존 모델 필요",
                                "먼저 「기존 best.pt 선택」에서 젯슨에 있는 .pt 를 골라주세요.")
            return
        epochs, imgsz = self.epochs_spin.value(), self.imgsz_spin.value()
        answer = QMessageBox.question(
            self, "젯슨에서 학습 시작",
            f"젯슨에서 전이학습을 시작합니다.\n\n기준 모델: {model}\nEpoch: {epochs}   이미지 크기: {imgsz}\n\n"
            "Epoch 30 기준 약 50분 걸립니다.\n그동안 이 창을 닫아도 젯슨에서는 계속 진행됩니다. 시작할까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer != QMessageBox.Yes:
            return

        def trained(result):
            best = result.get("best_pt", "")
            self.last_best_model = best
            if not best:
                self.set_phase(None)
                QMessageBox.warning(self, "best.pt 를 찾지 못했습니다",
                                    "학습은 끝났지만 결과 경로를 로그에서 찾지 못했습니다.\n"
                                    "「변환·적용만 다시 하기」 로 직접 골라 진행해주세요.")
                return
            self.ctx.settings["last_best_model"] = best
            self.ctx.save_settings()
            # 학습이 끝나면 곧바로 변환 → 적용까지 이어서 한다.
            self.set_phase(f"2 / 3  ENGINE 변환 중…   ({Path(best).parent.parent.name})")
            self.log.appendPlainText(f"[자동] 학습 완료 → ENGINE 변환으로 이어갑니다: {best}")
            self._start_job("engine", "ENGINE 변환", self.deploy_on_jetson, pt=best, imgsz=imgsz)

        self.set_phase(f"1 / 3  전이학습 중…   Epoch {epochs} · 약 {max(1, epochs * 100 // 60)}분 예상")
        self._start_job("train", "전이학습", trained, model=model, epochs=epochs, imgsz=imgsz)

    def _run_process(self, on_finished, args):
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(on_finished)
        self.process.start("bash", args)

    def _read_process_output(self):
        if not self.process:
            return
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self.log.appendPlainText(line)
            if line.startswith("BEST_MODEL="):
                self.last_best_model = line.split("=", 1)[1].strip()

    def _training_finished(self, exit_code, _exit_status):
        if exit_code != 0:
            self.log.appendPlainText(f"[TRAIN ERROR] exit_code={exit_code}")
            self.set_phase(None)
            QMessageBox.warning(self, "학습 실패", "YOLO 학습이 정상적으로 끝나지 않았습니다.\n"
                                                  "아래 로그와 ultralytics 설치 여부를 확인해주세요.")
            return
        self.log.appendPlainText("[TRAIN COMPLETE]")
        best = self.find_last_best_model()
        if best is None:
            self.set_phase(None)
            QMessageBox.information(self, "YOLO 학습 완료", "학습은 끝났지만 best.pt 를 찾지 못했습니다.\n"
                                                           "「변환·적용만 다시 하기」 로 직접 진행해주세요.")
            return
        self.set_phase(f"2 / 3  ENGINE 변환 중…   ({best.parent.parent.name})")
        self.log.appendPlainText(f"[자동] 학습 완료 → ENGINE 변환으로 이어갑니다: {best}")
        self._start_local_engine_build(best, self.imgsz_spin.value())

    def find_last_best_model(self):
        """이번 실행에서 학습한 best.pt. 없으면 runs/ 에서 가장 최근 것 (앱을 다시 켠 뒤에도 변환할 수 있게)."""
        if self.last_best_model and Path(self.last_best_model).exists():
            return Path(self.last_best_model)
        candidates = sorted(self.ctx.paths.runs.glob("*/weights/best.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    # ── 변환·적용 ────────────────────────────────────────────────────────
    def apply_last_best_model(self):
        """best.pt → ONNX → TensorRT ENGINE 변환 후 POS 활성 모델로 적용."""
        if self.ctx.can_run_on_jetson():
            self._build_engine_on_jetson()
            return
        if not self.require_jetson("ENGINE 변환"):
            return
        best = self.find_last_best_model()
        if best is None:
            QMessageBox.warning(self, "best.pt 없음", "학습된 best.pt 를 찾을 수 없습니다. 먼저 모델 학습을 완료해주세요.")
            return
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "작업 진행 중", "학습 또는 변환이 이미 진행 중입니다.")
            return
        imgsz = self.imgsz_spin.value()
        answer = QMessageBox.question(
            self, "ENGINE 변환",
            f"아래 모델을 TensorRT ENGINE 으로 변환하고 POS 에 적용합니다.\n\n{best}\n이미지 크기: {imgsz}\n\n"
            "몇 분 걸립니다. 진행할까요?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            self._start_local_engine_build(best, imgsz)

    def _start_local_engine_build(self, best, imgsz):
        self.pending_engine = str(Path(best).with_suffix(".engine"))
        self.log.appendPlainText(f"\n[ENGINE BUILD START]\n입력: {best}")
        self._run_process(self._engine_build_finished, [
            str(self.ctx.paths.base / "training" / "export_and_build_engine.sh"), str(best), str(imgsz)])

    def _build_engine_on_jetson(self):
        if self.busy_with_job():
            return
        best = self.last_best_model or self.ctx.settings.get("last_best_model", "")
        if not best:
            best = self.pick_jetson_file("pt", "변환할 .pt 선택")
            if not best:
                return
        imgsz = self.imgsz_spin.value()
        answer = QMessageBox.question(
            self, "ENGINE 변환 및 적용",
            f"젯슨에서 아래 모델을 TensorRT ENGINE 으로 만들고 POS 에 적용합니다.\n\n"
            f"{best}\n이미지 크기: {imgsz}\n\n몇 분 걸립니다. 진행할까요?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            self._start_job("engine", "ENGINE 변환", self.deploy_on_jetson, pt=best, imgsz=imgsz)

    def deploy_on_jetson(self, result):
        """변환된 엔진을 젯슨 추론 서버에 즉시 물린다."""
        engine = result.get("engine", "")
        if not engine:
            self.set_phase(None)
            QMessageBox.warning(self, "ENGINE 없음", "변환은 끝났지만 .engine 경로를 로그에서 찾지 못했습니다.")
            return
        self.set_phase("3 / 3  POS 에 적용 중…")
        try:
            res = self.ctx.ops.deploy(engine)
        except OPS_ERRORS as e:
            self.set_phase(None)
            QMessageBox.critical(self, "적용 실패", str(e))
            return
        self.ctx.settings["active_engine"] = res.get("deployed", engine)
        self.ctx.save_settings()
        self._sync_remote_model()
        self.refresh_model_labels()
        self.log.appendPlainText(f"[젯슨] POS 적용 완료 → {res.get('deployed')}")
        self.set_phase(None)
        QMessageBox.information(self, "완료", f"새 모델이 POS 에 적용됐습니다. 지금부터 바로 이 모델로 인식합니다.\n\n"
                                            f"{res.get('deployed')}\n\n프로그램을 다시 켜도 이 모델이 유지됩니다.")

    def _engine_build_finished(self, exit_code, _exit_status):
        if exit_code != 0:
            self.set_phase(None)
            self.log.appendPlainText(f"[ENGINE BUILD ERROR] exit_code={exit_code}")
            QMessageBox.warning(self, "변환 실패", "ENGINE 변환이 정상적으로 끝나지 않았습니다.\n로그를 확인해주세요.")
            return
        engine = self.pending_engine
        if not engine or not Path(engine).exists():
            self.log.appendPlainText("[ENGINE BUILD] 파일을 찾지 못했습니다.")
            QMessageBox.warning(self, "ENGINE 없음", "변환은 끝났지만 .engine 파일을 찾지 못했습니다.")
            return
        self.log.appendPlainText(f"[ENGINE BUILD COMPLETE] {engine}")
        self.set_phase("3 / 3  POS 에 적용 중…")
        try:
            self.ctx.detector.load(engine)
        except Exception as e:          # noqa: BLE001
            log.warning("변환한 ENGINE 로드 실패: %s (%s)", engine, e)
            self.set_phase(None)
            QMessageBox.warning(self, "ENGINE 로드 실패", str(e))
            return
        self._set_active_engine(engine)
        self.set_phase(None)
        QMessageBox.information(self, "완료", f"새 모델이 POS 에 적용됐습니다. 지금부터 바로 이 모델로 인식합니다.\n\n"
                                            f"{Path(engine).name}\n\n프로그램을 다시 켜도 이 모델이 유지됩니다.")

    def shutdown(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.kill()
