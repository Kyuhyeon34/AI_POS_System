"""jetson_ops.py — PC 가 보낸 요청을 젯슨이 실행하기 전에 막는 안전장치."""
import pytest

import jetson_ops


def test_safe_path_accepts_project_relative_and_windows_slashes():
    p = jetson_ops.safe_path("models\\m13_A.engine")
    assert p == (jetson_ops.BASE_DIR / "models" / "m13_A.engine").resolve()


@pytest.mark.parametrize("bad", ["../outside.pt", "models/../../etc/passwd", ""])
def test_safe_path_rejects_escape(bad):
    with pytest.raises(ValueError):
        jetson_ops.safe_path(bad)


def test_safe_path_rejects_absolute_outside(tmp_path):
    with pytest.raises(ValueError):
        jetson_ops.safe_path(str(tmp_path / "x.pt"))


@pytest.fixture
def upload_dirs(tmp_path, monkeypatch):
    dest = {".zip": tmp_path / "import", ".onnx": tmp_path / "models", ".pt": tmp_path / "models"}
    monkeypatch.setattr(jetson_ops, "UPLOAD_DEST", dest)
    monkeypatch.setattr(jetson_ops, "BASE_DIR", tmp_path)
    return tmp_path


def test_save_upload_strips_directories_from_name(upload_dirs):
    res = jetson_ops.save_upload("../../evil.zip", b"data")
    assert (upload_dirs / "import" / "evil.zip").read_bytes() == b"data"
    assert res["bytes"] == 4


def test_save_upload_rejects_unknown_extension(upload_dirs):
    with pytest.raises(ValueError):
        jetson_ops.save_upload("run.sh", b"rm -rf /")


def test_save_upload_replace_prefix_removes_old_files(upload_dirs):
    jetson_ops.save_upload("collected_1.zip", b"old")
    jetson_ops.save_upload("keep.zip", b"keep")
    res = jetson_ops.save_upload("collected_2.zip", b"new", replace_prefix="collected_")
    names = sorted(p.name for p in (upload_dirs / "import").iterdir())
    assert names == ["collected_2.zip", "keep.zip"]
    assert res["removed"] == ["collected_1.zip"]


def test_build_command_rejects_unknown_task():
    with pytest.raises(ValueError):
        jetson_ops._build_command("rm", {})


def test_build_command_rejects_bad_run_name():
    with pytest.raises(ValueError):
        jetson_ops._build_command("train", {"model": "models/m13_A.pt", "name": "x; reboot"})


def test_build_command_rejects_synth_count_out_of_range():
    with pytest.raises(ValueError):
        jetson_ops._build_command("synth", {"count": 999999})


def test_build_command_train_uses_whitelisted_script():
    cmd = jetson_ops._build_command("train", {"model": "models/m13_A.pt", "name": "run_1"})
    assert cmd[0] == "bash"
    assert cmd[1].endswith("run_training_container.sh")
    assert cmd[-1] == "run_1"


def test_start_task_refuses_second_job(monkeypatch):
    monkeypatch.setattr(jetson_ops, "_jobs", {"1": {"running": True}})
    with pytest.raises(RuntimeError):
        jetson_ops.start_task("split", {})
