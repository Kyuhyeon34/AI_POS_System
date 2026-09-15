"""core/dataset_io.py — 수집 사진·라벨 파일."""
import zipfile

import numpy as np

from core.dataset_io import (class_count_detail, collected_pairs, count_collected, delete_files,
                             labels_per_class, save_sample, write_training_zip)


def test_save_sample_numbers_files_and_handles_korean_path(tmp_path):
    """윈도우에서 cv2.imwrite 는 한글 경로에 조용히 실패한다. 사진이 실제로 생겼는지 본다."""
    img_dir, lbl_dir = tmp_path / "빵사진" / "images", tmp_path / "빵사진" / "labels"
    frame = np.full((48, 64, 3), 200, np.uint8)
    stem1, before1 = save_sample(img_dir, lbl_dir, "salt", frame, ["0 0.5 0.5 0.2 0.2"])
    stem2, before2 = save_sample(img_dir, lbl_dir, "salt", frame, ["0 0.5 0.5 0.2 0.2"])
    assert (stem1, before1, stem2, before2) == ("salt_00001", 0, "salt_00002", 1)
    assert (img_dir / "salt_00002.jpg").stat().st_size > 0
    assert (lbl_dir / "salt_00002.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
    assert count_collected(img_dir, "salt") == 2


def test_pairs_counts_zip_and_delete(tmp_path):
    img, lbl = tmp_path / "images", tmp_path / "labels"
    img.mkdir()
    lbl.mkdir()
    for name, text in (("a", "0 .5 .5 .1 .1\n1 .2 .2 .1 .1\n"), ("b", "bad line\n0 .5 .5 .1 .1\n")):
        (img / f"{name}.jpg").write_bytes(b"jpg")
        (lbl / f"{name}.txt").write_text(text, encoding="utf-8")
    (img / "no_label.jpg").write_bytes(b"jpg")

    pairs = collected_pairs(img, lbl)
    assert [p[0].name for p in pairs] == ["a.jpg", "b.jpg"]

    per_class = labels_per_class([l for _, l in pairs])
    assert per_class == {0: 2, 1: 1}
    products = {"0": {"display_name": "소금빵"}}
    assert class_count_detail(per_class, products) == "  0 소금빵: 2장\n  1 ?: 1장"

    out = write_training_zip(tmp_path / "import" / "c.zip", pairs, ["salt", "pie"])
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ["a.jpg", "a.txt", "b.jpg", "b.txt", "classes.txt"]
        assert z.read("classes.txt").decode() == "salt\npie\n"

    removed, failed = delete_files([img / "a.jpg", img / "missing.jpg"])
    assert removed == 1
    assert len(failed) == 1
