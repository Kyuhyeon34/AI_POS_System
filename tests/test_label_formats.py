"""YOLO 라벨 파일 읽기·쓰기 (autolabel.py, ai/compose.py)."""
import pytest

import autolabel
from ai.compose import to_yolo


def test_write_then_read_yolo_roundtrip(tmp_path):
    txt = tmp_path / "a.txt"
    boxes = [{"c": 3, "x1": 64.0, "y1": 48.0, "x2": 320.0, "y2": 240.0}]
    kept = autolabel.write_yolo(txt, boxes, 640, 480)
    assert len(kept) == 1
    assert txt.read_text(encoding="utf-8") == "3 0.300000 0.300000 0.400000 0.400000\n"

    back = autolabel.read_yolo(txt, 640, 480)
    assert back[0]["c"] == 3
    assert back[0]["x1"] == pytest.approx(64.0)
    assert back[0]["y2"] == pytest.approx(240.0)


def test_write_yolo_clips_and_drops_tiny_boxes(tmp_path):
    txt = tmp_path / "b.txt"
    boxes = [{"c": 0, "x1": -10, "y1": -10, "x2": 50, "y2": 50},
             {"c": 1, "x1": 10, "y1": 10, "x2": 11, "y2": 11}]
    kept = autolabel.write_yolo(txt, boxes, 100, 100)
    assert [b["c"] for b in kept] == [0]
    assert kept[0]["x1"] == 0.0


def test_read_yolo_skips_broken_lines(tmp_path):
    txt = tmp_path / "c.txt"
    txt.write_text("0 0.5 0.5 0.2 0.2\nbroken\n1 a b c d\n", encoding="utf-8")
    assert len(autolabel.read_yolo(txt, 100, 100)) == 1


def test_read_yolo_missing_file(tmp_path):
    assert autolabel.read_yolo(tmp_path / "none.txt", 10, 10) == []


def test_compose_to_yolo():
    lines = to_yolo([(2, 0, 0, 50, 100), (1, 10, 10, 10, 20)], 100, 200)
    assert lines == ["2 0.250000 0.250000 0.500000 0.500000"]
