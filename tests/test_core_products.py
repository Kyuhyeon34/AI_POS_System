"""core/products.py, core/settings.py — 상품 목록과 설정 파일."""
import json

import pytest

from core.products import (add_product, class_name_map, data_yaml_text, display_name,
                           load_products, next_class_id, ordered)
from core.settings import load_settings


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_first_run_copies_default_list(tmp_path):
    write(tmp_path / "products.default.json", {"0": {"class_name": "salt", "display_name": "소금빵", "price": 0}})
    products = load_products(tmp_path / "products.json")
    assert products["0"]["display_name"] == "소금빵"
    assert (tmp_path / "products.json").exists()


def test_no_products_at_all(tmp_path):
    assert load_products(tmp_path / "products.json") == {}


def test_legacy_format_is_converted_and_saved(tmp_path):
    f = tmp_path / "products.json"
    write(f, {"salt": {"class_id": 4, "name": "소금빵", "price": "3000"}})
    products = load_products(f)
    assert products == {"4": {"class_name": "salt", "display_name": "소금빵", "price": 3000}}
    assert json.loads(f.read_text(encoding="utf-8")) == products


def test_broken_products_file_is_backed_up(tmp_path):
    f = tmp_path / "products.json"
    f.write_text("{broken", encoding="utf-8")
    assert load_products(f) == {}
    assert (tmp_path / "products.json.broken.bak").exists()


def test_add_product_assigns_next_id_and_normalizes_name():
    products = {"0": {"class_name": "salt", "display_name": "소금빵", "price": 0},
                "5": {"class_name": "pie", "display_name": "파이", "price": 0}}
    cid = add_product(products, " 크림빵 ", "Cream Bread", 2500)
    assert cid == 6
    assert products["6"] == {"class_name": "cream_bread", "display_name": "크림빵", "price": 2500}


@pytest.mark.parametrize("display,cls", [("", "x"), ("크림빵", "  "), ("소금빵2", "SALT")])
def test_add_product_rejects_empty_or_duplicate(display, cls):
    products = {"0": {"class_name": "salt", "display_name": "소금빵", "price": 0}}
    with pytest.raises(ValueError):
        add_product(products, display, cls, 0)
    assert len(products) == 1


def test_helpers():
    products = {"10": {"class_name": "b", "display_name": "비", "price": 1},
                "2": {"class_name": "a", "display_name": "에이", "price": 1}}
    assert next_class_id({}) == 0
    assert [cid for cid, _ in ordered(products)] == ["2", "10"]
    assert class_name_map(products) == {10: "b", 2: "a"}
    assert display_name(products, 2) == "에이"
    assert display_name(products, 99) == "?"


def test_data_yaml_follows_class_id_order(tmp_path):
    products = {"1": {"class_name": "muffine"}, "0": {"class_name": "baggette"}}
    text = data_yaml_text(products, tmp_path / "master")
    assert "\\" not in text.splitlines()[0]
    assert text.endswith("names:\n  0: baggette\n  1: muffine\n")


def test_settings_start_from_example_and_fill_defaults(tmp_path):
    write(tmp_path / "settings.example.json", {"active_engine": "models/m13_A.engine"})
    s = load_settings(tmp_path / "settings.json")
    assert s["active_engine"] == "models/m13_A.engine"
    assert s["training_base_model"] == "models/m12_A.pt"
    assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8")) == s


def test_settings_migrate_old_active_model(tmp_path):
    write(tmp_path / "settings.json", {"active_model": "models/old.engine"})
    assert load_settings(tmp_path / "settings.json")["active_engine"] == "models/old.engine"


def test_broken_settings_are_not_silently_lost(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text('{"camera_source": "jetson"', encoding="utf-8")
    load_settings(f)
    assert (tmp_path / "settings.json.broken.bak").read_text(encoding="utf-8").startswith('{"camera_source"')
