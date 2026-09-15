"""수집한 사진·라벨 파일 다루기 (저장, 개수 세기, 학습용 ZIP, 삭제).

폴더 구조
    dataset/collected/images/<class_name>_00001.jpg
    dataset/collected/labels/<class_name>_00001.txt   (YOLO 형식, 같은 이름)
"""
import zipfile
from pathlib import Path

import cv2

from core.log import get_logger

log = get_logger("dataset")


def count_collected(img_dir, class_name):
    return len(list(Path(img_dir).glob(f"{class_name}_*.jpg")))


def save_sample(img_dir, lbl_dir, class_name, frame, lines):
    """사진 한 장과 라벨을 저장한다. (저장한 이름, 이 품목의 저장 전 장수) 를 돌려준다.

    cv2.imwrite 는 윈도우에서 한글이 들어간 경로에 쓰면 오류 없이 False 만 돌려준다.
    그러면 라벨만 남고 사진이 없는 짝이 생긴다. 메모리에서 JPEG 로 만든 뒤 직접 쓴다.
    """
    img_dir, lbl_dir = Path(img_dir), Path(lbl_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    existing = count_collected(img_dir, class_name)
    stem = f"{class_name}_{existing + 1:05d}"
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise OSError(f"사진을 JPEG 로 바꾸지 못했습니다: {stem}")
    (img_dir / f"{stem}.jpg").write_bytes(buf.tobytes())
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stem, existing


def collected_pairs(img_dir, lbl_dir):
    """[(사진, 라벨)] — 라벨이 있는 사진만."""
    pairs = [(f, Path(lbl_dir) / f"{f.stem}.txt") for f in sorted(Path(img_dir).glob("*.jpg"))]
    return [(i, l) for i, l in pairs if l.exists()]


def labels_per_class(label_files):
    """{class_id: 박스 수}. 읽을 수 없는 파일·줄은 기록만 남기고 건너뛴다."""
    per_class = {}
    for lab in label_files:
        try:
            text = Path(lab).read_text(encoding="utf-8")
        except OSError as e:
            log.warning("라벨을 읽지 못했습니다: %s (%s)", lab, e)
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                cid = int(line.split()[0])
            except ValueError:
                log.warning("라벨 형식이 잘못된 줄을 건너뜁니다: %s: %r", Path(lab).name, line)
                continue
            per_class[cid] = per_class.get(cid, 0) + 1
    return per_class


def class_count_detail(per_class, products):
    return "\n".join(
        f"  {cid} {products.get(str(cid), {}).get('display_name', '?')}: {n}장"
        for cid, n in sorted(per_class.items())
    )


def write_training_zip(out, pairs, class_names):
    """import_master_dataset.py 가 읽는 평면 구조(jpg 옆에 같은 이름 txt)로 압축한다.

    classes.txt 는 products.json 순서 그대로 쓴다 (신규 상품이 그대로 반영된다).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for img, lab in pairs:
            z.write(img, img.name)
            z.write(lab, lab.name)
        z.writestr("classes.txt", "\n".join(class_names) + "\n")
    return out


def delete_files(paths):
    """(지운 개수, 실패 목록). 이미지와 라벨은 항상 같이 지운다 — 짝이 안 맞으면 나중에 조용히 문제가 된다."""
    removed, failed = 0, []
    for f in paths:
        try:
            Path(f).unlink()
            removed += 1
        except OSError as e:
            failed.append(f"{Path(f).name}: {e}")
    return removed, failed
