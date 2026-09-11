#!/usr/bin/env python3
"""Master 데이터셋에 '여러 개 놓인' 합성 사진을 채워 넣는다.

왜 필요한가
    신상품은 턴테이블로 혼자 있는 모습만 찍는다. 그 사진만으로 학습하면
    매대에서 겹쳐 놓인 빵을 못 잡는다.
    (실측 기록: 한 개짜리 329장을 넣었더니 겹친 빵 검출 87.5% → 54.2%)

    그래서 찍은 사진에서 빵만 오려내 여러 개짜리 장면으로 합성한다.
    찍는 건 한 개, 학습하는 건 여러 개.

라벨을 사람이 안 그린다
    어디에 무엇을 붙였는지 우리가 알고 있으므로 붙인 마스크에서 박스가 바로 나온다.

언제 도는가
    「Master 데이터셋 준비」(split) 다음이다. split 이 만든
    dataset/master/images/train 을 재료로 쓰고, 결과를 같은 곳에 넣는다.
    val 은 건드리지 않는다 — 평가는 실제 사진으로만 해야 믿을 수 있다.

여러 번 눌러도 안전하다
    지난번에 만든 synth_*.jpg 를 먼저 지우고 새로 만든다. 쌓이지 않는다.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def rows_of(path):
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        f = line.split()
        if len(f) == 5:
            try:
                out.append((int(f[0]), *map(float, f[1:])))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--count", type=int, default=400, help="기본으로 만들 합성 사진 수")
    ap.add_argument("--target", type=int, default=250,
                    help="신상품이 합성 안에 최소 몇 번 등장하게 할지")
    ap.add_argument("--max-count", type=int, default=1600, help="합성 장수 상한")
    ap.add_argument("--min-obj", type=int, default=1)
    ap.add_argument("--max-obj", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = Path(args.project).resolve()
    sys.path.insert(0, str(base))
    from ai.cutout import cut                      # noqa: E402
    from ai.compose import compose, to_yolo        # noqa: E402

    img_dir = base / "dataset" / "master" / "images" / "train"
    lbl_dir = base / "dataset" / "master" / "labels" / "train"
    if not img_dir.is_dir():
        raise SystemExit(f"먼저 「Master 데이터셋 준비」를 하세요: {img_dir} 가 없습니다.")

    names = []
    yml = base / "dataset" / "data.yaml"
    if yml.exists():
        for line in yml.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s[:1].isdigit() and ":" in s:
                names.append(s.split(":", 1)[1].strip())

    def label(cid):
        return names[cid] if cid < len(names) else f"id{cid}"

    def check_names(used):
        """data.yaml 의 이름 수가 라벨보다 적으면 학습이 깨진다. 미리 잡는다."""
        missing = sorted(c for c in used if c >= len(names))
        if names and missing:
            raise SystemExit(
                f"[합성] 중단 — data.yaml 에 클래스 이름이 {len(names)}개뿐인데 "
                f"라벨에는 {missing} 번이 있습니다.\n"
                f"        상품 관리에서 품목을 등록한 뒤 「Master 데이터셋 준비」 를 "
                f"다시 눌러주세요.")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ── 지난번 합성본을 지운다. 안 지우면 눌 때마다 쌓여서 실제 사진을 덮어버린다.
    old = list(img_dir.glob("synth_*.jpg"))
    for p in old:
        p.unlink()
        t = lbl_dir / (p.stem + ".txt")
        if t.exists():
            t.unlink()
    if old:
        print(f"[합성] 지난번 합성본 {len(old)}장 지움", flush=True)

    # ── 실제 사진에서만 오려낸다
    reals = sorted(p for p in img_dir.glob("*.jpg") if not p.name.startswith("synth_"))
    print(f"[합성] 실제 사진 {len(reals)}장에서 재료를 뽑습니다", flush=True)

    library, backgrounds = {}, []
    real_rows = []
    used_ids = set()
    tried = ok = 0
    for ip in reals:
        lp = lbl_dir / (ip.stem + ".txt")
        if not lp.exists():
            continue
        rows = rows_of(lp)
        if not rows:
            continue
        real_rows.append(rows)
        used_ids.update(r[0] for r in rows)
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for cid, cx, cy, bw, bh in rows:
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h,
                   (cx + bw / 2) * w, (cy + bh / 2) * h)
            boxes.append(box)
            tried += 1
            got = cut(img, box)
            if got is not None:
                ok += 1
                library.setdefault(cid, []).append(got)
        # 물체를 지운 자리를 배경으로 쓴다 (진짜 트레이 질감이 남는다)
        if len(backgrounds) < 80:
            m = np.zeros((h, w), np.uint8)
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(m, (int(x1) - 6, int(y1) - 6),
                              (int(x2) + 6, int(y2) + 6), 255, -1)
            if m.mean() < 200:          # 화면 대부분이 물체면 배경으로 못 쓴다
                backgrounds.append(cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA))

    check_names(used_ids)

    if not library or not backgrounds:
        raise SystemExit("[합성] 오려낼 재료가 없습니다. 빵이 트레이(밝은 배경) 위에 있는지 확인하세요.")

    print(f"[합성] 오려냄 {ok}/{tried}개  " +
          "  ".join(f"{label(k)}:{len(v)}" for k, v in sorted(library.items())), flush=True)
    print(f"[합성] 배경 {len(backgrounds)}장", flush=True)

    thin = [label(k) for k, v in sorted(library.items()) if len(v) < 20]
    if thin:
        print(f"[합성] 주의 — 재료가 20개 미만인 품목: {', '.join(thin)}\n"
              f"        그 품목은 각도를 더 다양하게 찍어주세요.", flush=True)

    # ── 어느 품목을 많이 붙일 것인가
    #
    # 신상품은 '여러 개 섞인 실제 사진' 이 없다 (턴테이블로 혼자 찍은 것뿐).
    # 바로 그 품목이 합성에서 많이 나와야 한다. 기존 품목은 실제 사진이
    # 이미 충분하므로 합성에서까지 자리를 차지할 이유가 없다.
    #
    # 실측(moca 를 신상품으로 가정한 실험) — 합성 안 등장 횟수와 재현율:
    #     115회 0.82 · 177회 0.92 · 231회 0.98 (실제 사진으로 배운 것과 동일)
    # 그래서 target 만큼 나올 때까지 더 만든다.
    multi = Counter()
    for rows in real_rows:
        if len(rows) > 1:
            for r in rows:
                multi[r[0]] += 1
    top = max(multi.values()) if multi else 1
    weights = {c: 1.0 + 6.0 * (1.0 - multi.get(c, 0) / top) for c in library}
    need = [c for c in library if weights[c] > 3.0]
    if need:
        print("[합성] 보강할 품목: " + ", ".join(label(c) for c in sorted(need)) +
              f"  (목표 {args.target}회 등장)", flush=True)

    # ── 합성
    made, counts = 0, Counter()
    i = 0
    while True:
        if made >= args.count:
            short = [c for c in need if counts[c] < args.target]
            if not short or made >= args.max_count:
                break
        i += 1
        bg = rng.choice(backgrounds)
        canvas, labs = compose(bg, library, n_min=args.min_obj,
                               n_max=args.max_obj, weights=weights, rng=rng)
        if not labs:
            continue
        h, w = canvas.shape[:2]
        stem = f"synth_{i:05d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), canvas,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(to_yolo(labs, w, h)) + "\n", encoding="utf-8")
        made += 1
        for cid, *_ in labs:
            counts[cid] += 1
        if made % 100 == 0:
            print(f"[합성] {made} 장", flush=True)

    print(f"[합성] 완료 — 합성 {made}장 추가", flush=True)
    print("[합성] 합성본 객체 수  " +
          "  ".join(f"{label(k)}:{counts[k]}" for k in sorted(counts)), flush=True)
    total = len(list(img_dir.glob("*.jpg")))
    print(f"[합성] 학습에 쓰일 train = 실제 {len(reals)} + 합성 {made} = {total}장", flush=True)
    print(f"SYNTH_MADE={made}", flush=True)


if __name__ == "__main__":
    main()
