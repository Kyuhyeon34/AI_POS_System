"""model_compare_colab.ipynb 방식의 채점을 앙상블까지 넓힌 것 — 젯슨 TensorRT 엔진으로 잰다.

노트북과 같은 것
  · 빵 그룹 3개(기존 빵 / 휘낭시에·와플 / 12종 전체). 그룹 밖의 빵은 회색(114)으로 가리고 채점
    (박스를 4% 넓혀 가리고, 채점할 빵에 걸친 부분은 원래 사진으로 되돌림)
  · mAP50-95 · mAP50 · precision · recall (Ultralytics 계산 방식: 101점 보간, F1 최대 지점의 P/R)
  · 젯슨 기준 오류 — 신뢰도 0.45 · NMS 0.6 · 겹친 박스 제거 → 착각 / 놓침 / 없는빵찾음 / F1
    (가린 자리에 나온 예측은 세지 않음)
  · 클래스별 mAP50-95, CSV, 차트
노트북에 없는 것
  · 앙상블 — 서버(inference_server.py)와 같은 ai/fuse.py (agree, cross_class, min_votes=1)
  · 규칙 — "rotus 2표": 합친 결과의 rotus 는 두 모델 이상 동의해야 인정
  · TTA(좌우반전, 「빵 인식」 버튼) 끔/켬 둘 다
  · 실제 계산 기준(POS 0.55) 오류 수도 같이

다른 점: 노트북의 mAP 는 Ultralytics val(클래스별 NMS 0.7)이고, 여기는 서버 엔진의 NMS(클래스 무관 0.6)
결과로 계산한다. 같은 방식으로 모든 구성을 재므로 구성끼리 비교는 공정하다.

사용:
  python3 training/compare_ensemble.py --test ~/work/examples/breadtest_0913.zip
  python3 training/compare_ensemble.py --test dataset/exp/D0914/images/test     # 폴더도 된다
"""
import argparse
import hashlib
import itertools
import json
import pickle
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from ai.fuse import fuse  # noqa: E402

TEST_NAMES = ["baggette", "muffine", "pizza", "moca", "salt", "pie", "fruit", "rotus", "plain", "waffle", "almond", "maple"]
OLD_BREADS = TEST_NAMES[:6]
FINANCIER_WAFFLE = TEST_NAMES[6:]
PASSES = {"old": ("기존 빵", OLD_BREADS), "fw": ("휘낭시에·와플", FINANCIER_WAFFLE), "all": ("12종 전체", TEST_NAMES)}
CONF, NMS_IOU, IOU_MATCH, POS = 0.45, 0.60, 0.5, 0.55      # 노트북 기준 + 실제 계산 문턱
MAP_CONF, MAX_DET, MASK_PAD = 0.001, 100, 0.04
IOUV = np.linspace(0.5, 0.95, 10)
ROT = TEST_NAMES.index("rotus")

# 엔진 (표시 이름 → 파일). 없는 엔진은 건너뛴다.
ENGINES = {"0914": "bread_0914", "0915_26n": "bread_0915_26n", "m13B": "m13_B", "m13A": "m13_A"}
TWELVE = ["0914", "0915_26n", "m13B", "m13A"]              # 서로 앙상블로 묶을 수 있는 12종 엔진
ROT2_FOR = [("0914", "0915_26n"), ("0914", "m13B"), ("0914", "0915_26n", "m13B")]


# ── 테스트셋 ────────────────────────────────────────────────────────────
def load_test(src, work):
    src = Path(src).expanduser()
    if src.suffix == ".zip":
        raw = work / "raw"
        shutil.rmtree(raw, ignore_errors=True)
        zipfile.ZipFile(src).extractall(raw)
        src = raw
    imgs, labels = [], {}
    for img in sorted(src.rglob("*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lab = img.with_suffix(".txt")
        if not lab.exists():
            lab = Path(str(lab).replace("/images/", "/labels/"))
        if lab.exists():
            imgs.append(img); labels[img.name] = lab
    GT = {}
    for p in imgs:
        rows = []
        for l in labels[p.name].read_text().splitlines():
            q = l.split()
            if len(q) == 5 and int(q[0]) < len(TEST_NAMES):
                rows.append((TEST_NAMES[int(q[0])], *map(float, q[1:])))
        GT[p.name] = rows
    return imgs, GT


def build_pass_images(key, imgs, GT, present, work):
    """그룹 밖의 빵을 회색으로 칠한 사진 (노트북 build_pass_images 와 같음)"""
    target = [c for c in PASSES[key][1] if c in present]
    out = {}
    d = work / "pass" / key
    d.mkdir(parents=True, exist_ok=True)
    for p in imgs:
        hide = [g for g in GT[p.name] if g[0] not in target]
        orig = cv2.imread(str(p))
        if not hide:
            out[p.name] = orig; continue
        im = orig.copy(); H, W = im.shape[:2]
        for c, x, y, w, h in hide:
            w2, h2 = w * (1 + MASK_PAD * 2), h * (1 + MASK_PAD * 2)
            im[int(max(0, (y - h2 / 2) * H)):int(min(H, (y + h2 / 2) * H)),
               int(max(0, (x - w2 / 2) * W)):int(min(W, (x + w2 / 2) * W))] = 114
        for c, x, y, w, h in GT[p.name]:
            if c in target:
                y1, y2 = int(max(0, (y - h / 2) * H)), int(min(H, (y + h / 2) * H))
                x1, x2 = int(max(0, (x - w / 2) * W)), int(min(W, (x + w / 2) * W))
                im[y1:y2, x1:x2] = orig[y1:y2, x1:x2]
        out[p.name] = im
    return target, out


# ── 추론 (엔진·그룹·TTA 별 캐시) ────────────────────────────────────────
def engine_names(stem):
    f = BASE / "models" / f"{stem}.engine.names.json"
    return json.loads(f.read_text()) if f.exists() else TEST_NAMES


def run_engine(stem, pass_imgs, cache_file):
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())
    from ai.detector import JetsonTensorRTDetector
    names = engine_names(stem)
    d = JetsonTensorRTDetector(conf=MAP_CONF, iou=NMS_IOU)
    d.load(str(BASE / "models" / f"{stem}.engine"))
    res = {}
    for key, (target, ims) in pass_imgs.items():
        for name, im in ims.items():
            for tta in (False, True):
                dets = sorted(d.infer(im, draw=False, tta=tta)[1], key=lambda x: -x["confidence"])[:MAX_DET]
                out = []
                for x in dets:
                    n = names[int(x["class_id"])] if int(x["class_id"]) < len(names) else "?"
                    if n in TEST_NAMES:                  # 테스트셋에 없는 빵(peanut 등)은 12종 번호로 못 바꾸므로 뺀다
                        out.append({"class_id": TEST_NAMES.index(n), "confidence": float(x["confidence"]),
                                    "xyxy": [float(v) for v in x["xyxy"]]})
                res[(key, tta, name)] = out
    d.close()
    cache_file.write_bytes(pickle.dumps(res))
    return res


# ── 지표 ───────────────────────────────────────────────────────────────
def box_iou(a, b):
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_predictions(pred_cls, gt_cls, iou):
    """Ultralytics DetectionValidator.match_predictions (use_scipy=False)"""
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    if not len(pred_cls) or not len(gt_cls):
        return correct
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])          # gt x pred
    for i, t in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1], i] = True
    return correct


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0])); mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return np.trapz(np.interp(x, mrec, mpre), x)


def ap_per_class(tp, conf, pred_cls, target_cls):
    """Ultralytics metrics.ap_per_class 요약판 → mp, mr, map50, map50-95, 클래스별 map50-95"""
    i = np.argsort(-conf); tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    classes, nt = np.unique(target_cls, return_counts=True)
    px = np.linspace(0, 1, 1000)
    ap = np.zeros((len(classes), tp.shape[1])); p_curve = np.zeros((len(classes), 1000)); r_curve = np.zeros((len(classes), 1000))
    for ci, c in enumerate(classes):
        idx = pred_cls == c; n_l = nt[ci]; n_p = idx.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[idx]).cumsum(0); tpc = tp[idx].cumsum(0)
        recall = tpc / (n_l + 1e-16); precision = tpc / (tpc + fpc)
        r_curve[ci] = np.interp(-px, -conf[idx], recall[:, 0], left=0)
        p_curve[ci] = np.interp(-px, -conf[idx], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    f1 = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
    fm = f1.mean(0); nf = round(len(fm) * 0.1 * 2) // 2 + 1
    fm = np.convolve(np.concatenate((np.ones(nf // 2) * fm[0], fm, np.ones(nf // 2) * fm[-1])), np.ones(nf) / nf, mode="valid")
    k = fm.argmax()
    return dict(precision=float(p_curve[:, k].mean()), recall=float(r_curve[:, k].mean()),
                mAP50=float(ap[:, 0].mean()), **{"mAP50-95": float(ap.mean())},
                per_class={TEST_NAMES[int(c)]: float(ap[ci].mean()) for ci, c in enumerate(classes)})


def config_dets(cfg, rule, raw, key, tta, name):
    lists = [raw[e][(key, tta, name)] for e in cfg]
    if len(lists) == 1:
        return lists[0]
    out = fuse(lists, cross_class=True, min_votes=1)
    if rule == "rot2":
        out = [d for d in out if not (d["class_id"] == ROT and d.get("votes", 1) < 2)]
    return out


def score(cfg, rule, raw, key, target, imgs, GT, shapes, tta):
    tidx = {TEST_NAMES.index(c) for c in target}
    stats_tp, stats_conf, stats_cls, tcls = [], [], [], []
    cnt = {thr: Counter() for thr in (CONF, POS)}
    for p in imgs:
        H, W = shapes[p.name]
        gts, ignore = [], []
        for c, x, y, w, h in GT[p.name]:
            b = [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
            (gts if c in target else ignore).append((TEST_NAMES.index(c), b))
        dets = config_dets(cfg, rule, raw, key, tta, p.name)
        # mAP (그룹 빵 라벨만 정답. 노트북처럼 가린 자리 예측도 그대로 둔다)
        pc = np.array([d["class_id"] for d in dets], int); pconf = np.array([d["confidence"] for d in dets], float)
        gc = np.array([g[0] for g in gts], int)
        iou = box_iou([g[1] for g in gts], [d["xyxy"] for d in dets])
        stats_tp.append(match_predictions(pc, gc, iou)); stats_conf.append(pconf); stats_cls.append(pc); tcls.append(gc)
        # 젯슨 기준 개수 (가린 자리 예측은 세지 않음, 같은 빵 먼저 매칭 → 다른 빵 매칭)
        for thr in (CONF, POS):
            preds = [d for d in sorted(dets, key=lambda d: -d["confidence"]) if d["confidence"] >= thr]
            preds = [d for d in preds if not any(box_iou([d["xyxy"]], [g[1]])[0, 0] >= IOU_MATCH for g in ignore)]
            used, match = set(), {}
            for same in (True, False):
                for pi, d in enumerate(preds):
                    if pi in match:
                        continue
                    best, bi = IOU_MATCH, None
                    for gi, g in enumerate(gts):
                        if gi in used or (same and g[0] != d["class_id"]):
                            continue
                        v = box_iou([d["xyxy"]], [g[1]])[0, 0]
                        if v >= best:
                            best, bi = v, gi
                    if bi is not None:
                        used.add(bi); match[pi] = bi
            c = cnt[thr]
            c["박스"] += len(gts)
            c["tp"] += sum(preds[pi]["class_id"] == gts[gi][0] for pi, gi in match.items())
            c["착각"] += sum(preds[pi]["class_id"] != gts[gi][0] for pi, gi in match.items())
            c["놓침"] += len(gts) - len(used)
            c["없는빵찾음"] += len(preds) - len(match)
    m = ap_per_class(np.concatenate(stats_tp), np.concatenate(stats_conf), np.concatenate(stats_cls), np.concatenate(tcls))
    row = {k: m[k] for k in ("mAP50-95", "mAP50", "precision", "recall")}
    for thr, tag in ((CONF, ""), (POS, "POS0.55 ")):
        c = cnt[thr]
        P = c["tp"] / max(1, c["tp"] + c["착각"] + c["없는빵찾음"]); R = c["tp"] / max(1, c["박스"])
        row.update({f"{tag}착각": c["착각"], f"{tag}놓침": c["놓침"], f"{tag}없는빵찾음": c["없는빵찾음"],
                    f"{tag}오류합계": c["착각"] + c["놓침"] + c["없는빵찾음"], f"{tag}F1": round(2 * P * R / max(1e-9, P + R), 3)})
    row["박스"] = cnt[CONF]["박스"]
    return row, m["per_class"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="테스트셋 zip 또는 폴더 (YOLO 라벨, 12종 번호)")
    ap.add_argument("--out", default=str(BASE / "runs" / "model_compare_ens"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"; work.mkdir(exist_ok=True)

    imgs, GT = load_test(args.test, work)
    if not imgs:
        raise SystemExit(f"테스트 사진을 찾지 못했습니다: {args.test}")
    present = [n for n in TEST_NAMES if any(g[0] == n for v in GT.values() for g in v)]
    shapes = {p.name: cv2.imread(str(p)).shape[:2] for p in imgs}
    tag = hashlib.md5("".join(sorted(p.name for p in imgs)).encode()).hexdigest()[:8]
    print(f"테스트 사진 {len(imgs)}장, 박스 {sum(len(v) for v in GT.values())}개, 빵 {len(present)}종 "
          f"(없는 빵: {[n for n in TEST_NAMES if n not in present]})", flush=True)

    pass_imgs = {k: build_pass_images(k, imgs, GT, present, work) for k in PASSES}
    raw = {}
    for label, stem in ENGINES.items():
        if not (BASE / "models" / f"{stem}.engine").exists():
            print(f"(엔진 없음: {stem} — 건너뜀)"); continue
        raw[label] = run_engine(stem, pass_imgs, work / f"cache_{stem}_{tag}.pkl")
        print(f"  추론 완료 {label} ({stem})", flush=True)

    have12 = [e for e in TWELVE if e in raw]
    configs = [((e,), "") for e in raw]
    configs += [(c, "") for r in (2, 3) for c in itertools.combinations(have12, r)]
    configs += [(c, "rot2") for c in ROT2_FOR if all(e in raw for e in c)]
    cfg_name = lambda c, r: "+".join(c) + (" · rotus 2표" if r else "")

    import pandas as pd
    rows, per_class = [], {}
    for key, (title, _) in PASSES.items():
        target = pass_imgs[key][0]
        for tta in (False, True):
            for cfg, rule in configs:
                row, pc = score(cfg, rule, raw, key, target, imgs, GT, shapes, tta)
                rows.append({"그룹": title, "TTA": "켬" if tta else "끔", "구성": cfg_name(cfg, rule),
                             "모델 수": len(cfg), **row})
                per_class[(title, "켬" if tta else "끔", cfg_name(cfg, rule))] = pc
        print(f"  채점 완료 {title}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "results_ensemble.csv", index=False, encoding="utf-8-sig")
    pcdf = pd.DataFrame(per_class).T
    pcdf.index.names = ["그룹", "TTA", "구성"]
    pcdf.to_csv(out / "per_class_map50-95.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["구성", "mAP50-95", "mAP50", "precision", "recall", "착각", "놓침", "없는빵찾음", "오류합계", "F1",
            "POS0.55 오류합계", "POS0.55 F1"]
    for key, (title, _) in PASSES.items():
        for tta in ("끔", "켬"):
            sub = res[(res["그룹"] == title) & (res["TTA"] == tta)].sort_values(["mAP50-95", "오류합계"], ascending=[False, True])
            print(f"\n================ {title} · TTA {tta} (박스 {int(sub['박스'].iloc[0])}개) — mAP50-95 높은 순 ================")
            print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ── 차트: 그룹마다 TTA 끔/켬 두 칸, 윗줄 mAP50-95 막대 · 아랫줄 오류 누적 막대
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if "Nanum" in f or "NotoSansCJK" in f:
            font_manager.fontManager.addfont(f)
    for fam in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if any(fam == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam; break
    plt.rcParams["axes.unicode_minus"] = False
    BROWN, TAN, SAND, INK = "#B5651D", "#D9A66C", "#E3CBA7", "#3E2723"
    for key, (title, _) in PASSES.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.55})
        for j, tta in enumerate(("끔", "켬")):
            sub = res[(res["그룹"] == title) & (res["TTA"] == tta)].sort_values("mAP50-95", ascending=False).reset_index(drop=True)
            x = np.arange(len(sub)); single = sub["모델 수"] == 1
            ax = axes[0, j]
            ax.bar(x, sub["mAP50-95"], color=[TAN if s else BROWN for s in single])
            for i, v in enumerate(sub["mAP50-95"]):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
            lo = float(sub["mAP50-95"].min())
            ax.set_ylim(max(0, lo - 0.05), min(1, float(sub["mAP50-95"].max()) + 0.02))
            ax.set_title(f"{title} · TTA {tta} — mAP50-95 (연한색=단독, 진한색=앙상블)", fontsize=10, color=INK)
            ax.set_xticks(x); ax.set_xticklabels(sub["구성"], rotation=55, ha="right", fontsize=7)
            bx = axes[1, j]
            bx.bar(x, sub["착각"], color=BROWN, label="착각")
            bx.bar(x, sub["놓침"], bottom=sub["착각"], color=TAN, label="놓침")
            bx.bar(x, sub["없는빵찾음"], bottom=sub["착각"] + sub["놓침"], color=SAND, label="없는 빵 찾음")
            for i, v in enumerate(sub["오류합계"]):
                bx.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=7, color=INK)
            bx.set_title(f"젯슨 기준 오류 (신뢰도 {CONF}, NMS {NMS_IOU})", fontsize=10, color=INK)
            bx.set_xticks(x); bx.set_xticklabels(sub["구성"], rotation=55, ha="right", fontsize=7)
            bx.legend(frameon=False, fontsize=8)
            for a in (ax, bx):
                a.grid(axis="y", alpha=.3)
                for sp in ("top", "right"):
                    a.spines[sp].set_visible(False)
        fig.savefig(out / f"chart_{key}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n저장: {out}/results_ensemble.csv, per_class_map50-95.csv, chart_old.png, chart_fw.png, chart_all.png")


if __name__ == "__main__":
    main()
