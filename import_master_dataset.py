#!/usr/bin/env python3
import argparse, hashlib, random, shutil, zipfile
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project', default=str(Path(__file__).resolve().parent))
    args=ap.parse_args()
    base=Path(args.project).resolve()
    imp=base/'dataset'/'import'
    imp.mkdir(parents=True, exist_ok=True)   # 없으면 만들어 준다 (어디에 넣을지 바로 보이게)
    master=base/'dataset'/'master'
    tmp=base/'dataset'/'_import_tmp'
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    zips=sorted(imp.glob('*.zip'))
    if not zips:
        raise SystemExit(f'ZIP 없음: {imp} 에 dataset ZIP을 넣어주세요.')
    pairs=[]; class_lists=[]; max_id=-1
    # 같은 사진이 여러 ZIP 에 들어 있으면 한 번만 쓴다.
    #   「학습용 ZIP으로 내보내기」는 수집 폴더 '전체'를 담으므로,
    #   옛 ZIP 이 남아 있으면 같은 사진이 두 번 학습에 들어간다.
    #   파일명이 아니라 '사진 내용'(해시)으로 판단하므로 이름을 바꿔 넣어도 걸러진다.
    seen={}                      # 사진해시 -> (ZIP이름, 파일명, 라벨내용)
    dup_same=0; dup_diff=[]      # 라벨까지 같은 중복 / 라벨이 다른 중복
    for zi,zp in enumerate(zips):
        out=tmp/f'z{zi}'
        out.mkdir()
        with zipfile.ZipFile(zp) as z: z.extractall(out)
        cls=list(out.rglob('classes.txt'))
        if cls:
            names=[x.strip() for x in cls[0].read_text(encoding='utf-8',errors='replace').splitlines() if x.strip()]
            class_lists.append(names)
        for img in list(out.rglob('*.jpg'))+list(out.rglob('*.jpeg'))+list(out.rglob('*.png')):
            lab=img.with_suffix('.txt')
            if not lab.exists(): continue
            valid=True
            for line in lab.read_text(encoding='utf-8',errors='replace').splitlines():
                if not line.strip(): continue
                p=line.split()
                if len(p)<5: valid=False; break
                try:
                    cid=int(float(p[0])); vals=list(map(float,p[1:5])); max_id=max(max_id,cid)
                    if any(v<0 or v>1 for v in vals): valid=False
                except: valid=False
            if not valid: continue

            digest=hashlib.md5(img.read_bytes()).hexdigest()
            label=lab.read_bytes().strip()
            if digest in seen:
                prev_zip, prev_name, prev_label = seen[digest]
                if prev_label == label:
                    dup_same += 1
                else:
                    # 같은 사진인데 라벨이 다르다 = 다시 라벨링했을 가능성. 먼저 것을 쓰고 알린다.
                    dup_diff.append(f'{img.name} ({zips[zi].name}) = {prev_name} ({prev_zip})')
                continue
            seen[digest]=(zips[zi].name, img.name, label)
            # ZIP 안에 train/ val/ 폴더가 있으면 그 분할을 따른다 (기본 데이터는 장면 단위로 미리 나눠 둠)
            parts=[x.lower() for x in img.relative_to(out).parts[:-1]]
            hint='val' if 'val' in parts else 'train' if 'train' in parts else None
            pairs.append((zi,img,lab,hint))
    if not pairs: raise SystemExit('유효한 image/txt 쌍을 찾지 못했습니다.')
    # choose class list long enough for actual max label id; extra unused trailing classes are ignored
    names=max(class_lists,key=len) if class_lists else [f'class_{i}' for i in range(max_id+1)]
    names=names[:max_id+1]
    if len(names)<max_id+1: names += [f'class_{i}' for i in range(len(names),max_id+1)]
    for sub in ['images/train','images/val','labels/train','labels/val']:
        d=master/sub
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)
    # 폴더 표시가 없는 것(수집 ZIP 등)만 무작위 80:20 으로 나눈다.
    # 기본 데이터까지 섞어 버리면 연속 촬영한 비슷한 사진이 train/val 에 갈라져 점수가 부풀려진다.
    rng=random.Random(42); rng.shuffle(pairs)
    free=[i for i,p in enumerate(pairs) if p[3] is None]
    nval=max(1,round(len(free)*0.2)) if free else 0; valset=set(free[:nval])
    tr=va=0
    for i,(zi,img,lab,hint) in enumerate(pairs):
        split=hint or ('val' if i in valset else 'train')
        stem=f'z{zi}_{img.stem}'
        shutil.copy2(img, master/'images'/split/(stem+img.suffix.lower()))
        shutil.copy2(lab, master/'labels'/split/(stem+'.txt'))
        if split=='train': tr+=1
        else: va+=1
    yaml=base/'dataset'/'data.yaml'
    lines=[f'path: {master.as_posix()}','train: images/train','val: images/val','','names:']
    lines += [f'  {i}: {n}' for i,n in enumerate(names)]
    yaml.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.rmtree(tmp,ignore_errors=True)
    if dup_same:
        print(f'[중복 제외] 같은 사진 {dup_same}장을 한 번만 사용했습니다.')
    if dup_diff:
        print(f'[주의] 같은 사진인데 라벨이 다른 것 {len(dup_diff)}장 — 먼저 나온 것을 사용했습니다.')
        for d in dup_diff[:5]:
            print(f'          {d}')
        if len(dup_diff) > 5:
            print(f'          ... 외 {len(dup_diff)-5}장')
    print(f'[MASTER READY] ZIP={len(zips)}, total={len(pairs)}, train={tr}, val={va}')
    print('[CLASSES] ' + ', '.join(f'{i}={n}' for i,n in enumerate(names)))
    print(f'[YAML] {yaml}')

if __name__=='__main__': main()
