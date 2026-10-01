"""Reproducible 10k paired selection with family-disjoint 80/10/10 splits."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
QUOTAS = dict(main=6000, jump=1500, squat=1000, crawl=700, kneel=600, acrobatic=200)


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def select(output):
    roots = dict(human=Path('dataset/Human_22d_50hz'), g1=Path('dataset/g1_50hz'), roban=Path('dataset/roban_50hz'))
    ext = dict(human='.npz', g1='.csv', roban='.npz')
    inventory = {s: {p.relative_to(ROOT/r).with_suffix('').as_posix() for p in (ROOT/r).rglob('*'+ext[s])}
                 for s,r in roots.items()}
    common = set.intersection(*inventory.values())
    families = defaultdict(list)
    for key in sorted(common):
        families[re.sub(r'_\d+$', '', Path(key).name.split('__')[0]).casefold()].append(key)
    splits = dict(train=.8, val=.1, test=.1)
    for seed in range(1000):
        pools = defaultdict(list)
        for family, keys in families.items():
            bucket = int(digest(f'{seed}:{family}')[:16],16)%100
            split = 'train' if bucket<80 else 'val' if bucket<90 else 'test'
            for key in keys:
                pools[split,key.split('/')[0]].append((family,key))
        if all(len(pools[s,c]) >= round(n*frac) for c,n in QUOTAS.items() for s,frac in splits.items()):
            break
    else:
        raise ValueError('Dataset does not supply the requested family/category quotas')
    rows=[]
    for split,frac in splits.items():
        for category,total in QUOTAS.items():
            grouped=defaultdict(list)
            for family,key in pools[split,category]:
                grouped[family].append(key)
            ranked=[]
            for family,keys in grouped.items():
                for rank,key in enumerate(sorted(keys,key=lambda k:digest('member:'+k))):
                    ranked.append((rank,digest('family:'+family),family,key))
            for _,_,family,key in sorted(ranked)[:round(total*frac)]:
                rows.append(dict(motion_id=key,family_id=family,category=category,split=split,
                                 paths={s:str(r/(key+ext[s])) for s,r in roots.items()}))
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report=dict(count=len(rows),split_seed=seed,split_counts=dict(Counter(r['split'] for r in rows)),
                category_quotas=QUOTAS,family_rule='casefold stem before __, strip trailing numeric take',
                inventories={s:len(v) for s,v in inventory.items()},common=len(common),
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    select(parser.parse_args().output)
