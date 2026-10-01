"""Aggregate frozen evaluation results across variants and optimization seeds."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    root=Path(parser.parse_args().root);groups=defaultdict(list);digests=set();seen=set()
    for file in sorted(root.glob('*/eval_*/summary.json')):
        run=file.parent.parent
        config=json.loads((run/'config.json').read_text()); summary=json.loads(file.read_text())
        digests.add(summary['dataset_digest'])
        if len(digests)>1:
            raise ValueError('Comparison includes different dataset versions')
        for route,metrics in summary['matrix'].items():
            key=(config['variant'],summary['split'],route,config['seed'])
            if key in seen:
                raise ValueError(f'Duplicate evaluation seed: {key}')
            seen.add(key)
            groups[config['variant'],summary['split'],route].append((config['seed'],metrics))
    if not groups:
        raise ValueError('No completed evaluation summaries found')
    rows=[]
    for (variant,split,route),values in sorted(groups.items()):
        row=dict(variant=variant,split=split,route=route,seeds=','.join(str(s) for s,_ in values))
        for metric in ['success_rate','completion','mpkpe_world_m','mpkpe_root_relative_m','velocity_mps','macro_category_success']:
            samples=[m[metric] for _,m in values]
            row[metric+'_mean']=statistics.mean(samples)
            row[metric+'_seed_std']=statistics.stdev(samples) if len(samples)>1 else 0.
        rows.append(row)
    with (root/'comparison.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (root/'comparison.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print(root/'comparison.csv')


if __name__=='__main__':main()
