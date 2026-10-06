"""Plot paired family confidence intervals for the two predefined refinement variants."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    args=p.parse_args()
    rows=[]
    for split in ('val','test'):
        for name,label,color in [('teacher_correction_run01','Frozen source encoders','#2563a6'),
                                 ('all_sources_run01','Train source encoders','#16846b')]:
            file=args.study/name/('comparison_val_300.json' if split=='val' else 'comparison_test.json')
            rows.append((f'{label} / {split}',color,json.loads(file.read_text())))
    metrics=[('success_rate','Success rate change (pp)','pooled_success_delta_ci95_pp',100),
             ('completion','Completion change (pp)','completion_pp',100),
             ('mpkpe_root_relative_m','Root-relative error change (cm)','mpkpe_root_relative_m',100),
             ('mpkpe_world_m','World error change (cm)','mpkpe_world_m',100)]
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for ax,(metric,title,interval_key,scale) in zip(axes.flat,metrics):
        ax.axvline(0,color='#777777',linestyle='--',linewidth=1)
        for index,(label,color,data) in enumerate(rows):
            value=scale*(data['means'][metric]['candidate']-data['means'][metric]['baseline'])
            interval=data['pooled_success_delta_ci95_pp'] if metric=='success_rate' else data['mean_delta_family_ci95'][interval_key]
            lo,hi=[x*(100 if metric.startswith('mpkpe') else 1) for x in interval]
            y=len(rows)-1-index
            ax.hlines(y,lo,hi,color=color,linewidth=2)
            ax.scatter([value],[y],color=color,marker='o' if index<2 else 's',s=45,zorder=3)
            ax.annotate(f'{value:+.2f}',(value,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        ax.set_yticks(list(reversed(range(len(rows)))),[r[0] for r in rows],fontsize=9)
        ax.set_ylim(-.55,3.65)
        ax.set_title(title+'\n'+('Higher is better' if not metric.startswith('mpkpe') else 'Lower is better'),fontsize=11)
        ax.grid(axis='x',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Teacher correction vs stage C: paired-family 95% intervals\nOne training seed; full validation and held-out test, six routes each',fontsize=13)
    for extension in ('png','svg'):
        path=args.study/f'paired_improvements.{extension}'
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
