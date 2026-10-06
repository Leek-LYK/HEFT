import datetime,json,math,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path.cwd();out=root/'logs/v0.2/transformer_20261006/analysis_5600';end=5600;window=500
bases={'MLP':root/'logs/v0.2/staged_run01','Transformer':root/'logs/v0.2/transformer_20261006'}
data={};stats={};configs={};snapshots={}
keys=['reward','policy_loss','value_loss','kl','entropy','grad_norm','done_fraction','recovery_fraction']
for label,base in bases.items():
 for robot in ['g1','roban']:
  run=base/f'teacher_{robot}';raw=(run/'metrics.jsonl').read_text();rows=[json.loads(x) for x in raw[:raw.rfind('\n')].splitlines()]
  assert len(rows)>=end
  assert [r['update'] for r in rows[:end]]==list(range(1,end+1))
  assert all(math.isfinite(v) for r in rows[:end] for v in r.values())
  assert rows[end-1]['transitions']==end*4096*32
  configs[label,robot]=json.loads((run/'config.json').read_text());data[label,robot]=rows[:end]
  snapshots[label+'/'+robot]=dict(observed_update=rows[-1]['update'],compared_update=end)
  recent=rows[end-window:end]
  s={k:statistics.mean(r[k] for r in recent) for k in keys}
  s.update(recovery_events_window=rows[end-1]['recoveries']-rows[end-window-1]['recoveries'],recoveries_total=rows[end-1]['recoveries'],elapsed_seconds=rows[end-1]['elapsed_seconds'])
  stats[label+'/'+robot]=s
  (out/f'{label}_{robot}_metrics_through_5600.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[:end])+'\n')
for robot in ['g1','roban']:
 for k in ['num_envs','rollout_steps','learning_rate','epochs','minibatch_size','clip','entropy_weight','value_weight','seed','dataset','history_steps','action_history','command_offsets','max_grad_norm']:
  assert configs['MLP',robot][k]==configs['Transformer',robot][k],k
 assert json.loads((bases['MLP']/f'teacher_{robot}/environment.json').read_text())==json.loads((bases['Transformer']/f'teacher_{robot}/environment.json').read_text())
report=dict(checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),common_endpoint=end,window=[5101,5600],transitions_per_teacher=734003200,ppo_optimizer_steps_per_teacher=268800,configs_and_environments_matched=True,snapshot_progress=snapshots,means=stats)
(out/'comparison.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
for row,robot in enumerate(['g1','roban']):
 for ax,k,title in zip(axes[row],['reward','kl','entropy','recovery_fraction'],['Reward','Approx. KL','Policy entropy','Recoveries / 100k transitions']):
  for label in bases:
   y=np.array([r[k] for r in data[label,robot]])
   if k=='recovery_fraction':y*=100000
   w=200;ax.plot(np.arange(w,end+1),np.convolve(y,np.ones(w)/w,'valid'),label=label)
  ax.set_title(robot.upper()+' — '+title);ax.set_xlabel('PPO updates');ax.axvspan(5101,5600,color='grey',alpha=.1);ax.grid(alpha=.2)
  if row==0 and k=='reward':ax.legend()
fig.suptitle('A vs A_T: matched updates, 200-update moving averages; final window shaded')
fig.savefig(out/'training_comparison.png',dpi=160)
lines=['# A_T与MLP教师同期训练比较：第5600轮','',
       '统计时间：'+report['checked_at']+'（北京时间）。只读分析训练记录，未暂停训练、修改超参数或启动额外仿真。','',
       '## 结论','',
       '当前相同采样预算下，Transformer教师的训练奖励落后于MLP，KL与熵更高；曲线仍持续改善，没有看到数值发散证据。尚不能由随机策略训练奖励直接判断最终确定性策略的完整验证成功率，也不能推广到尚未开始的B_T/E_T学生。','',
       '## 比较口径','',
       '统一截止第5600轮，每教师734,003,200 transitions、268,800次PPO优化。表格采用第5101–5600轮500轮均值，曲线采用200轮滑动平均。指标来自metrics.jsonl，与TensorBoard使用同一log_metrics写入，避免网页平滑系数不同造成误读。','',
       '两路线每机型4096环境、rollout32步、3 epochs、minibatch8192、学习率1e-4、clip0.2、entropy_weight0.005、value_weight0.5、seed0、数据/时序上下文/梯度裁剪相同；环境JSON逐值一致。网络结构与参数量不同，实际轨迹也随策略不同。未把MLP最终10000轮与当前Transformer作比较。','',
       '| 指标（最近500轮） | G1 MLP | G1 Transformer | Roban MLP | Roban Transformer |','|---|---:|---:|---:|---:|']
for k,title in [('reward','平均奖励'),('policy_loss','Policy loss'),('value_loss','Value loss'),('kl','Approx. KL'),('entropy','策略熵'),('grad_norm','裁剪前梯度范数'),('done_fraction','Done fraction'),('recovery_fraction','Recovery fraction')]:
 vals=[stats[l+'/'+r][k] for r in ['g1','roban'] for l in ['MLP','Transformer']]
 lines.append('| '+title+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
for k,title in [('recovery_events_window','窗口新增恢复事件'),('recoveries_total','累计恢复事件')]:
 vals=[stats[l+'/'+r][k] for r in ['g1','roban'] for l in ['MLP','Transformer']];lines.append('| '+title+' | '+' | '.join(str(v) for v in vals)+' |')
lines+=['','![同期训练曲线](../../logs/v0.2/transformer_20261006/analysis_5600/training_comparison.png)','',
        '## 解读','',
        '1. G1奖励0.154270→0.135552（约-12.1%）；Roban 0.174663→0.160004（约-8.4%）。500轮窗口在1000、3000、5000轮的比较也呈落后，差距并非单轮噪声。两种结构奖励均继续上升，不能解释为已经完全不再学习。','',
        '2. G1 KL约2.50倍、Roban约2.58倍。相同学习率与PPO clip并不意味着策略更新幅度相同；目前Transformer的行为分布变化更大。但这是近似KL，不是clip fraction，也没有预设据此自动提前停止的阈值。不能仅凭这一项宣称训练崩溃。','',
        '3. Transformer策略熵G1高约16.1%、Roban高约13.9%，提示随机策略保持更大的探索噪声。训练奖励受该噪声影响，最终动作均值策略可能有不同表现，须按既定完整验证评价。不同机器人动作维数不同，熵不在G1与Roban之间直接比较。','',
        '4. Transformer policy loss更负，不代表控制能力更好；PPO surrogate受旧策略、优势和当前采样分布影响。Value loss G1较低、Roban近似，但也不能直接说明actor更好，因为双方回报分布不同。梯度范数是clip_grad_norm_返回的裁剪前范数，数值大于max_grad_norm=1不表示裁剪失效。','',
        '5. G1双方恢复事件为0。Roban窗口新增恢复1183→1553（+31.3%）；累计16851→20059（+19.0%）。窗口约为每10万transitions 1.81→2.37次。原始恢复计数、recovery_fraction与验证恢复片段含义不同，不能替代完整动作失败率。Done fraction同时包含完成与其他终止，不能等同摔倒率。','',
        '6. 第5600轮累计墙钟，G1 MLP约9.93h、Transformer约7.75h；Roban约9.86h、7.91h。仅是两次实际运行观测：历史GPU并发/负载、不同参数量等未受控，不能当作严格架构加速比，更不能外推G1本机ONNX的50Hz时延。','',
        '## 对当前实验的建议','',
        '继续按原10000轮预算运行，不在中途依据这些指标修改学习率或熵系数，以保留原先固定超参数的架构对照。完整验证后再决定是否有必要另开Transformer专属超参数实验。若需要定位KL偏高，后续可单独研究学习率/目标KL控制；若研究探索噪声，则另做熵系数对照，不把这些改变混进本轮。','',
        '## 可复查产物','',
        '- [窗口统计与可比性核验](../../logs/v0.2/transformer_20261006/analysis_5600/comparison.json)',
        '- [生成脚本](../../logs/v0.2/transformer_20261006/analysis_5600/compare_training.py)',
        '- 该目录保留四份截止5600轮的指标快照，后续训练继续写入不改变本报告输入。','']
(root/'科研/v0.2/15_Transformer与MLP同期训练对比.md').write_text('\n'.join(lines))
print(json.dumps(report,ensure_ascii=False))
