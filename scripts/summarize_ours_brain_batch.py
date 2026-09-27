"""Summarize completed ours HEFT batch artifacts without changing evaluation."""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


METRICS = {
    'joint_mae_deg': ('joint_mae_rad', 180 / np.pi),
    'joint_rmse_deg': ('joint_rmse_rad', 180 / np.pi),
    'root_relative_body_pos_mm': ('root_relative_body_pos_mm', 1),
    'root_relative_body_ori_rad': ('root_relative_body_ori_rad', 1),
}


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'count': 0, 'mean': None, 'median': None, 'case_mean_p95': None}
    assert np.isfinite(a).all()
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'case_mean_p95': float(np.quantile(a, .95))}


def summarize(out, conversion_report=None):
    results = read(out / 'batch_results.json')
    config = read(out / 'run_config.json')
    assert len(results) == config['motion_count'], 'Batch is incomplete'
    ids = [r['case'].split('_', 1)[0] for r in results]
    assert len(ids) == len(set(ids)), 'Duplicate case IDs'
    issues, rows = [], []
    manifest = read(conversion_report) if conversion_report else None
    conversion = {}
    if manifest is not None:
        assert len(manifest['motions']) == len(results), 'Conversion/batch counts differ'
        for m in manifest['motions']:
            mid = Path(m['output']).stem.split('_', 1)[0]
            assert mid not in conversion
            conversion[mid] = m
            for kind in ('source', 'output'):
                assert sha256(m[kind]) == m[kind + '_sha256'], (mid, kind, 'hash mismatch')
        assert set(conversion) == set(ids)
    for r, mid in zip(results, ids):
        folder = out / mid
        row = {'id': mid, 'case': r['case'], 'success': False,
               'status': 'execution_error', 'sample_count': 0}
        suspicious = []
        log = folder / 'console.log'
        if log.exists():
            for line in log.read_text(errors='replace').splitlines():
                # JSON report fields such as fall_or_nonfinite are normal output;
                # detect actual warnings and explicit instability/reset messages.
                if re.search(r'WARNING|unstable|instability|simulation.*reset|reset.*simulation|NaN|Inf\b', line, re.I):
                    suspicious.append(line[:1000])
        if suspicious:
            issues.append({'case': r['case'], 'log': str(log), 'lines': suspicious})
        row['suspicious_log'] = bool(suspicious)
        if not r.get('execution_error'):
            m, report = read(folder / 'metrics.json'), read(folder / 'report.json')
            assert m['case'] == r['case']
            assert m['policy_sha256'] == config['policy_sha256']
            assert m['source_csv_sha256'] == report['source_csv_sha256']
            assert sha256(report['csv']) == m['source_csv_sha256']
            if conversion:
                assert conversion[mid]['output_sha256'] == m['source_csv_sha256']
            success = report['playback_started'] and report['playback_complete'] and not report['fell']
            assert success == m['success'] == r['success']
            if success:
                status = 'success'
            elif not report['playback_started']:
                status = 'not_started'
            elif report['playback_complete']:
                status = 'stopped_during_final_hold'
            else:
                status = 'stopped_during_playback'
            with np.load(folder / 'tracking_samples.npz') as s:
                n = m['sample_count']
                assert s['joint_error_rad'].shape == (n, 21)
                assert s['position_mm'].shape == s['orientation_rad'].shape == (n, len(config['bodies']))
                for key in ('joint_error_rad', 'position_mm', 'orientation_rad'):
                    assert np.isfinite(s[key]).all(), (mid, key)
                if success:
                    assert n == m['motion_frames'] - 1
                if n:
                    q = s['joint_error_rad']
                    computed = [np.abs(q).mean(), np.sqrt((q*q).mean(1)).mean(), s['position_mm'].mean(), s['orientation_rad'].mean()]
                    for (key, _), value in zip(METRICS.values(), computed):
                        assert np.isclose(m[key], value, rtol=1e-10, atol=1e-12), (mid, key)
            row.update(success=success, status=status, sample_count=m['sample_count'],
                       playback_started=report['playback_started'], playback_complete=report['playback_complete'],
                       fell=report['fell'], source_csv_sha256=m['source_csv_sha256'])
            for dest, (src, scale) in METRICS.items():
                row[dest] = None if m[src] is None else m[src] * scale
        else:
            row['error'] = r.get('error', 'returncode=' + str(r.get('returncode')))
        rows.append(row)
    successful = [r for r in rows if r['success']]
    summary = {
        'motion_count': len(rows), 'success_count': len(successful),
        'success_rate': len(successful) / len(rows) if rows else None,
        'status_counts': dict(collections.Counter(r['status'] for r in rows)),
        'success_metrics': {k: describe([r[k] for r in successful]) for k in METRICS},
        'suspicious_log_count': len(issues), 'suspicious_logs': issues,
        'successful_with_suspicious_logs': sum(r['success'] and r['suspicious_log'] for r in rows),
        'provenance': config, 'conversion_report': str(conversion_report) if conversion_report else None,
        'conversion_source_output_hashes_verified': bool(conversion),
        'success_definition': 'Playback started and completed, no fall/nonfinite stop including final hold; no tracking-error threshold.',
        'aggregation': 'Equal weight per successful motion; P95 is across motion means, not pooled frame errors.',
        'root_assumption': 'Original root treated as training chest base_link, with policy-coordinate joints (--source-root-frame chest); producer semantics not independently verified.',
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / 'per_motion.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    rate = 'N/A' if summary['success_rate'] is None else f"{100*summary['success_rate']:.2f}%"
    lines = ['# Ours HEFT 大脑动作批量 sim2sim 评测', '',
             f"稳定完成 **{len(successful)}/{len(rows)}（{rate}）**。分母为 batch_results.json 中全部动作。", '',
             '成功定义：已起播、播完且包括末帧保持阶段在内没有触发停止条件；没有人为添加跟踪误差阈值。完整播放不表示所有动作都具有高精度，也不表示文本任务完成或真机通过。', '',
             '| 状态 | 数量 |', '|---|---:|']
    labels = {'success': '成功', 'execution_error': '执行错误', 'not_started': '未起播',
              'stopped_during_playback': '播放期间停止', 'stopped_during_final_hold': '末帧保持期间停止'}
    lines.extend(f"| {label} | {summary['status_counts'].get(key, 0)} |" for key, label in labels.items())
    lines += ['', '## 成功动作跟踪性能', '', '| 指标 | 案例等权均值 | 案例均值中位数 | 案例均值 P95 |', '|---|---:|---:|---:|']
    for key, stats in summary['success_metrics'].items():
        vals = ['N/A' if stats[s] is None else f'{stats[s]:.6f}' for s in ('mean', 'median', 'case_mean_p95')]
        lines.append('| ' + key + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', 'Joint MAE 为实际关节角与参考角绝对差均值；Joint RMSE 为每帧先对 21 关节求 RMSE，再对时间平均。上述角度以 degree 表示。失败动作截断前误差保留在逐例文件中，不混入成功均值；未起播不是零误差。', '',
              'Body 指标在实际/参考各自胸部 base_link 坐标系中计算，14 个节点等权；位置为 body 原点欧氏距离（mm），姿态为最短旋转角（rad）。这些指标消除整体平移与旋转，不评价全局路径、绝对朝向或根倾角。', '',
              '## 数据和控制约定', '',
              '- 25 Hz 转换为 50 Hz；参考 root 按训练胸部 base_link、关节按策略坐标处理（chest）。这是沿用假设，未独立核实大脑生成端 root 定义。',
              '- 本批 ours HEFT 使用 ONNX Runtime CPU 推理。模型路径与 SHA-256 见 run_config.json。',
              '- 物理 500 Hz、策略 50 Hz；先执行 3 s AMP 站立，然后用 2 s 混合切换至 HEFT，沿用 PD/EMA 控制链。',
              '- 切换后陀螺仪模长 <0.3 rad/s 连续 0.5 s 才起播；仿真时间达到 20 s 仍未起播则超时停止。',
              '- 胸部高度 <0.4 m 或 qpos 非有限则停止；播放结束后保持末帧约 2 s，保持期间停止仍判失败。',
              '- 指标窗口仅为起播后参考帧 0 至 N−2 的物理积分前采样；排除 AMP 站立、混合、等待和保持。',
              f"- 转换源文件与输出文件 SHA-256 全量核验：{'通过' if conversion else '未提供 conversion report'}。样本 NPZ 与 metrics 数值、形状、有限值已核验。", '',
              '## 仿真日志检查', '',
              f"发现 {len(issues)} 个案例包含可疑 warning/instability/reset 日志，其中 {summary['successful_with_suspicious_logs']} 个按原停止条件计为成功。详见 summary.json 的 suspicious_logs；日志标记不改变控制或原成功判定。", '',
              '## 产物', '', '- [汇总 JSON](summary.json)', '- [逐动作 CSV](per_motion.csv)', '- [运行来源](run_config.json)',
              '- numeric ID 子目录保存 report.json、metrics.json、tracking_samples.npz、console.log。', '',
              '## 复现汇总', '', '在项目根目录执行（核验现有结果，不重跑仿真）：', '', '```bash',
              f'.venv/bin/python -B scripts/summarize_ours_brain_batch.py --output-dir {out}' +
              (f' --conversion-report {conversion_report}' if conversion_report else ''), '```', '']
    if conversion_report:
        lines += ['## 复现评测', '',
                  '在项目根目录执行。已有 metrics.json 且策略和 CSV 哈希匹配的案例将复用；需要全新重跑时将 output-dir 改为新的目录。', '', '```bash',
                  f'.venv/bin/python -B scripts/evaluate_ours_brain_batch.py --input-dir {conversion_report.parent} --output-dir {out} --workers {config["workers"]}',
                  '```', '']
    (out / 'README.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--conversion-report', type=Path)
    args = parser.parse_args()
    result = summarize(args.output_dir.resolve(), args.conversion_report)
    print(json.dumps({k: result[k] for k in ('motion_count', 'success_count', 'success_rate', 'status_counts', 'suspicious_log_count')}, ensure_ascii=False))
