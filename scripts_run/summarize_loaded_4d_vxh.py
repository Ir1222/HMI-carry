"""Summarize an evaluated checkpoint and guard a bounded local training loop.

Selection is based on fixed-target control and Track retention, never training
return or checkpoint number. This utility does not change acceptance thresholds.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def summarize(folder, updates):
    gen = json.loads((folder / 'targets.json').read_text())
    track = json.loads((folder / 'track.json').read_text())
    targets = []
    for goal in gen['sequences']:
        goal = goal[0]
        trials = [t for t in gen['trials'] if t['segments'][0]['target'] == goal]
        segments = [t['segments'][0] for t in trials]
        steady = [s for s in segments if s.get('mae') is not None and s.get('steady_seconds', 0) >= gen['criteria']['min_steady_s']]
        reasons = Counter(reason for s in segments for reason in s['failure_reasons'])
        reasons.update(term for t in trials for term in t['termination_terms'])
        targets.append({'target': goal, 'trials': len(trials), 'successes': sum(t['task_success'] for t in trials),
                        'horizon_complete': sum(t['outcome'] == 'horizon_complete' for t in trials),
                        'measured_mean': np.mean([s['measured_mean'] for s in steady], axis=0).tolist() if steady else None,
                        'mae': np.mean([s['mae'] for s in steady], axis=0).tolist() if steady else None,
                        'p95_error_mean_across_trials': np.mean([s['p95'] for s in steady], axis=0).tolist() if steady else None,
                        'forward_net_displacement_mean': float(np.mean([s.get('forward_net_displacement', 0) for s in segments])),
                        'pelvis_change_mean': float(np.mean([s.get('pelvis_change', 0) for s in segments])),
                        'double_support_fraction': float(np.mean([s.get('double_support_fraction', 0) for s in steady])) if steady else None,
                        'hand_geometry_fraction': float(np.mean([s.get('hand_geometry_fraction', 0) for s in steady])) if steady else None,
                        'survival_s_mean': float(np.mean([t['seconds'] for t in trials])),
                        'resets': sum(t['physical_resets_after_initial'] for t in trials),
                        'failure_reasons': dict(reasons)})
    refs = []
    for name in sorted({t['motion'] for t in track['trials']}):
        rows = [t for t in track['trials'] if t['motion'] == name]
        refs.append({'motion': name, 'completed': sum(t['outcome'] == 'clip_complete' for t in rows),
                     'trials': len(rows), 'body_object_mae': np.mean([t['tracking_errors_body_object']['mae'] for t in rows], axis=0).tolist()})
    # Screening score, not acceptance: retain survival information when a target
    # never settles. Lower is better. Explicit Track penalty prevents forgetting.
    score = sum((min(5., r['mae'][0] / max(.03, .15 * r['target'][0])) + min(5., r['mae'][3] / .02))
                if r['mae'] else 10. for r in targets) / len(targets)
    baseline_path = Path(__file__).resolve().parents[1] / 'docs/loaded_4d_vxh/calibration_summary.json'
    baseline = {r['motion']: r['body_object_mae'] for r in json.loads(baseline_path.read_text())['track_results']}
    ratios = [np.mean(np.array(r['body_object_mae']) / np.maximum(baseline[r['motion']], 1e-6)) for r in refs if r['motion'] in baseline]
    track_error_ratio = float(np.mean(ratios)) if ratios else 1.
    score += max(0., track_error_ratio - 1.)
    score += 5 * (1 - track['completion_fraction'])
    score += 3 * (1 - gen['completion_fraction'])
    forward = [r['measured_mean'][0] for r in targets if r['target'][0] > 0 and r['measured_mean']]
    heights = [r['measured_mean'][3] for r in targets if r['target'][0] == 0 and r['measured_mean']]
    no_response = (not forward or max(forward) < .05) and (len(heights) < 2 or max(heights)-min(heights) < .03)
    severe_track_loss = track['completion_fraction'] < .5 or any('locomotion' in r['motion'] and r['completed'] == 0 for r in refs)
    return {'checkpoint': gen['checkpoint'], 'checkpoint_iteration': gen['checkpoint_iteration'], 'additional_updates': updates,
            'criteria_sha256': gen['criteria_sha256'], 'targets': targets, 'track': refs,
            'task_success_fraction': gen['task_success_fraction'], 'track_completion_fraction': track['completion_fraction'],
            'screening_score': score, 'track_error_ratio_to_stage1': track_error_ratio, 'no_response': no_response, 'severe_track_loss': severe_track_loss,
            'continuous': None, 'continuous_status': 'not_run_no_stable_target_response' if no_response else 'pending'}


def checkpoint_updates(checkpoint, seen=None):
    """Count actual updates along saved segment ancestry despite repeated labels."""
    import torch
    checkpoint = Path(checkpoint).resolve()
    seen = set() if seen is None else seen
    if checkpoint in seen:
        raise ValueError('Cycle in checkpoint ancestry')
    seen.add(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = payload['experiment_config']
    if cfg['command']['setup_terms']['motion_command']['params']['loaded_4d'].get('profile') != 'vxh':
        return 0
    ledger = json.loads((Path(cfg['logger']['base_dir']) / 'segment.json').read_text())
    added = payload['iter'] - ledger['first_display_iteration'] + 1
    if not 0 < added <= ledger['requested_updates']:
        raise ValueError('Saved iteration does not fit its segment ledger')
    return checkpoint_updates(ledger['checkpoint'], seen) + added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?')
    parser.add_argument('--count-updates', type=Path, help='Count saved updates along segment ancestry for recovery')
    parser.add_argument('--updates', type=int)
    parser.add_argument('--history', type=Path)
    args = parser.parse_args()
    if args.count_updates:
        print(checkpoint_updates(args.count_updates))
        return
    if args.folder is None or args.updates is None or args.history is None:
        parser.error('folder, --updates and --history are required for a summary')
    report = summarize(args.folder, args.updates)
    sequence = args.folder / 'sequence.json'
    if sequence.exists():
        data = json.loads(sequence.read_text())
        report['continuous_status'] = 'evaluated'
        report['continuous'] = {'task_success_fraction': data['task_success_fraction'], 'completion_fraction': data['completion_fraction'], 'trials': len(data['trials'])}
    history = [json.loads(line) for line in args.history.read_text().splitlines()] if args.history.exists() else []
    # Same folder can be summarized again after its sequence run without double-counting a window.
    history = [r for r in history if r['checkpoint'] != report['checkpoint']]
    history.append(report)
    flat = len(history) >= 3 and all(r['no_response'] for r in history[-3:])
    report['decision'] = 'stop_for_diagnosis' if report['severe_track_loss'] or flat else 'continue_within_budget'
    report['decision_reason'] = 'severe Track loss' if report['severe_track_loss'] else ('three consecutive windows with no forward/height response' if flat else 'finite evaluated rollout; short nonconvergence is not a stop condition')
    (args.folder / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    args.history.write_text(''.join(json.dumps(r) + '\n' for r in history))
    eligible = [r for r in history if r['track_completion_fraction'] >= .8 and r['track_error_ratio_to_stage1'] <= 1.5 and not r['severe_track_loss']]
    best = min(eligible or history, key=lambda r: (-r['task_success_fraction'], r['screening_score']))
    (args.history.parent / 'best_checkpoint.txt').write_text(best['checkpoint'] + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
