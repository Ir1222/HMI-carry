"""Evaluate Loaded-4D checkpoints through upstream PPO and simulation setup.

Examples (the first performs CPU data checks only):
  python scripts_run/evaluate_loaded_4d.py --validate-data
  python scripts_run/evaluate_loaded_4d.py --checkpoint /path/model.pt --mode track --trials 10
  python scripts_run/evaluate_loaded_4d.py --checkpoint /path/model.pt --mode gen --seconds 60

An episode ends on its first physical/track failure. Reset fragments are never
concatenated into a successful trial. No trainer or motion generator is added.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from holosoma.config_types.command import EvalFixedResetConfig
from holosoma.config_values.wbt.g1.loaded_4d import DATA_ROOT, MOTIONS
from holosoma.managers.command.terms.loaded_4d import heading_state


VXH_TARGETS = [[v, 0., 0., .73] for v in (.1, .4, .8)] + [[0., 0., 0., h] for h in (.60, .69, .80)]
VXH_SEQUENCE = [[0., 0., 0., .73], [.1, 0., 0., .73], [.4, 0., 0., .73],
                [0., 0., 0., .73], [0., 0., 0., .60], [0., 0., 0., .80], [.4, 0., 0., .73]]

DEFAULT_COMMANDS = [
    [0., 0., 0., .73], [.15, 0., 0., .73], [-.15, 0., 0., .73],
    [0., .10, 0., .73], [0., -.10, 0., .73], [0., 0., .20, .73],
    [0., 0., -.20, .73], [0., 0., 0., .66], [0., 0., 0., .73],
    [.10, .05, .10, .70], [-.10, -.05, -.10, .70], [0., 0., 0., .73],
]


def validate_data(directory=DATA_ROOT / "joint"):
    directory = Path(directory)
    if {p.name for p in directory.glob('*.npz')} != set(MOTIONS):
        raise ValueError(f"Data directory must contain exactly the three audited files: {directory}")
    result, joint_commands = {}, []
    for name, expected in MOTIONS.items():
        path = directory / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected['sha256']:
            raise ValueError(f"Hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            if len(data['joint_pos']) != expected['frames'] or float(data['fps'][0]) != 50:
                raise ValueError(f"Frame count or fps mismatch: {path}")
            for key in data.files:
                if np.issubdtype(data[key].dtype, np.number) and not np.isfinite(data[key]).all():
                    raise ValueError(f"Non-finite motion field: {path}: {key}")
            pelvis = list(data['body_names']).index('pelvis')
            quat = torch.from_numpy(data['body_quat_w'][:, pelvis])[:, [1, 2, 3, 0]]
            values = heading_state(torch.from_numpy(data['body_pos_w'][:, pelvis]), quat,
                torch.from_numpy(data['body_lin_vel_w'][:, pelvis]), torch.from_numpy(data['body_ang_vel_w'][:, pelvis]))
            joint_commands.append(values)
            result[name] = {'sha256': digest, 'frames': expected['frames'], 'fps': 50,
                'duration_s': (expected['frames'] - 1) / 50,
                'command_min': values.amin(0).tolist(), 'command_max': values.amax(0).tolist()}
    values = torch.cat(joint_commands)
    result['joint_distribution_counts'] = {
        'frames': len(values),
        'low_forward': int(((values[:, 3] < .70) & (values[:, 0] > .10)).sum()),
        'backward': int((values[:, 0] < -.10).sum()),
        'lateral_without_forward': int(((values[:, 1].abs() > .10) & (values[:, 0].abs() < .05)).sum()),
        'turn_without_forward': int(((values[:, 2].abs() > .30) & (values[:, 0].abs() < .05)).sum()),
    }
    return result


def json_safe(value):
    """Represent failed non-finite physics samples as null, preserving the trial."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_samples(samples):
    if not samples:
        return {'samples': 0, 'mae': None, 'p95': None, 'max': None, 'nonfinite_samples': 0}
    values = torch.stack(samples).float()
    finite = torch.isfinite(values).all(dim=-1)
    good = values[finite]
    return {'samples': len(values), 'nonfinite_samples': int((~finite).sum()),
            'mae': good.mean(0).tolist() if len(good) else None,
            'p95': torch.quantile(good, .95, dim=0).tolist() if len(good) else None,
            'max': good.amax(0).tolist() if len(good) else None}


def read_sequence(path):
    sequence = DEFAULT_COMMANDS if path is None else json.loads(Path(path).read_text())
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Sequence must be a nonempty JSON array of finite [vx, vy, yaw_rate, pelvis_height] rows")
    return values.tolist()


def summarize_segment(rows, target, criteria, exclusion_s):
    result = {'target': target, 'executed': bool(rows), 'task_success': False}
    if not rows:
        return {**result, 'failure_reasons': ['not_executed']}
    reached = next((r['time_s'] for r in rows if max(abs(a-b) for a,b in zip(r['applied'], target)) < 1e-3), None)
    steady = [r for r in rows if reached is not None and r['time_s'] >= reached + exclusion_s
              and max(abs(a-b) for a,b in zip(r['applied'], target)) < 1e-3]
    result.update(target_reached_s=reached, steady_samples=len(steady),
                  pelvis_start=rows[0]['measured'][3], pelvis_end=rows[-1]['measured'][3],
                  pelvis_change=rows[-1]['measured'][3]-rows[0]['measured'][3])
    if 'root_position' in rows[0]:
        delta = np.asarray(rows[-1]['root_position'])-rows[0]['root_position']
        result['world_net_displacement'] = delta.tolist()
        result['forward_net_displacement'] = float(np.dot(delta[:2], rows[0]['root_heading']))
    if rows[0].get('reward_rates'):
        result['reward_rate_mean'] = {k: float(np.mean([r['reward_rates'][k] for r in rows])) for k in rows[0]['reward_rates']}
    if not steady:
        return {**result, 'failure_reasons': ['no_steady_samples']}
    values = np.array([r['measured'] for r in steady])
    if not np.isfinite(values).all():
        return {**result, 'failure_reasons': ['nonfinite_measurement']}
    error = np.abs(values-np.array([r['applied'] for r in steady]))
    result.update(measured_mean=values.mean(0).tolist(), mae=error.mean(0).tolist(),
                  p95=np.quantile(error,.95,axis=0).tolist(), steady_seconds=len(steady)*rows[0]['dt'])
    if steady[0].get('reward_rates'):
        result['steady_reward_rate_mean'] = {k: float(np.mean([r['reward_rates'][k] for r in steady])) for k in steady[0]['reward_rates']}
    reasons=[]
    if criteria is None:
        return {**result, 'task_success': None, 'failure_reasons': ['criteria_not_supplied']}
    c=criteria
    if result['steady_seconds'] < c['min_steady_s']: reasons.append('insufficient_steady_time')
    if result['mae'][0] > max(.03, .15*abs(target[0])): reasons.append('vx_mae')
    if result['mae'][3] > .02: reasons.append('height_mae')
    if np.abs(values[:,1]).mean() > c['vy_mae_max']: reasons.append('lateral_velocity')
    if np.abs(values[:,2]).mean() > c['yaw_mae_max']: reasons.append('yaw_rate')
    if target[0] > .03 and (result.get('forward_net_displacement',0) <= .02 or values[:,0].mean() < target[0]*.5):
        reasons.append('no_forward_motion')
    requested_change = target[3]-result['pelvis_start']
    if abs(requested_change) > .03 and np.sign(requested_change)*result['pelvis_change'] < abs(requested_change)-.02:
        reasons.append('height_response')
    if 'foot_force_z' not in steady[0]:
        reasons.append('missing_support_carry_metrics')
    else:
        forces=np.array([r['foot_force_z'] for r in steady])
        heights=np.array([r['foot_height'] for r in steady])
        slips=np.array([r['foot_slip'] for r in steady])
        supported=(forces>=c['foot_force_min']) & (heights<=c['foot_height_max']) & (slips<=c['foot_slip_max'])
        local=np.array([r['hand_box_local'] for r in steady])
        geometry=((local>=np.array(c['hand_local_min'])-c['hand_margin']) &
                  (local<=np.array(c['hand_local_max'])+c['hand_margin'])).all(axis=(1,2))
        tilt=np.array([r['box_tilt'] for r in steady])
        hand_speed=np.array([r['hand_box_speed'] for r in steady])
        result.update(double_support_fraction=float(supported.all(1).mean()),
                      foot_contact_fraction=(forces>=c['foot_force_min']).mean(0).tolist(),
                      foot_height_p95=np.quantile(heights,.95,axis=0).tolist(),
                      foot_slip_p95=np.quantile(slips,.95,axis=0).tolist(),
                      hand_geometry_fraction=float(geometry.mean()), hand_box_speed_p95=np.quantile(hand_speed,.95,axis=0).tolist(),
                      box_tilt_p95=float(np.quantile(tilt,.95)), min_box_clearance=min(r['box_clearance'] for r in steady))
        if abs(target[0]) < .03 and result['double_support_fraction'] < c['support_fraction_min']: reasons.append('double_support')
        if geometry.mean() < c['carry_fraction_min']: reasons.append('hand_box_geometry')
        if (hand_speed.max(1)<=c['hand_speed_max']).mean() < c['carry_fraction_min']: reasons.append('hand_box_slip')
        if (tilt<=c['box_tilt_max']).mean() < c['carry_fraction_min']: reasons.append('box_tilt')
        if result['min_box_clearance'] <= .02: reasons.append('box_clearance')
    return {**result, 'task_success': not reasons, 'failure_reasons': reasons}


def run_trial(env, algo, *, mode, steps, sequence, stochastic, seed, transition_exclusion_s, criteria=None):
    """One continuous trial. Uses pre-reset snapshots supplied by the thin task."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    obs = env.reset_all()
    cmd = env.command_manager.get_state('motion_command')
    initial_reset_count = cmd.physical_reset_count.clone()
    initial_measured = cmd.measured_state()[0].detach().cpu().tolist() if hasattr(cmd, 'measured_state') else None
    policy = algo.get_inference_policy()
    samples, settled_samples, tracking_samples, clearances, trace = [], [], [], [], []
    segment_rows = [[] for _ in sequence]
    outcome = 'horizon_complete' if mode == 'gen' else 'incomplete'
    last_segment, segment_start, hidden_reset = -1, 0, False
    termination_names = []
    # Managers retain tensors created during step(), then mutate them in the
    # next trial's reset_all(). Inference tensors forbid that outside their
    # context; use the same no-grad policy as upstream PPO.evaluate_policy.
    with torch.no_grad():
        for step in range(steps):
            segment = min(len(sequence) - 1, step * len(sequence) // steps)
            if mode == 'gen' and segment != last_segment:
                cmd.set_command(sequence[segment])
                last_segment, segment_start = segment, step
            actor_obs = torch.cat([obs[key] for key in algo.actor_obs_keys], dim=1)
            if stochastic:
                normalized = algo._normalize_actor_obs(actor_obs, update=False)
                actions = algo.actor.act({'actor_obs': normalized})
            else:
                actions = policy({'actor_obs': actor_obs})
            obs, _, dones, extras = env.step({'actions': actions})
            snapshot = extras['loaded_4d']
            error = (snapshot['measured'][0] - snapshot['applied_command'][0]).abs().cpu()
            samples.append(error)
            tracking_samples.append(torch.stack((snapshot['body_position_error'][0], snapshot['object_position_error'][0])).cpu())
            clearances.append(float(snapshot['box_clearance'][0]))
            settled = (snapshot['applied_command'][0] - snapshot['target_command'][0]).abs().max() < 1e-3
            if mode == 'gen' and settled and (step - segment_start) * env.dt >= transition_exclusion_s:
                settled_samples.append(error)
            row = {'time_s': (step+1)*env.dt, 'dt': env.dt,
                   'target': snapshot['target_command'][0].cpu().tolist(),
                   'applied': snapshot['applied_command'][0].cpu().tolist(),
                   'measured': snapshot['measured'][0].cpu().tolist(),
                   'reference_frame': int(snapshot['reference_frame'][0]), 'box_clearance': clearances[-1]}
            for key in ('foot_force_z','foot_height','foot_min_height','foot_slip','hand_box_local',
                        'hand_box_speed','box_tilt','root_position','root_heading','box_quaternion',
                        'upright','heading_projection','box_separation'):
                if key in snapshot: row[key] = snapshot[key][0].cpu().tolist()
            row['reward_rates'] = {k.split('/')[-1]: float(v) for k, v in getattr(env, 'log_dict', {}).items()
                                   if k.startswith('loaded_4d/gen_reward_rate/')}
            segment_rows[segment].append(row)
            # Store diagnostics at 10 Hz; success uses every control step.
            if step % max(1, round(.1 / env.dt)) == 0 or bool(dones[0]):
                trace.append(row)
            reset_delta = cmd.physical_reset_count - initial_reset_count
            if bool((reset_delta > 0).any()) and not bool(dones[0]):
                hidden_reset, outcome = True, 'hidden_reset'
                break
            if bool(dones[0]):
                termination_names = [name for name, mask in env.termination_manager.last_term_results.items() if bool(mask[0])]
                if mode == 'track' and bool(snapshot['clip_finished'][0]) and bool(extras['time_outs'][0]):
                    outcome = 'clip_complete'
                else:
                    outcome = 'terminated'
                break
    segments = [summarize_segment(rows, target, criteria, transition_exclusion_s)
                for rows, target in zip(segment_rows, sequence)] if mode == 'gen' else []
    success = outcome == 'horizon_complete' and all(r['task_success'] is True for r in segments) if mode == 'gen' else outcome == 'clip_complete'
    return {'initial_measured': initial_measured, 'first_failure_s': (step+1)*env.dt if outcome in ('terminated','hidden_reset') else None,
            'task_success': success, 'segments': segments, 'seed': seed, 'outcome': outcome, 'steps': step + 1, 'seconds': (step + 1) * env.dt,
            'hidden_reset': hidden_reset, 'physical_resets_after_initial': int((cmd.physical_reset_count - initial_reset_count)[0]),
            'termination_terms': termination_names, 'command_errors_all': summarize_samples(samples),
            'command_errors_settled': summarize_samples(settled_samples),
            'tracking_errors_body_object': summarize_samples(tracking_samples),
            'min_box_clearance': min(clearances), 'trace': trace}


def evaluate(args):
    # Deferred imports keep --help and --validate-data independent of simulation startup.
    from holosoma.config_values.logger import disabled
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    required = ('actor_model_state_dict', 'critic_model_state_dict', 'actor_obs_normalizer_state_dict',
                'critic_obs_normalizer_state_dict', 'env_state')
    if any(payload.get(key) is None for key in required) or 'loaded_4d_contract' not in payload['env_state']:
        raise ValueError('Evaluation requires a complete Loaded-4D checkpoint with both normalizers and task contract')
    cfg, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint)))
    if cfg.env_class != 'holosoma.envs.wbt.loaded_4d.Loaded4DTrackingManager':
        raise ValueError('Checkpoint is not a Loaded-4D policy')
    if not cfg.algo.config.empirical_normalization:
        raise ValueError('Loaded-4D evaluation requires the training observation normalizers')
    term = cfg.command.setup_terms['motion_command']
    # Saved motion params are plain dictionaries; preserve all other fields.
    motion = dict(term.params['motion_config'])
    motion['hybrid'] = {**motion['hybrid'], 'track_ratio': 1.0 if args.mode == 'track' else 0.0}
    motion['eval_fixed_reset'] = as_fixed_reset(0, 0)
    options = {**term.params['loaded_4d'], 'diagnostics': True}
    if args.vxh_protocol:
        from holosoma.config_values.wbt.g1.loaded_4d import g1_loaded_4d_stage2_vxh
        options = dict(g1_loaded_4d_stage2_vxh.command.setup_terms['motion_command'].params['loaded_4d'])
    term = replace(term, params={**term.params, 'motion_config': motion, 'loaded_4d': options})
    cfg = replace(cfg,
        command=replace(cfg.command, setup_terms={'motion_command': term}),
        reward=g1_loaded_4d_stage2_vxh.reward if args.vxh_protocol else cfg.reward,
        training=replace(cfg.training, num_envs=1, headless=not args.viewer, checkpoint=str(checkpoint), seed=args.seed),
        simulator=replace(cfg.simulator, config=replace(cfg.simulator.config,
            sim=replace(cfg.simulator.config.sim, max_episode_length_s=max(16., args.seconds + 1.)))),
        logger=replace(disabled, base_dir=str(args.output.parent)),
    )
    env, device, app = setup_simulation_environment(cfg)
    algo = None
    try:
        env.set_is_evaluating()
        algo = get_class(cfg.algo._target_)(env=env, config=cfg.algo.config, device=device,
                                           log_dir=str(args.output.parent), multi_gpu_cfg=None)
        algo.setup()
        # Compare the contract before loading any network/optimizer state.
        env.load_checkpoint_state(payload['env_state'])
        del payload
        algo.load(str(checkpoint))
        algo._eval_mode()
        cmd = env.command_manager.get_state('motion_command')
        sequence = VXH_SEQUENCE if args.suite == 'vxh-sequence' else read_sequence(args.sequence)
        sequences = [[goal] for goal in VXH_TARGETS] if args.suite == 'vxh-targets' else [sequence]
        criteria = json.loads(args.criteria.read_text()) if args.criteria else None
        if args.mode == 'gen':
            cmd.set_command(sequence[0])  # validate envelope before any steps
            for seq in sequences:
                for goal in seq: cmd.set_command(goal)
        trials = []
        selected = range(cmd.motion.num_motions) if args.motion_id is None else [args.motion_id]
        if args.mode == 'gen':
            selected = [0 if args.motion_id is None else args.motion_id]
        for motion_id in selected:
            if not 0 <= motion_id < cmd.motion.num_motions:
                raise ValueError('motion-id is outside the loaded dataset')
            length = int(cmd.motion.motion_end_idx[motion_id] - cmd.motion.motion_start_idx[motion_id])
            for frame in args.start_frames:
                if not 0 <= frame < length - 1:
                    raise ValueError(f'Invalid start frame {frame} for {cmd.motion_names[motion_id]}')
                cmd.motion_cfg = replace(cmd.motion_cfg, eval_fixed_reset=EvalFixedResetConfig(
                    enabled=True, motion_id=motion_id, local_frame=frame))
                steps = length - 1 - frame if args.mode == 'track' else math.ceil(args.seconds / env.dt)
                for sequence in sequences:
                    for trial in range(args.trials):
                        print(f'[{args.mode}] motion={cmd.motion_names[motion_id]} start_frame={frame} '
                              f'trial={trial + 1}/{args.trials} target={sequence if args.mode == "gen" else "reference"}', flush=True)
                        result = run_trial(env, algo, mode=args.mode, steps=steps, sequence=sequence,
                            stochastic=args.stochastic, seed=args.seed + trial,
                            transition_exclusion_s=args.transition_exclusion_s, criteria=criteria)
                        result.update(motion=cmd.motion_names[motion_id], start_frame=frame)
                        trials.append(result)
                        print(f"  outcome={result['outcome']} seconds={result['seconds']:.2f} "
                              f"task_success={result['task_success']} termination_terms={result['termination_terms']}", flush=True)
        complete = sum(row['outcome'] in ('clip_complete', 'horizon_complete') for row in trials)
        summary = {'checkpoint': str(checkpoint), 'checkpoint_iteration': algo.current_learning_iteration,
            'mode': args.mode, 'stochastic': args.stochastic, 'observation_noise': 'preserved training schema',
            'control_dt': env.dt, 'sequences': sequences if args.mode == 'gen' else None,
            'command_options': cmd.options, 'suite': args.suite,
            'criteria_sha256': hashlib.sha256(args.criteria.read_bytes()).hexdigest() if args.criteria else None,
            'criteria': criteria, 'task_success_fraction': sum(r['task_success'] for r in trials)/len(trials),
            'completion_fraction': complete / len(trials), 'trials': trials,
            'interpretation': 'Completion is separate from command accuracy; no hand-contact or convergence claim.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + '\n')
        print(f'Saved {len(trials)} trials to {args.output}; completion={complete}/{len(trials)}', flush=True)
    except Exception:
        # Kit shutdown may terminate Python before an uncaught exception is
        # printed. Emit and flush it before entering simulator cleanup.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if algo is not None:
            algo.writer.close()
        close_simulation_app(app)


def as_fixed_reset(motion_id, frame):
    return {'enabled': True, 'motion_id': motion_id, 'bin_id': None, 'local_frame': frame, 'randomize_within_bin': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--validate-data', action='store_true', help='CPU-only immutable data checks; no simulator')
    parser.add_argument('--data-dir', type=Path, default=DATA_ROOT / 'joint')
    parser.add_argument('--checkpoint')
    parser.add_argument('--mode', choices=['track', 'gen'], default='track')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--motion-id', type=int)
    parser.add_argument('--start-frames', type=int, nargs='+', default=[0])
    parser.add_argument('--sequence', type=Path, help='JSON array of 4D commands, equally spaced across --seconds')
    parser.add_argument('--suite', choices=['vxh-targets', 'vxh-sequence'])
    parser.add_argument('--vxh-protocol', action='store_true', help='Use new Gen command protocol with Stage1 for baseline diagnostics')
    parser.add_argument('--criteria', type=Path, help='Frozen pre-training acceptance thresholds JSON')
    parser.add_argument('--stochastic' , action='store_true')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--transition-exclusion-s', type=float, default=.8)
    parser.add_argument('--output', type=Path, default=Path('logs_eval/loaded_4d/results.json'))
    args = parser.parse_args()
    if args.validate_data:
        print(json.dumps(validate_data(args.data_dir), indent=2))
        return
    if not args.checkpoint:
        parser.error('--checkpoint is required unless --validate-data is selected')
    if args.trials <= 0 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--trials and --seconds must be positive')
    if not math.isfinite(args.transition_exclusion_s) or args.transition_exclusion_s < 0:
        parser.error('--transition-exclusion-s must be finite and nonnegative')
    evaluate(args)


if __name__ == '__main__':
    main()
