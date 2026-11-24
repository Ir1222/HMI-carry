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


def run_trial(env, algo, *, mode, steps, sequence, stochastic, seed, transition_exclusion_s):
    """One continuous trial. Uses pre-reset snapshots supplied by the thin task."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    obs = env.reset_all()
    cmd = env.command_manager.get_state('motion_command')
    initial_reset_count = cmd.physical_reset_count.clone()
    policy = algo.get_inference_policy()
    samples, settled_samples, tracking_samples, clearances, trace = [], [], [], [], []
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
            # Store command response at 10 Hz plus every terminal transition.
            if step % max(1, round(.1 / env.dt)) == 0 or bool(dones[0]):
                trace.append({'time_s': (step + 1) * env.dt,
                    'target': snapshot['target_command'][0].cpu().tolist(),
                    'applied': snapshot['applied_command'][0].cpu().tolist(),
                    'measured': snapshot['measured'][0].cpu().tolist(),
                    'reference_frame': int(snapshot['reference_frame'][0]),
                    'box_clearance': clearances[-1]})
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
    return {'seed': seed, 'outcome': outcome, 'steps': step + 1, 'seconds': (step + 1) * env.dt,
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
    term = replace(term, params={**term.params, 'motion_config': motion})
    cfg = replace(cfg,
        command=replace(cfg.command, setup_terms={'motion_command': term}),
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
        sequence = read_sequence(args.sequence)
        if args.mode == 'gen':
            cmd.set_command(sequence[0])  # validate envelope before any steps
            for goal in sequence:
                cmd.set_command(goal)
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
                for trial in range(args.trials):
                    print(f'[{args.mode}] motion={cmd.motion_names[motion_id]} start_frame={frame} '
                          f'trial={trial + 1}/{args.trials}', flush=True)
                    result = run_trial(env, algo, mode=args.mode, steps=steps, sequence=sequence,
                        stochastic=args.stochastic, seed=args.seed + trial,
                        transition_exclusion_s=args.transition_exclusion_s)
                    result.update(motion=cmd.motion_names[motion_id], start_frame=frame)
                    trials.append(result)
                    print(f"  outcome={result['outcome']} seconds={result['seconds']:.2f} "
                          f"termination_terms={result['termination_terms']}", flush=True)
        complete = sum(row['outcome'] in ('clip_complete', 'horizon_complete') for row in trials)
        summary = {'checkpoint': str(checkpoint), 'checkpoint_iteration': algo.current_learning_iteration,
            'mode': args.mode, 'stochastic': args.stochastic, 'observation_noise': 'preserved training schema',
            'control_dt': env.dt, 'sequence': sequence if args.mode == 'gen' else None,
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
    parser.add_argument('--stochastic', action='store_true')
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
