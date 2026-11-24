"""CPU protocol tests: these do not claim PhysX stability or learned task success."""
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import json

import numpy as np
import pytest
import torch

from holosoma.config_types.command import EvalFixedResetConfig
from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_hybrid_object
from holosoma.config_values.wbt.g1.loaded_4d import DATA_ROOT, MOTIONS, loaded_4d_preset
from holosoma.envs.wbt.hybrid_size_randomization import collision_bounds_corners, load_urdf_collision_bounds
from holosoma.envs.wbt.loaded_4d import (
    Loaded4DTrackingManager, command_tracking_reward, physical_failure, physical_metrics,
)
from holosoma.managers.command.terms.loaded_4d import Loaded4DCommand, heading_state, validate_exact_joint_positions
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.manager import ObservationManager
from holosoma.managers.observation.terms import wbt as observations
from holosoma.utils.rotations import quat_from_euler_xyz, quat_mul
from holosoma.utils.simulator_config import SimulatorType


@pytest.fixture
def make_command(tmp_path):
    """Use the real NPZ loader/reset path with a tiny, independently built clip."""
    def build(stage=2):
        config = loaded_4d_preset(stage)
        robot = config.robot
        n, frames = 4, 5
        body_count = len(robot.body_names)
        joints = np.zeros((frames, 36), dtype=np.float32)
        joints[:, 2], joints[:, 3] = 0.72, 1.0
        joints[:, 7:] = [robot.init_state.default_joint_angles[name] for name in robot.dof_names]
        body_pos = np.zeros((frames, body_count, 3), dtype=np.float32)
        body_pos[:, :, 2] = 0.72
        body_pos[:, :, 0] = np.arange(frames)[:, None] * 0.004
        body_quat = np.zeros((frames, body_count, 4), dtype=np.float32)
        body_quat[:, :, 0] = 1.0
        velocity = np.zeros_like(body_pos)
        velocity[:, :, 0] = 0.2
        box_pos = np.tile(np.array([0.35, 0.0, 0.85], dtype=np.float32), (frames, 1))
        box_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, 1))
        path = tmp_path / 'synthetic.npz'
        np.savez(path, fps=np.array([50.0]), joint_names=np.array(robot.dof_names), body_names=np.array(robot.body_names),
                 joint_pos=joints, joint_vel=np.zeros((frames, 35), dtype=np.float32),
                 body_pos_w=body_pos, body_quat_w=body_quat, body_lin_vel_w=velocity, body_ang_vel_w=np.zeros_like(body_pos),
                 object_pos_w=box_pos, object_quat_w=box_quat,
                 object_lin_vel_w=np.zeros_like(box_pos), object_ang_vel_w=np.zeros_like(box_pos))
        term = config.command.setup_terms['motion_command']
        motion = replace(term.params['motion_config'], motion_file=str(path), motion_dir='',
                         use_adaptive_timesteps_sampler=False, start_at_timestep_zero_prob=0.0,
                         eval_fixed_reset=EvalFixedResetConfig(enabled=True, motion_id=0, local_frame=0))
        term = replace(term, params={**term.params, 'motion_config': motion,
            'expected_motions': {path.name: {'frames': frames, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}})
        all_states = torch.zeros(n * 2, 13)
        all_states[:, 6] = 1.0
        limits = torch.tensor(list(zip(robot.dof_pos_lower_limit_list, robot.dof_pos_upper_limit_list)))
        sim = SimpleNamespace(
            _body_list=list(robot.body_names), dof_names=list(robot.dof_names),
            scene=SimpleNamespace(env_origins=torch.zeros(n, 3)),
            get_actor_indices=lambda name, env_ids=None: torch.arange(n, 2*n),
            get_simulator_type=lambda: SimulatorType.ISAACSIM,
            robot_root_states=all_states[:n], all_root_states=all_states,
            dof_pos=torch.zeros(n, 29), dof_vel=torch.zeros(n, 29),
            dof_pos_limits=limits * 0.95, hard_dof_pos_limits=limits,
            _rigid_body_pos=torch.zeros(n, body_count, 3),
            _rigid_body_rot=torch.zeros(n, body_count, 4),
            _rigid_body_vel=torch.zeros(n, body_count, 3),
            _rigid_body_ang_vel=torch.zeros(n, body_count, 3),
            writes=0,
        )
        def write_box(names, ids, states):
            sim.all_root_states[n + ids] = states
            sim.writes += 1
        sim.set_actor_states = write_box
        env = SimpleNamespace(num_envs=n, device='cpu', dt=0.02, simulator=sim, is_evaluating=True, viewer=None,
                              episode_length_buf=torch.zeros(n, dtype=torch.long), robot_config=robot)
        cmd = Loaded4DCommand(term, env)
        env.command_manager = SimpleNamespace(get_state=lambda name: cmd)
        cmd.setup()
        cmd.reset(torch.arange(n))
        # Only the test double copies reference body transforms for initial FK.
        sim._rigid_body_pos[:] = cmd.motion.body_pos_w[cmd.time_steps]
        sim._rigid_body_rot[:] = cmd.motion.body_quat_w[cmd.time_steps]
        sim._rigid_body_vel[:] = cmd.motion.body_lin_vel_w[cmd.time_steps]
        sim._rigid_body_ang_vel[:] = cmd.motion.body_ang_vel_w[cmd.time_steps]
        sim.base_quat = sim.robot_root_states[:, 3:7]
        env.base_quat = sim.base_quat
        env.default_dof_pos = torch.tensor([robot.init_state.default_joint_angles[name] for name in robot.dof_names]).expand(n, -1)
        env.action_manager = SimpleNamespace(action=torch.zeros(n, 29), prev_action=torch.zeros(n, 29))
        bounds = torch.tensor(load_urdf_collision_bounds(robot.object.object_urdf_path))
        env._loaded_collision_corners = collision_bounds_corners(bounds)
        cmd.step()
        return cmd, env, config
    return build


def test_heading_rate_matches_rotating_forward_axis_and_not_world_z():
    q = quat_from_euler_xyz(torch.tensor([0.5]), torch.tensor([0.6]), torch.tensor([1.1]))
    omega = torch.tensor([[0.7, -0.4, 0.3]])
    position = torch.tensor([[1.0, 2.0, 1.7]])
    velocity = torch.tensor([[0.4, -0.2, 0.8]])
    state = heading_state(position, q, velocity, omega, 1.0)
    dt = 0.001
    angle = omega.norm(dim=-1) * dt
    delta = torch.cat((omega / omega.norm(dim=-1, keepdim=True) * torch.sin(angle[:, None] / 2), torch.cos(angle[:, None] / 2)), dim=-1)
    q_next = quat_mul(delta, q, w_last=True)
    from holosoma.utils.rotations import quat_apply
    x = torch.tensor([[1.0, 0.0, 0.0]])
    before, after = quat_apply(q, x, w_last=True), quat_apply(q_next, x, w_last=True)
    difference = torch.atan2(after[:, 1], after[:, 0]) - torch.atan2(before[:, 1], before[:, 0])
    assert torch.allclose(state[:, 2], difference / dt, atol=5e-4)
    assert not torch.allclose(state[:, 2], omega[:, 2], atol=0.01)
    assert state[0, 3] == pytest.approx(0.7)


def test_heading_translation_rotation_invariance_and_wrap():
    yaw = torch.tensor([3.1415, -3.1415])
    q = quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), yaw)
    velocity = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros(2)), dim=1)
    result = heading_state(torch.tensor([[0., 0., .7], [10., -3., 2.7]]), q, velocity,
                           torch.tensor([[0., 0., .2], [0., 0., .2]]), torch.tensor([0., 2.]))
    assert torch.allclose(result, torch.tensor([[1., 0., .2, .7]]).expand(2, -1), atol=1e-6)


def test_exact_reset_keeps_legal_soft_limit_exceedance():
    positions = torch.tensor([[0.98]])
    limits = torch.tensor([[-1., 1.]])
    assert validate_exact_joint_positions(positions, limits) is positions
    with pytest.raises(ValueError, match='hard joint'):
        validate_exact_joint_positions(torch.tensor([[1.1]]), limits)
    with pytest.raises(ValueError, match='finite'):
        validate_exact_joint_positions(torch.tensor([[float('nan')]]), limits)
    upstream = object.__new__(MotionCommand)
    upstream._env = SimpleNamespace(simulator=SimpleNamespace(dof_pos_limits=limits * .95))
    assert upstream._prepare_reset_joint_positions(positions).item() == pytest.approx(.95)


def test_real_reset_path_sets_robot_and_free_box_pose_velocity(make_command):
    cmd, env, _ = make_command()
    assert torch.equal(env.simulator.dof_pos, cmd.joint_pos)
    assert torch.equal(env.simulator.robot_root_states[:, :3], cmd.root_pos_w)
    assert torch.equal(env.simulator.robot_root_states[:, 7:10], cmd.root_lin_vel_w)
    assert torch.equal(cmd.simulator_object_pos_w, cmd.object_pos_w)
    assert torch.equal(cmd.simulator_object_quat_w, cmd.object_quat_w)
    assert torch.equal(cmd.simulator_object_lin_vel_w, cmd.object_lin_vel_w)
    assert torch.equal(cmd.time_steps, torch.zeros(4, dtype=torch.long))


def test_reference_clock_is_once_per_token_and_gen_never_clip_resets(make_command):
    cmd, env, _ = make_command()
    writes, state = env.simulator.writes, env.simulator.all_root_states.clone()
    # Episode lengths are deliberately unrelated to the monotonic control clock.
    env.episode_length_buf[:] = torch.tensor([91, 22, 13, 4])
    for token in range(1, 50):
        cmd.advance_reference_to_post_action(token)
        before = cmd.time_steps.clone()
        cmd.advance_reference_to_post_action(token)
        cmd.step()
        cmd.step()
        assert torch.equal(before, cmd.time_steps)
    assert (cmd.time_steps[cmd.track_env_mask] == 4).all()
    assert (cmd.time_steps[cmd.gen_env_mask] == 0).all()
    assert torch.equal(cmd.clip_finished, cmd.track_env_mask)
    assert env.simulator.writes == writes
    assert torch.equal(env.simulator.all_root_states, state)


def test_external_stream_changes_only_gen_targets_and_slews_without_stalling(make_command):
    cmd, env, _ = make_command()
    ids = torch.where(cmd.gen_env_mask)[0]
    physical = env.simulator.all_root_states.clone()
    ref = cmd.time_steps.clone()
    history = env.action_manager.action.clone()
    episode = env.episode_length_buf.clone()
    old_obs_command = cmd.task_command.clone()
    cmd.set_command([0.0, 0.1, 0.2, 0.66], ids)
    assert torch.equal(cmd.task_command, old_obs_command)
    assert torch.equal(cmd.time_steps, ref)
    cmd.begin_control_step()
    cmd.advance_reference_to_post_action(1)
    assert torch.equal(cmd.applied_command, old_obs_command)
    for token in range(2, 151):
        cmd.set_command([0.0, 0.1, 0.2, 0.66], ids)
        cmd.begin_control_step()
        last_h = cmd.task_command[ids, 3].clone()
        cmd.advance_reference_to_post_action(token)
        assert ((cmd.task_command[ids, 3] - last_h).abs() <= .06 * .02 + 1e-6).all()
    assert torch.allclose(cmd.task_command[ids], torch.tensor([0., .1, .2, .66]).expand(len(ids), -1), atol=1e-5)
    assert torch.equal(env.simulator.all_root_states, physical)
    assert torch.equal(env.action_manager.action, history)
    assert torch.equal(env.episode_length_buf, episode)
    assert (cmd.time_steps[ids] == ref[ids]).all()
    with pytest.raises(ValueError, match='Gen'):
        cmd.set_command([0., 0., 0., .7], torch.where(cmd.track_env_mask)[0])
    with pytest.raises(ValueError, match='finite'):
        cmd.set_command([float('nan'), 0., 0., .7])
    with pytest.raises(ValueError, match='envelope'):
        cmd.set_command([10., 0., 0., .7])
    cmd.release_external_command(ids)
    assert not cmd.external_command_mask.any()


def test_command_reward_uses_applied_goal_and_dropped_box_cannot_score(make_command):
    cmd, env, _ = make_command()
    cmd.applied_command[:] = cmd.measured_state()
    cmd.task_command[:] = 999.0
    reward = command_tracking_reward(env, 'velocity', .2)
    assert torch.equal(reward, cmd.gen_env_mask.float())
    env.simulator.all_root_states[cmd.object_indices_in_simulator, 2] = 0.0
    assert physical_failure(env).all()
    assert not command_tracking_reward(env, 'velocity', .2).any()


def test_clearance_is_rotation_aware_and_failure_is_reference_independent(make_command):
    cmd, env, _ = make_command()
    before = physical_metrics(env)['box_clearance']
    q = quat_from_euler_xyz(torch.full((4,), .7), torch.zeros(4), torch.zeros(4))
    env.simulator.all_root_states[cmd.object_indices_in_simulator, 3:7] = q
    after = physical_metrics(env)['box_clearance']
    assert not torch.allclose(before, after)
    failures = physical_failure(env)
    cmd.motion._body_pos_w[:, :, 2] += 10.0
    cmd.motion._object_pos_w[:, 2] += 10.0
    assert torch.equal(failures, physical_failure(env))


def test_observation_dimensions_order_and_gen_privileged_mask(make_command):
    cmd, env, config = make_command()
    manager = ObservationManager(config.observation, env, 'cpu')
    obs = manager.compute()
    assert obs['actor_obs'].shape == (4, 113)
    assert obs['critic_obs'].shape == (4, 315)
    expected = []
    for key in sorted(config.observation.groups['critic_obs'].terms):
        term = config.observation.groups['critic_obs'].terms[key]
        expected.append(manager._term_funcs['critic_obs'][key](env, **term.params) * term.scale)
    assert torch.equal(obs['critic_obs'], torch.cat(expected, dim=1))
    for fn in (observations.hybrid_motion_command, observations.hybrid_motion_ref_pos_b,
               observations.hybrid_motion_ref_ori_b, observations.hybrid_obj_ref_pos_b, observations.hybrid_obj_ref_ori_b):
        assert not fn(env)[cmd.gen_env_mask].any()
    assert 'obj_goal_pos_b' not in config.observation.groups['actor_obs'].terms
    assert torch.equal(cmd.command, torch.cat((cmd.joint_pos, cmd.joint_vel), dim=1))


def test_presets_preserve_upstream_and_share_stage_interface():
    original = asdict(g1_29dof_wbt_hybrid_object)
    one, two = loaded_4d_preset(1), loaded_4d_preset(2)
    assert asdict(one.observation) == asdict(two.observation)
    assert asdict(one.action) == asdict(two.action)
    assert asdict(one.robot.control) == asdict(two.robot.control)
    assert asdict(one.algo) == asdict(two.algo)
    assert one.algo.config.empirical_normalization
    assert not two.algo.config.reset_policy_std_on_resume
    assert (one.simulator.config.sim.fps, one.simulator.config.sim.control_decimation) == (200, 4)
    assert original == asdict(g1_29dof_wbt_hybrid_object)
    assert 'goal_object_reached_once' in original['reward']['terms']
    assert 'goal_object_reached_once' not in one.reward.terms
    assert one.reward.terms['limits_dof_pos'].params['soft_dof_pos_limit'] == 1.0


def test_stage_transfer_contract_checks_schema_without_restoring_partition(make_command):
    cmd1, env1, cfg1 = make_command(1)
    cmd2, env2, cfg2 = make_command(2)
    def manager(env, cfg):
        task = object.__new__(Loaded4DTrackingManager)
        task._loaded_config = get_tyro_env_config(cfg)
        task.command_manager = env.command_manager
        task.dt = .02
        return task
    one, two = manager(env1, cfg1), manager(env2, cfg2)
    state = one.get_checkpoint_state()
    before = cmd2.track_env_mask.clone()
    two.load_checkpoint_state(state)
    assert cmd1.track_env_mask.all()
    assert torch.equal(cmd2.track_env_mask, before) and before.sum() == 2
    changed = json.loads(state['loaded_4d_contract'])
    changed['timing'] = 'old_clock'
    with pytest.raises(ValueError, match='contract mismatch'):
        two.load_checkpoint_state({'loaded_4d_contract': json.dumps(changed)})


def test_failure_wins_over_clip_timeout_and_advance_precedes_termination(make_command):
    cmd, env, _ = make_command()
    task = object.__new__(Loaded4DTrackingManager)
    task.command_manager = env.command_manager
    task._loaded_control_token = 1
    task.is_evaluating = False
    task.reset_buf = torch.zeros(4, dtype=torch.long)
    task.time_out_buf = torch.zeros(4, dtype=torch.bool)
    def check():
        assert (cmd.time_steps[cmd.track_env_mask] == 1).all()
        return torch.tensor([True, False, False, False]), torch.tensor([True, True, False, False])
    task.termination_manager = SimpleNamespace(check=check, time_outs=torch.zeros(4, dtype=torch.bool))
    task._record_termination_metrics = Mock()
    task._check_termination()
    assert task.time_out_buf.tolist() == [False, True, False, False]
    assert task.reset_buf.tolist() == [1, 1, 0, 0]


def test_reset_all_has_no_hidden_physics_step(make_command):
    cmd, env, _ = make_command()
    task = object.__new__(Loaded4DTrackingManager)
    task.num_envs, task.device = 4, 'cpu'
    task._loaded_control_token = 13
    task.command_manager = env.command_manager
    task.reset_buf = torch.ones(4, dtype=torch.long)
    task.time_out_buf = torch.ones(4, dtype=torch.bool)
    task.reset_envs_idx = cmd.reset
    task._refresh_envs_after_reset = Mock()
    task._compute_observations = Mock()
    task._clip_observations = Mock()
    task.obs_buf_dict = {'actor_obs': torch.zeros(4, 113)}
    task.step = Mock(side_effect=AssertionError('hidden physics step'))
    task.reset_all()
    task.step.assert_not_called()
    assert not cmd.time_steps.any()
    assert cmd._last_advanced_token == 13
    assert not task.reset_buf.any()


def test_dataset_views_match_audited_bytes_and_counts():
    if not all((DATA_ROOT / 'joint' / name).is_file() for name in MOTIONS):
        pytest.skip('Audited external NPZ dataset is not available on this machine')
    for name, metadata in MOTIONS.items():
        path = DATA_ROOT / 'joint' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256']
        with np.load(path, allow_pickle=False) as data:
            assert int(data['fps'][0]) == 50
            assert len(data['joint_pos']) == metadata['frames']
    assert sorted(p.name for p in (DATA_ROOT / 'joint').glob('*.npz')) == sorted(MOTIONS)
    assert len(list((DATA_ROOT / 'squat').glob('*.npz'))) == 2


def test_hash_fps_and_missing_clip_fail_fast(make_command):
    cmd, env, _ = make_command()
    name = next(iter(cmd.expected_motions))
    expected = cmd.expected_motions[name]
    cmd.expected_motions = {name: {**expected, 'sha256': '0' * 64}}
    with pytest.raises(ValueError, match='hash mismatch'):
        cmd._load_motion_data(env.simulator._body_list, env.simulator.dof_names)
    cmd.expected_motions = {name: expected, 'missing.npz': expected}
    with pytest.raises(ValueError, match='exactly'):
        cmd._load_motion_data(env.simulator._body_list, env.simulator.dof_names)
    cmd.expected_motions = {name: expected}
    env.dt = .01
    with pytest.raises(ValueError, match='fps'):
        cmd.setup()


def test_real_upstream_ppo_restores_networks_normalizers_optimizers_std_and_new_partition(make_command, tmp_path):
    from holosoma.agents.base_algo.base_algo import BaseAlgo
    from holosoma.agents.ppo.ppo import PPO

    def create(stage):
        cmd, env, cfg = make_command(stage)
        task = object.__new__(Loaded4DTrackingManager)
        task._loaded_config = get_tyro_env_config(cfg)
        task.command_manager, task.dt = env.command_manager, env.dt
        env.get_checkpoint_state = task.get_checkpoint_state
        env.load_checkpoint_state = task.load_checkpoint_state
        env.observation_manager = ObservationManager(cfg.observation, env, 'cpu')
        algo = object.__new__(PPO)
        BaseAlgo.__init__(algo, env, cfg.algo.config, 'cpu')
        algo._init_config()
        algo._setup_models_and_optimizer()
        algo.attach_checkpoint_metadata(cfg)
        algo.logging_helper = SimpleNamespace(save_checkpoint_artifact=lambda payload, path: torch.save(payload, path))
        algo.current_learning_iteration = 42
        return algo, cmd
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        one, cmd1 = create(1)
        one.actor_obs_normalizer.update(torch.randn(8, 113) + .4)
        one.critic_obs_normalizer.update(torch.randn(8, 315) - .6)
        # Populate AdamW moment/step state using synthetic tensors, not RL rollouts.
        for network, optimizer in ((one.actor, one.actor_optimizer), (one.critic, one.critic_optimizer)):
            optimizer.zero_grad()
            sum(parameter.square().sum() for parameter in network.parameters()).backward()
            optimizer.step()
        with torch.no_grad():
            one.actor.std.fill_(.37)
        path = tmp_path / 'transfer.pt'
        one.save(str(path))
        two, cmd2 = create(2)
        partition = cmd2.track_env_mask.clone()
        two.load(str(path))
        assert torch.equal(cmd2.track_env_mask, partition)
        assert cmd1.track_env_mask.sum() == 4 and cmd2.track_env_mask.sum() == 2
        for left, right in ((one.actor, two.actor), (one.critic, two.critic),
                            (one.actor_obs_normalizer, two.actor_obs_normalizer),
                            (one.critic_obs_normalizer, two.critic_obs_normalizer)):
            for key, value in left.state_dict().items():
                assert torch.equal(value, right.state_dict()[key])
        for left, right in ((one.actor_optimizer, two.actor_optimizer), (one.critic_optimizer, two.critic_optimizer)):
            a, b = left.state_dict(), right.state_dict()
            assert a['param_groups'] == b['param_groups']
            for key, values in a['state'].items():
                for field, value in values.items():
                    assert torch.equal(value, b['state'][key][field])
        x = torch.randn(4, 113)
        assert torch.equal(one.get_inference_policy()({'actor_obs': x}), two.get_inference_policy()({'actor_obs': x}))
        assert two.current_learning_iteration == 42
        assert torch.allclose(two.actor.std, torch.full((29,), .37))
    finally:
        torch.set_num_threads(old_threads)


def test_full_environment_callback_preserves_terminal_obs_before_reset(make_command):
    from holosoma.envs.base_task.base_task import BaseTask

    cmd, env, cfg = make_command(1)
    task = object.__new__(Loaded4DTrackingManager)
    task.__dict__.update(env.__dict__)
    cmd._env = task
    task._loaded_control_token = 1
    task._update_tasks_before_termination = False
    task.reset_buf = torch.zeros(4, dtype=torch.long)
    task.time_out_buf = torch.zeros(4, dtype=torch.bool)
    task.extras, task.log_dict = {}, {}
    task.observation_manager = ObservationManager(cfg.observation, task, 'cpu')
    task._refresh_sim_tensors = lambda: None
    task._update_counters_each_step = lambda: None
    task._record_termination_metrics = lambda **kwargs: None
    task._update_log_dict = lambda: None
    task._post_compute_observations_callback = lambda: None
    task._compute_reward = lambda: None
    task._get_envs_to_refresh = lambda: torch.empty(0, dtype=torch.long)
    task._update_tasks_callback = cmd.step
    # This transition arrives at the final frame and should truncate normally.
    cmd.time_steps[:] = 3
    def check():
        assert (cmd.time_steps == 4).all()
        return torch.zeros(4, dtype=torch.bool), cmd.clip_finished
    task.termination_manager = SimpleNamespace(check=check, time_outs=torch.zeros(4, dtype=torch.bool))
    def reset(ids):
        cmd.reset(ids)
        task.simulator._rigid_body_pos[ids] = cmd.motion.body_pos_w[cmd.time_steps[ids]]
        task.episode_length_buf[ids] = 0
        task.extras['time_outs'] = task.time_out_buf
    task.reset_envs_idx = reset
    BaseTask._post_physics_step(task)
    assert not cmd.time_steps.any()
    assert (task.extras['loaded_4d']['reference_frame'] == 4).all()
    assert task.time_out_buf.all()
    # Critic contains the advanced reference position before reset, while the
    # returned observation has frame 0. This protects timeout value bootstrapping.
    final = task.extras['final_observations']['critic_obs']
    current = task.obs_buf_dict['critic_obs']
    assert not torch.equal(final, current)
