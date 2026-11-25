"""Loaded-4D presets composed from the original Object HMI preset.

Data views contain immutable symlinks, not rewritten motion files. Override
motion_dir/motion_file for a relocated view; the manifest is checked at setup.
"""
from dataclasses import replace
from pathlib import Path

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, GoalPoseNoiseConfig, NoiseToInitialPoseConfig
from holosoma.config_types.observation import ObsTermCfg
from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_types.termination import TerminationTermCfg
from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum
from holosoma.config_values.wbt.g1.experiment import g1_29dof_wbt_hybrid_object

MOTIONS = {
    "loaded_locomotion.npz": {"frames": 110, "sha256": "fe03f9072560cc75096c5a0cbcd1ae781a1fe9607c23fa8781ed257ebed98550"},
    "continuous_grasped_squat_stand.npz": {"frames": 351, "sha256": "412681d73670e24c4a32401970710513ba9e3c2dd8b3205e2bf4c4a6a4f33a90"},
    "repeated_grasped_squat_stand.npz": {"frames": 651, "sha256": "d6238a97fa785d94cdf9579f82c002f2b088c34267ba35465e0ed350932ca835"},
}
DATA_ROOT = Path(__file__).resolve().parents[6] / "datasets" / "loaded_4d"
COMMAND_CLASS = "holosoma.managers.command.terms.loaded_4d:Loaded4DCommand"
TASK_MODULE = "holosoma.envs.wbt.loaded_4d"


def loaded_4d_preset(stage=1, data="joint", axes="single"):
    if stage not in (1, 2) or data not in ("joint", "locomotion", "squat") or axes not in ("single", "pair", "combined"):
        raise ValueError("Invalid Loaded-4D stage, data variant or command sampling mode")
    base = g1_29dof_wbt_hybrid_object
    source = base.command.setup_terms["motion_command"].params["motion_config"]
    names = list(MOTIONS) if data == "joint" else (["loaded_locomotion.npz"] if data == "locomotion" else list(MOTIONS)[1:])
    motion_cfg = replace(
        source,
        motion_file=str(DATA_ROOT / "joint" / "loaded_locomotion.npz"),
        motion_dir="" if data == "locomotion" else str(DATA_ROOT / data),
        load_fixed_scene_from_motion=False,
        enable_default_pose_prepend=False,
        enable_default_pose_append=False,
        enable_terminal_pose_hold=False,
        freeze_at_timestep_zero_prob=0.0,
        start_at_timestep_zero_prob=0.05,
        noise_to_initial_pose=NoiseToInitialPoseConfig(),
        hybrid=replace(
            source.hybrid, track_ratio=1.0 if stage == 1 else 0.5,
            gen_step_zero_root_noise=None,
            robot_goal_noise=GoalPoseNoiseConfig(), object_goal_noise=GoalPoseNoiseConfig(),
            size_randomization=replace(source.hybrid.size_randomization, enabled=False),
        ),
    )
    options = {
        "ranges": [[-0.15, 0.15], [-0.10, 0.10], [-0.20, 0.20], [0.66, 0.73]],
        "external_ranges": [[-0.8, 0.8], [-0.4, 0.4], [-1.0, 1.0], [0.60, 0.78]],
        "stand_height": 0.73, "stand_probability": 0.2,
        "axes": axes, "dwell_s": [3.0, 5.0], "transition_s": 0.8, "height_rate": 0.06,
        "fall_height": 0.40, "min_upright": 0.20, "box_clearance": 0.02, "max_box_separation": 1.0,
    }
    command = CommandManagerCfg(
        setup_terms={"motion_command": CommandTermCfg(func=COMMAND_CLASS, params={
            "motion_config": motion_cfg, "loaded_4d": options,
            "expected_motions": {name: dict(MOTIONS[name]) for name in names},
        })},
        reset_terms={"motion_command": CommandTermCfg(func=COMMAND_CLASS)},
        step_terms={"motion_command": CommandTermCfg(func=COMMAND_CLASS)},
    )
    groups = {}
    for name, group in base.observation.groups.items():
        terms = {key: value for key, value in group.terms.items() if key not in ("obj_goal_pos_b", "obj_goal_ori_b")}
        terms["task_command"] = ObsTermCfg(func=f"{TASK_MODULE}:task_command", noise=0.0)
        terms["pelvis_height"] = ObsTermCfg(func=f"{TASK_MODULE}:pelvis_height", noise=0.0)
        for key in ("base_lin_vel", "projected_gravity"):
            if key not in terms:
                terms[key] = ObsTermCfg(func=f"holosoma.managers.observation.terms.wbt:{key}", noise=0.0)
        groups[name] = replace(group, terms=terms, history_length=1)
    observation = replace(base.observation, groups=groups, history_augmentation=False)
    rewards = {key: value for key, value in base.reward.terms.items() if key != "goal_object_reached_once"}
    rewards["limits_dof_pos"] = replace(rewards["limits_dof_pos"], params={"soft_dof_pos_limit": 1.0})
    # Both stages have identical terms; the Gen mask gives zero during Stage 1.
    for component, sigma in (("velocity", 0.2), ("yaw_rate", 0.3), ("height", 0.05)):
        rewards[f"command_{component}"] = RewardTermCfg(
            func=f"{TASK_MODULE}:command_tracking_reward", weight=1.0,
            params={"component": component, "sigma": sigma},
        )
    bad = base.termination.terms["bad_tracking"]
    terms = {
        **base.termination.terms,
        "bad_tracking": replace(bad, params={
            **bad.params, "min_root_height": float("-inf"),
            "gen_bad_ref_pos_z_threshold": None, "gen_bad_ref_pos_xyz_threshold": None,
            "gen_bad_object_pos_z_threshold": None, "gen_bad_object_ori_threshold": None,
        }),
        "clip_finished": TerminationTermCfg(func=f"{TASK_MODULE}:clip_finished", is_timeout=True),
        "physical_failure": TerminationTermCfg(func=f"{TASK_MODULE}:physical_failure"),
    }
    setup = {}
    for key, term in base.randomization.setup_terms.items():
        params = dict(term.params)
        if "enabled" in params:
            params["enabled"] = False
        if key == "randomize_robot_rigid_body_material_startup":
            params.update(enabled=True, static_friction_range=[1.0, 1.0], dynamic_friction_range=[0.9, 0.9], restitution_range=[0.0, 0.0])
        elif key == "randomize_object_rigid_body_material_startup":
            params.update(enabled=True, static_friction_range=[0.6, 0.6], dynamic_friction_range=[0.6, 0.6], restitution_range=[0.0, 0.0])
        elif key == "randomize_object_rigid_body_mass_startup":
            params.update(enabled=True, mass_distribution_params=[1.0, 1.0])  # add to URDF 0.1 kg
        setup[key] = replace(term, params=params)
    return replace(
        base, env_class=f"{TASK_MODULE}.Loaded4DTrackingManager",
        training=replace(base.training, name=f"g1_loaded_4d_stage{stage}_{data}", num_envs=8192),
        algo=replace(base.algo, config=replace(base.algo.config, init_at_random_ep_len=False, reset_policy_std_on_resume=False, num_learning_iterations=15000)),
        robot=replace(base.robot, control=replace(base.robot.control, action_scales_by_effort_limit_over_p_gain=True)),
        simulator=replace(base.simulator, config=replace(base.simulator.config,
            sim=replace(base.simulator.config.sim, max_episode_length_s=15.0),
            scene=replace(base.simulator.config.scene, env_spacing=4.0))),
        command=command, observation=observation,
        reward=replace(base.reward, terms=rewards), termination=replace(base.termination, terms=terms),
        randomization=replace(base.randomization, setup_terms=setup),
        curriculum=g1_29dof_wbt_curriculum,
    )


g1_loaded_4d_stage1 = loaded_4d_preset()
g1_loaded_4d_stage1_locomotion = loaded_4d_preset(data="locomotion")
g1_loaded_4d_stage1_squat = loaded_4d_preset(data="squat")
g1_loaded_4d_stage2 = loaded_4d_preset(stage=2)


def loaded_4d_vxh_preset():
    """Independent Stage2 experiment: Track remains reference-derived in all axes."""
    base = loaded_4d_preset(stage=2)
    term = base.command.setup_terms['motion_command']
    options = {**term.params['loaded_4d'],
        'profile': 'vxh', 'diagnostics': True,
        'ranges': [[0., .8], [0., 0.], [0., 0.], [.60, .80]],
        'external_ranges': [[0., .8], [0., 0.], [0., 0.], [.60, .80]],
        'forward_range': [.1, .8], 'category_probabilities': [.5, .4, .1, 0.],
        'combined_targets': [], 'slew_rates': [.3, .3, .6, .06],
        # Frozen from docs/loaded_4d_vxh/calibration_summary.json before Stage2 training.
        'constraints': {'support_vx_max': 0.03,
         'support_grace_s': 0.5,
         'support_ramp_s': 0.5,
         'foot_force_min': 20.0,
         'foot_height_max': 0.025,
         'foot_slip_max': 0.15,
         'hand_local_min': [[0.2, -0.02, 0.06], [-0.25, -0.21, 0.01]],
         'hand_local_max': [[0.25, 0.13, 0.13], [-0.12, -0.09, 0.09]],
         'hand_margin': 0.05,
         'hand_speed_max': 0.35,
         'box_tilt_max': 0.32},
    }
    rewards = {k: v for k,v in base.reward.terms.items() if not k.startswith('command_')}
    for component, sigma, weight in [('vx', .10, 2.), ('height', .05, 1.), ('vy', .10, .3), ('yaw_rate', .30, .3)]:
        rewards['command_'+component] = RewardTermCfg(
            func=f'{TASK_MODULE}:command_tracking_reward', weight=weight,
            params={'component': component, 'sigma': sigma})
    for component, weight in [('support', -.5), ('carry', -.5)]:
        rewards['gen_'+component] = RewardTermCfg(func=f'{TASK_MODULE}:gen_constraint_penalty',
                                                weight=weight, params={'component': component})
    return replace(base, training=replace(base.training, name='g1_loaded_4d_stage2_vxh'),
                   command=replace(base.command, setup_terms={'motion_command': replace(term, params={**term.params, 'loaded_4d': options})}),
                   reward=replace(base.reward, terms=rewards))


g1_loaded_4d_stage2_vxh = loaded_4d_vxh_preset()
