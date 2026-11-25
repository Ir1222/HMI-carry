"""Thin Loaded-4D WBT task and its observation/reward/termination terms."""
from __future__ import annotations

from dataclasses import asdict
import json

import torch

from holosoma.envs.wbt.hybrid_size_randomization import (
    collision_bounds_corners,
    load_urdf_collision_bounds,
    rotated_scaled_lower_bound_z,
)
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.utils.rotations import quat_apply, quat_conjugate


def command(env):
    return env.command_manager.get_state("motion_command")


def task_command(env):
    return command(env).task_command


def pelvis_height(env):
    return (env.simulator.robot_root_states[:, 2] - env.simulator.scene.env_origins[:, 2]).unsqueeze(-1)


def physical_metrics(env):
    """Actual state only. Clearance uses rotated collision bounds of the box mesh."""
    cmd = command(env)
    root = env.simulator.robot_root_states[:, :13]
    box_pos, box_quat = cmd.simulator_object_pos_w, cmd.simulator_object_quat_w
    lower = rotated_scaled_lower_bound_z(
        env._loaded_collision_corners, torch.ones_like(box_pos), box_quat,
    )
    up = torch.zeros_like(root[:, :3])
    up[:, 2] = 1.0
    forward = torch.zeros_like(up)
    forward[:, 0] = 1.0
    up = quat_apply(root[:, 3:7], up, w_last=True)
    forward = quat_apply(root[:, 3:7], forward, w_last=True)
    values = (root, env.simulator.dof_pos, env.simulator.dof_vel, box_pos, box_quat,
              cmd.simulator_object_lin_vel_w, cmd.simulator_object_ang_vel_w)
    finite = torch.stack([torch.isfinite(value).all(dim=-1) for value in values]).all(dim=0)
    finite &= (root[:, 3:7].norm(dim=-1) - 1.0).abs() < 0.05
    finite &= (box_quat.norm(dim=-1) - 1.0).abs() < 0.05
    return {
        "finite": finite,
        "pelvis_height": pelvis_height(env).squeeze(-1),
        "upright": up[:, 2],
        "heading_projection": forward[:, :2].square().sum(dim=-1),
        "box_clearance": box_pos[:, 2] + lower - env.simulator.scene.env_origins[:, 2],
        "box_separation": (box_pos - root[:, :3]).norm(dim=-1),
    }


def carry_support_metrics(env):
    """Actual ankle contact and box-local wrist geometry; no hand contact claim.

    Contact forces are reordered by IsaacSim into simulator._body_list order.
    The collisionless *_foot_contact_point bodies MUST NOT supply the forces.
    Four sole probes approximate the ankle cylinders' bottom (URDF z=-.035).
    Requiring the highest probe near ground rejects toe-only contact.
    """
    cmd, sim = command(env), env.simulator
    feet, hands = env._loaded_feet, env._loaded_hands
    pos, rot = sim._rigid_body_pos[:, feet], sim._rigid_body_rot[:, feet]
    offsets = quat_apply(rot[:, :, None, :].expand(-1, -1, 4, -1),
                         env._loaded_sole_points.expand(env.num_envs, 2, -1, -1), w_last=True)
    heights = (pos[:, :, None, 2] + offsets[..., 2] - sim.scene.env_origins[:, None, None, 2])
    velocities = sim._rigid_body_vel[:, feet, None, :] + torch.linalg.cross(
        sim._rigid_body_ang_vel[:, feet, None, :].expand_as(offsets), offsets, dim=-1)
    # The audited box local z points down in the references; measure axis tilt,
    # not its directed angle to world up (which would report normal carry as pi).
    box_q = cmd.simulator_object_quat_w
    inv = quat_conjugate(box_q, w_last=True)
    rel = sim._rigid_body_pos[:, hands] - cmd.simulator_object_pos_w[:, None, :]
    local = quat_apply(inv[:, None, :].expand(-1, 2, -1), rel, w_last=True)
    relative_velocity = sim._rigid_body_vel[:, hands] - cmd.simulator_object_lin_vel_w[:, None, :]
    relative_velocity -= torch.linalg.cross(cmd.simulator_object_ang_vel_w[:, None, :].expand_as(rel), rel, dim=-1)
    up = torch.zeros_like(cmd.simulator_object_pos_w)
    up[:, 2] = 1
    forward = torch.zeros_like(up)
    forward[:, 0] = 1
    heading = quat_apply(sim.robot_root_states[:, 3:7], forward, w_last=True)[:, :2]
    heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return {
        "root_heading": heading,
        "foot_force_z": sim.contact_forces[:, feet, 2].clamp_min(0),
        "foot_height": heights.amax(-1),
        "foot_min_height": heights.amin(-1),
        "foot_slip": velocities[..., :2].norm(dim=-1).amax(-1),
        "hand_box_local": local,
        "hand_box_speed": relative_velocity.norm(dim=-1),
        "box_tilt": torch.acos(quat_apply(box_q, up, w_last=True)[:, 2].abs().clamp(0, 1)),
        "root_position": sim.robot_root_states[:, :3],
        "box_quaternion": box_q,
    }


def gen_constraint_penalty(env, component):
    """Dimensionless soft costs; RewardManager alone applies weight and dt."""
    cmd = command(env)
    cfg = cmd.options['constraints']
    m = carry_support_metrics(env)
    if component == 'support':
        force = (m['foot_force_z'] / cfg['foot_force_min']).clamp(0, 1)
        height = torch.exp(-(m['foot_height'].clamp_min(0) / cfg['foot_height_max']).square())
        slip = torch.exp(-(m['foot_slip'] / cfg['foot_slip_max']).square())
        quality = torch.minimum(torch.minimum(force, height), slip).amin(-1)
        gate = ((cmd._stationary_time - cfg['support_grace_s']) / cfg['support_ramp_s']).clamp(0, 1)
        cost = gate * (1 - quality)
    elif component == 'carry':
        low = torch.as_tensor(cfg['hand_local_min'], device=env.device)
        high = torch.as_tensor(cfg['hand_local_max'], device=env.device)
        distance = ((low - m['hand_box_local']).clamp_min(0) +
                    (m['hand_box_local'] - high).clamp_min(0)).norm(dim=-1).amax(-1)
        slip = (m['hand_box_speed'].amax(-1) - cfg['hand_speed_max']).clamp_min(0)
        tilt = (m['box_tilt'] - cfg['box_tilt_max']).clamp_min(0)
        cost = (distance / cfg['hand_margin']).square() + (slip / cfg['hand_speed_max']).square() + (tilt / .3).square()
        cost = cost.clamp_max(4)
    else:
        raise ValueError(component)
    return torch.where(cmd.gen_env_mask & ~physical_failure(env), cost, torch.zeros_like(cost))


def physical_failure(env):
    metrics = physical_metrics(env)
    cfg = command(env).options
    limits = env.simulator.hard_dof_pos_limits
    joint_failure = ((env.simulator.dof_pos < limits[:, 0] - 0.03)
                     | (env.simulator.dof_pos > limits[:, 1] + 0.03)).any(dim=-1)
    return (
        ~metrics["finite"] | joint_failure
        | (metrics["pelvis_height"] < float(cfg.get("fall_height", 0.40)))
        | (metrics["upright"] < float(cfg.get("min_upright", 0.20)))
        | (metrics["heading_projection"] < 1.0e-4)
        | (metrics["box_clearance"] <= float(cfg.get("box_clearance", 0.02)))
        | (metrics["box_separation"] > float(cfg.get("max_box_separation", 1.0)))
    )


def clip_finished(env):
    return command(env).clip_finished


def command_tracking_reward(env, component: str, sigma: float):
    """Gen reward evaluates the command acted upon, before the next publication."""
    if sigma <= 0 or component not in {"velocity", "vx", "vy", "yaw_rate", "height"}:
        raise ValueError("Invalid Loaded-4D command reward configuration")
    cmd = command(env)
    error = cmd.measured_state() - cmd.applied_command
    if component == "velocity":
        error_sq = error[:, :2].square().sum(dim=-1)
    else:
        error_sq = error[:, {"vx": 0, "vy": 1, "yaw_rate": 2, "height": 3}[component]].square()
    valid = cmd.gen_env_mask & ~physical_failure(env)
    return torch.where(valid, torch.exp(-error_sq / sigma**2), torch.zeros_like(error_sq))


class Loaded4DTrackingManager(WholeBodyTrackingManager):
    def __init__(self, tyro_config, *, device):
        self._loaded_config = tyro_config
        self._loaded_control_token = 0
        self._checkpoint_contract_loaded = False
        if not tyro_config.robot.control.action_scales_by_effort_limit_over_p_gain:
            raise ValueError("Loaded-4D requires the upstream WBT action scaling")
        if not tyro_config.robot.asset.enable_self_collisions:
            raise ValueError("Loaded-4D requires the upstream WBT self-collision setting")
        if tyro_config.terrain.terrain_term.mesh_type != "plane":
            raise ValueError("Loaded-4D currently defines ground height on a plane")
        super().__init__(tyro_config, device=device)
        path = self.robot_config.object.object_urdf_path
        if not path:
            raise ValueError("Loaded-4D requires the original free dynamic object URDF")
        bounds = torch.tensor(load_urdf_collision_bounds(path), device=self.device, dtype=torch.float32)
        self._loaded_collision_corners = collision_bounds_corners(bounds)
        if command(self).options.get('diagnostics', False):
            names = self.simulator._body_list
            self._loaded_feet = [names.index(side + '_ankle_roll_link') for side in ('left', 'right')]
            self._loaded_hands = [names.index(side + '_wrist_yaw_link') for side in ('left', 'right')]
            self._loaded_sole_points = torch.tensor(
                [[-.04, -.026, -.035], [-.04, .026, -.035], [.12, -.026, -.035], [.12, .026, -.035]],
                device=self.device).reshape(1, 1, 4, 3)

    def reset_all(self):
        # Deliberately bypass WBT.reset_all -> BaseTask.reset_all's zero-action step.
        cmd = command(self)
        cmd.init_buffers()
        cmd._last_advanced_token = self._loaded_control_token
        ids = torch.arange(self.num_envs, device=self.device)
        self.reset_envs_idx(ids)
        self._refresh_envs_after_reset(ids)
        cmd.step()
        self.reset_buf.zero_()
        self.time_out_buf.zero_()
        self._compute_observations()
        self._clip_observations()
        return self.obs_buf_dict

    def _pre_physics_step(self, actions):
        if self.training_config.checkpoint and not self._checkpoint_contract_loaded:
            raise RuntimeError("Loaded-4D checkpoint is missing its task contract")
        command(self).begin_control_step()
        self._loaded_control_token += 1
        super()._pre_physics_step(actions)

    def _check_termination(self):
        command(self).advance_reference_to_post_action(self._loaded_control_token)
        failures, timeouts = self.termination_manager.check()
        # A fall on the last reference frame is a failure, not a bootstrap target.
        timeouts &= ~failures
        self.termination_manager.time_outs.copy_(timeouts)
        self.reset_buf[:] = failures | timeouts
        self.time_out_buf[:] = timeouts
        self._record_termination_metrics(reset_flags=failures, timeout_flags=timeouts)
        if self.is_evaluating:
            # Snapshot before BaseTask auto-resets terminal rows. Consumers must
            # never mistake the fresh reset state for a successful final state.
            cmd = command(self)
            self.extras["loaded_4d"] = {
                "measured": cmd.measured_state().clone(),
                "applied_command": cmd.applied_command.clone(),
                "published_command": cmd.task_command.clone(),
                "target_command": cmd.target_command.clone(),
                "reference_frame": cmd.time_steps.clone(),
                "motion_id": cmd.motion_ids.clone(),
                "clip_finished": cmd.clip_finished.clone(),
                "physical_failure": physical_failure(self).clone(),
                "body_position_error": (cmd.body_pos_relative_w - cmd.robot_body_pos_w).norm(dim=-1).mean(dim=-1),
                "object_position_error": (cmd.object_pos_w - cmd.simulator_object_pos_w).norm(dim=-1),
                **{key: value.clone() for key, value in physical_metrics(self).items()},
                **({key: value.clone() for key, value in carry_support_metrics(self).items()}
                   if cmd.options.get('diagnostics', False) else {}),
            }

    def _compute_reward(self):
        detailed = command(self).options.get('profile') == 'vxh'
        if detailed:
            before = {k: v.clone() for k, v in self.reward_manager._episode_sums.items()}
        super()._compute_reward()
        if detailed:
            mask = command(self).gen_env_mask
            if mask.any():
                for key, value in self.reward_manager._episode_sums.items():
                    self.log_dict[f'loaded_4d/gen_reward_rate/{key}'] = ((value - before[key])[mask].mean() / self.dt).detach()
        bad = ~torch.isfinite(self.rew_buf)
        if (bad & ~self.reset_buf.bool()).any():
            raise RuntimeError("Non-finite Loaded-4D reward without a physical termination")
        self.rew_buf[bad] = 0.0

    def _compute_final_observations(self):
        observations = super()._compute_final_observations()
        # Failed/non-finite physical states never bootstrap. Keep the stored
        # terminal tensors finite so a batched critic forward cannot spread NaNs.
        for values in observations.values():
            values[self.reset_buf.bool() & ~self.time_out_buf] = torch.nan_to_num(
                values[self.reset_buf.bool() & ~self.time_out_buf], nan=0.0, posinf=0.0, neginf=0.0,
            )
        return observations

    def _update_log_dict(self):
        super()._update_log_dict()
        cmd = command(self)
        if cmd.options.get('profile') == 'vxh':
            for labels, counts in ((('forward','height','stand','combined'), cmd.sample_counts),
                                   (tuple(f'vx_bin_{i}' for i in range(7)), cmd.sample_vx_hist),
                                   (tuple(f'height_bin_{i}' for i in range(5)), cmd.sample_height_hist)):
                for label, count in zip(labels, counts):
                    self.log_dict[f'loaded_4d/sampled/{label}'] = (count / counts.sum().clamp_min(1)).detach()
        errors = (cmd.measured_state() - cmd.applied_command).abs()
        for name, mask in (("track", cmd.track_env_mask), ("gen", cmd.gen_env_mask)):
            if mask.any():
                for index, axis in enumerate(("vx", "vy", "yaw_rate", "pelvis_height")):
                    self.log_dict[f"loaded_4d/{name}/mae_{axis}"] = errors[mask, index].mean().detach()
        self.log_dict["loaded_4d/gen_tracking_baseline_per_step"] = cmd.gen_env_mask.float().mean() * (3.5 * self.dt)

    def _task_contract(self):
        cfg = self._loaded_config
        cmd = command(self)
        # Exclude task-stage ratio, sampling schedule, physical and episode state.
        # All policy semantics stay fixed across Stage 1 -> Stage 2.
        return {
            "version": 1,
            "command": "pelvis_heading_vxy_heading_derivative_ground_height",
            "timing": "actor_k_tracking_k_plus_1_gen_applied_k",
            "observations": asdict(cfg.observation),
            "observation_order": {name: sorted(group.terms) for name, group in cfg.observation.groups.items()},
            "normalization": "upstream_empirical_per_observation_group",
            "action": asdict(cfg.action),
            "control": asdict(cfg.robot.control),
            "joint_names": list(cfg.robot.dof_names),
            "default_joint_angles": dict(cfg.robot.init_state.default_joint_angles),
            "control_dt": self.dt,
            "reference_fps": float(cmd.motion.fps),
            "data": cmd.data_manifest,
            "tracking_anchor": cmd.motion_cfg.body_name_ref,
            "tracked_bodies": cmd.motion_cfg.body_names_to_track,
        }

    def get_checkpoint_state(self):
        return {"loaded_4d_contract": json.dumps(self._task_contract(), sort_keys=True)}

    def load_checkpoint_state(self, state):
        expected = self.get_checkpoint_state()
        if state != expected:
            raise ValueError("Loaded-4D checkpoint contract mismatch: observation/action/timing/data must agree")
        self._checkpoint_contract_loaded = True
        # Intentionally restore no masks, physical tensors or sampler buffers.
