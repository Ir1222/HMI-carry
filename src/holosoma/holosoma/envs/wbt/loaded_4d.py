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
from holosoma.utils.rotations import quat_apply


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
    if sigma <= 0 or component not in {"velocity", "yaw_rate", "height"}:
        raise ValueError("Invalid Loaded-4D command reward configuration")
    cmd = command(env)
    error = cmd.measured_state() - cmd.applied_command
    if component == "velocity":
        error_sq = error[:, :2].square().sum(dim=-1)
    else:
        error_sq = error[:, 2 if component == "yaw_rate" else 3].square()
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
            }

    def _compute_reward(self):
        super()._compute_reward()
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
