"""Loaded-4D command, using upstream motion loading, reset sampling and partitioning.

Runtime quaternions are xyzw. ``command`` remains the privileged joint reference;
``task_command`` is the four-dimensional policy goal. No code here attaches or
moves the object except through the inherited episode-reset path.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from holosoma.managers.command.terms.wbt import HybridMotionCommand
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import quat_apply


def heading_state(position, quaternion, linear_velocity, angular_velocity, ground_z=0.0):
    """Return heading-frame vx/vy, d(heading)/dt and ground-relative pelvis height.

    Angular velocity is in world coordinates, not Euler-angle derivatives.
    Near a vertical forward axis the denominator is bounded; the physical
    termination separately rejects that singular posture.
    """
    forward = torch.zeros_like(position)
    forward[..., 0] = 1.0
    forward = quat_apply(quaternion, forward, w_last=True)
    derivative = torch.linalg.cross(angular_velocity, forward, dim=-1)
    norm_sq = forward[..., :2].square().sum(-1).clamp_min(1.0e-8)
    cos_yaw = forward[..., 0] / norm_sq.sqrt()
    sin_yaw = forward[..., 1] / norm_sq.sqrt()
    vx = cos_yaw * linear_velocity[..., 0] + sin_yaw * linear_velocity[..., 1]
    vy = -sin_yaw * linear_velocity[..., 0] + cos_yaw * linear_velocity[..., 1]
    yaw_rate = (forward[..., 0] * derivative[..., 1] - forward[..., 1] * derivative[..., 0]) / norm_sq
    return torch.stack((vx, vy, yaw_rate, position[..., 2] - ground_z), dim=-1)


def validate_exact_joint_positions(positions, limits, tolerance=1.0e-5):
    """Reject an invalid reset; never silently modify an otherwise legal reference."""
    if not torch.isfinite(positions).all():
        raise ValueError("Loaded-4D reference joint positions must be finite")
    if ((positions < limits[..., 0] - tolerance) | (positions > limits[..., 1] + tolerance)).any():
        raise ValueError("Loaded-4D reference exceeds hard joint limits")
    return positions


class Loaded4DCommand(HybridMotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.options = dict(cfg.params.get("loaded_4d", {}))
        self.expected_motions = dict(cfg.params["expected_motions"])

    def setup(self):
        cfg = self.motion_cfg
        if (cfg.enable_default_pose_prepend or cfg.enable_default_pose_append
                or cfg.enable_terminal_pose_hold or cfg.freeze_at_timestep_zero_prob):
            raise ValueError("Loaded-4D requires unextended clips and one frame per control step")
        if cfg.noise_to_initial_pose.overall_noise_scale != 0 or cfg.noise_to_initial_pose.step_zero_root_noise is not None:
            raise ValueError("Loaded-4D starts from exact reference states")
        if cfg.hybrid is None or cfg.hybrid.size_randomization.enabled or cfg.hybrid.gen_step_zero_root_noise is not None:
            raise ValueError("Loaded-4D requires fixed geometry and exact Gen reset poses")
        if self._env.simulator._body_list[0] != "pelvis":
            raise ValueError("Loaded-4D requires pelvis as the simulator root body")
        super().setup()
        if not self.motion.has_object or not math.isclose(float(self.motion.fps) * self._env.dt, 1.0, abs_tol=1e-6):
            raise ValueError("Loaded-4D requires object data and motion fps * control dt == 1")
        self._prepare_reset_joint_positions(self.motion.joint_pos)
        # PhysX order can differ from policy order. Check the mapped hard limits,
        # without changing either the simulator limits or upstream regularization.
        robot = getattr(self._env.simulator, "_robot", None)
        if robot is not None:
            limits = robot.data.joint_pos_limits[0, self._env.simulator.dof_ids]
            validate_exact_joint_positions(self.motion.joint_pos, limits)

    def _load_motion_data(self, robot_body_names_alias, robot_joint_names):
        if self.motion_cfg.motion_dir:
            paths = []
            for directory in self.motion_cfg.motion_dir.split(","):
                paths.extend(sorted(Path(resolve_data_file_path(str(Path(directory.strip()).expanduser()))).glob("*.npz")))
        else:
            paths = [Path(resolve_data_file_path(self.motion_cfg.motion_file))]
        names = [p.name for p in paths]
        if len(names) != len(set(names)) or set(names) != set(self.expected_motions):
            raise ValueError(f"Loaded-4D data view must contain exactly {sorted(self.expected_motions)}; found {names}")
        self.data_manifest = {}
        for path in paths:
            expected = self.expected_motions[path.name]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected["sha256"]:
                raise ValueError(f"Loaded-4D data hash mismatch: {path}")
            self.data_manifest[path.name] = digest
        motion = super()._load_motion_data(robot_body_names_alias, robot_joint_names)
        lengths = (motion.motion_end_idx - motion.motion_start_idx).tolist()
        if lengths != [self.expected_motions[name]["frames"] for name in names]:
            raise ValueError("Motion loader skipped a required clip or changed its frame count")
        self.motion_names = names
        return motion

    def init_buffers(self):
        super().init_buffers()
        self.task_command = torch.zeros(self.num_envs, 4, device=self.device)
        self.applied_command = torch.zeros_like(self.task_command)
        self.target_command = torch.zeros_like(self.task_command)
        self._transition_start = torch.zeros_like(self.task_command)
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._dwell = torch.zeros_like(self._elapsed)
        self.external_command_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_advanced_token = -1
        self.physical_reset_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ranges = torch.tensor(self.options.get("ranges", [[-0.15, 0.15], [-0.10, 0.10], [-0.20, 0.20], [0.66, 0.73]]), device=self.device)
        self._external_ranges = torch.tensor(self.options.get("external_ranges", [[-0.8, 0.8], [-0.4, 0.4], [-1.0, 1.0], [0.60, 0.78]]), device=self.device)
        for ranges in (self._ranges, self._external_ranges):
            if ranges.shape != (4, 2) or not torch.isfinite(ranges).all() or (ranges[:, 0] > ranges[:, 1]).any():
                raise ValueError("Loaded-4D ranges must be finite ordered [4, 2] bounds")
        if (self._ranges[:, 0] < self._external_ranges[:, 0]).any() or (self._ranges[:, 1] > self._external_ranges[:, 1]).any():
            raise ValueError("Sampler ranges must fit inside the external command envelope")
        self.transition_s = float(self.options.get("transition_s", 0.8))
        self.height_rate = float(self.options.get("height_rate", 0.06))
        self.stand_height = float(self.options.get("stand_height", 0.73))
        self.stand_probability = float(self.options.get("stand_probability", 0.2))
        self.axes = str(self.options.get("axes", "single"))
        self.dwell_s = tuple(self.options.get("dwell_s", [3.0, 5.0]))
        if (not math.isfinite(self.transition_s) or self.transition_s <= 0
                or not math.isfinite(self.height_rate) or self.height_rate <= 0
                or not 0 <= self.stand_probability <= 1
                or self.axes not in {"single", "pair", "combined"}
                or len(self.dwell_s) != 2 or not 0 < self.dwell_s[0] <= self.dwell_s[1]
                or not all(math.isfinite(x) for x in self.dwell_s)):
            raise ValueError("Invalid Loaded-4D sampling/interpolation options")
        if not self._ranges[3, 0] <= self.stand_height <= self._ranges[3, 1]:
            raise ValueError("stand_height must be inside the sampled height range")

    def _prepare_reset_joint_positions(self, positions):
        return validate_exact_joint_positions(positions, self._env.simulator.hard_dof_pos_limits)

    def _assign_goal_time_steps(self, env_ids):
        # Inherited endpoint buffers are unused. In particular, do not sample
        # endpoint noise or refresh endpoint rewards/curriculum during resets.
        return

    def reference_command(self):
        return heading_state(self.root_pos_w, self.root_quat_w, self.root_lin_vel_w,
                             self.root_ang_vel_w, self._env.simulator.scene.env_origins[:, 2])

    def measured_state(self):
        root = self._env.simulator.robot_root_states
        return heading_state(root[:, :3], root[:, 3:7], root[:, 7:10], root[:, 10:13],
                             self._env.simulator.scene.env_origins[:, 2])

    def reset(self, env_ids):
        ids = self._ensure_index_tensor(env_ids)
        if not ids.numel():
            return
        super().reset(ids)
        self.physical_reset_count[ids] += 1
        self.external_command_mask[ids] = False
        self.task_command[ids] = self.reference_command()[ids]
        self.applied_command[ids] = self.task_command[ids]
        self.target_command[ids] = self.task_command[ids]
        gen_ids = ids[self.gen_env_mask[ids]]
        self._sample_gen_targets(gen_ids)
        # The caller flushes FK before step() refreshes relative reference terms.

    def begin_control_step(self):
        """Snapshot exactly the command present in the observation acted upon."""
        self.applied_command.copy_(self.task_command)

    def advance_reference_to_post_action(self, token):
        if token == self._last_advanced_token:
            return
        if token < self._last_advanced_token:
            raise RuntimeError("Loaded-4D control clock must be monotonic")
        self._last_advanced_token = token
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self._increment_adaptive_sampler_global_env_step()
        end = self.motion.motion_end_idx[self.motion_ids] - 1
        self.time_steps[:] = torch.minimum(self.time_steps + self.track_env_mask.long(), end)
        self._update_relative_reference()
        self.task_command[self.track_env_mask] = self.reference_command()[self.track_env_mask]
        self._advance_gen_commands()

    def step(self):
        # BaseTask calls this after reset. Never advance or trigger clip-end reset.
        self._update_relative_reference()
        self.task_command[self.track_env_mask] = self.reference_command()[self.track_env_mask]

    @property
    def clip_finished(self):
        return self.track_env_mask & (self.time_steps >= self.motion.motion_end_idx[self.motion_ids] - 1)

    def _sample_gen_targets(self, ids):
        if not ids.numel():
            return
        target = self._ranges[:, 0] + torch.rand(len(ids), 4, device=self.device) * (self._ranges[:, 1] - self._ranges[:, 0])
        count = {"single": 1, "pair": 2, "combined": 4}[self.axes]
        chosen = torch.rand(len(ids), 4, device=self.device).argsort(dim=1)[:, :count]
        mask = torch.zeros_like(target, dtype=torch.bool).scatter_(1, chosen, True)
        standing = torch.zeros_like(target)
        standing[:, 3] = self.stand_height
        target = torch.where(mask, target, standing)
        stand = torch.rand(len(ids), device=self.device) < self.stand_probability
        target[stand] = standing[stand]
        # Low pelvis heights get a smaller velocity envelope; combinations are
        # learned explicitly rather than inferred from independent data extrema.
        factor = ((target[:, 3] - 0.60) / 0.13).clamp(0.25, 1.0)
        target[:, :3] *= factor[:, None]
        self._transition_start[ids] = self.task_command[ids]
        self.target_command[ids] = target
        self._elapsed[ids] = 0.0
        low, high = self.dwell_s
        self._dwell[ids] = low + (high - low) * torch.rand(len(ids), device=self.device)

    def _advance_gen_commands(self):
        ids = torch.where(self.gen_env_mask)[0]
        if not ids.numel():
            return
        dt = float(self._env.dt)
        self._elapsed[ids] += dt
        self._dwell[ids] -= dt
        random_ids = ids[~self.external_command_mask[ids]]
        if random_ids.numel():
            alpha = (self._elapsed[random_ids] / self.transition_s).clamp(0.0, 1.0)
            alpha = (alpha.square() * (3.0 - 2.0 * alpha))[:, None]
            self.task_command[random_ids, :3] = (
                self._transition_start[random_ids, :3] * (1 - alpha)
                + self.target_command[random_ids, :3] * alpha
            )
        external_ids = ids[self.external_command_mask[ids]]
        if external_ids.numel():
            # Retargetable rate limiting avoids repeatedly restarting smoothstep
            # at zero slope when an external producer updates every control step.
            rate = (self._external_ranges[:3, 1] - self._external_ranges[:3, 0]) / self.transition_s
            delta = self.target_command[external_ids, :3] - self.task_command[external_ids, :3]
            self.task_command[external_ids, :3] += torch.clamp(delta, -rate * dt, rate * dt)
        height_delta = self.target_command[ids, 3] - self.task_command[ids, 3]
        self.task_command[ids, 3] += height_delta.clamp(-self.height_rate * dt, self.height_rate * dt)
        due = random_ids[self._dwell[random_ids] <= 0.0]
        self._sample_gen_targets(due)

    def _gen_ids(self, env_ids):
        ids = torch.where(self.gen_env_mask)[0] if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("env_ids must be a one-dimensional integer index tensor")
        ids = ids.long()
        if (ids < 0).any() or (ids >= self.num_envs).any() or ids.unique().numel() != ids.numel():
            raise ValueError("env_ids must be unique valid indices")
        if not self.gen_env_mask[ids].all():
            raise ValueError("External commands are only accepted by Gen environments")
        return ids

    def set_command(self, command, env_ids=None):
        """Set Gen targets without advancing time, observations or physical state."""
        ids = self._gen_ids(env_ids)
        values = torch.as_tensor(command, device=self.device, dtype=self.task_command.dtype)
        if values.shape == (4,):
            values = values.expand(len(ids), -1)
        if values.shape != (len(ids), 4) or not torch.isfinite(values).all():
            raise ValueError("command must be finite [4] or [len(env_ids), 4]")
        if ((values < self._external_ranges[:, 0]) | (values > self._external_ranges[:, 1])).any():
            raise ValueError("External command is outside the configured envelope")
        self.target_command[ids] = values
        self.external_command_mask[ids] = True

    def release_external_command(self, env_ids=None):
        ids = self._gen_ids(env_ids)
        self.external_command_mask[ids] = False
        self._sample_gen_targets(ids)

    def update_metrics(self):
        super().update_metrics()
        error = (self.measured_state() - self.applied_command).abs()
        for i, name in enumerate(("vx", "vy", "yaw_rate", "pelvis_height")):
            self.metrics[f"loaded_4d/error/{name}"] = error[:, i]
        self.metrics["loaded_4d/gen_tracking_baseline_per_second"] = self.gen_env_mask.float() * 3.5
