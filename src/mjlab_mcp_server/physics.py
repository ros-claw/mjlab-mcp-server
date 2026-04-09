"""
Physics Sandbox for e-URDF Safety Validation.

This module provides a MuJoCo-based physics simulation environment for
validating robot trajectories before execution on real hardware.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from .semantic_translator import SemanticTranslator, SemanticViolation


@dataclass
class SafetyCheckResult:
    """Result of a physics safety check simulation."""

    is_safe: bool
    reason: str
    collision_detected: bool
    joint_limit_violated: bool
    torque_limit_exceeded: bool
    collision_details: list[str] = field(default_factory=list)
    joint_limit_details: list[str] = field(default_factory=list)
    torque_limit_details: list[str] = field(default_factory=list)
    simulation_steps: int = 0
    final_qpos: list[float] = field(default_factory=list)
    max_torque_observed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_safe": self.is_safe,
            "reason": self.reason,
            "collision_detected": self.collision_detected,
            "joint_limit_violated": self.joint_limit_violated,
            "torque_limit_exceeded": self.torque_limit_exceeded,
            "collision_details": self.collision_details,
            "joint_limit_details": self.joint_limit_details,
            "torque_limit_details": self.torque_limit_details,
            "simulation_steps": self.simulation_steps,
            "final_qpos": self.final_qpos,
            "max_torque_observed": self.max_torque_observed,
        }


@dataclass
class SafetyPolicy:
    """Safety policy configuration for a robot."""

    joint_names: list[str] = field(default_factory=list)
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_velocity_limits: dict[str, float] = field(default_factory=dict)
    torque_limits: dict[str, float] = field(default_factory=dict)
    collision_exclude_pairs: list[tuple[str, str]] = field(default_factory=list)
    environment_collision_enabled: bool = True
    self_collision_enabled: bool = True
    safety_margin: float = 0.05  # 5% safety margin

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SafetyPolicy":
        """Load safety policy from YAML file."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        return cls(
            joint_names=config.get("joint_names", []),
            joint_limits={
                k: tuple(v) for k, v in config.get("joint_limits", {}).items()
            },
            joint_velocity_limits=config.get("joint_velocity_limits", {}),
            torque_limits=config.get("torque_limits", {}),
            collision_exclude_pairs=[
                tuple(pair) for pair in config.get("collision_exclude_pairs", [])
            ],
            environment_collision_enabled=config.get(
                "environment_collision_enabled", True
            ),
            self_collision_enabled=config.get("self_collision_enabled", True),
            safety_margin=config.get("safety_margin", 0.05),
        )


class PhysicsSandbox:
    """
    MuJoCo-based physics sandbox for trajectory validation.

    This class provides a controlled simulation environment to validate
    robot movements before execution on real hardware.
    """

    _lock = threading.RLock()

    def __init__(
        self,
        model_path: str,
        policy_path: str | None = None,
        safety_margin: float = 0.05,
        e_urdf_config: dict | None = None,
    ):
        """
        Initialize the physics sandbox.

        Args:
            model_path: Path to MuJoCo XML model file (MJCF/URDF)
            policy_path: Optional path to safety policy YAML file
            safety_margin: Safety margin as fraction (0.05 = 5%)
        """
        self.model_path = model_path
        self.safety_margin = safety_margin

        # Load MuJoCo model
        self._load_model()

        # Store e-URDF config early so _create_default_policy can use it
        self.e_urdf_config = e_urdf_config or {}

        # Load or create safety policy
        if policy_path and os.path.exists(policy_path):
            self.policy = SafetyPolicy.from_yaml(policy_path)
        else:
            self.policy = self._create_default_policy()

        # Initialize semantic translator
        self.translator = SemanticTranslator(self, self.e_urdf_config)

        # Extract joint information
        self._extract_joint_info()

        # Initialize violations tracking
        self.violations: list[SemanticViolation] = []

        print(f"[PhysicsSandbox] Loaded model: {model_path}")
        print(f"[PhysicsSandbox] Joints: {self.joint_names}")
        print(f"[PhysicsSandbox] DOF: nq={self.nq}, nv={self.nv}, nu={self.nu}")
        print(f"[PhysicsSandbox] Semantic translator: {'enabled' if self.e_urdf_config else 'disabled'}")

    def _load_model(self) -> None:
        """Load MuJoCo model from file."""
        try:
            if self.model_path.endswith((".xml", ".mjcf")):
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
            elif self.model_path.endswith(".urdf"):
                # MuJoCo can load URDF directly
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
            else:
                raise ValueError(f"Unsupported model format: {self.model_path}")

            self.data = mujoco.MjData(self.model)

            # Get model dimensions
            self.nq = self.model.nq  # Number of position coordinates
            self.nv = self.model.nv  # Number of velocity coordinates
            self.nu = self.model.nu  # Number of actuators

        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model: {e}") from e

    def _create_default_policy(self) -> SafetyPolicy:
        """Create default safety policy from model or e_URDF config."""
        joint_names = []
        joint_limits = {}
        joint_velocity_limits = {}
        torque_limits = {}
        collision_exclude_pairs = []

        # If e_URDF config is available, prefer it over raw MJCF limits
        if self.e_urdf_config:
            joints_cfg = self.e_urdf_config.get("joints", {})
            joint_names = joints_cfg.get("names", [])
            limits_cfg = joints_cfg.get("limits", {})
            joint_limits = {
                k: tuple(v)
                for k, v in limits_cfg.get("position_rad", {}).items()
            }
            joint_velocity_limits = {
                k: float(v)
                for k, v in limits_cfg.get("velocity_rad_s", {}).items()
            }
            torque_limits = {
                k: float(v)
                for k, v in limits_cfg.get("torque_nm", {}).items()
            }
            fw_cfg = self.e_urdf_config.get("physical_firewall", {})
            collision_exclude_pairs = [
                tuple(pair)
                for pair in fw_cfg.get("excluded_collision_pairs", [])
            ]

        # Fallback to MuJoCo model for anything missing
        if not joint_names:
            for i in range(self.model.njnt):
                joint_id = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, i
                )
                joint_names.append(joint_id)

        if not joint_limits:
            for i in range(self.model.njnt):
                joint_id = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, i
                )
                jnt_range = self.model.jnt_range[i]
                if jnt_range[0] < jnt_range[1]:  # Valid range
                    joint_limits[joint_id] = (
                        float(jnt_range[0]), float(jnt_range[1])
                    )

        if not torque_limits:
            default_torque = 100.0
            for i in range(min(self.nu, len(joint_names))):
                torque_limits[joint_names[i]] = default_torque

        return SafetyPolicy(
            joint_names=joint_names,
            joint_limits=joint_limits,
            joint_velocity_limits=joint_velocity_limits,
            torque_limits=torque_limits,
            collision_exclude_pairs=collision_exclude_pairs,
            safety_margin=self.safety_margin,
        )

    def _extract_joint_info(self) -> None:
        """Extract joint information from model."""
        self.joint_names = []
        self.joint_name_to_id = {}
        self.geom_names = []
        self.geom_name_to_id = {}

        # Extract joint names
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.joint_names.append(name)
            self.joint_name_to_id[name] = i

        # Extract geometry names
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            self.geom_names.append(name)
            self.geom_name_to_id[name] = i

    def reset(self) -> None:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_joint_positions(self, positions: np.ndarray | list[float]) -> None:
        """Set joint positions in simulation."""
        pos_array = np.array(positions)
        if len(pos_array) != self.nq:
            raise ValueError(
                f"Expected {self.nq} positions, got {len(pos_array)}"
            )
        self.data.qpos[:] = pos_array
        mujoco.mj_forward(self.model, self.data)

    def set_joint_velocities(self, velocities: np.ndarray | list[float]) -> None:
        """Set joint velocities in simulation."""
        vel_array = np.array(velocities)
        if len(vel_array) != self.nv:
            raise ValueError(
                f"Expected {self.nv} velocities, got {len(vel_array)}"
            )
        self.data.qvel[:] = vel_array

    def apply_control(self, ctrl: np.ndarray | list[float]) -> None:
        """Apply control signals to actuators."""
        ctrl_array = np.array(ctrl)
        if len(ctrl_array) != self.nu:
            raise ValueError(
                f"Expected {self.nu} controls, got {len(ctrl_array)}"
            )
        self.data.ctrl[:] = ctrl_array

    def step(self, n_steps: int = 1) -> None:
        """Step physics simulation forward."""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def check_collision(self) -> tuple[bool, list[str], float]:
        """
        Check for collisions in current state.

        Returns:
            Tuple of (collision_detected, collision_details, min_distance)
        """
        # Update contact information
        mujoco.mj_collision(self.model, self.data)

        collision_detected = False
        min_distance = float("inf")
        collision_details = []

        # Check all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Get geometry names
            name1 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id
            )
            name2 = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id
            )

            # Check if this pair should be excluded
            if self._is_collision_excluded(name1, name2):
                continue

            # Contact distance < 0 means penetration
            if contact.dist < 0:
                collision_detected = True
                collision_details.append(f"{name1} collided with {name2}")

            min_distance = min(min_distance, abs(contact.dist))

        return collision_detected, collision_details, min_distance

    def _is_collision_excluded(self, name1: str, name2: str) -> bool:
        """Check if collision between two geoms should be excluded."""
        for excluded_pair in self.policy.collision_exclude_pairs:
            if (name1 in excluded_pair and name2 in excluded_pair):
                return True
        return False

    def check_joint_limits(self) -> tuple[bool, list[str]]:
        """
        Check if any joint exceeds its limits.

        Returns:
            Tuple of (limit_violated, violation_details)
        """
        limit_violated = False
        violations = []

        for joint_name, (min_limit, max_limit) in self.policy.joint_limits.items():
            if joint_name not in self.joint_name_to_id:
                continue

            joint_id = self.joint_name_to_id[joint_name]
            if joint_id >= len(self.data.qpos):
                continue

            qpos = self.data.qpos[joint_id]

            # Apply safety margin
            margin = (max_limit - min_limit) * self.policy.safety_margin

            if qpos < min_limit + margin:
                limit_violated = True
                violations.append(
                    f"Joint '{joint_name}': {qpos:.4f} rad < "
                    f"{min_limit + margin:.4f} rad (min limit)"
                )
            elif qpos > max_limit - margin:
                limit_violated = True
                violations.append(
                    f"Joint '{joint_name}': {qpos:.4f} rad > "
                    f"{max_limit - margin:.4f} rad (max limit)"
                )

        return limit_violated, violations

    def check_joint_velocity_limits(self) -> tuple[bool, list[str]]:
        """
        Check if any joint velocity exceeds its limits.

        Returns:
            Tuple of (limit_violated, violation_details)
        """
        limit_violated = False
        violations = []

        for joint_name, vel_limit in self.policy.joint_velocity_limits.items():
            if joint_name not in self.joint_name_to_id:
                continue

            joint_id = self.joint_name_to_id[joint_name]
            if joint_id >= len(self.data.qvel):
                continue

            qvel = abs(self.data.qvel[joint_id])
            margin = vel_limit * self.policy.safety_margin

            if qvel > vel_limit - margin:
                limit_violated = True
                violations.append(
                    f"Joint '{joint_name}': {qvel:.4f} rad/s > "
                    f"{vel_limit - margin:.4f} rad/s (velocity limit)"
                )

        return limit_violated, violations

    def check_torque_limits(self) -> tuple[bool, float, list[str]]:
        """
        Check if any joint torque exceeds limits.

        Returns:
            Tuple of (limit_exceeded, max_torque, violation_details)
        """
        limit_exceeded = False
        max_torque = 0.0
        violations = []

        # Get joint torques from actuator forces
        for i in range(min(self.nu, len(self.joint_names))):
            joint_name = self.joint_names[i]
            if i < len(self.data.qfrc_actuator):
                torque = abs(self.data.qfrc_actuator[i])
                max_torque = max(max_torque, torque)

                if joint_name in self.policy.torque_limits:
                    limit = self.policy.torque_limits[joint_name] * (
                        1 - self.policy.safety_margin
                    )
                    if torque > limit:
                        limit_exceeded = True
                        violations.append(
                            f"Joint '{joint_name}': torque {torque:.2f} Nm > "
                            f"limit {limit:.2f} Nm"
                        )

        return limit_exceeded, max_torque, violations

    def simulate_safety_check(
        self,
        current_qpos: list[float],
        target_qpos: list[float],
        duration_sec: float = 2.0,
        control_mode: str = "position",
        kp: float = 100.0,
        kd: float = 10.0,
    ) -> SafetyCheckResult:
        """
        Simulate trajectory and check for safety violations.

        Args:
            current_qpos: Current joint positions (radians)
            target_qpos: Target joint positions (radians)
            duration_sec: Simulation duration in seconds
            control_mode: Control mode ('position' or 'velocity')
            kp: Proportional gain for PD controller
            kd: Derivative gain for PD controller

        Returns:
            SafetyCheckResult with detailed safety assessment
        """
        # Validate inputs
        if len(current_qpos) != self.nq:
            return SafetyCheckResult(
                is_safe=False,
                reason=f"Invalid current_qpos: expected {self.nq} joints, got {len(current_qpos)}",
                collision_detected=False,
                joint_limit_violated=False,
                torque_limit_exceeded=False,
            )

        if len(target_qpos) != self.nq:
            return SafetyCheckResult(
                is_safe=False,
                reason=f"Invalid target_qpos: expected {self.nq} joints, got {len(target_qpos)}",
                collision_detected=False,
                joint_limit_violated=False,
                torque_limit_exceeded=False,
            )

        # Reset and set initial state
        self.reset()
        self.set_joint_positions(current_qpos)

        # Pre-check: validate target positions against limits
        for joint_name, (min_limit, max_limit) in self.policy.joint_limits.items():
            if joint_name not in self.joint_name_to_id:
                continue
            joint_id = self.joint_name_to_id[joint_name]
            if joint_id >= len(target_qpos):
                continue

            target = target_qpos[joint_id]
            if target < min_limit or target > max_limit:
                return SafetyCheckResult(
                    is_safe=False,
                    reason=f"Target position violates joint limits",
                    collision_detected=False,
                    joint_limit_violated=True,
                    joint_limit_details=[
                        f"Joint '{joint_name}': target {target:.4f} rad "
                        f"outside limits [{min_limit:.4f}, {max_limit:.4f}]"
                    ],
                    torque_limit_exceeded=False,
                )

        # Calculate number of simulation steps
        steps = int(duration_sec / self.model.opt.timestep)

        # Tracking variables
        collision_detected = False
        joint_limit_violated = False
        velocity_limit_violated = False
        torque_limit_exceeded = False
        all_collision_details = []
        all_joint_limit_details = []
        all_velocity_limit_details = []
        all_torque_limit_details = []
        max_torque_observed = 0.0

        current_array = np.array(current_qpos)
        target_array = np.array(target_qpos)

        # Run simulation
        for step in range(steps):
            # Calculate progress (0 to 1)
            progress = (step + 1) / steps

            # Generate control input based on mode
            if control_mode == "position":
                # PD control toward interpolated position
                desired_pos = current_array * (1 - progress) + target_array * progress

                # Check if MuJoCo actuators already have built-in position control
                # (common in MuJoCo Menagerie models: GAIN_FIXED + BIAS_AFFINE)
                has_builtin_pos_ctrl = (
                    self.nu > 0
                    and all(
                        self.model.actuator_gaintype[i]
                        == mujoco.mjtGain.mjGAIN_FIXED
                        and self.model.actuator_biastype[i]
                        == mujoco.mjtBias.mjBIAS_AFFINE
                        for i in range(self.nu)
                    )
                )

                if has_builtin_pos_ctrl:
                    # For models with built-in PD, ctrl represents target position
                    self.apply_control(desired_pos[: self.nu])
                else:
                    pos_error = desired_pos - self.data.qpos[: self.nq]
                    vel_error = -self.data.qvel[: self.nv]
                    control = kp * pos_error + kd * vel_error
                    self.apply_control(control[: self.nu])
            elif control_mode == "velocity":
                # Velocity control toward target
                direction = target_array - self.data.qpos[: self.nq]
                if np.linalg.norm(direction) > 0.01:
                    desired_vel = direction / np.linalg.norm(direction) * 0.5  # rad/s
                else:
                    desired_vel = np.zeros(self.nv)
                vel_error = desired_vel - self.data.qvel[: self.nv]
                control = kd * vel_error
                self.apply_control(control[: self.nu])
            else:
                # Direct interpolation of control
                control = current_array * (1 - progress) + target_array * progress
                self.apply_control(control[: self.nu])

            # Step physics
            self.step()

            # Check collisions
            has_collision, collision_details, _ = self.check_collision()
            if has_collision and not collision_detected:
                collision_detected = True
                all_collision_details.extend(collision_details)

            # Check joint limits
            j_limit_violated, j_violations = self.check_joint_limits()
            if j_limit_violated:
                joint_limit_violated = True
                all_joint_limit_details.extend(j_violations)

            # Check velocity limits
            v_limit_violated, v_violations = self.check_joint_velocity_limits()
            if v_limit_violated:
                velocity_limit_violated = True
                all_velocity_limit_details.extend(v_violations)

            # Check torque limits
            t_limit_exceeded, step_max_torque, t_violations = self.check_torque_limits()
            max_torque_observed = max(max_torque_observed, step_max_torque)
            if t_limit_exceeded:
                torque_limit_exceeded = True
                all_torque_limit_details.extend(t_violations)

        # Compile final result
        is_safe = not (
            collision_detected
            or joint_limit_violated
            or velocity_limit_violated
            or torque_limit_exceeded
        )

        # Generate semantic violations for better error messages
        self.violations = []
        if self.translator:
            if collision_detected and all_collision_details:
                # Get semantic collision info
                contacts = self.translator.get_collision_summary(duration_sec)
                for contact in contacts[:3]:  # Limit to first 3
                    violation = self.translator.translate_violation(
                        violation_type="COLLISION",
                        raw_message=contact.to_natural_language(),
                        context={"contact": contact},
                    )
                    self.violations.append(violation)

            if joint_limit_violated:
                for detail in all_joint_limit_details[:3]:
                    violation = self.translator.translate_violation(
                        violation_type="JOINT_LIMIT",
                        raw_message=detail,
                    )
                    self.violations.append(violation)

        # Generate summary reason
        if is_safe:
            reason = "Physics simulation passed. No safety violations detected."
        else:
            violations = []
            if collision_detected:
                violations.append(f"{len(all_collision_details)} collision(s)")
            if joint_limit_violated:
                violations.append(f"{len(all_joint_limit_details)} joint limit violation(s)")
            if velocity_limit_violated:
                violations.append(f"{len(all_velocity_limit_details)} velocity limit violation(s)")
            if torque_limit_exceeded:
                violations.append(f"{len(all_torque_limit_details)} torque limit violation(s)")
            reason = f"Safety violations detected: {', '.join(violations)}"

        return SafetyCheckResult(
            is_safe=is_safe,
            reason=reason,
            collision_detected=collision_detected,
            joint_limit_violated=joint_limit_violated,
            torque_limit_exceeded=torque_limit_exceeded,
            collision_details=all_collision_details,
            joint_limit_details=all_joint_limit_details,
            torque_limit_details=all_torque_limit_details,
            simulation_steps=steps,
            final_qpos=self.data.qpos[: self.nq].tolist(),
            max_torque_observed=max_torque_observed,
        )

    def get_semantic_error_message(self) -> str:
        """Get human-readable error message using semantic translator."""
        if not self.translator or not self.violations:
            return "No violations to report."
        return self.translator.format_error_response(self.violations)

    def add_environment_object(
        self,
        obj_type: str,
        position: list[float],
        size: list[float],
        name: str = "env_obj",
    ) -> None:
        """
        Add a simple environment object to the simulation.

        Args:
            obj_type: Type of object ('box', 'sphere', 'cylinder')
            position: [x, y, z] position
            size: Size parameters [x, y, z] for box, [radius] for sphere
            name: Object name
        """
        # Note: This would require modifying the MJCF model
        # For now, this is a placeholder for future implementation
        raise NotImplementedError(
            "Dynamic object addition not yet implemented. "
            "Please define environment objects in the MJCF file."
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "nq": self.nq,
            "nv": self.nv,
            "nu": self.nu,
            "joint_names": self.joint_names,
            "geom_names": self.geom_names,
            "timestep": float(self.model.opt.timestep),
            "gravity": self.model.opt.gravity.tolist(),
            "policy": {
                "joint_limits": self.policy.joint_limits,
                "torque_limits": self.policy.torque_limits,
                "safety_margin": self.policy.safety_margin,
            },
        }
