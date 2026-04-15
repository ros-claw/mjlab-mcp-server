"""
Advanced MuJoCo Tools for MJLab MCP Server.

Provides advanced robotics capabilities:
- Inverse Kinematics (IK)
- Contact force analysis
- Sensor simulation (IMU, force/torque)
- Grasp pose sampling
- Domain randomization
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from .physics import PhysicsSandbox


@dataclass
class IKSolution:
    """Result of inverse kinematics computation."""

    success: bool
    joint_angles: list[float]
    error_pos: float
    error_rot: float
    iterations: int
    message: str


@dataclass
class ContactInfo:
    """Contact force information."""

    body1: str
    body2: str
    contact_point: list[float]
    contact_normal: list[float]
    penetration_depth: float
    force_normal: float
    force_tangent: list[float]
    torque: list[float]


@dataclass
class SensorData:
    """Simulated sensor readings."""

    accelerometer: list[float] | None = None
    gyroscope: list[float] | None = None
    magnetometer: list[float] | None = None
    force: list[float] | None = None
    torque: list[float] | None = None
    joint_positions: list[float] | None = None
    joint_velocities: list[float] | None = None
    joint_torques: list[float] | None = None


@dataclass
class GraspPose:
    """Sampled grasp pose candidate."""

    pose: list[float]  # [x, y, z, qx, qy, qz, qw]
    joint_angles: list[float]
    quality_score: float
    approach_direction: list[float]


class MuJoCoAdvancedTools:
    """
    Advanced MuJoCo-based tools for robot manipulation.
    """

    def __init__(self, sandbox: "PhysicsSandbox"):
        self.sandbox = sandbox
        self.model = sandbox.model
        self.data = sandbox.data

    # ==========================================================================
    # Inverse Kinematics
    # ==========================================================================

    def solve_ik(
        self,
        target_pos: list[float],
        target_quat: list[float] | None = None,
        body_name: str | None = None,
        joint_names: list[str] | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> IKSolution:
        """
        Solve inverse kinematics for target end-effector pose.

        Args:
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [qx, qy, qz, qw] (optional)
            body_name: End-effector body name (default: last body)
            joint_names: Joint names to control (default: all joints)
            max_iterations: Maximum optimization iterations
            tolerance: Position error tolerance

        Returns:
            IKSolution with joint angles and success status
        """
        # Get body ID
        if body_name is None:
            # Use last body as end-effector
            body_id = self.model.nbody - 1
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
        else:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )

        # Get joint indices to control and their DOF addresses
        if joint_names is None:
            # Use all actuated joints (assume first nu joints control first nu DOF)
            joint_ids = list(range(self.model.nu))
            dof_ids = list(range(self.model.nu))
        else:
            joint_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ]
            # Map joint IDs to their DOF addresses (handles joint qpos/dof mapping)
            dof_ids = [
                self.model.jnt_dofadr[jid] if jid < self.model.njnt else i
                for i, jid in enumerate(joint_ids)
            ]

        # Save current state
        qpos_init = self.data.qpos.copy()

        # Use MuJoCo's built-in IK (via optimization)
        target_pos = np.array(target_pos)

        # Simple Jacobian-based IK
        qpos = self.data.qpos.copy()
        best_error = float("inf")
        best_qpos = qpos.copy()

        for iteration in range(max_iterations):
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)

            # Get current end-effector position
            current_pos = self.data.xpos[body_id].copy()

            # Compute position error
            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)

            if error_norm < best_error:
                best_error = error_norm
                best_qpos = qpos.copy()

            if error_norm < tolerance:
                # Success! Return joint angles at controlled DOF positions
                solution_qpos = [best_qpos[did] for did in dof_ids]
                return IKSolution(
                    success=True,
                    joint_angles=solution_qpos,
                    error_pos=error_norm,
                    error_rot=0.0,
                    iterations=iteration + 1,
                    message=f"IK converged in {iteration + 1} iterations",
                )

            # Compute Jacobian (position part only for now)
            jac_pos = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac_pos, None, body_id)

            # Use only controlled DOFs (columns corresponding to our joints)
            jac_reduced = jac_pos[:, dof_ids]

            # Damped least squares
            damping = 0.1
            jac_jac_T = jac_reduced @ jac_reduced.T
            damping_matrix = damping**2 * np.eye(3)
            delta_q = jac_reduced.T @ np.linalg.solve(
                jac_jac_T + damping_matrix, pos_error
            )

            # Update joint positions at correct DOF indices
            for i, did in enumerate(dof_ids):
                qpos[did] += delta_q[i] * 0.5  # Step size

            # Apply to simulation
            self.data.qpos[:] = qpos

        # Restore original state
        self.data.qpos[:] = qpos_init
        mujoco.mj_forward(self.model, self.data)

        # Return best solution found (extract values at controlled DOF positions)
        solution_qpos = [best_qpos[did] for did in dof_ids]
        return IKSolution(
            success=best_error < tolerance * 10,  # Looser tolerance
            joint_angles=solution_qpos,
            error_pos=best_error,
            error_rot=0.0,
            iterations=max_iterations,
            message=f"IK {'converged' if best_error < tolerance * 10 else 'did not converge'} "
            f"(best error: {best_error:.6f})",
        )

    # ==========================================================================
    # Contact Force Analysis
    # ==========================================================================

    def get_contact_forces(self) -> list[ContactInfo]:
        """
        Get detailed contact force information.

        Returns:
            List of ContactInfo for all active contacts
        """
        contacts = []

        # Update contact information
        mujoco.mj_collision(self.model, self.data)

        for i in range(self.data.ncon):
            con = self.data.contact[i]

            # Get body names
            body1_id = self.model.geom_bodyid[con.geom1]
            body2_id = self.model.geom_bodyid[con.geom2]

            body1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            # Get contact force (requires contact force computation)
            contact_force = np.zeros(6)  # [force_x, force_y, force_z, torque_x, torque_y, torque_z]
            if i < self.data.ncon:
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)

            contacts.append(
                ContactInfo(
                    body1=body1,
                    body2=body2,
                    contact_point=con.pos.tolist(),
                    contact_normal=con.frame[:3].tolist(),
                    penetration_depth=float(con.dist),
                    force_normal=float(np.dot(contact_force[:3], con.frame[:3])),
                    force_tangent=contact_force[:3].tolist(),
                    torque=contact_force[3:].tolist(),
                )
            )

        return contacts

    def analyze_contact_stability(self) -> dict:
        """
        Analyze contact stability for grasp/manipulation.

        Returns:
            Dictionary with stability metrics
        """
        contacts = self.get_contact_forces()

        if not contacts:
            return {
                "stable": False,
                "num_contacts": 0,
                "total_normal_force": 0.0,
                "force_closure": False,
            }

        # Sum forces
        total_normal_force = sum(c.force_normal for c in contacts)

        # Check force closure (simplified: enough contacts and force)
        force_closure = len(contacts) >= 2 and total_normal_force > 1.0

        return {
            "stable": force_closure,
            "num_contacts": len(contacts),
            "total_normal_force": total_normal_force,
            "force_closure": force_closure,
            "contacts": [
                {
                    "body1": c.body1,
                    "body2": c.body2,
                    "force": c.force_normal,
                }
                for c in contacts
            ],
        }

    # ==========================================================================
    # Sensor Simulation
    # ==========================================================================

    def get_sensor_data(self, sensor_types: list[str] | None = None) -> SensorData:
        """
        Get simulated sensor readings.

        Args:
            sensor_types: List of sensor types to read
                Options: "imu", "force", "joint"

        Returns:
            SensorData with requested readings
        """
        if sensor_types is None:
            sensor_types = ["imu", "joint"]

        data = SensorData()

        # Ensure physics is up to date
        mujoco.mj_forward(self.model, self.data)

        # Root body for IMU/force sensors (typically body 0 / world or base)
        root_body_id = 0

        if "imu" in sensor_types:
            # Accelerometer (linear acceleration)
            data.accelerometer = self.data.cacc[root_body_id * 6 : root_body_id * 6 + 3].tolist()

            # Gyroscope (angular velocity)
            data.gyroscope = self.data.cvel[root_body_id * 6 + 3 : root_body_id * 6 + 6].tolist()

            # Magnetometer (simplified - just return a reference direction)
            data.magnetometer = [1.0, 0.0, 0.0]  # North

        if "force" in sensor_types:
            # Get total external force on root body
            data.force = self.data.cfrc_ext[root_body_id * 6 : root_body_id * 6 + 3].tolist()
            data.torque = self.data.cfrc_ext[root_body_id * 6 + 3 : root_body_id * 6 + 6].tolist()

        if "joint" in sensor_types:
            # Joint positions
            data.joint_positions = self.data.qpos[: self.model.nq].tolist()
            data.joint_velocities = self.data.qvel[: self.model.nv].tolist()
            data.joint_torques = self.data.qfrc_actuator[: self.model.nu].tolist()

        return data

    # ==========================================================================
    # Grasp Pose Sampling
    # ==========================================================================

    def sample_grasp_poses(
        self,
        object_pos: list[float],
        num_samples: int = 10,
        approach_radius: float = 0.2,
    ) -> list[GraspPose]:
        """
        Sample candidate grasp poses around an object.

        Args:
            object_pos: Object center position [x, y, z]
            num_samples: Number of grasp poses to sample
            approach_radius: Distance from object to sample poses

        Returns:
            List of GraspPose candidates sorted by quality
        """
        object_pos = np.array(object_pos)
        poses = []

        for i in range(num_samples):
            # Sample approach direction (hemisphere above object)
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi / 2)  # Upper hemisphere

            # Spherical to cartesian
            dx = approach_radius * np.sin(phi) * np.cos(theta)
            dy = approach_radius * np.sin(phi) * np.sin(theta)
            dz = approach_radius * np.cos(phi)

            grasp_pos = object_pos + np.array([dx, dy, dz])

            # Compute orientation (approach direction is -Z of gripper)
            approach_dir = -(grasp_pos - object_pos)
            approach_dir = approach_dir / np.linalg.norm(approach_dir)

            # Create rotation matrix with Z pointing along approach
            z_axis = approach_dir
            x_axis = np.cross([0, 0, 1], z_axis)
            if np.linalg.norm(x_axis) < 0.001:
                x_axis = np.array([1, 0, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
            quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]

            pose = [
                grasp_pos[0],
                grasp_pos[1],
                grasp_pos[2],
                quat[0],
                quat[1],
                quat[2],
                quat[3],
            ]

            # Compute quality score (simplified)
            # Prefer approaches from above and farther from obstacles
            quality = 1.0 - phi / (np.pi / 2)  # Higher = more vertical

            poses.append(
                GraspPose(
                    pose=pose,
                    joint_angles=[],  # Would be filled by IK
                    quality_score=quality,
                    approach_direction=approach_dir.tolist(),
                )
            )

        # Sort by quality
        poses.sort(key=lambda p: p.quality_score, reverse=True)

        return poses

    # ==========================================================================
    # Domain Randomization
    # ==========================================================================

    def apply_domain_randomization(
        self,
        body_mass_range: tuple[float, float] = (0.9, 1.1),
        joint_friction_range: tuple[float, float] = (0.8, 1.2),
        gravity_range: tuple[float, float] = (9.0, 10.0),
        seed: int | None = None,
    ) -> dict:
        """
        Apply domain randomization for sim-to-real transfer.

        Randomizes physical parameters within specified ranges.

        Args:
            body_mass_range: Multiplier range for body masses
            joint_friction_range: Multiplier range for joint friction
            gravity_range: Range for gravity magnitude
            seed: Random seed for reproducibility

        Returns:
            Dictionary with applied randomization values
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        changes = {
            "body_masses": [],
            "joint_frictions": [],
            "gravity": None,
        }

        # Randomize body masses
        for i in range(1, self.model.nbody):  # Skip world body
            mass_mult = random.uniform(*body_mass_range)
            original_mass = self.model.body_mass[i]
            self.model.body_mass[i] = original_mass * mass_mult
            changes["body_masses"].append(
                {
                    "body_id": i,
                    "original": float(original_mass),
                    "new": float(self.model.body_mass[i]),
                    "multiplier": mass_mult,
                }
            )

        # Randomize joint friction
        for i in range(self.model.njnt):
            friction_mult = random.uniform(*joint_friction_range)
            # Note: MuJoCo doesn't directly expose joint friction, it's in dof_frictionloss
            if i < self.model.nv:
                original_friction = self.model.dof_frictionloss[i]
                self.model.dof_frictionloss[i] = original_friction * friction_mult
                changes["joint_frictions"].append(
                    {
                        "joint_id": i,
                        "original": float(original_friction),
                        "new": float(self.model.dof_frictionloss[i]),
                        "multiplier": friction_mult,
                    }
                )

        # Randomize gravity
        gravity_mag = random.uniform(*gravity_range)
        self.model.opt.gravity[2] = -gravity_mag
        changes["gravity"] = float(gravity_mag)

        # Forward to apply changes
        mujoco.mj_forward(self.model, self.data)

        return changes

    def reset_domain(self) -> None:
        """Reset domain randomization to original values (requires model reload)."""
        # This is a placeholder - in practice, you'd reload the model
        # or store original values to restore
        pass
