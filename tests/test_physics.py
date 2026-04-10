"""
Unit tests for PhysicsSandbox class.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mjlab_mcp_server.physics import PhysicsSandbox, SafetyCheckResult, SafetyPolicy

# Resolve menagerie path relative to this test file:
# tests/test_physics.py -> mjlab-mcp-server -> e-urdf -> mujoco_menagerie
MENAGERIE_PATH = Path(__file__).resolve().parent.parent.parent / "mujoco_menagerie"
UR5E_MODEL_PATH = MENAGERIE_PATH / "universal_robots_ur5e" / "ur5e.xml"


class TestSafetyPolicy:
    """Test SafetyPolicy dataclass."""

    def test_default_policy(self):
        """Test default policy creation."""
        policy = SafetyPolicy()
        assert policy.safety_margin == 0.05
        assert policy.environment_collision_enabled is True
        assert policy.self_collision_enabled is True

    def test_policy_with_values(self):
        """Test policy with custom values."""
        policy = SafetyPolicy(
            joint_names=["j1", "j2"],
            joint_limits={"j1": (-1.0, 1.0)},
            torque_limits={"j1": 50.0},
            safety_margin=0.1,
        )
        assert policy.safety_margin == 0.1
        assert "j1" in policy.joint_limits
        assert policy.joint_limits["j1"] == (-1.0, 1.0)


class TestPhysicsSandbox:
    """Test PhysicsSandbox with UR5e model."""

    @pytest.fixture(scope="class")
    def sandbox(self):
        """Create sandbox with UR5e model."""
        if not UR5E_MODEL_PATH.exists():
            pytest.skip(f"UR5e model not found at {UR5E_MODEL_PATH}")

        return PhysicsSandbox(str(UR5E_MODEL_PATH))

    def test_model_loading(self, sandbox):
        """Test that model loads correctly."""
        assert sandbox.nq == 6  # UR5e has 6 joints
        assert sandbox.nv == 6
        assert len(sandbox.joint_names) == 6

    def test_joint_names(self, sandbox):
        """Test joint name extraction."""
        expected_joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        for joint in expected_joints:
            assert joint in sandbox.joint_names

    def test_reset(self, sandbox):
        """Test simulation reset."""
        # Set some position
        sandbox.set_joint_positions([0.5] * 6)
        # Reset
        sandbox.reset()
        # Check reset to default
        assert np.allclose(sandbox.data.qpos, 0)

    def test_set_joint_positions(self, sandbox):
        """Test setting joint positions."""
        positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        sandbox.set_joint_positions(positions)
        assert np.allclose(sandbox.data.qpos[:6], positions)

    def test_set_joint_positions_wrong_size(self, sandbox):
        """Test error on wrong number of positions."""
        with pytest.raises(ValueError, match="Expected 6 positions"):
            sandbox.set_joint_positions([0.1, 0.2, 0.3])  # Only 3

    def test_simulation_step(self, sandbox):
        """Test physics step."""
        sandbox.reset()
        initial_time = sandbox.data.time
        sandbox.step(10)
        assert sandbox.data.time > initial_time


class TestSafetyCheck:
    """Test safety check simulation."""

    @pytest.fixture(scope="class")
    def sandbox(self):
        """Create sandbox with UR5e model."""
        if not UR5E_MODEL_PATH.exists():
            pytest.skip(f"UR5e model not found at {UR5E_MODEL_PATH}")

        return PhysicsSandbox(str(UR5E_MODEL_PATH))

    def test_safe_trajectory(self, sandbox):
        """Test validation of safe trajectory."""
        current = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        target = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]

        result = sandbox.simulate_safety_check(
            current, target, duration_sec=0.5  # Short for testing
        )

        assert isinstance(result, SafetyCheckResult)
        assert result.simulation_steps > 0

    def test_joint_limit_pre_check(self, sandbox):
        """Test pre-check for joint limit violations."""
        current = [0.0] * 6
        # Target exceeds 2*pi limit
        target = [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]

        result = sandbox.simulate_safety_check(current, target, duration_sec=0.1)

        assert not result.is_safe
        assert result.joint_limit_violated
        assert "wrist_3_joint" in str(result.joint_limit_details)

    def test_result_to_dict(self, sandbox):
        """Test result serialization."""
        current = [0.0] * 6
        target = [0.1] * 6

        result = sandbox.simulate_safety_check(current, target, duration_sec=0.1)
        d = result.to_dict()

        assert "is_safe" in d
        assert "reason" in d
        assert "collision_details" in d


class TestModelInfo:
    """Test model information extraction."""

    def test_get_model_info(self):
        """Test model info extraction."""
        if not UR5E_MODEL_PATH.exists():
            pytest.skip(f"UR5e model not found at {UR5E_MODEL_PATH}")

        sandbox = PhysicsSandbox(str(UR5E_MODEL_PATH))
        info = sandbox.get_model_info()

        assert "model_path" in info
        assert "nq" in info
        assert "nv" in info
        assert "joint_names" in info
        assert "policy" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
