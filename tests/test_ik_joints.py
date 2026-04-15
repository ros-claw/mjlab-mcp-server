#!/usr/bin/env python3
"""Test IK with custom joint names to verify joint mapping fix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mjlab_mcp_server.physics import PhysicsSandbox
from mjlab_mcp_server.advanced_tools import MuJoCoAdvancedTools

MENAGERIE_PATH = Path(__file__).resolve().parent.parent.parent / "mujoco_menagerie"
UR5E_MODEL_PATH = MENAGERIE_PATH / "universal_robots_ur5e" / "ur5e.xml"


def test_ik_with_custom_joints():
    """Test IK with specific joint subset."""
    if not UR5E_MODEL_PATH.exists():
        print(f"SKIP: UR5e model not found")
        return
    
    sandbox = PhysicsSandbox(str(UR5E_MODEL_PATH))
    tools = MuJoCoAdvancedTools(sandbox)
    
    print("=" * 60)
    print("TEST: IK with custom joint names")
    print("=" * 60)
    
    # Test with specific joints (e.g., only first 3 joints - shoulder + elbow)
    target_pos = [0.4, 0.2, 0.4]  # reachable position
    
    # Test 1: default (all joints)
    print("\n1. IK with all joints (default)...")
    result1 = tools.solve_ik(
        target_pos=target_pos,
        body_name="wrist_3_link",
        max_iterations=100,
        tolerance=1e-3,
    )
    print(f"   Success: {result1.success}, Error: {result1.error_pos:.6f}, Joints: {len(result1.joint_angles)}")
    
    # Test 2: only first 3 joints
    print("\n2. IK with first 3 joints only...")
    result2 = tools.solve_ik(
        target_pos=target_pos,
        body_name="wrist_3_link",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
        max_iterations=100,
        tolerance=1e-3,
    )
    print(f"   Success: {result2.success}, Error: {result2.error_pos:.6f}, Joints: {len(result2.joint_angles)}")
    
    # Verify the number of returned joints matches the requested count
    assert len(result2.joint_angles) == 3, f"Expected 3 joints, got {len(result2.joint_angles)}"
    print(f"   ✓ Correct joint count: {len(result2.joint_angles)}")
    
    # Test 3: specific non-consecutive joints (1, 2, 4)
    print("\n3. IK with non-consecutive joints...")
    result3 = tools.solve_ik(
        target_pos=target_pos,
        body_name="wrist_3_link",
        joint_names=["shoulder_lift_joint", "elbow_joint", "wrist_2_joint"],
        max_iterations=100,
        tolerance=1e-3,
    )
    print(f"   Success: {result3.success}, Error: {result3.error_pos:.6f}, Joints: {len(result3.joint_angles)}")
    assert len(result3.joint_angles) == 3, f"Expected 3 joints, got {len(result3.joint_angles)}"
    print(f"   ✓ Correct joint count: {len(result3.joint_angles)}")
    
    print("\n" + "=" * 60)
    print("✓ All IK joint mapping tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_ik_with_custom_joints()
