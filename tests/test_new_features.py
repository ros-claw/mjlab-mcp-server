#!/usr/bin/env python3
"""Functional tests for rendering and advanced MuJoCo tools."""

import sys
from pathlib import Path
import json
import base64

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mjlab_mcp_server.physics import PhysicsSandbox
from mjlab_mcp_server.renderer import MuJoCoRenderer
from mjlab_mcp_server.advanced_tools import MuJoCoAdvancedTools

MENAGERIE_PATH = Path(__file__).resolve().parent.parent.parent / "mujoco_menagerie"
UR5E_MODEL_PATH = MENAGERIE_PATH / "universal_robots_ur5e" / "ur5e.xml"


def test_renderer():
    print("=" * 60)
    print("TESTING RENDERER")
    print("=" * 60)
    
    if not UR5E_MODEL_PATH.exists():
        print(f"SKIP: UR5e model not found at {UR5E_MODEL_PATH}")
        return False
    
    sandbox = PhysicsSandbox(str(UR5E_MODEL_PATH))
    renderer = MuJoCoRenderer(sandbox)
    
    # Test 1: render_current_state
    print("\n1. render_current_state...")
    try:
        result = renderer.render_current_state(width=320, height=240)
        assert "image_base64" in result, "Missing image_base64"
        assert "format" in result, "Missing format"
        assert result["format"] == "png", "Format should be png"
        assert result["width"] == 320, "Width mismatch"
        assert result["height"] == 240, "Height mismatch"
        # Verify base64 is valid
        img_data = base64.b64decode(result["image_base64"])
        assert len(img_data) > 0, "Empty image data"
        print(f"   ✓ OK - image size: {len(img_data)} bytes")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    
    # Test 2: render_trajectory_preview
    print("\n2. render_trajectory_preview...")
    try:
        trajectory = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.1, 0.1, -0.1, 0.1, -0.1],
            [0.2, -0.2, 0.2, -0.2, 0.2, -0.2],
        ]
        result = renderer.render_trajectory_preview(trajectory, width=320, height=240, frames=3)
        assert "frames" in result, "Missing frames"
        assert len(result["frames"]) == 3, f"Expected 3 frames, got {len(result['frames'])}"
        print(f"   ✓ OK - {len(result['frames'])} frames generated")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    
    # Test 3: render_collision_debug
    print("\n3. render_collision_debug...")
    try:
        result = renderer.render_collision_debug(width=320, height=240)
        assert "image_base64" in result, "Missing image_base64"
        assert "contacts_detected" in result, "Missing contacts_detected"
        print(f"   ✓ OK - contacts detected: {result['contacts_detected']}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    
    # Test 4: render_depth_map
    print("\n4. render_depth_map...")
    try:
        result = renderer.render_depth_map(width=320, height=240, max_depth=5.0)
        assert "image_base64" in result, "Missing image_base64"
        assert "min_depth_m" in result, "Missing min_depth_m"
        assert "max_depth_m" in result, "Missing max_depth_m"
        print(f"   ✓ OK - depth range: [{result['min_depth_m']:.3f}, {result['max_depth_m']:.3f}]")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    
    # Test 5: save_screenshot
    print("\n5. save_screenshot...")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        result = renderer.save_screenshot(tmp_path, width=640, height=480)
        assert result["saved"] is True, "Screenshot not saved"
        assert Path(result["filepath"]).exists(), "File not created"
        print(f"   ✓ OK - saved to {result['filepath']} ({result['size_bytes']} bytes)")
        Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False
    
    return True


def test_advanced_tools():
    print("\n" + "=" * 60)
    print("TESTING ADVANCED TOOLS")
    print("=" * 60)
    
    if not UR5E_MODEL_PATH.exists():
        print(f"SKIP: UR5e model not found at {UR5E_MODEL_PATH}")
        return False
    
    sandbox = PhysicsSandbox(str(UR5E_MODEL_PATH))
    tools = MuJoCoAdvancedTools(sandbox)
    
    # Test 1: solve_ik
    print("\n1. solve_ik...")
    try:
        # Target near the default position of the end effector
        result = tools.solve_ik(
            target_pos=[0.4, 0.2, 0.5],
            body_name="wrist_3_link",
            max_iterations=100,
            tolerance=1e-3,
        )
        assert result is not None, "IK returned None"
        print(f"   ✓ OK - success={result.success}, error={result.error_pos:.6f}, iters={result.iterations}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: get_contact_forces
    print("\n2. get_contact_forces...")
    try:
        contacts = tools.get_contact_forces()
        assert isinstance(contacts, list), "Should return list"
        print(f"   ✓ OK - {len(contacts)} contacts")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: analyze_contact_stability
    print("\n3. analyze_contact_stability...")
    try:
        result = tools.analyze_contact_stability()
        assert "stable" in result, "Missing stable key"
        assert "num_contacts" in result, "Missing num_contacts"
        print(f"   ✓ OK - stable={result['stable']}, contacts={result['num_contacts']}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: get_sensor_data
    print("\n4. get_sensor_data...")
    try:
        data = tools.get_sensor_data(sensor_types=["imu", "force", "joint"])
        assert data is not None, "Sensor data is None"
        assert data.joint_positions is not None, "Missing joint_positions"
        assert len(data.joint_positions) == sandbox.nq, "Joint positions length mismatch"
        print(f"   ✓ OK - IMU: {data.accelerometer is not None}, force: {data.force is not None}, joints: {len(data.joint_positions)}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: sample_grasp_poses
    print("\n5. sample_grasp_poses...")
    try:
        poses = tools.sample_grasp_poses(
            object_pos=[0.5, 0.0, 0.3],
            num_samples=5,
            approach_radius=0.2,
        )
        assert len(poses) == 5, f"Expected 5 poses, got {len(poses)}"
        assert poses[0].quality_score >= poses[-1].quality_score, "Poses not sorted by quality"
        print(f"   ✓ OK - {len(poses)} poses, best quality={poses[0].quality_score:.3f}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: apply_domain_randomization
    print("\n6. apply_domain_randomization...")
    try:
        result = tools.apply_domain_randomization(
            body_mass_range=(0.9, 1.1),
            joint_friction_range=(0.8, 1.2),
            gravity_range=(9.0, 10.0),
            seed=42,
        )
        assert "body_masses" in result, "Missing body_masses"
        assert "joint_frictions" in result, "Missing joint_frictions"
        assert "gravity" in result, "Missing gravity"
        assert len(result["body_masses"]) > 0, "No body masses randomized"
        print(f"   ✓ OK - randomized {len(result['body_masses'])} bodies, gravity={result['gravity']:.3f}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_server_tools():
    print("\n" + "=" * 60)
    print("TESTING SERVER MCP TOOLS INTEGRATION")
    print("=" * 60)
    
    # Just test that the server module imports correctly and tools are registered
    try:
        from mjlab_mcp_server.server import mcp, render_current_state, solve_inverse_kinematics
        print("   ✓ OK - server imports successfully")
        
        # Check that mcp has the tools
        tool_names = list(mcp._tool_manager._tools.keys())
        expected_tools = [
            "render_current_state",
            "render_trajectory_preview",
            "render_collision_debug",
            "render_depth_map",
            "save_screenshot",
            "solve_inverse_kinematics",
            "get_contact_forces",
            "analyze_contact_stability",
            "get_sensor_data",
            "sample_grasp_poses",
            "apply_domain_randomization",
        ]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"
        print(f"   ✓ OK - all {len(expected_tools)} new tools registered")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    results = []
    results.append(("Renderer", test_renderer()))
    results.append(("Advanced Tools", test_advanced_tools()))
    results.append(("Server Integration", test_server_tools()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_ok = all(ok for _, ok in results)
    sys.exit(0 if all_ok else 1)
