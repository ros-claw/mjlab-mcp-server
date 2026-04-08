"""
MJLab MCP Server - Physics Sandbox Firewall for Robot Safety Validation.

This module provides an MCP server that exposes MuJoCo physics simulation
capabilities to LLM agents for safety validation before executing real robot commands.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .physics import PhysicsSandbox, SafetyCheckResult

# Initialize MCP Server
mcp = FastMCP("mjlab-e-urdf-firewall")

# Global sandbox instance (initialized lazily)
_sandbox: PhysicsSandbox | None = None
_default_model_path: str | None = None


def get_sandbox() -> PhysicsSandbox:
    """Get or initialize the physics sandbox."""
    global _sandbox, _default_model_path

    if _sandbox is None:
        # Try to find a default model
        model_path = _default_model_path or os.environ.get("MUJOCO_MODEL_PATH")

        if model_path is None:
            # Try to find UR5e in menagerie
            menagerie_path = Path("/root/workspace/rosclaw/e-urdf/mujoco_menagerie")
            if menagerie_path.exists():
                ur5e_path = menagerie_path / "universal_robots_ur5e" / "ur5e.xml"
                if ur5e_path.exists():
                    model_path = str(ur5e_path)

        if model_path is None:
            raise RuntimeError(
                "No model path specified. Set MUJOCO_MODEL_PATH environment variable "
                "or call load_model() first."
            )

        policy_path = os.environ.get("SAFETY_POLICY_PATH")
        _sandbox = PhysicsSandbox(model_path, policy_path)

    return _sandbox


@mcp.tool()
def load_model(model_path: str, policy_path: str | None = None) -> str:
    """
    Load a MuJoCo model for safety validation.

    Args:
        model_path: Path to the MJCF or URDF model file
        policy_path: Optional path to safety policy YAML file

    Returns:
        Success message with model information
    """
    global _sandbox, _default_model_path

    try:
        _default_model_path = model_path
        _sandbox = PhysicsSandbox(model_path, policy_path)

        info = _sandbox.get_model_info()
        return (
            f"✅ Model loaded successfully!\n"
            f"  Path: {info['model_path']}\n"
            f"  Joints: {info['nq']} (DOF: nq={info['nq']}, nv={info['nv']}, nu={info['nu']})\n"
            f"  Joint names: {', '.join(info['joint_names'][:5])}"
            f"{'...' if len(info['joint_names']) > 5 else ''}\n"
            f"  Timestep: {info['timestep'] * 1000:.2f} ms\n"
            f"  Safety margin: {info['policy']['safety_margin'] * 100:.0f}%"
        )
    except Exception as e:
        return f"❌ Failed to load model: {str(e)}"


@mcp.tool()
def verify_action_safety(
    current_joints: list[float],
    target_joints: list[float],
    duration_sec: float = 2.0,
    control_mode: str = "position",
) -> str:
    """
    CRITICAL SAFETY TOOL: Verify if a robot action is safe before execution.

    Before sending ANY physical movement commands to the real robot, the LLM MUST use this tool
    to simulate the trajectory. It returns whether the planned movement will cause collisions
    or exceed physical limits.

    Args:
        current_joints: List of current joint angles (radians)
        target_joints: List of desired target joint angles (radians)
        duration_sec: Simulation duration in seconds (default: 2.0)
        control_mode: Control mode - 'position' or 'velocity' (default: 'position')

    Returns:
        Detailed safety assessment result
    """
    try:
        sandbox = get_sandbox()
        result = sandbox.simulate_safety_check(
            current_joints, target_joints, duration_sec, control_mode
        )

        if result.is_safe:
            return (
                f"✅ [SAFE] Physics simulation passed!\n\n"
                f"No collisions or limit violations detected over {duration_sec}s simulation.\n"
                f"Simulation steps: {result.simulation_steps}\n"
                f"Max torque observed: {result.max_torque_observed:.2f} Nm\n\n"
                f"You may proceed with execution on the real robot."
            )
        else:
            response = f"❌ [DANGER] Physical simulation failed!\n\n"
            response += f"Reason: {result.reason}\n"
            response += f"Simulation steps before failure: {result.simulation_steps}\n\n"

            if result.collision_details:
                response += "🔴 COLLISIONS DETECTED:\n"
                for detail in result.collision_details[:5]:  # Limit to first 5
                    response += f"  - {detail}\n"
                if len(result.collision_details) > 5:
                    response += f"  ... and {len(result.collision_details) - 5} more\n"
                response += "\n"

            if result.joint_limit_details:
                response += "🟡 JOINT LIMIT VIOLATIONS:\n"
                for detail in result.joint_limit_details[:5]:
                    response += f"  - {detail}\n"
                if len(result.joint_limit_details) > 5:
                    response += f"  ... and {len(result.joint_limit_details) - 5} more\n"
                response += "\n"

            if result.torque_limit_details:
                response += "🟠 TORQUE LIMIT VIOLATIONS:\n"
                for detail in result.torque_limit_details[:5]:
                    response += f"  - {detail}\n"
                if len(result.torque_limit_details) > 5:
                    response += f"  ... and {len(result.torque_limit_details) - 5} more\n"
                response += "\n"

            response += (
                f"⚠️  ACTION BLOCKED - DO NOT EXECUTE ON REAL HARDWARE!\n"
                f"Please replan your trajectory to avoid the above violations."
            )
            return response

    except Exception as e:
        return f"⚠️ [ERROR] Sandbox engine failed to compute: {str(e)}"


@mcp.tool()
def get_model_info() -> str:
    """
    Get information about the currently loaded model.

    Returns:
        Model details including joint names, limits, and simulation parameters
    """
    try:
        sandbox = get_sandbox()
        info = sandbox.get_model_info()

        response = f"📊 Model Information\n"
        response += f"{'=' * 50}\n\n"
        response += f"Path: {info['model_path']}\n"
        response += f"Dimensions: nq={info['nq']}, nv={info['nv']}, nu={info['nu']}\n"
        response += f"Timestep: {info['timestep'] * 1000:.2f} ms\n"
        response += f"Gravity: {info['gravity']}\n\n"

        response += f"Joint Names ({len(info['joint_names'])}):\n"
        for i, name in enumerate(info['joint_names']):
            response += f"  {i}: {name}\n"

        if info['policy']['joint_limits']:
            response += f"\nJoint Limits:\n"
            for name, (min_val, max_val) in info['policy']['joint_limits'].items():
                response += f"  {name}: [{min_val:.3f}, {max_val:.3f}] rad\n"

        if info['policy']['torque_limits']:
            response += f"\nTorque Limits:\n"
            for name, limit in info['policy']['torque_limits'].items():
                response += f"  {name}: {limit:.2f} Nm\n"

        response += f"\nSafety Margin: {info['policy']['safety_margin'] * 100:.0f}%"

        return response
    except Exception as e:
        return f"❌ Error getting model info: {str(e)}"


@mcp.tool()
def list_available_models() -> str:
    """
    List available robot models from the MuJoCo Menagerie.

    Returns:
        List of available models that can be loaded
    """
    menagerie_path = Path("/root/workspace/rosclaw/e-urdf/mujoco_menagerie")

    if not menagerie_path.exists():
        return "❌ MuJoCo Menagerie not found at expected path."

    models = []
    for item in menagerie_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Look for XML files
            xml_files = list(item.glob("*.xml"))
            if xml_files:
                models.append((item.name, xml_files[0].name))

    if not models:
        return "No models found in Menagerie."

    response = f"📁 Available Models from MuJoCo Menagerie ({len(models)}):\n"
    response += f"{'=' * 50}\n\n"

    for name, xml_file in sorted(models):
        response += f"  • {name}\n"
        response += f"    XML: {xml_file}\n"

    response += f"\n💡 To load a model, use:\n"
    response += f"   load_model with path: {menagerie_path}/[model_name]/[xml_file]"

    return response


@mcp.resource("safety://status")
def get_safety_status() -> str:
    """Get current safety validation system status."""
    try:
        sandbox = get_sandbox()
        info = sandbox.get_model_info()

        return f"""{{
    "status": "active",
    "model_loaded": true,
    "model_path": "{info['model_path']}",
    "joints": {info['nq']},
    "safety_margin": {info['policy']['safety_margin']},
    "ready_for_validation": true
}}"""
    except Exception:
        return """{
    "status": "inactive",
    "model_loaded": false,
    "error": "No model loaded. Call load_model() first."
}"""


@mcp.resource("safety://limits")
def get_safety_limits() -> str:
    """Get safety limits for the current model."""
    try:
        sandbox = get_sandbox()
        info = sandbox.get_model_info()

        import json

        return json.dumps(
            {
                "joint_limits": info["policy"]["joint_limits"],
                "torque_limits": info["policy"]["torque_limits"],
                "safety_margin": info["policy"]["safety_margin"],
            },
            indent=2,
        )
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'


def main() -> None:
    """Main entry point for the MCP server."""
    print("=" * 60, file=sys.stderr)
    print("MJLab MCP Server - Physics Sandbox Firewall", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("", file=sys.stderr)
    print("Available tools:", file=sys.stderr)
    print("  - load_model: Load a MuJoCo model", file=sys.stderr)
    print("  - verify_action_safety: Validate robot action safety", file=sys.stderr)
    print("  - get_model_info: Get model information", file=sys.stderr)
    print("  - list_available_models: List available models", file=sys.stderr)
    print("", file=sys.stderr)
    print("Starting MCP server (stdio mode)...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    mcp.run()


if __name__ == "__main__":
    main()
