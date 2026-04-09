"""
MJLab MCP Server - Physics Sandbox Firewall for Robot Safety Validation.

This module provides an MCP server that exposes MuJoCo physics simulation
capabilities to LLM agents for safety validation before executing real robot commands.

Enhanced with e-URDF-Zoo integration for dynamic robot asset loading.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .batch_validator import BatchValidator, TrajectoryCandidate
from .physics import PhysicsSandbox, SafetyCheckResult
from .reality_sync import RealitySync, RealitySyncManager
from .semantic_translator import SemanticTranslator

# Initialize MCP Server
mcp = FastMCP("mjlab-e-urdf-firewall")

# Global sandbox instance (initialized lazily)
_sandbox: PhysicsSandbox | None = None
_default_model_path: str | None = None
_current_e_urdf_config: dict[str, Any] | None = None
_current_robot_id: str | None = None

# Reality sync manager (initialized lazily)
_reality_sync_manager: RealitySyncManager | None = None

# Batch validator (initialized lazily)
_batch_validator: BatchValidator | None = None

# Semantic translator (initialized lazily)
_semantic_translator: SemanticTranslator | None = None

# Default e-URDF-Zoo path: resolve next to the package installation (e.g. ../e-urdf-zoo)
_DEFAULT_E_URDF_ZOO_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "e-urdf-zoo")
E_URDF_ZOO_PATH = Path(os.environ.get("E_URDF_ZOO_PATH", _DEFAULT_E_URDF_ZOO_PATH))
MUJOCO_MENAGERIE_PATH = Path(os.environ.get("MUJOCO_MENAGERIE_PATH", str(E_URDF_ZOO_PATH.parent / "mujoco_menagerie")))


def get_e_urdf_robot_path(robot_id: str) -> Path | None:
    """Get path to robot asset bundle in e-URDF-Zoo."""
    robot_path = E_URDF_ZOO_PATH / "robots" / robot_id
    if robot_path.exists() and (robot_path / "e_urdf.json").exists():
        return robot_path
    return None


def load_e_urdf_config(robot_path: Path) -> dict[str, Any]:
    """Load e_urdf.json configuration."""
    config_path = robot_path / "e_urdf.json"
    with open(config_path, "r") as f:
        return json.load(f)


def get_sandbox() -> PhysicsSandbox:
    """Get or initialize the physics sandbox."""
    global _sandbox, _default_model_path

    if _sandbox is None:
        # Try to find a default model
        model_path = _default_model_path or os.environ.get("MUJOCO_MODEL_PATH")

        if model_path is None:
            # Try to find UR5e in e-URDF-Zoo first, then menagerie
            zoo_path = get_e_urdf_robot_path("universal_robots_ur5e")
            if zoo_path:
                # Auto-load config
                load_e_urdf_config_internal(zoo_path)

                # Resolve actual model path (model.xml may include menagerie)
                actual_model = zoo_path / "model.xml"
                with open(actual_model, "r") as f:
                    content = f.read()
                    if "mujoco_menagerie" in content and "include" in content:
                        if MUJOCO_MENAGERIE_PATH.exists():
                            ur5e_path = MUJOCO_MENAGERIE_PATH / "universal_robots_ur5e" / "ur5e.xml"
                            if ur5e_path.exists():
                                actual_model = ur5e_path

                model_path = str(actual_model)
            else:
                if MUJOCO_MENAGERIE_PATH.exists():
                    ur5e_path = MUJOCO_MENAGERIE_PATH / "universal_robots_ur5e" / "ur5e.xml"
                    if ur5e_path.exists():
                        model_path = str(ur5e_path)

        if model_path is None:
            raise RuntimeError(
                "No model path specified. Set MUJOCO_MODEL_PATH environment variable, "
                "call load_model() or load_embodiment() first."
            )

        policy_path = os.environ.get("SAFETY_POLICY_PATH")
        _sandbox = PhysicsSandbox(
            model_path,
            policy_path,
            e_urdf_config=_current_e_urdf_config,
        )

    return _sandbox


def load_e_urdf_config_internal(robot_path: Path) -> None:
    """Internal function to load e-urdf config."""
    global _current_e_urdf_config, _current_robot_id
    _current_e_urdf_config = load_e_urdf_config(robot_path)
    _current_robot_id = robot_path.name


@mcp.tool()
def load_embodiment(robot_id: str) -> str:
    """
    Load a robot embodiment from the e-URDF-Zoo.

    This is the PREFERRED way to load robots as it includes safety configuration,
    semantic descriptions, and LLM prompts.

    Args:
        robot_id: Robot identifier (e.g., "universal_robots_ur5e", "unitree_g1")

    Returns:
        Success message with embodiment information
    """
    global _sandbox, _default_model_path, _current_e_urdf_config, _current_robot_id

    try:
        # Find robot in zoo
        robot_path = get_e_urdf_robot_path(robot_id)
        if robot_path is None:
            # List available robots
            available = list_available_robots_from_zoo()
            available_str = "\n  - ".join([""] + available) if available else "\n  (none found)"
            return (
                f"❌ Robot '{robot_id}' not found in e-URDF-Zoo.\n\n"
                f"Available robots:{available_str}\n\n"
                f"Zoo path: {E_URDF_ZOO_PATH}"
            )

        # Load e_urdf.json
        _current_e_urdf_config = load_e_urdf_config(robot_path)
        _current_robot_id = robot_id

        # Get model path from config
        model_path = robot_path / "model.xml"

        # Check if model.xml references menagerie
        with open(model_path, "r") as f:
            content = f.read()
            if "mujoco_menagerie" in content and "include" in content:
                # Use actual menagerie model
                if MUJOCO_MENAGERIE_PATH.exists():
                    # Parse the include path
                    if "universal_robots_ur5e" in robot_id:
                        actual_model = MUJOCO_MENAGERIE_PATH / "universal_robots_ur5e" / "ur5e.xml"
                    elif "unitree_g1" in robot_id:
                        actual_model = MUJOCO_MENAGERIE_PATH / "unitree_g1" / "g1.xml"
                    else:
                        # Try to find matching model
                        for item in MUJOCO_MENAGERIE_PATH.iterdir():
                            if item.is_dir() and robot_id.replace("_", "") in item.name.replace("_", ""):
                                xml_files = list(item.glob("*.xml"))
                                if xml_files:
                                    actual_model = xml_files[0]
                                    break
                        else:
                            actual_model = model_path
                else:
                    actual_model = model_path
            else:
                actual_model = model_path

        # Load safety config from e_urdf.json
        firewall_config = _current_e_urdf_config.get("physical_firewall", {})
        safety_margin = firewall_config.get("safety_margins", {}).get("joint_position", 0.05)

        # Load into sandbox with full e-URDF config
        _default_model_path = str(actual_model)
        _sandbox = PhysicsSandbox(
            str(actual_model),
            safety_margin=safety_margin,
            e_urdf_config=_current_e_urdf_config,
        )

        info = _sandbox.get_model_info()

        response = (
            f"✅ Embodiment loaded successfully!\n\n"
            f"🤖 {info['nq']}-DOF {info.get('policy', {}).get('robot_type', 'robot')} loaded\n"
            f"📦 Robot ID: {robot_id}\n"
            f"📁 Path: {robot_path}\n"
            f"🔧 Model: {actual_model}\n\n"
        )

        # Add semantic info
        semantics = _current_e_urdf_config.get("semantics", {})
        if semantics.get("description"):
            response += f"📝 Description: {semantics['description']}\n"
        if semantics.get("affordances"):
            response += f"💪 Capabilities: {', '.join(semantics['affordances'][:5])}\n"

        response += f"\n🛡️  Safety Configuration:\n"
        response += f"   - Validation Level: {firewall_config.get('validation_level', 'standard')}\n"
        response += f"   - Simulation Horizon: {firewall_config.get('max_simulation_horizon_sec', 2.0)}s\n"
        response += f"   - Speed Factor: {firewall_config.get('speed_up_factor', 100)}x\n"
        response += f"   - Safety Margin: {safety_margin * 100:.0f}%\n"

        response += f"\n✨ Prompts available via resources:\n"
        response += f"   - e_urdf://{robot_id}/system_prompt\n"
        response += f"   - e_urdf://{robot_id}/tools_usage\n"

        # Initialize semantic translator with loaded config
        global _semantic_translator
        _semantic_translator = SemanticTranslator(_sandbox, _current_e_urdf_config)

        return response

    except Exception as e:
        return f"❌ Failed to load embodiment: {str(e)}"


def list_available_robots_from_zoo() -> list[str]:
    """List all available robots in e-URDF-Zoo."""
    robots = []
    zoo_robots_path = E_URDF_ZOO_PATH / "robots"
    if zoo_robots_path.exists():
        for item in zoo_robots_path.iterdir():
            if item.is_dir() and (item / "e_urdf.json").exists():
                robots.append(item.name)
    return sorted(robots)


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
def sync_reality(use_ros2: bool = True) -> str:
    """
    Synchronize MuJoCo simulation to match real robot state.

    This ensures safety validation starts from the actual current state
    of the physical robot, making predictions meaningful.

    Args:
        use_ros2: Use ROS 2 to subscribe to /joint_states topic

    Returns:
        Sync status message
    """
    global _reality_sync_manager, _sandbox, _current_robot_id

    try:
        sandbox = get_sandbox()

        if _current_robot_id is None:
            _current_robot_id = "default_robot"

        # Initialize reality sync manager if needed
        if _reality_sync_manager is None:
            _reality_sync_manager = RealitySyncManager()

        # Check if we already have a sync for this robot
        existing_sync = _reality_sync_manager.get_sync(_current_robot_id)
        if existing_sync is None:
            # Create new reality sync
            from .reality_sync import RealitySync

            sync = RealitySync(
                sandbox=sandbox,
                joint_names=sandbox.joint_names,
                use_ros2=use_ros2,
            )
            _reality_sync_manager.register(_current_robot_id, sync)

            return (
                f"✅ Reality Sync initialized for '{_current_robot_id}'\n"
                f"   Subscribed to: /joint_states\n"
                f"   Joints tracked: {len(sandbox.joint_names)}\n"
                f"   ROS 2 mode: {use_ros2}\n\n"
                f"💡 The simulation will now automatically sync before each validation."
            )
        else:
            # Trigger manual sync
            success = existing_sync.sync_simulation_to_reality()
            state_age = existing_sync.get_state_age_ms()

            if success:
                return (
                    f"✅ Reality sync successful!\n"
                    f"   Simulation state updated to match real robot.\n"
                    f"   State age: {state_age:.1f}ms\n\n"
                    f"   Ready for trajectory validation."
                )
            else:
                return (
                    f"⚠️ Reality sync failed.\n"
                    f"   State age: {state_age:.1f}ms\n"
                    f"   Using default simulation state."
                )

    except Exception as e:
        return f"❌ Error initializing reality sync: {str(e)}"


@mcp.tool()
def validate_multiple_trajectories(
    current_joints: list[float],
    trajectory_targets: list[list[float]],
    trajectory_names: list[str] | None = None,
    duration_sec: float = 2.0,
) -> str:
    """
    Validate multiple trajectory candidates in parallel and recommend the best one.

    Use this when you have multiple ways to accomplish a task (e.g., different
    grasp poses) and want to find the safest, most efficient option.

    Args:
        current_joints: Current joint positions
        trajectory_targets: List of target joint positions (one per candidate)
        trajectory_names: Optional names for each trajectory
        duration_sec: Simulation duration for each trajectory

    Returns:
        Comparison results with recommendation
    """
    global _batch_validator, _sandbox

    try:
        sandbox = get_sandbox()

        # Initialize batch validator if needed
        if _batch_validator is None:
            _batch_validator = BatchValidator(sandbox, _semantic_translator)

        # Generate default names if not provided
        if trajectory_names is None:
            trajectory_names = [f"trajectory_{i+1}" for i in range(len(trajectory_targets))]

        # Create trajectory candidates
        import numpy as np

        candidates = []
        for i, (target, name) in enumerate(zip(trajectory_targets, trajectory_names)):
            candidate = TrajectoryCandidate(
                id=f"traj_{i+1}",
                name=name,
                waypoints=np.array([target]),  # Simplified: just target
                duration_sec=duration_sec,
                metadata={"description": f"Candidate trajectory: {name}"},
            )
            candidates.append(candidate)

        # Run batch validation
        results = _batch_validator.validate_multiple(
            candidates=candidates,
            current_qpos=np.array(current_joints),
            parallel=True,
        )

        # Get best recommendation
        best = _batch_validator.recommend_best(results, require_safe=True)

        # Format response
        return _batch_validator.format_recommendation(results, best)

    except Exception as e:
        return f"❌ Error in batch validation: {str(e)}"


@mcp.tool()
def list_available_models() -> str:
    """
    List available robot models from the MuJoCo Menagerie.

    Returns:
        List of available models that can be loaded
    """
    if not MUJOCO_MENAGERIE_PATH.exists():
        return "❌ MuJoCo Menagerie not found at expected path."

    models = []
    for item in MUJOCO_MENAGERIE_PATH.iterdir():
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
    response += f"   load_model with path: {MUJOCO_MENAGERIE_PATH}/[model_name]/[xml_file]"

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


@mcp.resource("e_urdf://{robot_id}/system_prompt")
def get_system_prompt(robot_id: str) -> str:
    """Get system prompt for a robot from e-URDF-Zoo."""
    try:
        robot_path = get_e_urdf_robot_path(robot_id)
        if robot_path is None:
            return f"Robot '{robot_id}' not found in e-URDF-Zoo."

        prompt_path = robot_path / "prompts" / "system.md"
        if not prompt_path.exists():
            return f"System prompt not found for {robot_id}."

        with open(prompt_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error loading system prompt: {str(e)}"


@mcp.resource("e_urdf://{robot_id}/tools_usage")
def get_tools_usage(robot_id: str) -> str:
    """Get tools usage guide for a robot from e-URDF-Zoo."""
    try:
        robot_path = get_e_urdf_robot_path(robot_id)
        if robot_path is None:
            return f"Robot '{robot_id}' not found in e-URDF-Zoo."

        guide_path = robot_path / "prompts" / "tools_usage.md"
        if not guide_path.exists():
            return f"Tools usage guide not found for {robot_id}."

        with open(guide_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error loading tools usage: {str(e)}"


@mcp.resource("e_urdf://{robot_id}/config")
def get_e_urdf_config_resource(robot_id: str) -> str:
    """Get e_urdf.json configuration for a robot."""
    try:
        robot_path = get_e_urdf_robot_path(robot_id)
        if robot_path is None:
            return json.dumps({"error": f"Robot '{robot_id}' not found"})

        config = load_e_urdf_config(robot_path)
        return json.dumps(config, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("e_urdf://{robot_id}/list")
def list_e_urdf_robots(robot_id: str) -> str:
    """List all robots available in e-URDF-Zoo (use robot_id='all')."""
    if robot_id != "all":
        return json.dumps({"error": "Use robot_id='all' to list all robots"})

    try:
        robots = []
        for rid in list_available_robots_from_zoo():
            robot_path = E_URDF_ZOO_PATH / "robots" / rid
            try:
                config = load_e_urdf_config(robot_path)
                robots.append({
                    "id": rid,
                    "name": config.get("embodiment_name", rid),
                    "type": config.get("semantics", {}).get("robot_type", "unknown"),
                    "dof": config.get("kinematics", {}).get("dof", 0),
                    "description": config.get("meta", {}).get("description", ""),
                })
            except Exception:
                robots.append({"id": rid, "error": "Failed to load config"})

        return json.dumps({"robots": robots, "count": len(robots)}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


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
    print("MJLab MCP Server - Physics Sandbox Firewall V2.0", file=sys.stderr)
    print("e-URDF-Zoo + Reality Sync + Semantic Translation Enabled", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("", file=sys.stderr)
    print("Available tools:", file=sys.stderr)
    print("  - load_embodiment: Load robot from e-URDF-Zoo (RECOMMENDED)", file=sys.stderr)
    print("  - load_model: Load a MuJoCo model directly", file=sys.stderr)
    print("  - verify_action_safety: Validate robot action safety", file=sys.stderr)
    print("  - validate_multiple_trajectories: Batch validate & recommend best", file=sys.stderr)
    print("  - sync_reality: Sync simulation to real robot state", file=sys.stderr)
    print("  - get_model_info: Get model information", file=sys.stderr)
    print("  - list_available_models: List MuJoCo Menagerie models", file=sys.stderr)
    print("", file=sys.stderr)
    print("e-URDF Resources:", file=sys.stderr)
    print("  - e_urdf://{robot_id}/system_prompt", file=sys.stderr)
    print("  - e_urdf://{robot_id}/tools_usage", file=sys.stderr)
    print("  - e_urdf://{robot_id}/config", file=sys.stderr)
    print("  - e_urdf://all/list", file=sys.stderr)
    print("", file=sys.stderr)
    print("Starting MCP server (stdio mode)...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    mcp.run()


if __name__ == "__main__":
    main()
