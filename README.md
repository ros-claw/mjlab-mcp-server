# MJLab MCP Server

🌐 **English** | [中文](./README.zh.md)

ROSClaw MCP Server for **MuJoCo Physics Simulation** - The e-URDF Safety Firewall.

Part of the [ROSClaw](https://github.com/ros-claw) Embodied Intelligence Operating System.

## Overview

This MCP server provides a "Semantic-Physical Firewall" that validates robot trajectories using MuJoCo physics simulation before execution on real hardware. It prevents LLM hallucinations from causing physical damage by simulating movements in a virtual sandbox.

```
LLM Agent ──MCP──► mjlab-mcp-server ──MuJoCo──► Virtual Simulation
                                          │
                                          ▼
                                    ✅ Safe → Execute on Real Robot
                                    ❌ Unsafe → Block & Report to LLM
```

## Core Concept: e-URDF Safety Validation

**e-URDF (Embodied URDF)** = Physical model + Safety policy + Simulation validation

Before any robot movement:
1. **Intercept** the planned trajectory
2. **Simulate** in MuJoCo sandbox (2 seconds of physics in ~10ms)
3. **Validate** collisions, joint limits, torque limits
4. **Decide**: Execute real hardware OR block with detailed feedback

## Features

| Tool | Description |
|------|-------------|
| `load_model` | Load MuJoCo MJCF/URDF model and safety policy |
| `verify_action_safety` | **Critical**: Validate trajectory before execution |
| `get_model_info` | Get loaded model details and joint limits |
| `list_available_models` | List models from MuJoCo Menagerie |

**MCP Resources**: `safety://status`, `safety://limits`

## Installation

```bash
# Clone
git clone https://github.com/ros-claw/mjlab-mcp-server.git
cd mjlab-mcp-server

# Install with uv (recommended)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

### 1. Run MCP Server

```bash
# Default: Uses UR5e from MuJoCo Menagerie
python -m mjlab_mcp_server.server

# With custom model
MUJOCO_MODEL_PATH=/path/to/robot.xml python -m mjlab_mcp_server.server
```

### 2. Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mjlab-firewall": {
      "command": "python",
      "args": ["/path/to/mjlab-mcp-server/src/mjlab_mcp_server/server.py"],
      "transportType": "stdio",
      "description": "MuJoCo Physics Safety Validation",
      "env": {
        "MUJOCO_MODEL_PATH": "/path/to/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
      }
    }
  }
}
```

## Usage Examples

### Example 1: UR5e Safety Check

```
User: Move UR5e to joint positions [0, -1.57, 1.57, 0, 0, 0]

LLM calls:
1. verify_action_safety(
     current_joints=[0, 0, 0, 0, 0, 0],
     target_joints=[0, -1.57, 1.57, 0, 0, 0],
     duration_sec=2.0
   )

Result: ✅ [SAFE] Physics simulation passed!
        You may proceed with execution.

2. (Real robot execution happens here)
```

### Example 2: Collision Detection

```
User: Move UR5e arm straight through the table

LLM calls:
1. verify_action_safety(
     current_joints=[0, 0, 0, 0, 0, 0],
     target_joints=[0, -2.5, 2.5, 0, 0, 0]  # Would hit table
   )

Result: ❌ [DANGER] Physical simulation failed!
        🔴 COLLISIONS DETECTED:
          - forearm_collision collided with table_top

        ⚠️ ACTION BLOCKED - DO NOT EXECUTE ON REAL HARDWARE!
        Please replan your trajectory.

2. LLM automatically replans with obstacle avoidance
```

### Example 3: Joint Limit Protection

```
User: Rotate wrist_3 to 10 radians

LLM calls:
1. verify_action_safety(
     current_joints=[0, 0, 0, 0, 0, 0],
     target_joints=[0, 0, 0, 0, 0, 10]  # Exceeds ±2π limit
   )

Result: ❌ [DANGER] Physical simulation failed!
        🟡 JOINT LIMIT VIOLATIONS:
          - Joint 'wrist_3_joint': target 10.0000 outside limits [-6.2832, 6.2832]
```

## Safety Policy Configuration

Create a `policy.yaml` file to define safety constraints:

```yaml
# Joint limits (radians)
joint_limits:
  shoulder_pan_joint: [-6.28319, 6.28319]  # ±360°
  shoulder_lift_joint: [-6.28319, 6.28319]
  # ...

# Torque limits (Nm)
torque_limits:
  shoulder_pan_joint: 150.0
  wrist_3_joint: 28.0

# Collision exclusions (adjacent links)
collision_exclude_pairs:
  - [base_link, shoulder_link]
  - [shoulder_link, upper_arm_link]

# Safety margin
safety_margin: 0.05  # 5%
```

## 3D Asset Integration

### e-URDF Zoo (Recommended)

This server integrates with **[e-URDF Zoo](https://github.com/ros-claw/e-urdf-zoo)** for pre-configured robot assets with safety policies:

| Robot | ID | DOF | Features |
|-------|-----|-----|----------|
| **Universal Robots UR5e** | `universal_robots_ur5e` | 6 | Full e-URDF config, collision semantics |
| **Unitree G1** | `unitree_g1` | 23 | Humanoid, balance checks, ZMP validation |
| **Franka FR3** | `franka_fr3` | 7 | Collaborative arm (skeleton) |
| **Boston Dynamics Spot** | `boston_dynamics_spot` | 12 | Quadruped (skeleton) |

**Total**: 63 robots from MuJoCo Menagerie with standardized configs

### Usage Examples

#### Option 1: Dynamic Loading (Recommended)
```python
# Use the load_embodiment tool to load from e-URDF-Zoo
load_embodiment(embodiment_id="universal_robots_ur5e")

# The server automatically:
# - Downloads model from e-URDF-Zoo if needed
# - Loads MuJoCo MJCF/XML
# - Applies safety policy from e_urdf.json
# - Enables semantic error translation
```

#### Option 2: Direct Model Path
```python
# Use local model file
load_model(
    model_path="/path/to/robot.xml",
    policy_path="/path/to/policy.yaml"
)
```

#### Option 3: Python API
```python
from mjlab_mcp_server.physics import PhysicsSandbox
from e_urdf_zoo import load_embodiment

# Load embodiment config
asset = load_embodiment("unitree_g1")

# Initialize sandbox with safety policy
sandbox = PhysicsSandbox(
    model_path=asset.model_xml,
    policy=asset.config.physical_firewall
)
```

## Architecture

```
mjlab_mcp_server/
├── src/mjlab_mcp_server/
│   ├── __init__.py
│   ├── server.py           # MCP Server with FastMCP
│   └── physics.py          # PhysicsSandbox class
├── assets/
│   └── ur5e_e_urdf/
│       └── policy.yaml     # Safety policy example
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

### PhysicsSandbox Class

Core simulation engine in `physics.py`:

```python
sandbox = PhysicsSandbox(
    model_path="ur5e.xml",
    policy_path="policy.yaml"
)

result = sandbox.simulate_safety_check(
    current_qpos=[0, 0, 0, 0, 0, 0],
    target_qpos=[0, -1.57, 1.57, 0, 0, 0],
    duration_sec=2.0
)

if result.is_safe:
    execute_on_real_robot()
else:
    print(result.collision_details)
```

## Safety Checks

The server performs multiple validation layers:

1. **Pre-check**: Target position within joint limits
2. **Collision Detection**: `data.ncon > 0` with contact penetration
3. **Joint Limit Violation**: Position outside soft limits (with margin)
4. **Velocity Limit Violation**: Joint velocity exceeds max
5. **Torque Limit Violation**: Actuator force exceeds rating

## Technical Details

- **Physics Engine**: MuJoCo 3.0+
- **Simulation Speed**: ~10ms for 2s of simulated time
- **Control Modes**: Position control (PD), Velocity control
- **Timestep**: Model-defined (typically 0.002s = 500Hz)

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MUJOCO_MODEL_PATH` | Default model path | Auto-detect UR5e |
| `SAFETY_POLICY_PATH` | Safety policy YAML | None (auto-generate) |

## Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=mjlab_mcp_server --cov-report=html
```

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [mjlab](https://github.com/mujocolab/mjlab) — GPU-accelerated MuJoCo batch simulation
- [mjlab Documentation](https://mujocolab.github.io/mjlab/)
- [e-URDF Zoo](https://github.com/ros-claw/e-urdf-zoo) — Pre-configured robot assets with safety policies
- [MCP Protocol](https://modelcontextprotocol.io/)
- [ROSClaw](https://github.com/ros-claw/rosclaw) — Embodied Intelligence Operating System
- [ROSClaw Paper (arXiv)](https://arxiv.org/pdf/2604.04664) — Technical Architecture Whitepaper

## Part of ROSClaw

- [rosclaw-g1-dds-mcp](https://github.com/ros-claw/g1-dds-mcp) — Unitree G1 humanoid
- [rosclaw-ur-ros2-mcp](https://github.com/ros-claw/ur-ros2-mcp) — UR5 via ROS2
- [rosclaw-ur-rtde-mcp](https://github.com/ros-claw/ur-rtde-mcp) — UR via RTDE
- [mjlab-mcp-server](https://github.com/ros-claw/mjlab-mcp-server) — Physics firewall (this repo)

---

**Safety Warning**: This server is a validation tool, not a substitute for physical safety systems. Always use proper emergency stops and safety cages with real robots.

*Generated by ROSClaw e-URDF Framework*
