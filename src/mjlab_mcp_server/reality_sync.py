"""
Reality-to-Sim Synchronization Module

This module ensures the physics simulation starts from the exact same
state as the real robot, making safety validation meaningful.

Key concept: Before validating any trajectory, we must sync the simulation
to match the real robot's current joint positions and velocities.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .physics import PhysicsSandbox


@dataclass
class JointState:
    """Represents the state of a single joint at a specific time."""

    name: str
    position: float  # radians or meters
    velocity: float = 0.0  # rad/s or m/s
    effort: float = 0.0  # Nm or N
    timestamp: float = field(default_factory=time.time)


@dataclass
class RobotState:
    """Complete state of a multi-joint robot."""

    joint_states: dict[str, JointState]
    timestamp: float = field(default_factory=time.time)

    def get_positions(self, joint_names: list[str]) -> np.ndarray:
        """Get positions array ordered by joint_names."""
        return np.array(
            [
                self.joint_states[j].position if j in self.joint_states else 0.0
                for j in joint_names
            ]
        )

    def get_velocities(self, joint_names: list[str]) -> np.ndarray:
        """Get velocities array ordered by joint_names."""
        return np.array(
            [
                self.joint_states[j].velocity if j in self.joint_states else 0.0
                for j in joint_names
            ]
        )


class RealitySync:
    """
    Synchronizes real robot state to MuJoCo simulation.

    This class maintains a background thread that subscribes to the real
    robot's joint state topic and caches the latest state. Before any
    safety validation, the simulation is synchronized to this real state.

    Attributes:
        sandbox: PhysicsSandbox instance to synchronize
        joint_names: List of joint names to track
        ros2_node: Optional ROS 2 node for subscription
        _latest_state: Cached latest robot state
        _lock: Thread lock for state access
        _running: Flag to control background thread
    """

    def __init__(
        self,
        sandbox: PhysicsSandbox,
        joint_names: list[str],
        use_ros2: bool = True,
    ):
        self.sandbox = sandbox
        self.joint_names = joint_names
        self.use_ros2 = use_ros2

        self._latest_state: RobotState | None = None
        self._lock = threading.RLock()
        self._running = False
        self._sync_thread: threading.Thread | None = None

        # ROS 2 components (lazy initialization)
        self._ros2_node = None
        self._subscriber = None

    def start(self) -> None:
        """Start the background synchronization thread."""
        if self._running:
            return

        self._running = True

        if self.use_ros2:
            self._init_ros2_subscription()
        else:
            # Mock mode for testing without ROS 2
            self._sync_thread = threading.Thread(
                target=self._mock_sync_loop, daemon=True
            )
            self._sync_thread.start()

    def stop(self) -> None:
        """Stop the background synchronization thread."""
        self._running = False
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=1.0)

        if self._ros2_node:
            # Cleanup ROS 2
            try:
                self._subscriber.destroy()
                self._ros2_node.destroy_node()
            except Exception:
                pass

    def _init_ros2_subscription(self) -> None:
        """Initialize ROS 2 subscription to /joint_states."""
        try:
            import rclpy
            from sensor_msgs.msg import JointState as JointStateMsg

            rclpy.init(args=None)
            self._ros2_node = rclpy.create_node("mjlab_reality_sync")

            self._subscriber = self._ros2_node.create_subscription(
                JointStateMsg,
                "/joint_states",
                self._on_joint_states,
                10,  # QoS depth
            )

            # Start ROS 2 spin in background thread
            self._sync_thread = threading.Thread(
                target=self._ros2_spin_loop, daemon=True
            )
            self._sync_thread.start()

        except ImportError:
            print("Warning: ROS 2 not available, falling back to mock mode")
            self.use_ros2 = False
            self._sync_thread = threading.Thread(
                target=self._mock_sync_loop, daemon=True
            )
            self._sync_thread.start()

    def _ros2_spin_loop(self) -> None:
        """ROS 2 spin loop running in background thread."""
        import rclpy

        while self._running:
            try:
                rclpy.spin_once(self._ros2_node, timeout_sec=0.01)
            except Exception as e:
                print(f"ROS 2 spin error: {e}")
                time.sleep(0.1)

    def _on_joint_states(self, msg) -> None:
        """Callback for incoming joint state messages."""
        joint_states = {}
        for name, pos, vel, eff in zip(msg.name, msg.position, msg.velocity, msg.effort):
            if name in self.joint_names:
                joint_states[name] = JointState(
                    name=name,
                    position=float(pos),
                    velocity=float(vel),
                    effort=float(eff),
                    timestamp=time.time(),
                )

        with self._lock:
            self._latest_state = RobotState(
                joint_states=joint_states, timestamp=time.time()
            )

    def _mock_sync_loop(self) -> None:
        """Mock sync loop for testing without ROS 2."""
        # In mock mode, we just keep the simulation state as-is
        while self._running:
            time.sleep(0.01)

    def get_current_state(self) -> RobotState | None:
        """Get the latest cached robot state."""
        with self._lock:
            return self._latest_state

    def sync_simulation_to_reality(self) -> bool:
        """
        Synchronize MuJoCo simulation to match real robot state.

        This is the critical function called before safety validation.
        It sets data.qpos to match the real robot's current joint positions.

        Returns:
            True if sync successful, False otherwise
        """
        if self._latest_state is None:
            print("Warning: No real robot state available, using simulation default")
            return False

        try:
            # Get current joint positions from real state
            qpos = self._latest_state.get_positions(self.joint_names)
            qvel = self._latest_state.get_velocities(self.joint_names)

            # Apply to MuJoCo simulation
            with self.sandbox._lock:
                self.sandbox.data.qpos[: len(qpos)] = qpos
                self.sandbox.data.qvel[: len(qvel)] = qvel
                # Forward kinematics to update body positions
                import mujoco

                mujoco.mj_forward(self.sandbox.model, self.sandbox.data)

            return True

        except Exception as e:
            print(f"Error syncing simulation to reality: {e}")
            return False

    def get_state_age_ms(self) -> float:
        """Get age of latest state in milliseconds."""
        with self._lock:
            if self._latest_state is None:
                return float("inf")
            return (time.time() - self._latest_state.timestamp) * 1000


class RealitySyncManager:
    """
    Manages multiple RealitySync instances for different robots.

    This is useful when validating trajectories for multiple robots
    or when switching between different embodiments.
    """

    def __init__(self):
        self._syncs: dict[str, RealitySync] = {}
        self._lock = threading.RLock()

    def register(self, robot_id: str, sync: RealitySync) -> None:
        """Register a RealitySync instance for a robot."""
        with self._lock:
            if robot_id in self._syncs:
                self._syncs[robot_id].stop()
            self._syncs[robot_id] = sync
            sync.start()

    def unregister(self, robot_id: str) -> None:
        """Unregister and stop a RealitySync instance."""
        with self._lock:
            if robot_id in self._syncs:
                self._syncs[robot_id].stop()
                del self._syncs[robot_id]

    def get_sync(self, robot_id: str) -> RealitySync | None:
        """Get RealitySync instance for a robot."""
        with self._lock:
            return self._syncs.get(robot_id)

    def sync_all(self) -> dict[str, bool]:
        """Synchronize all registered simulations to reality."""
        results = {}
        with self._lock:
            for robot_id, sync in self._syncs.items():
                results[robot_id] = sync.sync_simulation_to_reality()
        return results

    def stop_all(self) -> None:
        """Stop all RealitySync instances."""
        with self._lock:
            for sync in self._syncs.values():
                sync.stop()
            self._syncs.clear()
