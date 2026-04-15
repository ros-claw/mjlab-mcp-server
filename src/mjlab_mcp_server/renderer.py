"""
MuJoCo Rendering Module for MJLab MCP Server.

Provides visualization capabilities including:
- Static image rendering of robot states
- Trajectory preview animations
- Collision debug visualization
- Depth map generation
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .physics import PhysicsSandbox


@dataclass
class RenderConfig:
    """Configuration for rendering."""

    width: int = 640
    height: int = 480
    camera_name: str | None = None
    azimuth: float = -60.0
    elevation: float = -30.0
    distance: float | None = None
    lookat: tuple[float, float, float] | None = None


class MuJoCoRenderer:
    """
    MuJoCo-based renderer for robot visualization.

    Supports multiple camera modes and output formats.
    """

    def __init__(self, sandbox: "PhysicsSandbox"):
        self.sandbox = sandbox
        self.model = sandbox.model
        self.data = sandbox.data

        # Default camera configuration
        self.config = RenderConfig()

    def _setup_camera(self, config: RenderConfig | None = None) -> mujoco.MjvCamera:
        """Setup camera with given configuration."""
        if config is None:
            config = self.config

        camera = mujoco.MjvCamera()

        if config.camera_name:
            # Use named camera from model
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, config.camera_name
            )
            if camera_id >= 0:
                camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                camera.fixedcamid = camera_id
            else:
                # Camera not found, fall back to free camera
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.azimuth = config.azimuth
                camera.elevation = config.elevation
        else:
            # Use free camera with custom angles
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = config.azimuth
            camera.elevation = config.elevation

            if config.distance is not None:
                camera.distance = config.distance
            else:
                # Auto-compute distance based on model extent
                camera.distance = self.model.stat.extent * 2.0

            if config.lookat is not None:
                camera.lookat[:] = config.lookat
            else:
                # Center on model centroid
                camera.lookat[:] = self.model.stat.center

        return camera

    def render_current_state(
        self,
        width: int = 640,
        height: int = 480,
        camera_name: str | None = None,
        azimuth: float = -60.0,
        elevation: float = -30.0,
        show_labels: bool = False,
    ) -> dict:
        """
        Render the current robot state as an image.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            camera_name: Named camera to use (optional)
            azimuth: Camera azimuth angle (for free camera)
            elevation: Camera elevation angle (for free camera)
            show_labels: Whether to show joint/body labels

        Returns:
            Dictionary with base64-encoded image and metadata
        """
        # Update physics to ensure correct positions
        mujoco.mj_forward(self.model, self.data)

        # Setup renderer
        renderer = mujoco.Renderer(self.model, height=height, width=width)

        # Setup camera
        config = RenderConfig(
            width=width,
            height=height,
            camera_name=camera_name,
            azimuth=azimuth,
            elevation=elevation,
        )
        camera = self._setup_camera(config)

        # Render
        renderer.update_scene(self.data, camera=camera)
        pixels = renderer.render()

        # Convert to PIL Image
        image = Image.fromarray(pixels)

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        renderer.close()

        return {
            "image_base64": img_base64,
            "format": "png",
            "width": width,
            "height": height,
            "camera": camera_name or f"free(az={azimuth}, el={elevation})",
        }

    def render_trajectory_preview(
        self,
        trajectory: list[list[float]],
        width: int = 640,
        height: int = 480,
        frames: int = 8,
    ) -> dict:
        """
        Render a trajectory as a sequence of preview frames.

        Args:
            trajectory: List of joint positions (waypoints)
            width: Image width
            height: Image height
            frames: Number of frames to generate

        Returns:
            Dictionary with list of base64-encoded images
        """
        if not trajectory:
            return {"error": "Empty trajectory provided"}

        # Sample frames evenly from trajectory
        indices = np.linspace(0, len(trajectory) - 1, frames, dtype=int)
        selected_waypoints = [trajectory[i] for i in indices]

        frame_images = []

        for i, qpos in enumerate(selected_waypoints):
            # Set joint positions
            self.data.qpos[: len(qpos)] = qpos
            mujoco.mj_forward(self.model, self.data)

            # Render frame
            result = self.render_current_state(width=width, height=height)
            frame_images.append(
                {
                    "frame": i + 1,
                    "image_base64": result["image_base64"],
                    "waypoint_index": int(indices[i]),
                }
            )

        return {
            "frames": frame_images,
            "total_frames": len(frame_images),
            "trajectory_length": len(trajectory),
            "width": width,
            "height": height,
        }

    def render_collision_debug(
        self,
        width: int = 640,
        height: int = 480,
        highlight_contacts: bool = True,
    ) -> dict:
        """
        Render scene with collision debug visualization.

        Args:
            width: Image width
            height: Image height
            highlight_contacts: Whether to highlight contact points

        Returns:
            Dictionary with rendered image and collision info
        """
        # Update physics
        mujoco.mj_forward(self.model, self.data)

        # Get contact information
        contacts = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.dist < 0:  # Penetration
                body1 = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, contact.geom1
                )
                body2 = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, contact.geom2
                )
                contacts.append(
                    {
                        "body1": body1,
                        "body2": body2,
                        "distance": float(contact.dist),
                        "position": contact.pos.tolist(),
                    }
                )

        # Setup scene with contact visualization
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = self._setup_camera()

        # Update scene with flags
        renderer.update_scene(
            self.data,
            camera=camera,
            scene_option=mujoco.MjvOption(),
        )

        # Render
        pixels = renderer.render()
        image = Image.fromarray(pixels)

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        renderer.close()

        return {
            "image_base64": img_base64,
            "format": "png",
            "width": width,
            "height": height,
            "contacts_detected": len(contacts),
            "contacts": contacts,
        }

    def render_depth_map(
        self,
        width: int = 640,
        height: int = 480,
        camera_name: str | None = None,
        max_depth: float = 10.0,
    ) -> dict:
        """
        Render depth map from camera perspective.

        Args:
            width: Image width
            height: Image height
            camera_name: Named camera to use
            max_depth: Maximum depth value (clipping)

        Returns:
            Dictionary with depth map image and data
        """
        # Update physics
        mujoco.mj_forward(self.model, self.data)

        # Setup renderer for depth
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.enable_depth_rendering()
        config = RenderConfig(width=width, height=height, camera_name=camera_name)
        camera = self._setup_camera(config)

        # Render depth
        renderer.update_scene(self.data, camera=camera)
        depth = renderer.render()

        # Normalize depth for visualization (0-255)
        depth_normalized = np.clip(depth / max_depth, 0, 1)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Convert to grayscale image
        image = Image.fromarray(depth_uint8, mode="L")

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        renderer.close()

        return {
            "image_base64": img_base64,
            "format": "png",
            "width": width,
            "height": height,
            "max_depth": max_depth,
            "min_depth_m": float(np.min(depth)),
            "max_depth_m": float(np.max(depth)),
        }

    def save_screenshot(
        self,
        filepath: str,
        width: int = 1920,
        height: int = 1080,
        camera_name: str | None = None,
    ) -> dict:
        """
        Save high-quality screenshot to file.

        Args:
            filepath: Path to save the image
            width: Image width
            height: Image height
            camera_name: Named camera to use

        Returns:
            Dictionary with save status and file info
        """
        result = self.render_current_state(
            width=width, height=height, camera_name=camera_name
        )

        # Decode base64 and save
        img_data = base64.b64decode(result["image_base64"])

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            f.write(img_data)

        return {
            "saved": True,
            "filepath": str(path.absolute()),
            "width": width,
            "height": height,
            "size_bytes": len(img_data),
        }
