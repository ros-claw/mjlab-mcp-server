"""
Semantic Error Translator Module

Converts low-level MuJoCo geom/contact IDs into human-readable,
LLM-friendly error messages using e-URDF semantic annotations.

Key concept: Instead of reporting "geom 14 collided with geom 27",
we report "UR5e's 'wrist_3_link' will collide with 'table_obstacle' at t=1.2s".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .physics import PhysicsSandbox


@dataclass
class SemanticContact:
    """Human-readable contact information."""

    body1_name: str
    body2_name: str
    link1_name: str
    link2_name: str
    geom1_name: str
    geom2_name: str
    contact_time: float
    contact_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    penetration_depth: float = 0.0

    def to_natural_language(self) -> str:
        """Convert to natural language description."""
        return (
            f"'{self.link1_name}' will collide with '{self.link2_name}' "
            f"at t={self.contact_time:.2f}s"
        )


@dataclass
class SemanticViolation:
    """Generic safety violation with semantic context."""

    violation_type: str  # COLLISION, JOINT_LIMIT, VELOCITY_LIMIT, TORQUE_LIMIT
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    details: dict = field(default_factory=dict)
    recommendation: str = ""


class SemanticTranslator:
    """
    Translates raw MuJoCo errors into semantic, LLM-friendly messages.

    This class uses the e-URDF configuration to map geom IDs to meaningful
    link/body names, and generates actionable error messages for LLMs.

    Attributes:
        sandbox: PhysicsSandbox instance
        e_urdf_config: Parsed e-URDF configuration
        geom_to_link: Mapping from geom ID to link name
        geom_to_body: Mapping from geom ID to body name
    """

    # Violation type to severity mapping
    SEVERITY_MAP = {
        "COLLISION": "CRITICAL",
        "JOINT_LIMIT": "HIGH",
        "VELOCITY_LIMIT": "MEDIUM",
        "TORQUE_LIMIT": "HIGH",
        "SELF_COLLISION": "CRITICAL",
        "ENVIRONMENT_COLLISION": "CRITICAL",
        "BALANCE_LOST": "CRITICAL",
        "FALL_RISK": "CRITICAL",
    }

    def __init__(
        self,
        sandbox: PhysicsSandbox,
        e_urdf_config: dict | None = None,
    ):
        self.sandbox = sandbox
        self.e_urdf_config = e_urdf_config or {}
        self.model = sandbox.model
        self.data = sandbox.data

        # Build geom mappings
        self.geom_to_link: dict[int, str] = {}
        self.geom_to_body: dict[int, str] = {}
        self._build_geom_mappings()

    def _build_geom_mappings(self) -> None:
        """Build mappings from geom IDs to link/body names."""
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name:
                self.geom_to_link[geom_id] = geom_name

                # Try to find parent body
                body_id = self.model.geom_bodyid[geom_id]
                body_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                )
                if body_name:
                    self.geom_to_body[geom_id] = body_name

    def translate_collision(
        self, geom1_id: int, geom2_id: int, sim_time: float
    ) -> SemanticContact:
        """
        Translate geom IDs to semantic contact information.

        Args:
            geom1_id: First colliding geom ID
            geom2_id: Second colliding geom ID
            sim_time: Simulation time of collision

        Returns:
            SemanticContact with human-readable names
        """
        # Get geom names
        geom1_name = self.geom_to_link.get(geom1_id, f"geom_{geom1_id}")
        geom2_name = self.geom_to_link.get(geom2_id, f"geom_{geom2_id}")

        # Get body names
        body1_name = self.geom_to_body.get(geom1_id, "unknown")
        body2_name = self.geom_to_body.get(geom2_id, "unknown")

        # Try to get link names from e-URDF config
        link1_name = self._get_link_name(geom1_id, geom1_name, body1_name)
        link2_name = self._get_link_name(geom2_id, geom2_name, body2_name)

        # Get contact position from contact data
        contact_pos = self._get_contact_position(geom1_id, geom2_id)

        return SemanticContact(
            body1_name=body1_name,
            body2_name=body2_name,
            link1_name=link1_name,
            link2_name=link2_name,
            geom1_name=geom1_name,
            geom2_name=geom2_name,
            contact_time=sim_time,
            contact_position=contact_pos,
        )

    def _get_link_name(self, geom_id: int, geom_name: str, body_name: str) -> str:
        """Get human-readable link name from e-URDF config or MuJoCo names."""
        # Try e-URDF links first
        links = self.e_urdf_config.get("links", {})
        link_names = links.get("names", [])

        # Match geom name to link name
        for link_name in link_names:
            if link_name in geom_name or link_name in body_name:
                return link_name

        # Fallback to body name or geom name
        if body_name != "unknown":
            return body_name
        return geom_name

    def _get_contact_position(self, geom1_id: int, geom2_id: int) -> np.ndarray:
        """Get contact position from MuJoCo contact data."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == geom1_id and contact.geom2 == geom2_id) or (
                contact.geom1 == geom2_id and contact.geom2 == geom1_id
            ):
                return np.array(contact.pos)
        return np.zeros(3)

    def is_self_collision(self, geom1_id: int, geom2_id: int) -> bool:
        """Check if collision is between robot's own links."""
        body1 = self.geom_to_body.get(geom1_id)
        body2 = self.geom_to_body.get(geom2_id)

        if not body1 or not body2:
            return False

        # Check excluded pairs from e-URDF config
        excluded_pairs = self.e_urdf_config.get("physical_firewall", {}).get(
            "excluded_collision_pairs", []
        )
        for pair in excluded_pairs:
            if (body1 in pair and body2 in pair) or (body2 in pair and body1 in pair):
                return True

        return False

    def translate_violation(
        self,
        violation_type: str,
        raw_message: str,
        context: dict | None = None,
    ) -> SemanticViolation:
        """
        Translate a generic violation into semantic form.

        Args:
            violation_type: Type of violation (COLLISION, JOINT_LIMIT, etc.)
            raw_message: Original error message
            context: Additional context for translation

        Returns:
            SemanticViolation with human-readable message
        """
        context = context or {}
        severity = self.SEVERITY_MAP.get(violation_type, "MEDIUM")

        message = raw_message
        recommendation = ""

        if violation_type == "JOINT_LIMIT":
            joint_name = context.get("joint_name", "unknown")
            target = context.get("target_value", 0)
            limits = context.get("limits", [0, 0])
            message = (
                f"Joint '{joint_name}': target {target:.4f} rad "
                f"exceeds limits [{limits[0]:.4f}, {limits[1]:.4f}] rad"
            )
            recommendation = (
                f"Please adjust '{joint_name}' to be within "
                f"[{limits[0]:.2f}, {limits[1]:.2f}] radians"
            )

        elif violation_type == "VELOCITY_LIMIT":
            joint_name = context.get("joint_name", "unknown")
            velocity = context.get("velocity", 0)
            limit = context.get("limit", 0)
            message = (
                f"Joint '{joint_name}': velocity {velocity:.2f} rad/s "
                f"exceeds limit {limit:.2f} rad/s"
            )
            recommendation = f"Reduce velocity for '{joint_name}'"

        elif violation_type == "TORQUE_LIMIT":
            joint_name = context.get("joint_name", "unknown")
            torque = context.get("torque", 0)
            limit = context.get("limit", 0)
            message = (
                f"Joint '{joint_name}': torque {torque:.2f} Nm "
                f"exceeds limit {limit:.2f} Nm"
            )
            recommendation = f"Reduce load or slow down movement for '{joint_name}'"

        elif violation_type in ("COLLISION", "ENVIRONMENT_COLLISION"):
            contact = context.get("contact")
            if contact and isinstance(contact, SemanticContact):
                message = contact.to_natural_language()
                recommendation = (
                    f"Adjust trajectory to avoid '{contact.link1_name}' "
                    f"getting too close to '{contact.link2_name}'"
                )

        return SemanticViolation(
            violation_type=violation_type,
            severity=severity,
            message=message,
            details=context,
            recommendation=recommendation,
        )

    def format_error_response(self, violations: list[SemanticViolation]) -> str:
        """
        Format multiple violations into a comprehensive error response.

        This generates the message returned to the LLM when safety check fails.
        """
        if not violations:
            return "✅ [SAFE] Physics simulation passed!"

        lines = [
            "❌ [DANGER] Physical simulation failed!",
            "",
            "Violations detected:",
        ]

        # Group by severity
        critical = [v for v in violations if v.severity == "CRITICAL"]
        high = [v for v in violations if v.severity == "HIGH"]
        medium = [v for v in violations if v.severity == "MEDIUM"]

        if critical:
            lines.append("\n🔴 CRITICAL (Action BLOCKED):")
            for v in critical:
                lines.append(f"  • [{v.violation_type}] {v.message}")
                if v.recommendation:
                    lines.append(f"    → {v.recommendation}")

        if high:
            lines.append("\n🟡 HIGH (Recommend revision):")
            for v in high:
                lines.append(f"  • [{v.violation_type}] {v.message}")
                if v.recommendation:
                    lines.append(f"    → {v.recommendation}")

        if medium:
            lines.append("\n🟠 MEDIUM (Consider adjustment):")
            for v in medium:
                lines.append(f"  • [{v.violation_type}] {v.message}")

        lines.extend(
            [
                "",
                "⚠️ ACTION BLOCKED - DO NOT EXECUTE ON REAL HARDWARE!",
                "Please replan your trajectory with the above constraints in mind.",
            ]
        )

        return "\n".join(lines)

    def get_collision_summary(self, sim_time: float) -> list[SemanticContact]:
        """
        Get a summary of all collisions at the current simulation state.

        Args:
            sim_time: Current simulation time

        Returns:
            List of SemanticContact objects
        """
        contacts = []
        seen_pairs = set()

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Skip if we've seen this pair
            pair = tuple(sorted([geom1, geom2]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Skip excluded self-collisions
            if self.is_self_collision(geom1, geom2):
                continue

            semantic_contact = self.translate_collision(geom1, geom2, sim_time)
            contacts.append(semantic_contact)

        return contacts
