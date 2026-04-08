"""
GPU Batch Validator Module

Uses mjlab's GPU-accelerated batch simulation to validate multiple
trajectories in parallel, selecting the optimal one.

Key concept: When LLM generates multiple candidate trajectories for a task
(e.g., 3 different grasp poses), validate all of them simultaneously on GPU
and recommend the best one based on safety and efficiency criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .physics import PhysicsSandbox
    from .semantic_translator import SemanticTranslator


@dataclass
class TrajectoryCandidate:
    """A candidate trajectory for batch validation."""

    id: str
    name: str
    waypoints: np.ndarray  # Shape: (n_waypoints, n_joints)
    duration_sec: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validating a single trajectory."""

    candidate_id: str
    is_safe: bool
    safety_score: float  # 0.0 - 1.0
    efficiency_score: float  # 0.0 - 1.0
    total_cost: float  # Combined score (lower is better)
    violations: list = field(default_factory=list)
    sim_time_ms: float = 0.0


class BatchValidator:
    """
    Validates multiple trajectories in parallel using GPU batch simulation.

    This class provides an interface for validating multiple trajectory
    candidates simultaneously, returning a ranked list of results with
    recommendations for the optimal choice.

    Attributes:
        sandbox: PhysicsSandbox instance
        translator: SemanticTranslator for error messages
        use_gpu: Whether to use GPU acceleration (if available)
    """

    def __init__(
        self,
        sandbox: PhysicsSandbox,
        translator: SemanticTranslator | None = None,
        use_gpu: bool = True,
    ):
        self.sandbox = sandbox
        self.translator = translator
        self.use_gpu = use_gpu
        self._mjlab_available = self._check_mjlab()

    def _check_mjlab(self) -> bool:
        """Check if mjlab is available for GPU acceleration."""
        try:
            import mjlab

            return True
        except ImportError:
            return False

    def validate_multiple(
        self,
        candidates: list[TrajectoryCandidate],
        current_qpos: np.ndarray,
        parallel: bool = True,
    ) -> list[ValidationResult]:
        """
        Validate multiple trajectory candidates.

        Args:
            candidates: List of trajectory candidates to validate
            current_qpos: Current joint positions
            parallel: Use parallel validation if GPU available

        Returns:
            List of ValidationResult, sorted by total_cost (best first)
        """
        if parallel and self._mjlab_available and self.use_gpu:
            return self._validate_parallel(candidates, current_qpos)
        else:
            return self._validate_sequential(candidates, current_qpos)

    def _validate_sequential(
        self,
        candidates: list[TrajectoryCandidate],
        current_qpos: np.ndarray,
    ) -> list[ValidationResult]:
        """Validate trajectories sequentially (fallback mode)."""
        results = []

        for candidate in candidates:
            result = self._validate_single(candidate, current_qpos)
            results.append(result)

        # Sort by total cost (lower is better)
        results.sort(key=lambda r: r.total_cost)
        return results

    def _validate_parallel(
        self,
        candidates: list[TrajectoryCandidate],
        current_qpos: np.ndarray,
    ) -> list[ValidationResult]:
        """
        Validate trajectories in parallel using mjlab GPU batching.

        Note: This is a placeholder for actual mjlab integration.
        When mjlab is fully integrated, this will use GPU batch simulation.
        """
        # TODO: Implement actual mjlab batch simulation
        # For now, fall back to sequential
        return self._validate_sequential(candidates, current_qpos)

    def _validate_single(
        self,
        candidate: TrajectoryCandidate,
        current_qpos: np.ndarray,
    ) -> ValidationResult:
        """Validate a single trajectory candidate."""
        import time

        start_time = time.time()

        # Use the target as final waypoint
        target_qpos = candidate.waypoints[-1] if len(candidate.waypoints) > 0 else current_qpos

        # Run safety check
        result = self.sandbox.simulate_safety_check(
            current_qpos=current_qpos,
            target_qpos=target_qpos,
            duration_sec=candidate.duration_sec,
        )

        sim_time_ms = (time.time() - start_time) * 1000

        # Calculate scores
        if result.is_safe:
            safety_score = 1.0
            # Efficiency: prefer shorter trajectories and fewer waypoints
            efficiency_score = self._calculate_efficiency_score(candidate)
            total_cost = 1.0 - efficiency_score  # Lower cost = more efficient
        else:
            # Penalize based on violation severity
            safety_score = self._calculate_safety_score(result)
            efficiency_score = 0.0
            total_cost = 2.0 - safety_score  # Higher cost for unsafe

        return ValidationResult(
            candidate_id=candidate.id,
            is_safe=result.is_safe,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            total_cost=total_cost,
            violations=result.violations if hasattr(result, "violations") else [],
            sim_time_ms=sim_time_ms,
        )

    def _calculate_efficiency_score(self, candidate: TrajectoryCandidate) -> float:
        """Calculate efficiency score based on trajectory characteristics."""
        # Prefer fewer waypoints (simpler trajectories)
        waypoint_penalty = min(len(candidate.waypoints) * 0.05, 0.3)

        # Prefer shorter duration
        duration_penalty = min(candidate.duration_sec * 0.1, 0.3)

        # Calculate total joint movement (smaller is better for efficiency)
        if len(candidate.waypoints) > 1:
            total_movement = np.sum(np.abs(np.diff(candidate.waypoints, axis=0)))
            movement_penalty = min(total_movement * 0.01, 0.2)
        else:
            movement_penalty = 0.0

        efficiency = 1.0 - waypoint_penalty - duration_penalty - movement_penalty
        return max(0.0, efficiency)

    def _calculate_safety_score(self, result) -> float:
        """Calculate safety score from validation result."""
        # Start with base score
        score = 0.5

        # Penalize based on violations
        if hasattr(result, "collisions") and result.collisions:
            score -= len(result.collisions) * 0.2

        if hasattr(result, "joint_limit_violations") and result.joint_limit_violations:
            score -= len(result.joint_limit_violations) * 0.15

        if hasattr(result, "velocity_limit_violations") and result.velocity_limit_violations:
            score -= len(result.velocity_limit_violations) * 0.1

        return max(0.0, score)

    def recommend_best(
        self,
        results: list[ValidationResult],
        require_safe: bool = True,
    ) -> ValidationResult | None:
        """
        Recommend the best trajectory from validation results.

        Args:
            results: List of validation results
            require_safe: Only consider safe trajectories

        Returns:
            Best ValidationResult or None if no valid option
        """
        # Filter for safe trajectories if required
        candidates = results
        if require_safe:
            candidates = [r for r in results if r.is_safe]

        if not candidates:
            return None

        # Already sorted by total_cost
        return candidates[0]

    def format_recommendation(
        self,
        results: list[ValidationResult],
        best: ValidationResult | None,
    ) -> str:
        """
        Format validation results into human-readable recommendation.

        This generates the message returned to LLM with trajectory comparison.
        """
        lines = ["📊 Trajectory Validation Results", ""]

        # Summary statistics
        safe_count = sum(1 for r in results if r.is_safe)
        lines.append(f"Validated {len(results)} trajectories: {safe_count} safe, {len(results) - safe_count} unsafe")
        lines.append("")

        if best:
            lines.append(f"✅ RECOMMENDED: Trajectory '{best.candidate_id}'")
            lines.append(f"   Safety Score: {best.safety_score:.2%}")
            lines.append(f"   Efficiency Score: {best.efficiency_score:.2%}")
            lines.append(f"   Validation Time: {best.sim_time_ms:.1f}ms")
            lines.append("")

        # Detailed breakdown
        lines.append("Detailed Results:")
        for i, result in enumerate(results, 1):
            status = "✅ SAFE" if result.is_safe else "❌ UNSAFE"
            lines.append(f"  {i}. {result.candidate_id}: {status}")
            lines.append(f"     Safety: {result.safety_score:.2%}, Efficiency: {result.efficiency_score:.2%}")

            if result.violations and not result.is_safe:
                lines.append(f"     Issues: {len(result.violations)} violation(s)")

        if not best:
            lines.append("")
            lines.append("⚠️ No safe trajectory found. Please replan with different approach.")

        return "\n".join(lines)

    def create_candidates_from_poses(
        self,
        pose_targets: list[np.ndarray],
        durations: list[float] | None = None,
        names: list[str] | None = None,
    ) -> list[TrajectoryCandidate]:
        """
        Create trajectory candidates from target poses.

        Args:
            pose_targets: List of target joint position arrays
            durations: List of durations (default: 2.0s each)
            names: List of candidate names (default: candidate_1, candidate_2, ...)

        Returns:
            List of TrajectoryCandidate objects
        """
        if durations is None:
            durations = [2.0] * len(pose_targets)

        if names is None:
            names = [f"candidate_{i+1}" for i in range(len(pose_targets))]

        candidates = []
        for i, (pose, duration, name) in enumerate(zip(pose_targets, durations, names)):
            # Create simple linear interpolation waypoints
            n_waypoints = max(10, int(duration * 10))  # 10Hz waypoints
            waypoints = np.linspace(
                np.zeros_like(pose),  # Current assumed at zero for relative
                pose,
                n_waypoints,
            )

            candidate = TrajectoryCandidate(
                id=f"traj_{i+1}",
                name=name,
                waypoints=waypoints,
                duration_sec=duration,
                metadata={"description": f"Linear trajectory to {name}"},
            )
            candidates.append(candidate)

        return candidates
