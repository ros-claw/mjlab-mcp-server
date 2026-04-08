"""
MJLab MCP Server - Physics Sandbox Firewall for Robot Safety Validation.

This package provides an MCP server that exposes MuJoCo physics simulation
capabilities to LLM agents for safety validation before executing real robot commands.
"""

from __future__ import annotations

from .batch_validator import BatchValidator, TrajectoryCandidate, ValidationResult
from .physics import PhysicsSandbox, SafetyCheckResult, SafetyPolicy
from .reality_sync import JointState, RealitySync, RealitySyncManager, RobotState
from .semantic_translator import SemanticContact, SemanticTranslator, SemanticViolation
from .server import mcp, main

__version__ = "0.2.0"
__all__ = [
    "PhysicsSandbox",
    "SafetyCheckResult",
    "SafetyPolicy",
    "SemanticTranslator",
    "SemanticContact",
    "SemanticViolation",
    "RealitySync",
    "RealitySyncManager",
    "RobotState",
    "JointState",
    "BatchValidator",
    "TrajectoryCandidate",
    "ValidationResult",
    "mcp",
    "main",
]
