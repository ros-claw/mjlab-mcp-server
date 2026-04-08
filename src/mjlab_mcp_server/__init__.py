"""
MJLab MCP Server - Physics Sandbox Firewall for Robot Safety Validation.

This package provides an MCP server that exposes MuJoCo physics simulation
capabilities to LLM agents for safety validation before executing real robot commands.
"""

from __future__ import annotations

from .physics import PhysicsSandbox, SafetyCheckResult, SafetyPolicy
from .server import mcp, main

__version__ = "0.1.0"
__all__ = [
    "PhysicsSandbox",
    "SafetyCheckResult",
    "SafetyPolicy",
    "mcp",
    "main",
]
