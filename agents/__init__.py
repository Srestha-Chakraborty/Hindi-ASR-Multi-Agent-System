"""
LangGraph agent suite (canonical implementation lives in `asr_multiagent.agents`).

This package re-exports the orchestrator so `import agents` works when the repo root is on PYTHONPATH.
"""

from asr_multiagent.agents.orchestrator import build_graph, run_suite

__all__ = ["build_graph", "run_suite"]
