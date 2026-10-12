"""Acting policies behind one interface: ``act(state, seat) -> action``.

Re-exports the agent protocol and the concrete players used by the environment,
the evaluation scripts and the opponent league.
"""
from __future__ import annotations

from .base import Agent
from .policy_agent import PolicyAgent
from .pool import OpponentPool
from .random_agent import RandomAgent
from .rule_agent import RuleAgent, RuleWeights

__all__ = ["Agent", "OpponentPool", "PolicyAgent", "RandomAgent", "RuleAgent", "RuleWeights"]
