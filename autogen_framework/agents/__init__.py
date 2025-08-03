"""
AI Agents module for the AutoGen multi-agent framework.

This module contains all the AI agent implementations including the base abstract
class and specialized agents for different phases of the development workflow.
"""

from .base_agent import BaseLLMAgent

__all__ = [
    'BaseLLMAgent',
]