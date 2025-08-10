"""
Utility modules for the AutoGen Multi-Agent Framework.

This package contains various utility functions and classes used throughout
the framework for testing, context management, and other common operations.
"""

from .context_utils import dict_context_to_string, agent_context_to_string

# Import test_utils only when needed to avoid circular imports
def _get_manager_dependencies():
    from .test_utils import ManagerDependencies
    return ManagerDependencies

# Lazy import for ManagerDependencies
def __getattr__(name):
    if name == 'ManagerDependencies':
        return _get_manager_dependencies()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'ManagerDependencies',
    'dict_context_to_string', 
    'agent_context_to_string'
]