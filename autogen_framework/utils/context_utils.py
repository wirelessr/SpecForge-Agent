"""
Shared context utility functions to avoid duplication across modules.
"""

from typing import Any, Dict, List
import json


def dict_context_to_string(context: Dict[str, Any]) -> str:
    """
    Convert a generic dictionary-based context into a readable string.
    Mirrors previous logic used in the base agent while being reusable.
    """
    try:
        parts: List[str] = []
        for key, value in context.items():
            if isinstance(value, str):
                parts.append(f"## {key}\n{value}")
            elif isinstance(value, (dict, list)):
                parts.append(f"## {key}\n{json.dumps(value, indent=2)}")
            else:
                parts.append(f"## {key}\n{str(value)}")
        return "\n\n".join(parts)
    except Exception:
        # Fallback to simple string conversion on unexpected types
        return str(context)


def agent_context_to_string(context: Any) -> str:
    """
    Convert agent-specific context dataclasses (PlanContext, DesignContext,
    TasksContext, ImplementationContext) into a string for token estimation.
    Mirrors previous logic used in ContextManager._context_to_string.
    """
    parts: List[str] = []

    # Best-effort extraction without direct type checks to avoid circular imports
    if hasattr(context, 'user_request'):
        parts.append(getattr(context, 'user_request') or "")

    if hasattr(context, 'requirements') and getattr(context, 'requirements') is not None:
        requirements = getattr(context, 'requirements')
        parts.append(getattr(requirements, 'content', '') or '')

    if hasattr(context, 'design') and getattr(context, 'design') is not None:
        design = getattr(context, 'design')
        parts.append(getattr(design, 'content', '') or '')

    if hasattr(context, 'tasks') and getattr(context, 'tasks') is not None:
        tasks = getattr(context, 'tasks')
        parts.append(getattr(tasks, 'content', '') or '')

    if hasattr(context, 'project_structure') and getattr(context, 'project_structure') is not None:
        ps = getattr(context, 'project_structure')
        files = len(getattr(ps, 'files', []) or [])
        dirs = len(getattr(ps, 'directories', []) or [])
        parts.append(f"Files: {files}, Dirs: {dirs}")

    if hasattr(context, 'memory_patterns') and getattr(context, 'memory_patterns') is not None:
        for pattern in getattr(context, 'memory_patterns'):
            parts.append(getattr(pattern, 'content', '') or '')

    return ' '.join(parts)