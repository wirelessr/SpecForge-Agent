"""
Unit test specific fixtures and configuration.

This module provides fixtures specifically designed for unit tests,
ensuring they use only mocked dependencies and no real services.

Key Fixtures:
- llm_config: Mock LLM configuration for unit tests
- mock_memory_context: Predefined memory context data
- mock_autogen_agent: Mock AutoGen agent instance
- mock_autogen_client: Mock AutoGen client instance
- mock_shell_executor: Mock shell executor with execution results
- mock_memory_manager: Mock memory manager with standard operations
- mock_agent_manager: Mock agent manager with coordination methods
- disable_real_services: Auto-fixture preventing real service calls

For detailed usage patterns, see:
- tests/TEST_PATTERNS.md - Standard test patterns and conventions
- tests/MOCK_SETUP_GUIDE.md - Complex mock scenarios and explanations
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock

from autogen_framework.models import LLMConfig


@pytest.fixture
def llm_config():
    """
    Create a mock LLM configuration for unit tests.
    
    This configuration is designed for unit testing with mocked services only.
    It should never be used to make real API calls.
    """
    return LLMConfig(
        base_url="http://test.local:8888/openai/v1",
        model="test-model",
        api_key="test-key",
        temperature=0.7,
        max_output_tokens=4096,
        timeout=30
    )


@pytest.fixture
def mock_memory_context():
    """Create a mock memory context for unit tests."""
    return {
        "global": {
            "test_patterns.md": "# Test Patterns\n\nMocked test patterns content.",
            "best_practices.md": "# Best Practices\n\nMocked best practices content."
        },
        "projects": {
            "test_project": {
                "context.md": "# Project Context\n\nMocked project context."
            }
        },
        "root": {
            "general_notes.md": "# General Notes\n\nMocked general notes."
        }
    }


@pytest.fixture
def mock_autogen_agent():
    """Create a mock AutoGen agent for unit tests."""
    mock_agent = MagicMock()
    mock_agent.name = "MockAgent"
    mock_agent.on_messages = Mock()
    return mock_agent


@pytest.fixture
def mock_autogen_client():
    """Create a mock AutoGen client for unit tests."""
    mock_client = MagicMock()
    mock_client.model = "test-model"
    return mock_client


@pytest.fixture
def mock_shell_executor():
    """Create a mock shell executor for unit tests."""
    from autogen_framework.models import ExecutionResult
    
    mock_executor = Mock()
    mock_executor.execute_command = Mock()
    mock_executor.execute_multiple_commands = Mock()
    mock_executor.execution_history = []
    mock_executor.get_execution_stats = Mock(return_value={
        "total_executions": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "success_rate": 0.0,
        "average_execution_time": 0.0
    })
    
    # Default successful execution result
    success_result = ExecutionResult.create_success(
        command="echo 'test'",
        stdout="test",
        execution_time=0.1,
        working_directory="/tmp",
        approach_used="direct_execution"
    )
    mock_executor.execute_command.return_value = success_result
    mock_executor.execute_multiple_commands.return_value = [success_result]
    
    return mock_executor


@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager for unit tests."""
    mock_manager = Mock()
    mock_manager.load_memory = Mock(return_value={})
    mock_manager.save_memory = Mock(return_value=True)
    mock_manager.search_memory = Mock(return_value=[])
    mock_manager.get_memory_stats = Mock(return_value={
        "total_categories": 0,
        "total_files": 0,
        "total_size_chars": 0,
        "categories": {}
    })
    mock_manager.get_system_instructions = Mock(return_value="No memory content available.")
    mock_manager.export_memory = Mock(return_value=True)
    mock_manager.clear_cache = Mock()
    
    return mock_manager


@pytest.fixture
def mock_agent_manager():
    """Create a mock agent manager for unit tests."""
    mock_manager = Mock()
    mock_manager.setup_agents = Mock(return_value=True)
    mock_manager.coordinate_agents = Mock()
    mock_manager.get_agent_status = Mock(return_value={})
    mock_manager.get_agent_capabilities = Mock(return_value={})
    mock_manager.update_agent_memory = Mock()
    mock_manager.reset_coordination_state = Mock()
    mock_manager.export_coordination_report = Mock(return_value=True)
    mock_manager.get_coordination_log = Mock(return_value=[])
    mock_manager.get_agent_interactions = Mock(return_value=[])
    mock_manager.get_workflow_state = Mock(return_value=None)
    mock_manager.get_coordination_statistics = Mock(return_value={
        "total_interactions": 0,
        "total_coordination_events": 0,
        "agent_interaction_counts": {},
        "event_type_counts": {},
        "agents_initialized": 0,
        "current_workflow_phase": None
    })
    
    return mock_manager


@pytest.fixture
def mock_token_manager():
    """Create a mock TokenManager for unit tests."""
    mock_manager = Mock()
    
    # Configure common token management methods
    mock_manager.extract_token_usage_from_response = Mock(return_value=100)
    mock_manager.estimate_tokens_from_char_count = Mock(return_value=50)
    mock_manager.estimate_tokens_from_text = Mock(return_value=75)
    mock_manager.update_token_usage = Mock()
    mock_manager.check_token_limit = Mock(return_value=Mock(
        needs_compression=False,
        current_tokens=100,
        model_limit=4000,
        percentage_used=0.025
    ))
    mock_manager.get_model_limit = Mock(return_value=4000)
    mock_manager.get_usage_stats = Mock(return_value={
        "requests_made": 0,
        "total_tokens_used": 0,
        "compressions_triggered": 0,
        "average_tokens_per_request": 0.0
    })
    mock_manager.reset_context_size = Mock()
    mock_manager.increment_compression_count = Mock()
    
    return mock_manager


@pytest.fixture
def mock_context_manager():
    """Create a mock ContextManager for unit tests."""
    mock_manager = Mock()
    
    # Configure async methods with AsyncMock
    mock_manager.prepare_system_prompt = AsyncMock(return_value=Mock(
        system_prompt="prepared system prompt",
        estimated_tokens=100
    ))
    mock_manager.get_plan_context = AsyncMock(return_value=Mock(
        user_request="test request",
        memory_patterns=[],
        project_structure=None
    ))
    mock_manager.get_design_context = AsyncMock(return_value=Mock(
        user_request="test request",
        requirements=Mock(content="test requirements"),
        memory_patterns=[],
        project_structure=None
    ))
    mock_manager.get_tasks_context = AsyncMock(return_value=Mock(
        user_request="test request",
        requirements=Mock(content="test requirements"),
        design=Mock(content="test design"),
        memory_patterns=[],
        project_structure=None
    ))
    mock_manager.get_implementation_context = AsyncMock(return_value=Mock(
        task=Mock(description="test task"),
        requirements=Mock(content="test requirements"),
        design=Mock(content="test design"),
        tasks=Mock(content="test tasks"),
        execution_history=[],
        related_tasks=[],
        project_structure=None
    ))
    mock_manager.update_execution_history = AsyncMock()
    mock_manager.initialize = AsyncMock()
    
    # Configure sync methods
    mock_manager.get_memory_patterns = Mock(return_value=[])
    mock_manager.get_project_structure = Mock(return_value=None)
    
    return mock_manager


@pytest.fixture(autouse=True)
def disable_real_services():
    """
    Automatically disable real service calls in unit tests.
    
    This fixture runs automatically for all unit tests to ensure
    no real external services are called.
    """
    import os
    
    # Set environment flag to indicate unit testing
    original_testing = os.environ.get('UNIT_TESTING')
    os.environ['UNIT_TESTING'] = 'true'
    
    yield
    
    # Restore original value
    if original_testing is None:
        os.environ.pop('UNIT_TESTING', None)
    else:
        os.environ['UNIT_TESTING'] = original_testing


# Ensure unit tests don't accidentally use integration fixtures
@pytest.fixture
def real_llm_config():
    """
    Prevent unit tests from using real LLM configuration.
    
    This fixture raises an error if unit tests try to use real configuration.
    """
    raise RuntimeError(
        "Unit tests should not use real_llm_config. "
        "Use the 'llm_config' fixture instead for mocked configuration."
    )