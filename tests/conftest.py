"""
Shared test fixtures and configuration for all tests.

This module provides common test fixtures that can be used across
unit tests, integration tests, and e2e tests.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from autogen_framework.config_manager import ConfigManager, reset_config_manager
from autogen_framework.models import LLMConfig


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables for the entire test session."""
    # Set test environment flag
    os.environ['TESTING'] = 'true'
    
    yield
    
    # Clean up test environment
    os.environ.pop('TESTING', None)


@pytest.fixture
def test_config_manager():
    """Create a ConfigManager instance for testing."""
    # Reset global config manager to ensure clean state
    reset_config_manager()
    
    # Create test config manager that doesn't load .env files
    config_manager = ConfigManager(load_env=False)
    
    yield config_manager
    
    # Reset after test
    reset_config_manager()


@pytest.fixture
def test_llm_config():
    """
    Create a test LLM configuration for unit tests.
    
    This uses mock/test values suitable for unit testing with mocked services.
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
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="autogen_test_")
    
    # Create basic directory structure
    memory_dir = Path(temp_dir) / "memory"
    memory_dir.mkdir()
    (memory_dir / "global").mkdir()
    (memory_dir / "projects").mkdir()
    
    logs_dir = Path(temp_dir) / "logs"
    logs_dir.mkdir()
    
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_env_vars():
    """Provide a context manager for mocking environment variables in tests."""
    def _mock_env(**env_vars):
        return patch.dict(os.environ, env_vars, clear=False)
    
    return _mock_env


# Dependency Container Fixtures

@pytest.fixture
def mock_dependency_container(test_llm_config, temp_workspace):
    """
    Provide dependency container with mock managers for unit tests.
    
    This fixture creates a DependencyContainer configured for testing with
    all managers mocked. Each test gets a fresh container instance to ensure
    test isolation.
    
    Returns:
        DependencyContainer configured with mock managers
    """
    from autogen_framework.dependency_container import DependencyContainer
    
    container = DependencyContainer.create_test(temp_workspace, test_llm_config)
    return container





# Agent Fixtures using Dependency Containers

@pytest.fixture
def simple_plan_agent(mock_dependency_container, test_llm_config):
    """
    Provide PlanAgent with mocked dependencies for unit tests.
    
    This fixture creates a PlanAgent using the mock dependency container,
    making it suitable for fast unit tests that don't require real LLM calls.
    
    Returns:
        PlanAgent configured with mock dependencies
    """
    from autogen_framework.agents.plan_agent import PlanAgent
    
    return PlanAgent(
        container=mock_dependency_container,
        llm_config=test_llm_config
    )


# TODO: Update these fixtures when their respective agents are updated to use container-only initialization

@pytest.fixture
def simple_design_agent(mock_dependency_container, test_llm_config):
    """
    Provide DesignAgent with mocked dependencies for unit tests.
    
    This fixture creates a DesignAgent using the mock dependency container,
    making it suitable for fast unit tests that don't require real LLM calls.
    
    Returns:
        DesignAgent configured with mock dependencies
    """
    from autogen_framework.agents.design_agent import DesignAgent
    
    return DesignAgent(
        container=mock_dependency_container,
        llm_config=test_llm_config
    )


@pytest.fixture
def simple_tasks_agent(mock_dependency_container, test_llm_config):
    """
    Provide TasksAgent with mocked dependencies for unit tests.
    
    Returns:
        TasksAgent configured with mock dependencies
    """
    from autogen_framework.agents.tasks_agent import TasksAgent
    
    return TasksAgent(
        container=mock_dependency_container,
        name="TasksAgent",
        llm_config=test_llm_config
    )


@pytest.fixture
def simple_implement_agent(mock_dependency_container, test_llm_config):
    """
    Provide ImplementAgent with mocked dependencies for unit tests.
    
    Returns:
        ImplementAgent configured with mock dependencies
    """
    from autogen_framework.agents.implement_agent import ImplementAgent
    
    return ImplementAgent(
        container=mock_dependency_container,
        name="ImplementAgent",
        llm_config=test_llm_config,
        system_message="Test implementation agent"
    )


# Legacy fixture for backward compatibility
@pytest.fixture
def llm_config(test_llm_config):
    """Legacy fixture name for backward compatibility."""
    return test_llm_config