"""
Integration test fixtures and configuration.

These fixtures are specifically designed for integration tests that use real LLM endpoints
and test actual functionality with external services.

Key Fixtures:
- real_llm_config: Real LLM configuration loaded from .env.integration
- integration_config_manager: ConfigManager with real environment loading
- llm_config: Override that uses real_llm_config for integration tests
- enable_real_services: Auto-fixture enabling real external service calls
- mock_llm_config: Raises error to prevent accidental use in integration tests

Configuration Requirements:
- .env.integration file must exist with valid LLM credentials
- LLM_BASE_URL, LLM_MODEL, and LLM_API_KEY must be set
- External services must be accessible from test environment

For detailed usage patterns, see:
- tests/TEST_PATTERNS.md - Standard test patterns and conventions
- tests/README.md - Test organization and best practices
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

from autogen_framework.config_manager import ConfigManager
from autogen_framework.models import LLMConfig
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.token_manager import TokenManager
from autogen_framework.context_manager import ContextManager
from autogen_framework.context_compressor import ContextCompressor
from autogen_framework.utils import ManagerDependencies


@pytest.fixture
def real_llm_config():
    """
    Create real LLM configuration for integration testing.
    
    This loads configuration from .env.integration file for testing with real services.
    """
    # Load integration test environment
    load_dotenv('.env.integration')
    
    return LLMConfig(
        base_url=os.getenv('LLM_BASE_URL', 'http://ctwuhome.local:8888/openai/v1'),
        model=os.getenv('LLM_MODEL', 'models/gemini-2.0-flash'),
        api_key=os.getenv('LLM_API_KEY'),
        temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
        max_output_tokens=int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '4096')),
        timeout=int(os.getenv('LLM_TIMEOUT_SECONDS', '60'))
    )


@pytest.fixture
def integration_config_manager():
    """Create a ConfigManager for integration testing with real environment loading."""
    return ConfigManager(load_env=True)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for integration testing."""
    temp_dir = tempfile.mkdtemp(prefix="autogen_integration_test_")
    
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
def real_memory_manager(temp_workspace):
    """Create a real MemoryManager for integration testing."""
    return MemoryManager(workspace_path=temp_workspace)


@pytest.fixture
def real_token_manager(integration_config_manager):
    """Create a real TokenManager for integration testing."""
    return TokenManager(integration_config_manager)


@pytest.fixture
def real_managers(temp_workspace, real_memory_manager, real_llm_config):
    """Create real manager dependencies for integration testing."""
    return ManagerDependencies.create_for_workspace(
        workspace_path=temp_workspace,
        memory_manager=real_memory_manager,
        llm_config=real_llm_config,
        load_env=True
    )


@pytest.fixture
async def initialized_real_managers(temp_workspace, real_memory_manager, real_llm_config):
    """Create and initialize real manager dependencies for integration testing."""
    return await ManagerDependencies.create_and_initialize_for_workspace(
        workspace_path=temp_workspace,
        memory_manager=real_memory_manager,
        llm_config=real_llm_config,
        load_env=True
    )


@pytest.fixture
def real_context_manager(real_managers):
    """Create a real ContextManager for integration testing."""
    return real_managers.context_manager


@pytest.fixture
async def initialized_real_context_manager(initialized_real_managers):
    """Create and initialize a real ContextManager for integration testing."""
    return (await initialized_real_managers).context_manager


# Override the default llm_config fixture for integration tests
@pytest.fixture
def llm_config(real_llm_config):
    """Integration tests use real LLM configuration."""
    return real_llm_config


@pytest.fixture(autouse=True)
def enable_real_services():
    """
    Automatically enable real service calls in integration tests.
    
    This fixture runs automatically for all integration tests to ensure
    real external services can be called.
    """
    import os
    
    # Set environment flag to indicate integration testing
    original_integration = os.environ.get('INTEGRATION_TESTING')
    os.environ['INTEGRATION_TESTING'] = 'true'
    
    # Remove unit testing flag if present
    original_unit = os.environ.get('UNIT_TESTING')
    if 'UNIT_TESTING' in os.environ:
        del os.environ['UNIT_TESTING']
    
    yield
    
    # Restore original values
    if original_integration is None:
        os.environ.pop('INTEGRATION_TESTING', None)
    else:
        os.environ['INTEGRATION_TESTING'] = original_integration
        
    if original_unit is not None:
        os.environ['UNIT_TESTING'] = original_unit


# Dependency Container Fixtures for Integration Tests

@pytest.fixture
def real_dependency_container(temp_workspace, real_llm_config):
    """
    Provide dependency container with real managers for integration tests.
    
    This fixture creates a DependencyContainer configured for production use
    with real managers. It uses the real_llm_config fixture from integration
    testing configuration.
    
    Note: This fixture only creates the container - managers are created lazily
    when accessed to avoid hanging during fixture setup.
    
    Returns:
        DependencyContainer configured with real managers
    """
    from autogen_framework.dependency_container import DependencyContainer
    
    container = DependencyContainer.create_production(temp_workspace, real_llm_config)
    return container


# Agent Fixtures for Integration Tests

@pytest.fixture
def real_plan_agent(real_dependency_container):
    """
    Provide PlanAgent with real dependencies for integration tests.
    
    This fixture creates a PlanAgent using the real dependency container,
    making it suitable for integration tests that require real LLM calls
    and actual manager functionality.
    
    Returns:
        PlanAgent configured with real dependencies
    """
    from autogen_framework.agents.plan_agent import PlanAgent
    
    return PlanAgent(
        llm_config=real_dependency_container.llm_config,
        memory_manager=real_dependency_container.get_memory_manager(),
        token_manager=real_dependency_container.get_token_manager(),
        context_manager=real_dependency_container.get_context_manager(),
        config_manager=real_dependency_container.get_config_manager()
    )


@pytest.fixture
def real_design_agent(real_dependency_container):
    """
    Provide DesignAgent with real dependencies for integration tests.
    
    Returns:
        DesignAgent configured with real dependencies
    """
    from autogen_framework.agents.design_agent import DesignAgent
    
    return DesignAgent(
        llm_config=real_dependency_container.llm_config,
        token_manager=real_dependency_container.get_token_manager(),
        context_manager=real_dependency_container.get_context_manager(),
        config_manager=real_dependency_container.get_config_manager()
    )


@pytest.fixture
def real_tasks_agent(real_dependency_container):
    """
    Provide TasksAgent with real dependencies for integration tests.
    
    Returns:
        TasksAgent configured with real dependencies
    """
    from autogen_framework.agents.tasks_agent import TasksAgent
    
    return TasksAgent(
        llm_config=real_dependency_container.llm_config,
        token_manager=real_dependency_container.get_token_manager(),
        context_manager=real_dependency_container.get_context_manager(),
        config_manager=real_dependency_container.get_config_manager()
    )


@pytest.fixture
def real_implement_agent(real_dependency_container):
    """
    Provide ImplementAgent with real dependencies for integration tests.
    
    Returns:
        ImplementAgent configured with real dependencies
    """
    from autogen_framework.agents.implement_agent import ImplementAgent
    
    return ImplementAgent(
        name="ImplementAgent",
        llm_config=real_dependency_container.llm_config,
        system_message="Real implementation agent for integration testing",
        shell_executor=real_dependency_container.get_shell_executor(),
        token_manager=real_dependency_container.get_token_manager(),
        context_manager=real_dependency_container.get_context_manager(),
        task_decomposer=real_dependency_container.get_task_decomposer(),
        error_recovery=real_dependency_container.get_error_recovery(),
        config_manager=real_dependency_container.get_config_manager()
    )


# Ensure integration tests don't accidentally use unit test fixtures
@pytest.fixture
def mock_llm_config():
    """
    Prevent integration tests from using mock LLM configuration.
    
    This fixture raises an error if integration tests try to use mock configuration.
    """
    raise RuntimeError(
        "Integration tests should not use mock_llm_config. "
        "Use the 'real_llm_config' fixture instead for real configuration."
    )