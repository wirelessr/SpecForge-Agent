# Dependency Injection Guide

## Overview

The AutoGen Multi-Agent Framework uses a dependency injection system that dramatically simplifies agent initialization while maintaining clean architecture and testability. This guide explains how to use the new `DependencyContainer` system for both production code and testing.

## Core Concepts

### DependencyContainer

The `DependencyContainer` is the central component that manages all framework managers and provides them to agents on demand. It eliminates the need to manually pass multiple manager dependencies during agent creation.

```python
from autogen_framework.dependency_container import DependencyContainer

# Create container for production use
container = DependencyContainer.create_production(work_dir, llm_config)

# Create container for testing with mocks
container = DependencyContainer.create_test(work_dir, llm_config)
```

### Key Benefits

1. **Simplified Initialization**: Agents only need essential parameters
2. **Shared Instances**: All managers are created once and shared efficiently
3. **Lazy Loading**: Managers are only created when first accessed
4. **Thread Safety**: Safe for concurrent access across multiple agents
5. **Easy Testing**: Mock containers provide consistent test behavior

## Migration Guide

### Before: Complex Manual Initialization

```python
# Old pattern - complex and error-prone
from autogen_framework.token_manager import TokenManager
from autogen_framework.context_manager import ContextManager
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.shell_executor import ShellExecutor
from autogen_framework.agents.error_recovery import ErrorRecovery
from autogen_framework.agents.plan_agent import PlanAgent

# Manual dependency creation and wiring
token_manager = TokenManager(config)
memory_manager = MemoryManager(work_dir)
context_compressor = ContextCompressor(llm_config, token_manager)
context_manager = ContextManager(work_dir, memory_manager, context_compressor)
error_recovery = ErrorRecovery(llm_config, token_manager, context_manager)
shell_executor = ShellExecutor(error_recovery)

# Agent creation with many parameters
agent = PlanAgent(
    name="PlanAgent",
    llm_config=llm_config,
    system_message="Generate requirements",
    token_manager=token_manager,
    context_manager=context_manager,
    memory_manager=memory_manager,
    shell_executor=shell_executor
)
```

### After: Simple Container-Based Initialization

```python
# New pattern - clean and simple
from autogen_framework.dependency_container import DependencyContainer
from autogen_framework.agents.plan_agent import PlanAgent

# Single container creation
container = DependencyContainer.create_production(work_dir, llm_config)

# Simple agent creation
agent = PlanAgent(
    name="PlanAgent",
    llm_config=llm_config,
    system_message="Generate requirements",
    container=container
)
```

## Production Usage

### Creating Agents

```python
from autogen_framework.dependency_container import DependencyContainer
from autogen_framework.agents.plan_agent import PlanAgent
from autogen_framework.agents.design_agent import DesignAgent
from autogen_framework.agents.tasks_agent import TasksAgent
from autogen_framework.agents.implement_agent import ImplementAgent

# Create production container
container = DependencyContainer.create_production(work_dir, llm_config)

# Create all agents with the same container
plan_agent = PlanAgent(
    name="PlanAgent",
    llm_config=llm_config,
    system_message="Generate project requirements",
    container=container
)

design_agent = DesignAgent(
    name="DesignAgent", 
    llm_config=llm_config,
    system_message="Create technical design",
    container=container
)

tasks_agent = TasksAgent(
    name="TasksAgent",
    llm_config=llm_config,
    system_message="Generate implementation tasks",
    container=container
)

implement_agent = ImplementAgent(
    name="ImplementAgent",
    llm_config=llm_config,
    system_message="Execute implementation tasks",
    container=container
)
```

### Accessing Managers

Agents access managers through properties that delegate to the container:

```python
class PlanAgent(BaseLLMAgent):
    async def generate_requirements(self, user_request: str, work_dir: str) -> str:
        # Access managers through clean properties
        context = await self.context_manager.get_plan_context(user_request)
        
        # Use token manager for cost tracking
        self.token_manager.start_operation("generate_requirements")
        
        # Generate requirements using LLM
        response = await self._generate_llm_response(context)
        
        # Save to memory for future reference
        await self.memory_manager.save_requirements(response)
        
        return response
```

## Testing Usage

### Unit Tests with Mock Container

```python
import pytest
from autogen_framework.agents.plan_agent import PlanAgent

class TestPlanAgent:
    def test_generate_requirements(self, mock_dependency_container, test_llm_config):
        """Test requirements generation with mocked dependencies."""
        # Simple setup using fixture
        agent = PlanAgent(
            name="TestAgent",
            llm_config=test_llm_config,
            system_message="Test message",
            container=mock_dependency_container
        )
        
        # All dependencies are automatically mocked
        result = agent.generate_requirements("Create a web API", "/tmp/test")
        
        # Verify interactions with mocked managers
        agent.context_manager.get_plan_context.assert_called_once()
        agent.memory_manager.save_requirements.assert_called_once()
```

### Integration Tests with Real Container

```python
import pytest
from autogen_framework.agents.plan_agent import PlanAgent

class TestPlanAgentIntegration:
    @pytest.mark.integration
    async def test_real_requirements_generation(self, real_dependency_container, real_llm_config):
        """Test requirements generation with real LLM and managers."""
        agent = PlanAgent(
            name="IntegrationTestAgent",
            llm_config=real_llm_config,
            system_message="Generate project requirements",
            container=real_dependency_container
        )
        
        # Test with real LLM interaction
        result = await agent.generate_requirements(
            "Create a simple web API with user authentication", 
            "/tmp/integration_test"
        )
        
        # Validate real output quality
        assert "# Requirements Document" in result
        assert "User Story:" in result
        assert "Acceptance Criteria" in result
```

## Available Test Fixtures

### Mock Container Fixtures

```python
@pytest.fixture
def mock_dependency_container(test_llm_config, temp_workspace):
    """Dependency container with all managers mocked for unit tests."""
    return DependencyContainer.create_test(temp_workspace, test_llm_config)

@pytest.fixture
def simple_plan_agent(mock_dependency_container, test_llm_config):
    """PlanAgent with mocked dependencies for unit testing."""
    return PlanAgent(
        name="TestPlanAgent",
        llm_config=test_llm_config,
        system_message="Generate requirements",
        container=mock_dependency_container
    )
```

### Real Container Fixtures

```python
@pytest.fixture
def real_dependency_container(real_llm_config, temp_workspace):
    """Dependency container with real managers for integration tests."""
    return DependencyContainer.create_production(temp_workspace, real_llm_config)

@pytest.fixture
def real_plan_agent(real_dependency_container, real_llm_config):
    """PlanAgent with real dependencies for integration testing."""
    return PlanAgent(
        name="RealPlanAgent",
        llm_config=real_llm_config,
        system_message="Generate requirements",
        container=real_dependency_container
    )
```

## Best Practices

### Container Lifecycle

1. **Create Once**: Create the container once per application/test session
2. **Share Across Agents**: Use the same container for all agents that need to share state
3. **Fresh for Tests**: Each test should get a fresh container instance for isolation

### Error Handling

```python
from autogen_framework.dependency_container import DependencyError

try:
    container = DependencyContainer.create_production(work_dir, llm_config)
    agent = PlanAgent(container=container, llm_config=llm_config)
except DependencyError as e:
    logger.error(f"Failed to initialize dependencies: {e}")
    # Handle gracefully - don't hide the real error
    raise
```

### Custom Manager Configuration

```python
# For advanced use cases, you can customize manager creation
container = DependencyContainer.create_production(work_dir, llm_config)

# Access managers directly if needed (rare)
token_manager = container.get_token_manager()
context_manager = container.get_context_manager()

# Managers are singletons - same instance returned each time
assert container.get_token_manager() is token_manager
```

## Troubleshooting

### Common Issues

1. **Missing Work Directory**: Ensure the work directory exists before creating the container
2. **Invalid LLM Config**: Validate LLM configuration before passing to container
3. **Circular Dependencies**: The container handles dependency resolution automatically
4. **Thread Safety**: The container is thread-safe, but individual managers may have their own threading considerations

### Debug Information

```python
# Enable debug logging to see container initialization
import logging
logging.getLogger('autogen_framework.dependency_container').setLevel(logging.DEBUG)

# Check what managers are loaded
container = DependencyContainer.create_production(work_dir, llm_config)
print(f"Loaded managers: {list(container._managers.keys())}")
```

### Performance Considerations

- Managers are created lazily - only when first accessed
- All managers are singletons within a container instance
- Container creation is lightweight - actual manager creation happens on demand
- Thread-safe access may have slight overhead in high-concurrency scenarios

## Migration Checklist

When migrating existing code to use the dependency injection system:

- [ ] Replace manual manager creation with `DependencyContainer.create_production()`
- [ ] Update agent constructors to accept `container` parameter
- [ ] Remove explicit manager parameters from agent creation
- [ ] Update tests to use `mock_dependency_container` or `real_dependency_container` fixtures
- [ ] Remove manual mock setup in favor of container-provided mocks
- [ ] Verify all tests pass with the new initialization pattern
- [ ] Update documentation and examples to show new patterns

## Advanced Usage

### Custom Manager Factories

For specialized use cases, you can extend the container with custom manager factories:

```python
class CustomDependencyContainer(DependencyContainer):
    def get_custom_manager(self) -> CustomManager:
        """Get or create custom manager instance."""
        return self._get_or_create('custom_manager', self._create_custom_manager)
    
    def _create_custom_manager(self) -> CustomManager:
        """Factory method for custom manager."""
        return CustomManager(
            config=self.custom_config,
            dependency=self.get_token_manager()
        )
```

### Container Configuration

```python
from dataclasses import dataclass

@dataclass
class ContainerConfig:
    work_dir: str
    llm_config: LLMConfig
    use_mocks: bool = False
    enable_debug: bool = False

# Use configuration for container creation
config = ContainerConfig(
    work_dir="/path/to/workspace",
    llm_config=llm_config,
    enable_debug=True
)
container = DependencyContainer.create_from_config(config)
```

This dependency injection system provides a clean, maintainable, and testable foundation for the AutoGen Multi-Agent Framework. It eliminates complexity while maintaining flexibility and performance.