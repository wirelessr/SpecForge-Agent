# Dependency Container Fixtures Guide

This document explains how to use the new dependency container fixtures for simplified agent testing.

## Overview

The dependency container fixtures provide a clean, simple way to create agents for testing without manually managing multiple manager dependencies. They support both unit tests (with mocked dependencies) and integration tests (with real dependencies).

## Available Fixtures

### Container Fixtures

#### `mock_dependency_container`
- **Purpose**: Unit testing with mocked managers
- **Usage**: Provides a DependencyContainer with all managers mocked
- **Isolation**: Each test gets a fresh container instance

```python
def test_my_component(mock_dependency_container):
    # All managers are mocked and ready to use
    token_manager = mock_dependency_container.get_token_manager()
    context_manager = mock_dependency_container.get_context_manager()
```

#### `real_dependency_container`
- **Purpose**: Integration testing with real managers
- **Usage**: Provides a DependencyContainer with real managers
- **Configuration**: Loads from `.env.integration`
- **Note**: Managers are created lazily to avoid hanging during setup

```python
@pytest.mark.integration
def test_my_integration(real_dependency_container):
    # Real managers for integration testing
    config_manager = real_dependency_container.get_config_manager()
    memory_manager = real_dependency_container.get_memory_manager()
```

### Agent Fixtures

#### Unit Test Agent Fixtures (Mocked Dependencies)

- `simple_plan_agent`: PlanAgent with mocked dependencies
- `simple_design_agent`: DesignAgent with mocked dependencies  
- `simple_tasks_agent`: TasksAgent with mocked dependencies
- `simple_implement_agent`: ImplementAgent with mocked dependencies

```python
def test_plan_agent_functionality(simple_plan_agent):
    # Agent is ready to use with all dependencies mocked
    assert simple_plan_agent.name == "PlanAgent"
    # Test agent functionality without real LLM calls
```

#### Integration Test Agent Fixtures (Real Dependencies)

- `real_plan_agent`: PlanAgent with real dependencies
- `real_design_agent`: DesignAgent with real dependencies
- `real_tasks_agent`: TasksAgent with real dependencies  
- `real_implement_agent`: ImplementAgent with real dependencies

```python
@pytest.mark.integration
async def test_plan_agent_real_llm(real_plan_agent):
    # Agent with real LLM configuration for integration testing
    response = await real_plan_agent.generate_response("Test request")
    assert response is not None
```

## Migration from Old Patterns

### Before (Complex Manual Setup)

```python
def test_plan_agent_old_way(test_llm_config, temp_workspace):
    # Manual setup of all dependencies
    mock_token_manager = Mock(spec=TokenManager)
    mock_context_manager = Mock(spec=ContextManager)
    mock_memory_manager = Mock(spec=MemoryManager)
    
    # Configure mocks manually
    mock_token_manager.get_model_limit.return_value = 8192
    mock_context_manager.initialize.return_value = None
    
    agent = PlanAgent(
        llm_config=test_llm_config,
        memory_manager=mock_memory_manager,
        token_manager=mock_token_manager,
        context_manager=mock_context_manager
    )
```

### After (Simple Fixture-Based Setup)

```python
def test_plan_agent_new_way(simple_plan_agent):
    # Agent is ready to use - all dependencies automatically mocked
    agent = simple_plan_agent
    # Test functionality directly
```

## Best Practices

### Unit Tests
- Use `simple_*_agent` fixtures for fast unit tests
- All dependencies are automatically mocked with sensible defaults
- Focus on testing agent logic, not dependency integration

### Integration Tests
- Use `real_*_agent` fixtures for integration tests that need real LLM calls
- Mark tests with `@pytest.mark.integration`
- Be aware that these tests may be slower and require network access

### Container Usage
- Use `mock_dependency_container` when you need to create multiple agents or custom configurations
- Use `real_dependency_container` for integration tests that need real managers
- Containers provide lazy loading - managers are only created when accessed

### Test Isolation
- Each test gets a fresh container instance automatically
- No need to manually reset or clean up between tests
- Mocked managers have consistent behavior across tests

## Examples

### Simple Unit Test
```python
def test_agent_basic_functionality(simple_plan_agent):
    """Test basic agent functionality with mocked dependencies."""
    assert simple_plan_agent.name == "PlanAgent"
    assert simple_plan_agent.token_manager is not None
    # Test agent methods that don't require real LLM calls
```

### Custom Agent Creation
```python
def test_custom_agent_setup(mock_dependency_container, test_llm_config):
    """Create agent with custom configuration using container."""
    agent = PlanAgent(
        llm_config=test_llm_config,
        memory_manager=mock_dependency_container.get_memory_manager(),
        token_manager=mock_dependency_container.get_token_manager(),
        context_manager=mock_dependency_container.get_context_manager(),
        config_manager=mock_dependency_container.get_config_manager()
    )
    # Test custom agent behavior
```

### Integration Test
```python
@pytest.mark.integration
async def test_agent_real_llm_interaction(real_plan_agent):
    """Test agent with real LLM for integration validation."""
    # This test makes real LLM calls
    result = await real_plan_agent.generate_requirements(
        "Create a simple web API", 
        "/tmp/test_workspace"
    )
    assert "# Requirements Document" in result
```

## Troubleshooting

### Tests Hanging
- If integration tests hang, check that `.env.integration` has valid configuration
- Real agent fixtures may hang if AutoGen initialization fails
- Use container fixtures instead of agent fixtures for lightweight testing

### Mock Behavior
- All mocked managers have sensible default return values
- Customize mock behavior by accessing managers through the container
- Use `mock_dependency_container.get_manager_name()` to get specific managers

### Configuration Issues
- Unit tests use `test_llm_config` (mock configuration)
- Integration tests load from `.env.integration`
- Ensure environment files exist and have valid values

This fixture system dramatically simplifies agent testing while maintaining full flexibility for both unit and integration test scenarios.