# AutoGen Multi-Agent Framework Test Suite

This document describes the test organization, conventions, and best practices for the AutoGen Multi-Agent Framework.

## 📚 Quick Navigation

- **📋 [TESTING_STANDARDS_SUMMARY.md](./TESTING_STANDARDS_SUMMARY.md)** - Quick reference guide
- **🎯 [TEST_PATTERNS.md](./TEST_PATTERNS.md)** - Detailed patterns and conventions  
- **🔧 [MOCK_SETUP_GUIDE.md](./MOCK_SETUP_GUIDE.md)** - Complex mock scenarios with explanations

## Test Organization

The test suite is organized into two main categories:

### Unit Tests (`tests/unit/`)

Unit tests focus on testing individual components in isolation with mocked dependencies. They are designed to:

- **Run quickly** (target: under 10 seconds for the entire unit test suite)
- **Use only mocked dependencies** (no real external services)
- **Test logic paths and error handling** without external calls
- **Validate component interfaces and behavior**

**Key Characteristics:**
- All external dependencies are mocked (AutoGen, LLM APIs, file system operations)
- Tests complete in milliseconds, not seconds
- No network calls or real service dependencies
- Use `mock_llm_config` fixture for configuration
- Can run without `.env.integration` file

### Integration Tests (`tests/integration/`)

Integration tests verify that components work correctly with real external services. They are designed to:

- **Test actual service integration** (real AutoGen, LLM endpoints)
- **Verify end-to-end functionality** with real components
- **Validate configuration and service communication**
- **Test real file system operations and memory management**
- **Test autonomous execution components** (TaskDecomposer, ErrorRecovery, ContextManager)

**Key Characteristics:**
- Use real LLM configuration from `.env.integration`
- Make actual API calls to external services
- Test real AutoGen agent initialization and communication
- Test enhanced ImplementAgent with autonomous capabilities
- Use `real_llm_config` fixture for configuration
- May take longer to complete due to network calls

### Quality Tests (`tests/quality/`)

Quality tests measure and validate the quality of autonomous execution output. They are designed to:

- **Measure implementation quality** using objective metrics
- **Compare against established baselines** for regression detection
- **Validate quality gates** for different development phases
- **Test quality measurement framework** itself

**Key Characteristics:**
- Use real tasks.md files for quality assessment
- Measure functionality, maintainability, standards compliance
- Track quality improvements over time
- Integrate with quality gate management system

## Test Naming Conventions

### Test File Naming

- **Unit tests**: `test_<component_name>.py`
- **Integration tests**: `test_real_<component_name>.py`

### Test Class Naming

- **Unit test classes**: `Test<ComponentName>` (e.g., `TestBaseLLMAgent`)
- **Mocking test classes**: `Test<ComponentName>Mocking` (e.g., `TestBaseLLMAgentMocking`)
- **Integration test classes**: `Test<ComponentName>RealIntegration` (e.g., `TestBaseLLMAgentRealIntegration`)

### Test Method Naming

- **Unit tests**: `test_<functionality>` (e.g., `test_agent_initialization`)
- **Integration tests**: `test_real_<functionality>` (e.g., `test_real_autogen_initialization`)
- **Mocking tests**: `test_<functionality>_with_mocks` (e.g., `test_design_generation_with_mocks`)

## Test Fixtures and Configuration

### Unit Test Fixtures (`tests/unit/conftest.py`)

- `llm_config`: Mock LLM configuration for unit tests
- `mock_memory_context`: Mock memory context data
- `mock_autogen_agent`: Mock AutoGen agent instance
- `mock_autogen_client`: Mock AutoGen client instance
- `mock_shell_executor`: Mock shell executor
- `mock_memory_manager`: Mock memory manager
- `mock_agent_manager`: Mock agent manager
- `disable_real_services`: Auto-fixture that prevents real service calls

### Integration Test Fixtures (`tests/integration/conftest.py`)

- `real_llm_config`: Real LLM configuration from `.env.integration`
- `integration_config_manager`: Real configuration manager
- `enable_real_services`: Auto-fixture that enables real service calls

### Shared Fixtures (`tests/conftest.py`)

- `temp_workspace`: Temporary workspace directory
- `test_config_manager`: Test configuration manager
- `mock_env_vars`: Environment variable mocking helper

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Only Unit Tests (Fast)
```bash
pytest tests/unit/
```

### Run Only Integration Tests
```bash
pytest tests/integration/ -m integration
```

### Run Tests with Performance Timing
```bash
pytest tests/unit/ --durations=10
```

### Run Specific Test Categories
```bash
# Run only mocking tests
pytest tests/unit/ -k "Mocking"

# Run only real integration tests
pytest tests/integration/ -k "Real"
```

## Test Markers

- `@pytest.mark.integration`: Marks tests that require real services
- `@pytest.mark.asyncio`: Marks asynchronous tests

## Mock Pattern Standards

### Standard Mock Setup

All unit tests should follow these mock patterns for consistency:

```python
# ✅ Standard mock configuration
@pytest.fixture
def mock_component(self):
    """Create mock component with explicit return values."""
    mock_comp = Mock()
    mock_comp.method_name.return_value = expected_result
    mock_comp.async_method = AsyncMock(return_value=expected_async_result)
    return mock_comp

# ✅ Environment variable mocking
@patch.dict(os.environ, {
    'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
    'LLM_MODEL': 'test-model',
    'LLM_API_KEY': 'test-key'
})
def test_with_env_vars(self):
    """Test with mocked environment variables."""
    pass

# ✅ External dependency mocking
@patch('module.path.ExternalDependency')
def test_with_external_dep(self, mock_external):
    """Test with mocked external dependency."""
    mock_instance = Mock()
    mock_instance.method.return_value = expected_result
    mock_external.return_value = mock_instance
    
    # Test execution and verification
    result = subject_under_test.method()
    assert result == expected_result
    mock_instance.method.assert_called_once_with(expected_params)
```

### Mock Assertion Standards

Use consistent assertion patterns for mock verification:

```python
# ✅ Standard mock assertions
mock_method.assert_called_once()
mock_method.assert_called_once_with(expected_param)
mock_method.assert_not_called()
assert mock_method.call_count == expected_count

# ✅ Complex mock verification
mock_method.assert_has_calls([
    call(param1),
    call(param2),
    call(param3)
])
```

For detailed mock patterns and examples, see:
- [TEST_PATTERNS.md](./TEST_PATTERNS.md) - Standard patterns and conventions
- [MOCK_SETUP_GUIDE.md](./MOCK_SETUP_GUIDE.md) - Complex mock scenarios and explanations

## Best Practices

> **📋 For comprehensive test patterns and standards, see [TEST_PATTERNS.md](./TEST_PATTERNS.md)**

### Unit Test Best Practices

1. **Mock All External Dependencies**
   - Use the standardized mock patterns from [TEST_PATTERNS.md](./TEST_PATTERNS.md)
   - Always specify explicit return values for mocked methods
   - Use `AsyncMock` for async methods, `Mock` for sync methods

2. **Use Consistent Fixture Patterns**
   - Use `test_llm_config` for unit tests (never `real_llm_config`)
   - Use provided mock fixtures: `mock_shell_executor`, `mock_memory_manager`, etc.
   - Create component-specific fixtures following the standard pattern

3. **Follow Standard Test Structure**
   - Group tests into logical classes: `TestComponent` and `TestComponentMocking`
   - Use descriptive test method names that explain what is being tested
   - Structure tests with clear setup, execution, and verification phases

4. **Keep Tests Fast and Isolated**
   - Target: Complete unit test suite in under 10 seconds
   - Individual tests should complete in under 1 second
   - No real network calls, file operations, or external services

### Integration Test Best Practices

1. **Use Real Configuration and Services**
   - Use `real_llm_config` fixture for actual LLM endpoints
   - Test with real AutoGen agents and external services
   - Verify actual file system operations and memory persistence

2. **Handle Service Dependencies Gracefully**
   - Use `@pytest.mark.integration` marker for all integration tests
   - Skip tests gracefully when external services are unavailable
   - Provide clear error messages when tests fail due to configuration

3. **Test Complete Workflows**
   - Verify end-to-end functionality with real components
   - Test actual agent coordination and communication
   - Validate real memory management and persistence

4. **Maintain Test Environment**
   - Ensure `.env.integration` file exists with valid configuration
   - Clean up test artifacts after test completion
   - Use `temp_workspace` fixture for isolated test environments

## Troubleshooting

### Common Issues

1. **Slow Unit Tests**
   - Check for unmocked external dependencies
   - Look for real file operations or network calls
   - Ensure all async operations are properly mocked

2. **Integration Test Failures**
   - Verify `.env.integration` file exists and has correct values
   - Check network connectivity to LLM endpoints
   - Ensure API keys are valid and have sufficient quota

3. **Import Errors**
   - Avoid importing real AutoGen models in unit tests
   - Use mocks instead of real AutoGen classes
   - Check fixture dependencies and naming

### Performance Targets

- **Unit tests**: Complete in under 10 seconds total
- **Individual unit tests**: Complete in under 1 second each
- **Integration tests**: May take longer due to real service calls
- **Individual integration tests**: Should complete within reasonable time (30 seconds max)

## Migration Notes

This test structure was created by refactoring the original mixed test suite where unit tests contained real functionality testing. The key changes were:

1. **Separated real functionality tests** into `tests/integration/`
2. **Added comprehensive mocking** to unit tests
3. **Created separate fixture configurations** for unit vs integration tests
4. **Improved test naming conventions** to clearly distinguish test types
5. **Added performance optimizations** to ensure fast unit test execution

For more details on the refactoring process, see the test audit report in the project root.