# Test Patterns and Standards

This document defines the standardized patterns and conventions for writing tests in the AutoGen Multi-Agent Framework. Following these patterns ensures consistency, maintainability, and reliability across the test suite.

## Table of Contents

1. [Mock Patterns](#mock-patterns)
2. [Fixture Usage](#fixture-usage)
3. [Test Structure](#test-structure)
4. [Assertion Patterns](#assertion-patterns)
5. [Error Testing](#error-testing)
6. [Async Testing](#async-testing)
7. [Common Anti-Patterns](#common-anti-patterns)

## Mock Patterns

### Standard Mock Setup Pattern

Use this pattern for consistent mock setup across all unit tests:

```python
from unittest.mock import Mock, AsyncMock, patch, MagicMock

class TestMyComponent:
    """Test suite for MyComponent."""
    
    @pytest.fixture
    def mock_dependency(self):
        """Create a mock dependency with standard configuration."""
        mock_dep = Mock()
        mock_dep.method_name.return_value = expected_result
        mock_dep.async_method = AsyncMock(return_value=expected_async_result)
        return mock_dep
    
    @patch('module.path.ExternalDependency')
    def test_functionality(self, mock_external, mock_dependency):
        """Test with properly mocked dependencies."""
        # Configure mock behavior
        mock_external.return_value = mock_dependency
        
        # Execute test
        result = subject_under_test.method()
        
        # Verify behavior
        assert result == expected_result
        mock_dependency.method_name.assert_called_once_with(expected_params)
```

### Mock Configuration Standards

#### 1. Mock Return Values
Always specify explicit return values for mocked methods:

```python
# ✅ Good: Explicit return values
mock_agent.generate_response.return_value = "Expected response"
mock_manager.setup_agents.return_value = True
mock_executor.execute_command.return_value = ExecutionResult.create_success(...)

# ❌ Bad: Implicit/undefined return values
mock_agent.generate_response = Mock()  # Returns MagicMock, not expected type
```

#### 2. Async Mock Pattern
Use `AsyncMock` for async methods:

```python
# ✅ Good: Proper async mocking
mock_agent.generate_response = AsyncMock(return_value="Async response")

# ❌ Bad: Regular Mock for async methods
mock_agent.generate_response = Mock(return_value="Response")  # Won't work with await
```

#### 3. Complex Object Mocking
For complex return objects, create structured mocks:

```python
# ✅ Good: Structured mock objects
mock_result = Mock()
mock_result.success = True
mock_result.data = {"key": "value"}
mock_result.error = None
mock_method.return_value = mock_result

# ❌ Bad: Returning raw dictionaries when objects expected
mock_method.return_value = {"success": True, "data": {"key": "value"}}
```

### Environment Variable Mocking

Use `patch.dict` for environment variable mocking:

```python
@patch.dict(os.environ, {
    'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
    'LLM_MODEL': 'test-model',
    'LLM_API_KEY': 'test-key'
})
def test_with_env_vars(self):
    """Test functionality that depends on environment variables."""
    # Test implementation
    pass
```

## Fixture Usage

### Unit Test Fixtures

Use these standard fixtures in unit tests:

```python
def test_unit_functionality(self, test_llm_config, mock_shell_executor, mock_memory_manager):
    """Unit test using standard mocked fixtures."""
    component = MyComponent(
        llm_config=test_llm_config,
        shell_executor=mock_shell_executor,
        memory_manager=mock_memory_manager
    )
    # Test implementation
```

### Integration Test Fixtures

Use these fixtures for integration tests:

```python
@pytest.mark.integration
def test_integration_functionality(self, real_llm_config, temp_workspace):
    """Integration test using real configuration."""
    component = MyComponent(llm_config=real_llm_config)
    # Test with real services
```

### Custom Fixture Pattern

Create component-specific fixtures following this pattern:

```python
@pytest.fixture
def my_component(self, test_llm_config, mock_shell_executor):
    """Create MyComponent instance for testing."""
    return MyComponent(
        name="TestComponent",
        llm_config=test_llm_config,
        shell_executor=mock_shell_executor
    )
```

## Test Structure

### Standard Test Class Structure

```python
class TestMyComponent:
    """Test suite for MyComponent class."""
    
    # Fixtures specific to this test class
    @pytest.fixture
    def component_instance(self, test_llm_config):
        """Create component instance for testing."""
        return MyComponent(llm_config=test_llm_config)
    
    # Basic functionality tests
    def test_initialization(self, component_instance):
        """Test component initialization."""
        assert component_instance.name is not None
        assert component_instance.llm_config is not None
    
    # Success path tests
    def test_successful_operation(self, component_instance):
        """Test successful operation execution."""
        result = component_instance.perform_operation()
        assert result.success is True
    
    # Error handling tests
    def test_error_handling(self, component_instance):
        """Test error handling behavior."""
        with pytest.raises(ExpectedError):
            component_instance.invalid_operation()
    
    # Edge case tests
    def test_edge_case_handling(self, component_instance):
        """Test edge case behavior."""
        result = component_instance.handle_edge_case()
        assert result is not None


class TestMyComponentMocking:
    """Comprehensive mocking tests for MyComponent."""
    
    @patch('module.ExternalDependency')
    def test_external_dependency_integration(self, mock_dependency):
        """Test integration with external dependencies."""
        # Mock setup
        mock_instance = Mock()
        mock_dependency.return_value = mock_instance
        
        # Test execution
        component = MyComponent()
        result = component.use_external_dependency()
        
        # Verification
        mock_dependency.assert_called_once()
        assert result is not None
```

## Assertion Patterns

### Standard Assertions

Use these assertion patterns for consistency:

```python
# ✅ Success/failure assertions
assert result.success is True
assert result.error is None
assert result.data is not None

# ✅ Type assertions
assert isinstance(result, ExpectedType)
assert isinstance(result.data, dict)

# ✅ Content assertions
assert len(result.items) > 0
assert "expected_key" in result.data
assert result.message.startswith("Expected prefix")

# ✅ Mock call assertions
mock_method.assert_called_once()
mock_method.assert_called_once_with(expected_param)
mock_method.assert_not_called()
```

### Complex Assertion Patterns

For complex objects, use structured assertions:

```python
# ✅ Good: Structured assertions
result = component.complex_operation()
assert result is not None
assert result.status == "completed"
assert len(result.items) == expected_count
assert all(item.is_valid() for item in result.items)

# ❌ Bad: Single complex assertion
assert result and result.status == "completed" and len(result.items) == expected_count
```

## Error Testing

### Exception Testing Pattern

```python
def test_specific_error_handling(self, component):
    """Test specific error conditions."""
    # Test specific exception type
    with pytest.raises(SpecificError) as exc_info:
        component.operation_that_fails()
    
    # Verify error details
    assert "expected error message" in str(exc_info.value)
    assert exc_info.value.error_code == "EXPECTED_CODE"

def test_error_propagation(self, component):
    """Test that errors are properly propagated."""
    with patch.object(component, 'dependency') as mock_dep:
        mock_dep.method.side_effect = DependencyError("Dependency failed")
        
        with pytest.raises(ComponentError) as exc_info:
            component.operation_using_dependency()
        
        assert "Dependency failed" in str(exc_info.value)
```

### Error Recovery Testing

```python
def test_error_recovery(self, component):
    """Test error recovery mechanisms."""
    # Setup error condition
    with patch.object(component, 'unreliable_method') as mock_method:
        mock_method.side_effect = [Exception("First failure"), "Success"]
        
        # Test recovery
        result = component.operation_with_retry()
        assert result == "Success"
        assert mock_method.call_count == 2
```

## Async Testing

### Async Test Pattern

```python
class TestAsyncComponent:
    """Test suite for async component functionality."""
    
    @pytest.mark.asyncio
    async def test_async_operation(self, component):
        """Test async operation execution."""
        result = await component.async_operation()
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_async_with_mocks(self, component):
        """Test async operation with mocked dependencies."""
        with patch.object(component, 'async_dependency') as mock_dep:
            mock_dep.async_method = AsyncMock(return_value="Mocked result")
            
            result = await component.operation_using_async_dependency()
            assert result == "Mocked result"
            mock_dep.async_method.assert_called_once()
```

### Async Mock Configuration

```python
@pytest.fixture
def mock_async_component(self):
    """Create mock async component."""
    mock_comp = Mock()
    mock_comp.async_method = AsyncMock(return_value="Async result")
    mock_comp.sync_method = Mock(return_value="Sync result")
    return mock_comp
```

## Common Anti-Patterns

### Anti-Pattern 1: Over-Mocking

```python
# ❌ Bad: Mocking everything, including simple objects
@patch('builtins.str')
@patch('builtins.dict')
@patch('builtins.list')
def test_simple_operation(self, mock_list, mock_dict, mock_str):
    # Unnecessary mocking of built-in types
    pass

# ✅ Good: Mock only external dependencies
@patch('module.ExternalService')
def test_simple_operation(self, mock_service):
    # Mock only what's necessary
    pass
```

### Anti-Pattern 2: Inconsistent Mock Setup

```python
# ❌ Bad: Inconsistent mock configuration
def test_inconsistent_mocks(self):
    mock1 = Mock()
    mock1.method.return_value = "result1"
    
    mock2 = MagicMock()  # Different mock type
    mock2.method = "result2"  # Different assignment pattern
    
    mock3 = Mock()
    # No return value configured - will return MagicMock

# ✅ Good: Consistent mock configuration
def test_consistent_mocks(self):
    mock1 = Mock()
    mock1.method.return_value = "result1"
    
    mock2 = Mock()
    mock2.method.return_value = "result2"
    
    mock3 = Mock()
    mock3.method.return_value = "result3"
```

### Anti-Pattern 3: Testing Implementation Details

```python
# ❌ Bad: Testing internal implementation
def test_internal_implementation(self, component):
    component.public_method()
    assert component._private_method_called  # Testing internal state

# ✅ Good: Testing public behavior
def test_public_behavior(self, component):
    result = component.public_method()
    assert result.success is True  # Testing public outcome
```

### Anti-Pattern 4: Unclear Test Names

```python
# ❌ Bad: Unclear test names
def test_method1(self):
    pass

def test_stuff(self):
    pass

def test_it_works(self):
    pass

# ✅ Good: Descriptive test names
def test_agent_initialization_with_valid_config(self):
    pass

def test_error_handling_when_llm_unavailable(self):
    pass

def test_workflow_state_transitions_correctly(self):
    pass
```

## Documentation Standards

### Test Method Documentation

```python
def test_specific_functionality(self, fixture1, fixture2):
    """
    Test specific functionality under defined conditions.
    
    This test verifies that:
    1. Component initializes correctly with given parameters
    2. Method returns expected result format
    3. Dependencies are called with correct parameters
    4. Error conditions are handled appropriately
    
    Args:
        fixture1: Description of fixture purpose
        fixture2: Description of fixture purpose
    """
    # Test implementation with clear comments
    
    # Setup phase
    component = create_component(fixture1)
    
    # Execution phase
    result = component.method_under_test(fixture2)
    
    # Verification phase
    assert result.success is True
    assert result.data is not None
```

### Complex Mock Documentation

```python
@pytest.fixture
def complex_mock_setup(self):
    """
    Create complex mock setup for testing component interactions.
    
    This fixture provides:
    - Mock LLM client configured for test responses
    - Mock memory manager with predefined content
    - Mock shell executor with success/failure scenarios
    
    Returns:
        dict: Dictionary containing all configured mocks
    """
    mock_llm = Mock()
    mock_llm.generate.return_value = "Test LLM response"
    
    mock_memory = Mock()
    mock_memory.load_memory.return_value = {"test": "content"}
    
    mock_shell = Mock()
    mock_shell.execute_command.return_value = ExecutionResult.create_success(
        command="test command",
        stdout="test output",
        execution_time=0.1,
        working_directory="/tmp",
        approach_used="direct_execution"
    )
    
    return {
        "llm": mock_llm,
        "memory": mock_memory,
        "shell": mock_shell
    }
```

Following these patterns ensures consistent, maintainable, and reliable tests across the entire framework.