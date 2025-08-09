# Testing Standards Summary

This document provides a quick reference to the standardized testing patterns and documentation for the AutoGen Multi-Agent Framework.

## 📚 Documentation Structure

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [README.md](./README.md) | Main test documentation and organization | First read for understanding test structure |
| [TEST_PATTERNS.md](./TEST_PATTERNS.md) | Standard patterns and conventions | Writing new tests, following established patterns |
| [MOCK_SETUP_GUIDE.md](./MOCK_SETUP_GUIDE.md) | Complex mock scenarios with explanations | Setting up complex test scenarios |
| [TESTING_STANDARDS_SUMMARY.md](./TESTING_STANDARDS_SUMMARY.md) | Quick reference (this document) | Quick lookup of standards and patterns |

## 🏗️ Test Organization

### Directory Structure
```
tests/
├── unit/                    # Fast, isolated tests with mocked dependencies
│   ├── conftest.py         # Unit test fixtures and configuration
│   └── test_*.py           # Unit test files
├── integration/            # Tests with real services and components
│   ├── conftest.py         # Integration test fixtures and configuration
│   └── test_real_*.py      # Integration test files
├── e2e/                    # End-to-end workflow tests
└── conftest.py             # Shared fixtures across all test types
```

### Test File Naming
- **Unit tests**: `test_<component_name>.py`
- **Integration tests**: `test_real_<component_name>.py`
- **E2E tests**: `test_e2e_<workflow_name>.py`

### Test Class Naming
- **Unit test classes**: `Test<ComponentName>`
- **Mocking test classes**: `Test<ComponentName>Mocking`
- **Integration test classes**: `Test<ComponentName>RealIntegration`

## 🔧 Standard Fixtures

### Unit Test Fixtures (tests/unit/conftest.py)
| Fixture | Purpose | Returns |
|---------|---------|---------|
| `test_llm_config` | Mock LLM configuration | LLMConfig with test values |
| `mock_shell_executor` | Mock shell executor | Mock with execution methods |
| `mock_memory_manager` | Mock memory manager | Mock with memory operations |
| `mock_agent_manager` | Mock agent manager | Mock with coordination methods |
| `mock_autogen_agent` | Mock AutoGen agent | MagicMock AutoGen agent |
| `mock_context_manager` | Mock context manager | Mock with context operations |
| `mock_task_decomposer` | Mock task decomposer | Mock with decomposition methods |
| `mock_error_recovery` | Mock error recovery | Mock with recovery strategies |
| `disable_real_services` | Prevent real service calls | Auto-applied environment setup |

### Integration Test Fixtures (tests/integration/conftest.py)
| Fixture | Purpose | Returns |
|---------|---------|---------|
| `real_llm_config` | Real LLM configuration | LLMConfig from .env.integration |
| `real_context_manager` | Real context manager | ContextManager with real components |
| `real_memory_manager` | Real memory manager | MemoryManager with file operations |
| `test_workspace` | Temporary workspace | Path to isolated test directory |
| `quality_test_environment` | Quality testing setup | Complete test environment for quality measurement |

### Integration Test Fixtures (tests/integration/conftest.py)
| Fixture | Purpose | Returns |
|---------|---------|---------|
| `real_llm_config` | Real LLM configuration | LLMConfig from .env.integration |
| `integration_config_manager` | Real config manager | ConfigManager with env loading |
| `enable_real_services` | Enable real service calls | Auto-applied environment setup |

### Shared Fixtures (tests/conftest.py)
| Fixture | Purpose | Returns |
|---------|---------|---------|
| `temp_workspace` | Temporary test workspace | Path to temp directory |
| `test_config_manager` | Test configuration manager | ConfigManager for testing |
| `mock_env_vars` | Environment variable mocking | Context manager function |

## 🎯 Quick Reference Patterns

### Standard Mock Setup
```python
@pytest.fixture
def mock_component(self):
    """Create mock component with explicit return values."""
    mock_comp = Mock()
    mock_comp.method_name.return_value = expected_result
    mock_comp.async_method = AsyncMock(return_value=expected_async_result)
    return mock_comp
```

### Environment Variable Mocking
```python
@patch.dict(os.environ, {
    'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
    'LLM_MODEL': 'test-model',
    'LLM_API_KEY': 'test-key'
})
def test_with_env_vars(self):
    """Test with mocked environment variables."""
    pass
```

### Async Test Pattern
```python
@pytest.mark.asyncio
async def test_async_operation(self, component):
    """Test async operation execution."""
    component.async_method = AsyncMock(return_value="Expected result")
    result = await component.async_operation()
    assert result == "Expected result"
```

### Complex Mock with side_effect
```python
def test_sequential_responses(self, component):
    """Test component with different responses per call."""
    component.method = Mock()
    component.method.side_effect = [
        "First response",
        "Second response", 
        "Third response"
    ]
    
    # Each call returns different value
    assert component.method() == "First response"
    assert component.method() == "Second response"
    assert component.method() == "Third response"
```

## ✅ Standard Assertions

### Success/Failure Patterns
```python
# ✅ Good: Explicit boolean checks
assert result.success is True
assert result.error is None

# ✅ Good: Type assertions
assert isinstance(result, ExpectedType)
assert isinstance(result.data, dict)

# ✅ Good: Content assertions
assert len(result.items) > 0
assert "expected_key" in result.data
```

### Mock Verification Patterns
```python
# ✅ Standard mock assertions
mock_method.assert_called_once()
mock_method.assert_called_once_with(expected_param)
mock_method.assert_not_called()

# ✅ Multiple call verification
mock_method.assert_has_calls([
    call(param1),
    call(param2),
    call(param3)
])
```

## 🚫 Common Anti-Patterns to Avoid

### ❌ Over-Mocking
```python
# Bad: Mocking built-in types
@patch('builtins.str')
@patch('builtins.dict')
def test_simple_operation(self, mock_dict, mock_str):
    pass

# Good: Mock only external dependencies
@patch('module.ExternalService')
def test_simple_operation(self, mock_service):
    pass
```

### ❌ Inconsistent Mock Setup
```python
# Bad: Different mock types and patterns
mock1 = Mock()
mock1.method.return_value = "result1"
mock2 = MagicMock()
mock2.method = "result2"  # Different assignment pattern

# Good: Consistent mock configuration
mock1 = Mock()
mock1.method.return_value = "result1"
mock2 = Mock()
mock2.method.return_value = "result2"
```

### ❌ Testing Implementation Details
```python
# Bad: Testing internal state
def test_internal_implementation(self, component):
    component.public_method()
    assert component._private_method_called

# Good: Testing public behavior
def test_public_behavior(self, component):
    result = component.public_method()
    assert result.success is True
```

## 🏃‍♂️ Running Tests

### Quick Commands
```bash
# Run all tests
pytest tests/

# Run only unit tests (fast)
pytest tests/unit/

# Run only integration tests
pytest tests/integration/ -m integration

# Run with timing information
pytest tests/unit/ --durations=10

# Run specific test categories
pytest tests/unit/ -k "Mocking"
pytest tests/integration/ -k "Real"
```

### Performance Targets
- **Unit tests**: Complete in under 10 seconds total
- **Individual unit tests**: Complete in under 1 second each
- **Integration tests**: May take longer due to real service calls
- **Individual integration tests**: Should complete within 30 seconds max

## 📋 Test Documentation Standards

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
    pass
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
    # Mock setup implementation
    pass
```

## 🔍 Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow unit tests | Real service calls | Check for unmocked dependencies |
| Integration test failures | Missing .env.integration | Verify configuration file exists |
| Import errors | Wrong fixture usage | Use unit fixtures in unit tests, integration fixtures in integration tests |
| Mock assertion failures | Incorrect mock setup | Verify mock return values and call expectations |

### Debug Commands
```bash
# Run with verbose output
pytest tests/unit/ -v

# Run with detailed failure information
pytest tests/unit/ -vvv

# Run specific test with debugging
pytest tests/unit/test_main_controller.py::TestMainController::test_initialization -vvv
```

## 📖 Additional Resources

- **AutoGen Documentation**: [Official AutoGen Docs](https://microsoft.github.io/autogen/)
- **Pytest Documentation**: [Pytest Official Docs](https://docs.pytest.org/)
- **Python Mock Documentation**: [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

---

**Remember**: Always follow the established patterns for consistency and maintainability. When in doubt, refer to the detailed guides linked above.
## 🤖 
Autonomous Execution Component Testing

### Enhanced ImplementAgent Testing
- **Unit Tests**: Mock TaskDecomposer, ErrorRecovery, and ContextManager
- **Integration Tests**: Test with real components and LLM calls
- **Quality Tests**: Measure output quality against established baselines

### TaskDecomposer Testing
- **Unit Tests**: Mock ContextManager, test decomposition logic
- **Integration Tests**: Test with real context and LLM calls
- **Complexity Analysis**: Test task complexity assessment

### ErrorRecovery Testing
- **Unit Tests**: Mock error scenarios and strategy generation
- **Integration Tests**: Test with real error conditions
- **Pattern Learning**: Test successful pattern recording

### ContextManager Testing
- **Unit Tests**: Mock MemoryManager and ContextCompressor
- **Integration Tests**: Test with real file operations and memory
- **Context Compression**: Test automatic compression when approaching token limits

## 📊 Quality Measurement Testing

### Quality Metrics Testing
- **Functionality Tests**: Verify code works as intended
- **Maintainability Tests**: Check code structure and readability
- **Standards Compliance**: Validate adherence to project conventions
- **Test Coverage**: Ensure comprehensive test coverage
- **Documentation**: Verify implementation documentation

### Quality Gate Testing
- **Baseline Establishment**: Test baseline measurement creation
- **Gate Validation**: Test quality gate pass/fail logic
- **Regression Detection**: Test quality regression identification
- **Improvement Tracking**: Test quality improvement measurement

## 🔄 Test Execution Order

### Recommended Test Sequence
1. **Fast Unit Tests**: `pytest tests/unit/ -x --tb=short -q` (< 10 seconds)
2. **Integration Tests**: `pytest tests/integration/ -x --tb=short -q`
3. **End-to-End Tests**: `./tests/e2e/workflow_test.sh`
4. **Quality Measurement**: `pytest tests/quality/ -v`

### CI/CD Pipeline
```bash
# Fast feedback loop
pytest tests/unit/ -x --tb=short -q

# Component integration
pytest tests/integration/ -x --tb=short -q

# Complete workflow validation
./tests/e2e/workflow_test.sh

# Quality assessment
pytest tests/quality/ -v --tb=short
```

## 🎯 Testing Best Practices

### Unit Testing
- Use `test_llm_config` for all unit tests
- Mock all external dependencies
- Focus on logic paths and error handling
- Keep tests fast (< 1 second each)

### Integration Testing
- Use `real_llm_config` for integration tests
- Test actual service interactions
- Validate configuration and communication
- Allow longer execution times for real API calls

### Quality Testing
- Use real tasks.md files for quality assessment
- Compare against established baselines
- Track quality improvements over time
- Integrate with quality gate management

### Error Testing
- Test both expected and unexpected error conditions
- Verify error recovery strategies
- Test error pattern recognition
- Validate strategy learning and improvement

## 📝 Test Documentation Standards

### Test Method Documentation
```python
def test_specific_functionality(self, fixture):
    """Test description explaining what is being tested.
    
    This test verifies that [specific behavior] works correctly
    when [specific conditions] are met.
    """
    # Arrange
    # Act  
    # Assert
```

### Test Class Documentation
```python
class TestComponentName:
    """Test suite for ComponentName functionality.
    
    Tests cover:
    - Core functionality
    - Error handling
    - Integration with other components
    - Quality measurement (if applicable)
    """
```

## 🚨 Security and Safety

### Test Data Security
- Never use real API keys in unit tests
- Use mock configurations for sensitive data
- Clean up test artifacts after execution
- Isolate test environments from production

### Test Environment Safety
- Use temporary directories for file operations
- Clean up created files and directories
- Prevent accidental modification of source code
- Isolate test runs from each other

---

**Quick Reference**: For immediate testing needs, use `test_llm_config` for unit tests and `real_llm_config` for integration tests. Follow the recommended test execution order for comprehensive validation.