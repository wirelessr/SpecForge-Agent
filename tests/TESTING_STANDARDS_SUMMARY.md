# Testing Standards Summary

This document provides a quick reference to the standardized testing patterns and documentation for the AutoGen Multi-Agent Framework. For full details, please refer to the main testing documents.

## 📚 Documentation Structure

| Document | Purpose |
|----------|---------|
| [README.md](./README.md) | Main test documentation and organization. |
| [TEST_PATTERNS.md](./TEST_PATTERNS.md) | Standard patterns and conventions for writing tests. |
| [MOCK_SETUP_GUIDE.md](./MOCK_SETUP_GUIDE.md) | Guide for complex mock scenarios. |
| [TESTING_STANDARDS_SUMMARY.md](./TESTING_STANDARDS_SUMMARY.md) | Quick reference (this document). |

## 🏗️ Test Organization

- **`tests/unit/`**: Fast, isolated tests with mocked dependencies.
- **`tests/integration/`**: Tests with real services and components.
- **`tests/e2e/`**: End-to-end workflow test scripts.
- **`tests/quality/`**: Tests that measure the quality of the generated output.

### Naming Conventions
- **Unit tests**: `test_<component_name>.py`
- **Integration tests**: `test_real_<component_name>.py`

## 🔧 Standard Fixtures

### Unit Test Fixtures (`tests/unit/conftest.py`)
- `llm_config`: Mock LLM configuration for unit tests.
- `mock_shell_executor`, `mock_memory_manager`, etc.: Mocks for core components.
- `disable_real_services`: Auto-applied fixture to prevent real service calls.

### Integration Test Fixtures (`tests/integration/conftest.py`)
- `real_llm_config`: Real LLM configuration from `.env.integration`.
- `integration_config_manager`: Real configuration manager.
- `enable_real_services`: Auto-applied fixture to enable real service calls.

### Shared Fixtures (`tests/conftest.py`)
- `temp_workspace`: Creates a temporary workspace for a test.
- `test_config_manager`: A configuration manager for testing purposes.

## 🎯 Quick Reference Patterns

### Standard Mock Setup
```python
@pytest.fixture
def mock_component(self):
    mock_comp = Mock()
    mock_comp.method_name.return_value = "expected_result"
    mock_comp.async_method = AsyncMock(return_value="async_result")
    return mock_comp
```

### Async Test Pattern
```python
@pytest.mark.asyncio
async def test_async_operation(self, component):
    component.async_method = AsyncMock(return_value="Expected result")
    result = await component.async_operation()
    assert result == "Expected result"
```

## ✅ Standard Assertions
- **Success/Failure**: `assert result.success is True`
- **Type**: `assert isinstance(result, ExpectedType)`
- **Mock Verification**: `mock_method.assert_called_once_with(expected_param)`

## 🚫 Common Anti-Patterns to Avoid
- **Over-Mocking**: Don't mock built-in types or the component you are testing.
- **Testing Implementation Details**: Test public behavior, not private methods or internal state.

## 🏃‍♂️ Running Tests
- **All tests**: `pytest tests/`
- **Unit tests only**: `pytest tests/unit/`
- **Integration tests only**: `pytest tests/integration/ -m integration`

---
_This is a summary. For detailed standards, refer to the other testing documents._