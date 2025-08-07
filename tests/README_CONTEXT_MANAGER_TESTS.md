# ContextManager Test Structure

This document explains the test structure for the ContextManager component.

## Test Categories

### Unit Tests (`tests/unit/test_context_manager.py`)
- **Purpose**: Test individual ContextManager methods and data structures in isolation
- **Dependencies**: Uses mocks for MemoryManager and ContextCompressor
- **Coverage**: 23 tests covering core functionality
- **Speed**: Fast (< 1 second)

### Integration Tests (`tests/integration/`)

#### `test_context_manager_integration.py`
- **Purpose**: Test ContextManager with real MemoryManager and ContextCompressor instances
- **Dependencies**: Real components, temporary file system
- **Coverage**: 12 tests covering component interactions
- **Speed**: Medium (< 30 seconds)

#### `test_context_manager_memory_integration.py`
- **Purpose**: Test specific integration with MemoryManager and ContextCompressor
- **Dependencies**: Real components with comprehensive memory content
- **Coverage**: 11 tests covering memory patterns and compression
- **Speed**: Medium (< 30 seconds)

#### `test_context_manager_workflow_simulation.py`
- **Purpose**: Test ContextManager behavior in simulated workflow scenarios
- **Dependencies**: Real components, manually created workflow files
- **Coverage**: 5 tests covering workflow simulation
- **Speed**: Medium (< 30 seconds)
- **Note**: These are NOT true E2E tests - they simulate workflow rather than test actual framework integration

## True End-to-End Tests

For true end-to-end testing of the framework (including ContextManager when integrated), use:
- `tests/e2e/simple_workflow_test.sh` - Basic workflow test
- `tests/e2e/workflow_test.sh` - Complete workflow with revisions

## Current Status

✅ **ContextManager Implementation**: Complete and fully tested
✅ **Component Testing**: All 51 tests passing
❌ **Framework Integration**: ContextManager not yet integrated into WorkflowManager/AgentManager
❌ **True E2E Testing**: Cannot test ContextManager in real workflow until integration is complete

## Next Steps

1. Integrate ContextManager into WorkflowManager
2. Update AgentManager to use ContextManager
3. Modify agents to use agent-specific context interfaces
4. Run true E2E tests to verify integration

## Test Execution

```bash
# Run all ContextManager tests
python -m pytest tests/unit/test_context_manager.py tests/integration/test_context_manager_*.py -v

# Run only unit tests (fast)
python -m pytest tests/unit/test_context_manager.py -v

# Run only integration tests
python -m pytest tests/integration/test_context_manager_*.py -v

# Run true E2E tests (requires .env.integration)
./tests/e2e/simple_workflow_test.sh
```