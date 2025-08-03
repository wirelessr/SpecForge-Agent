# End-to-End Test Scripts

This directory contains the end-to-end test scripts for the AutoGen Framework, used to test the complete workflow.

## Test Scripts

### 1. `simple_workflow_test.sh` - Simplified Workflow Test

**Purpose**: To test the basic workflow: Requirements → Design → Tasks → Implementation

**Features**:
- Uses the integration configuration for real LLM testing.
- Tests each stage of the core workflow.
- Checks file generation and task completion status.
- Validates the generated code files.

**Usage**:
```bash
./tests/e2e/simple_workflow_test.sh
```

### 2. `workflow_test.sh` - Complete Workflow Test (including Revise)

**Purpose**: To test the complete workflow, including the revise functionality for each stage.

**Features**:
- Uses the integration configuration for real LLM testing.
- Tests the revise functionality for each stage (2 revisions per stage).
- Compares file content changes to verify that revise is effective.
- Complete workflow validation.
- Includes detailed debugging information and error handling.

**Usage**:
```bash
./tests/e2e/workflow_test.sh
```

## Fixed Issues

### 1. Configuration Issues
- **Problem**: The original script did not use the integration configuration.
- **Fix**: Added `.env.integration` file loading and environment variable settings.

### 2. Workspace Issues
- **Problem**: The original script created the work directory in the current directory, which could easily become cluttered.
- **Fix**: Used a dedicated `integration_test_workspace` directory.

### 3. Command Invocation Issues
- **Problem**: All `autogen-framework` commands did not specify a workspace.
- **Fix**: Added the `--workspace integration_test_workspace` parameter to all commands.

### 4. Syntax Errors
- **Problem**: Syntax errors in mathematical operations.
- **Fix**: Corrected the syntax for variable comparison and mathematical operations.

### 5. Cleanup Mechanism
- **Problem**: Incomplete test file cleanup.
- **Fix**: Added a complete cleanup function and a trap mechanism.

## Environment Requirements

### 1. Configuration File
An `.env.integration` file needs to be created in the project root directory:

```bash
# Integration Test Environment Configuration
LLM_BASE_URL=http://your-llm-endpoint/v1
LLM_MODEL=your-model-name
LLM_API_KEY=your-api-key
LLM_TEMPERATURE=0.7
LLM_MAX_OUTPUT_TOKENS=4096
LLM_TIMEOUT_SECONDS=60

# Framework Configuration
WORKSPACE_PATH=integration_test_workspace
LOG_LEVEL=DEBUG
SHELL_TIMEOUT_SECONDS=30
SHELL_MAX_RETRIES=2
```

### 2. Installation Requirements
- The project is installed: `pip install -e .`
- The `autogen-framework` command is available.
- The real LLM service is accessible.

## Test Flow

### Simplified Test Flow (`simple_workflow_test.sh`)
1. Environment check and configuration loading
2. Create test workspace
3. Reset session
4. Process initial request (generate Requirements)
5. Approve Requirements (generate Design)
6. Approve Design (generate Tasks)
7. Approve Tasks (execute Implementation)
8. Check execution results
9. Clean up test files

### Complete Test Flow (`workflow_test.sh`)
1. Environment check and configuration loading
2. Create test workspace
3. Reset session
4. Process initial request (generate Requirements)
5. First Revise of Requirements
6. Second Revise of Requirements
7. Approve Requirements (generate Design)
8. First Revise of Design
9. Second Revise of Design
10. Approve Design (generate Tasks)
11. First Revise of Tasks
12. Second Revise of Tasks
13. Approve Tasks (execute Implementation)
14. Check execution results
15. Clean up test files

## Expected Results

### Success Indicators
- ✅ All stage documents have been generated (requirements.md, design.md, tasks.md)
- ✅ Executable Python code files have been generated
- ✅ The code syntax is correct
- ✅ The task completion status is correctly marked
- ✅ The Revise functionality is working correctly (file content has changed)

### Output Example
```
🎉 End-to-end test completed successfully!
The framework correctly executed the complete workflow and generated the code files.
```

## Troubleshooting

### Common Issues
1. **LLM Connection Failed**: Check the LLM configuration in `.env.integration`.
2. **Command Not Found**: Make sure the project is installed with `pip install -e .`.
3. **Permission Issues**: Make sure the scripts have execution permissions `chmod +x tests/e2e/*.sh`.
4. **Work Directory Issues**: Make sure to run the tests from the project root directory.

### Debugging Tips
- Check the generated output files (`*.txt`) for detailed errors.
- Check the logs in the `integration_test_workspace/logs/` directory.
- Use `autogen-framework --status` to check the framework status.

## Contribution Guide

If you need to modify the test scripts:
1. Maintain backward compatibility.
2. Add appropriate error handling.
3. Update this README document.
4. Test all modifications.
5. Follow the user-requirements-first principle.
