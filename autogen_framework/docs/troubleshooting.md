# AutoGen Framework Troubleshooting Guide

This guide helps users diagnose and resolve common issues encountered while using the AutoGen Multi-Agent Framework.

## 🔧 Error Handling Approach

The framework uses a **two-level error handling strategy**:

1.  **Task-Level Recovery**: The `ImplementAgent` automatically handles individual task failures using its internal `TaskDecomposer` and `ErrorRecovery` components.
2.  **Workflow-Level Guidance**: When a workflow phase (e.g., requirements, design, tasks) fails, the system provides clear guidance for manual revision using the `--revise` command.

For example, if the design phase fails, you might see a message like:
`To fix this issue, use: autogen-framework --revise 'design:Please simplify the design and focus on core features'`

This approach gives you control over high-level decisions while automating low-level error recovery.

## ախ Diagnostic Tools

### Basic Commands
```bash
# Check the current status of the workflow
autogen-framework --workspace . --status

# Enable detailed logging for debugging
autogen-framework --workspace . --verbose --log-file debug.log --request "Your request"

# Reset the session to start from a clean state
autogen-framework --workspace . --reset-session
```

### Log Analysis
Logs are typically found in the `logs/` directory within your workspace.

```bash
# View the latest logs in real-time
tail -f logs/execution_log.md

# Search for errors and exceptions
grep -i "error\|failed\|exception" logs/execution_log.md

# View logs for a specific agent
grep "PlanAgent" logs/execution_log.md
```

## 🤯 Common Issues

### LLM Connection Issues

- **Symptoms**: Connection timeouts, `401 Unauthorized` errors, or `Model not found` errors.
- **Solution**:
    1.  Verify your LLM endpoint and API key in your `.env` file.
    2.  Test the connection directly using `curl` as shown in `autogen_framework/README.md`.
    3.  Increase the timeout with the `--llm-timeout` command-line argument if requests are timing out.

### Workflow Stuck or Behaving Erratically

- **Symptoms**: The framework is unresponsive, the status is not updating, or phases are repeating.
- **Solution**:
    1.  Check the status with `autogen-framework --workspace . --status`.
    2.  Inspect the session state file at `memory/session_state.json`. It might be corrupted.
    3.  If the session is corrupted or you want to start over, run `autogen-framework --workspace . --reset-session`. This is often the quickest fix.

### Task Execution Failures

- **Symptoms**: `ImplementAgent` fails to execute a task, code has syntax errors, or dependencies are missing.
- **Solution**:
    1.  Examine `logs/execution_log.md` for specific error messages from the `ImplementAgent`.
    2.  The framework uses a patch-first strategy. If a patch fails, it creates a `.backup` file. You can restore this file to revert the changes.
    3.  For syntax errors, you can manually fix the file and then re-run the task execution.
    4.  For dependency issues, manually install the missing package (`uv pip install <package>`) and consider revising the tasks to include the installation step.

### File Operation Issues

- **Symptoms**: Permission denied errors, patch application failures.
- **Solution**:
    1.  Ensure you have the correct read/write permissions for the workspace directory.
    2.  If a patch fails, check the `.patch` file for malformed content. You can restore the `.backup` file and attempt the task again.

## 🚀 Getting Help

If you can't resolve the issue, please file a bug report on GitHub. Include the following information:

- A clear description of the problem.
- Steps to reproduce the issue.
- The version of the framework you are using.
- Relevant snippets from your log files.

---

**Maintainer**: AutoGen Framework Support Team  
**Last Updated**: 2025-08-18
**Version**: 1.0.0