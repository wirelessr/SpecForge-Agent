# AutoGen Framework Troubleshooting Guide

This guide helps users diagnose and resolve common issues encountered while using the AutoGen Multi-Agent Framework.

## 🔧 Error Handling Approach

The framework uses a **two-level error handling strategy**:

1. **Task-Level Recovery**: ImplementAgent automatically handles individual task failures using TaskDecomposer and ErrorRecovery components
2. **Workflow-Level Guidance**: When workflow phases (requirements, design, tasks) fail, the system provides clear guidance for manual revision using `--revise`

### When Phases Fail
Instead of automatic retry, you'll receive specific guidance like:
```bash
# Requirements phase failure
"To fix this issue, use: autogen-framework --revise 'requirements:Please provide more specific requirements'"

# Design phase failure  
"To fix this issue, use: autogen-framework --revise 'design:Please simplify the design and focus on core features'"

# Tasks phase failure
"To fix this issue, use: autogen-framework --revise 'tasks:Please create simpler tasks with clearer descriptions'"
```

This approach ensures you maintain control over important design decisions while still benefiting from automatic task-level error recovery.

## 
 Quick Index of Common Issues

| Issue Type | Symptoms | Quick Solution |
|---|---|---|
| [LLM Connection](#llm-connection-issues) | Connection timeout, authentication failure | Check endpoint and API key |
| [Workflow Stuck](#workflow-issues) | Status not updating, no response | Check WorkflowManager state, reset session |
| [Task Generation Issues](#task-generation-issues) | TasksAgent failures, malformed tasks.md | Check TasksAgent logs, verify design.md |
| [Task Execution Failure](#task-execution-issues) | ImplementAgent errors, command failures | Check execution logs, manually fix |
| [TaskDecomposer Issues](#taskdecomposer-issues) | Task breakdown failures, invalid commands | Check task complexity, context availability |
| [Error Handling Issues](#error-handling-issues) | Phase failures, unclear error messages | Check error logs, use --revise guidance |
| [Context Management Issues](#context-management-issues) | Missing context, compression errors | Check ContextManager state, token limits |
| [Quality Measurement Issues](#quality-measurement-issues) | Quality scores failing, baseline missing | Check quality metrics, establish baseline |
| [Session Management Issues](#session-management-issues) | Session corruption, state inconsistency | Check SessionManager, reset session |
| [File Modification Issues](#file-operation-issues) | Patch failure, permission errors | Check file permissions, use backups |
| [Performance Issues](#performance-issues) | Slow response, insufficient memory | Optimize configuration, clear cache |
| [Configuration Issues](#configuration-issues) | Parameter errors, path issues | Validate configuration file |

## 
 Diagnostic Tools

### 1. Basic Diagnostic Commands

```bash
# Check framework status
autogen-framework --workspace . --status

# Enable detailed logging
autogen-framework --workspace . --verbose --log-file debug.log --status

# Check configuration
autogen-framework --workspace . --help-examples

# Reset session (clears all state)
autogen-framework --workspace . --reset-session
```

### 2. Log Analysis

```bash
# View the latest logs
tail -f logs/execution_log.md

# Search for errors
grep -i "error\|failed\|exception" logs/execution_log.md

# View logs for a specific agent
grep "PlanAgent\|DesignAgent\|ImplementAgent" logs/execution_log.md

# Analyze performance
grep -E "duration|time|performance" logs/execution_log.md
```

### 3. System Check Script

```bash
#!/bin/bash
# system_check.sh - System diagnostic script

echo "=== AutoGen Framework System Diagnostics ==="

# Check Python version
echo "Python version:"
python --version

# Check dependencies
echo "Checking key dependencies:"
python -c "import autogen_agentchat; print('AutoGen:', autogen_agentchat.__version__)" 2>/dev/null || echo "
 AutoGen not installed"
python -c "import click; print('Click:', click.__version__)" 2>/dev/null || echo "
 Click not installed"

# Check network connection
echo "Checking network connection:"
curl -s --max-time 5 http://google.com > /dev/null && echo "
 Network normal" || echo "
 Network issue"

# Check disk space
echo "Disk space:"
df -h . | tail -1

# Check permissions
echo "Directory permissions:"
ls -la . | head -5

echo "=== Diagnostics complete ==="
```

## 
 LLM Connection Issues

### Issue 1: Connection Timeout

**Symptoms**:
```
Error: Connection timeout when calling LLM endpoint
TimeoutError: Request timed out after 30 seconds
```

**Diagnostic Steps**:
```bash
# 1. Test endpoint connectivity
curl -X POST http://your-llm-endpoint/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"test"}]}'

# 2. Check network latency
ping your-llm-host

# 3. Test DNS resolution
nslookup your-llm-host
```

**Solutions**:
1.  **Increase timeout duration**:
    ```bash
    # Set environment variable
    export LLM_TIMEOUT=60
    
    # Or specify in the command
    autogen-framework --workspace . --llm-timeout 60 --request "your request"
    ```

2.  **Use a local proxy**:
    ```bash
    # If behind a firewall
    export HTTP_PROXY=http://your-proxy:port
    export HTTPS_PROXY=http://your-proxy:port
    ```

3.  **Switch to a backup endpoint**:
    ```bash
    autogen-framework \
     --workspace . \
     --llm-base-url http://backup-endpoint/v1 \
     --request "your request"
    ```

### Issue 2: Authentication Failure

**Symptoms**:
```
Error: Authentication failed
401 Unauthorized: Invalid API key
```

**Diagnostic Steps**:
```bash
# 1. Check API key format
echo $LLM_API_KEY | wc -c  # Check length

# 2. Test authentication
curl -H "Authorization: Bearer $LLM_API_KEY" \
     http://your-llm-endpoint/v1/models

# 3. Check key permissions
# Confirm the key has permission to call the model
```

**Solutions**:
1.  **Regenerate API key**:
    *   Log in to the LLM provider console.
    *   Generate a new API key.
    *   Update the environment variable.

2.  **Check key format**:
    ```bash
    # Ensure there are no extra spaces or newlines
    export LLM_API_KEY="$(echo $LLM_API_KEY | tr -d '\n\r ')"
    ```

3.  **Use a configuration file**:
    ```bash
    # Create ~/.autogen/config.json
    {
      "llm_base_url": "http://your-endpoint/v1",
      "llm_model": "your-model",
      "llm_api_key": "your-key"
    }
    ```

### Issue 3: Model Not Available

**Symptoms**:
```
Error: Model 'your-model' not found
404 Not Found: The model does not exist
```

**Solutions**:
```bash
# 1. List available models
curl -H "Authorization: Bearer $LLM_API_KEY" \
     http://your-llm-endpoint/v1/models

# 2. Use the correct model name
autogen-framework \
  --workspace . \
  --llm-model "correct-model-name" \
  --request "your request"
```

## 
 Workflow Issues

### Issue 1: Workflow Stuck

**Symptoms**:
- No response after command execution.
- Status not updating for a long time.
- Process consumes CPU but makes no progress.

**Diagnostic Steps**:
```bash
# 1. Check current status
autogen-framework --workspace . --status

# 2. Check process status
ps aux | grep autogen

# 3. Check logs for WorkflowManager issues
tail -f logs/execution_log.md | grep -i "workflow\|session"

# 4. Check session files managed by SessionManager
ls -la memory/session_state.json
cat memory/session_state.json
```

**Solutions**:
1.  **Force terminate and reset**:
    ```bash
    # Terminate the process
    pkill -f autogen-framework
    
    # Reset the session
    autogen-framework --workspace . --reset-session
    
    # Start over
    autogen-framework --workspace . --request "your request"
    ```

2.  **Check and fix session file**:
    ```bash
    # Backup the existing session
    cp .autogen_session_*.json .autogen_session_backup.json
    
    # Manually edit the session file to fix the state
    # Or delete the session file to start over
    rm .autogen_session_*.json
    ```

3.  **Execute step by step**:
    ```bash
    # Execute step by step instead of all at once
    autogen-framework --workspace . --request "your request"
    autogen-framework --workspace . --approve requirements
    autogen-framework --workspace . --approve design
    autogen-framework --workspace . --approve tasks
    autogen-framework --workspace . --execute-tasks
    ```

### Issue 2: Phase Approval Failed

**Symptoms**:
```
Error: Cannot approve phase 'design' - requirements not approved
Error: Phase transition not allowed
```

**Solutions**:
```bash
# 1. Check the current phase
autogen-framework --workspace . --status

# 2. Approve in order
autogen-framework --workspace . --approve requirements
autogen-framework --workspace . --approve design
autogen-framework --workspace . --approve tasks

# 3. If revision is needed
autogen-framework --workspace . --revise "requirements:your feedback"
```

### Issue 3: Inconsistent Session State

**Symptoms**:
- The displayed status does not match the actual files.
- The same phase is executed repeatedly.
- Cannot continue the workflow.

**Solutions**:
```bash
# 1. Check file status
ls -la requirements.md design.md tasks.md

# 2. Manually synchronize state
# If the file exists but the status shows incomplete, you can manually approve
autogen-framework --workspace . --approve requirements
autogen-framework --workspace . --approve design
autogen-framework --workspace . --approve tasks

# 3. Full reset (last resort)
autogen-framework --workspace . --reset-session
# Then restart the entire process
```

## 
 Task Generation Issues

### Issue 1: TasksAgent Generation Failures

**Symptoms**:
```
Error: TasksAgent failed to generate task list
Error: Malformed tasks.md generated
```

**Diagnostic Steps**:
```bash
# 1. Check TasksAgent logs
grep -i "tasksagent\|task.*generation" logs/execution_log.md

# 2. Verify input documents exist
ls -la requirements.md design.md

# 3. Check design document quality
wc -l design.md
head -20 design.md
```

**Solutions**:
1.  **Improve design document**:
    ```bash
    # Revise design to be more detailed
    autogen-framework --workspace . --revise "design:Add more implementation details"
    ```

2.  **Re-run task generation**:
    ```bash
    # Force TasksAgent to regenerate tasks
    rm -f tasks.md
    autogen-framework --workspace . --continue-workflow
    ```

3.  **Check TasksAgent context**:
    ```bash
    # Ensure TasksAgent has access to both requirements and design
    grep -A 5 -B 5 "context.*requirements\|context.*design" logs/execution_log.md
    ```

### Issue 2: Task List Format Issues

**Symptoms**:
- Tasks.md has incorrect format
- ImplementAgent cannot parse tasks
- Missing task references to requirements

**Solutions**:
```bash
# 1. Validate tasks.md format
grep -E "^- \[.\]" tasks.md | head -10

# 2. Check requirement references
grep -i "requirement" tasks.md

# 3. Regenerate with better prompts
autogen-framework --workspace . --revise "tasks:Ensure proper markdown format and requirement references"
```

## 
 Task Execution Issues

### Issue 1: Code Generation Errors

**Symptoms**:
```
Error: Generated code has syntax errors
SyntaxError: invalid syntax
```

**Diagnostic Steps**:
```bash
# 1. Check the generated files
find . -name "*.py" -exec python -m py_compile {} \;

# 2. View the specific error
python -m py_compile problematic_file.py

# 3. Check the task execution log
grep -A 10 -B 10 "SyntaxError" logs/execution_log.md
```

**Solutions**:
1.  **Manually fix syntax errors**:
    ```bash
    # Use an editor to fix syntax errors
vim problematic_file.py
    
    # Validate the fix
    python -m py_compile problematic_file.py
    ```

2.  **Re-execute the specific task**:
    ```bash
    # Re-execute the failed task
    autogen-framework --workspace . --execute-task "Fix syntax errors"
    ```

3.  **Use more specific requirements**:
    ```bash
    # Provide more detailed requirements to avoid errors
    autogen-framework --workspace . --revise "tasks:Add syntax validation step"
    ```

### Issue 2: Dependency Installation Failed

**Symptoms**:
```
Error: Failed to install package 'some-package'
ModuleNotFoundError: No module named 'some_package'
```

**Solutions**:
```bash
# 1. Manually install the dependency
pip install some-package

# 2. Check requirements.txt
cat requirements.txt
pip install -r requirements.txt

# 3. Use a virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Update the task to include dependency installation
autogen-framework --workspace . --revise "tasks:Add dependency installation step"
```

### Issue 3: Command Execution Permission Issues

**Symptoms**:
```
Error: Permission denied
bash: ./script.sh: Permission denied
```

**Solutions**:
```bash
# 1. Check file permissions
ls -la script.sh

# 2. Add execution permission
chmod +x script.sh

# 3. Check directory permissions
ls -la .

# 4. If it's a system command permission issue
sudo autogen-framework --workspace . --execute-task "your task"
```

## 
 Session Management Issues

### Issue 1: Session State Corruption

**Symptoms**:
```
Error: Cannot load session state
JSONDecodeError: Expecting property name
Error: Session file corrupted
```

**Diagnostic Steps**:
```bash
# 1. Check session file integrity
python -m json.tool memory/session_state.json

# 2. Check SessionManager logs
grep -i "sessionmanager\|session.*state" logs/execution_log.md

# 3. Check file permissions
ls -la memory/session_state.json
```

**Solutions**:
1.  **Backup and reset session**:
    ```bash
    # Backup corrupted session
    cp memory/session_state.json memory/session_state.json.backup
    
    # Reset session through SessionManager
    autogen-framework --workspace . --reset-session
    ```

2.  **Manually fix session file**:
    ```bash
    # Create minimal valid session
    echo '{"session_id": "new_session", "workflow_state": "initial"}' > memory/session_state.json
    ```

### Issue 2: Session Persistence Failures

**Symptoms**:
- Session state not saved between commands
- Workflow progress lost
- Repeated initialization

**Solutions**:
```bash
# 1. Check memory directory permissions
ls -la memory/
chmod 755 memory/

# 2. Check disk space
df -h .

# 3. Verify SessionManager initialization
grep -i "sessionmanager.*init" logs/execution_log.md
```

## 
 File Operation Issues

### Issue 1: Patch Application Failed

**Symptoms**:
```
Error: Patch application failed
patch: **** malformed patch at line 10
```

**Diagnostic Steps**:
```bash
# 1. Check the patch file
ls -la *.patch
cat file.patch

# 2. Check the original file
ls -la file.backup_*

# 3. Manually test the patch
patch --dry-run file.py < file.patch
```

**Solutions**:
1.  **Restore from backup**:
    ```bash
    # Find the backup file
    ls -la *.backup_*
    
    # Restore the original file
    cp file.py.backup_20250801_150000 file.py
    ```

2.  **Manually apply changes**:
    ```bash
    # View the patch content
    cat file.patch
    
    # Manually edit the file
    vim file.py
    ```

3.  **Regenerate the patch**:
    ```bash
    # Re-execute the task to have the system regenerate the patch
    autogen-framework --workspace . --execute-task "your task"
    ```

### Issue 2: File Permission Issues

**Symptoms**:
```
Error: Permission denied when writing to file
OSError: [Errno 13] Permission denied: 'file.py'
```

**Solutions**:
```bash
# 1. Check file permissions
ls -la file.py

# 2. Modify permissions
chmod 644 file.py

# 3. Check directory permissions
ls -la .
chmod 755 .

# 4. Check file owner
sudo chown $USER:$USER file.py
```

### Issue 3: Insufficient Disk Space

**Symptoms**:
```
Error: No space left on device
OSError: [Errno 28] No space left on device
```

**Solutions**:
```bash
# 1. Check disk space
df -h .

# 2. Clean up temporary files
find . -name "*.tmp" -delete
find . -name "*.backup_*" -mtime +7 -delete

# 3. Clean up log files
find logs/ -name "*.log" -mtime +30 -delete

# 4. Clean up cache
rm -rf ~/.cache/autogen_framework/
```

## 
 Performance Issues

### Issue 1: Slow Response Speed

**Symptoms**:
- LLM calls take too long.
- Task execution is slow.
- The system response is sluggish.

**Diagnostic Steps**:
```bash
# 1. Check network latency
ping your-llm-endpoint

# 2. Monitor system resources
top
htop
iostat 1

# 3. Analyze performance logs
grep -E "duration|time" logs/execution_log.md | tail -20
```

**Solutions**:
1.  **Optimize LLM configuration**:
    ```bash
    # Use a faster model
    autogen-framework \
     --workspace . \
     --llm-model "faster-model" \
     --request "your request"
    
    # Reduce context length
    export MAX_CONTEXT_LENGTH=4000
    ```

2.  **Parallel processing**:
    ```bash
    # Decompose large tasks into smaller tasks and execute them in parallel
    autogen-framework --workspace . --execute-task "Task 1" &
    autogen-framework --workspace . --execute-task "Task 2" &
    wait
    ```

3.  **Use local cache**:
    ```bash
    # Enable response caching
    export ENABLE_RESPONSE_CACHE=true
    ```

### Issue 2: High Memory Usage

**Symptoms**:
```
MemoryError: Unable to allocate memory
Process killed (OOM)
```

**Solutions**:
```bash
# 1. Monitor memory usage
free -h
ps aux --sort=-%mem | head

# 2. Clean up memory
# Restart the framework process
pkill -f autogen-framework

# 3. Optimize configuration
export MAX_CONTEXT_SIZE=2000
export CLEANUP_TEMP_FILES=true

# 4. Batch processing
# Decompose large projects into multiple smaller projects
```

### Issue 3: High CPU Usage

**Symptoms**:
- Slow system response.
- Loud fan noise.
- Difficulty running other programs.

**Solutions**:
```bash
# 1. Limit CPU usage
nice -n 10 autogen-framework --workspace . --request "your request"

# 2. Reduce concurrency
export MAX_CONCURRENT_TASKS=1

# 3. Use a more lightweight model
autogen-framework \
  --workspace . \
  --llm-model "lightweight-model" \
  --request "your request"
```

## 
 Configuration Issues

### Issue 1: Configuration File Errors

**Symptoms**:
```
Error: Invalid configuration
JSONDecodeError: Expecting ',' delimiter
```

**Solutions**:
```bash
# 1. Validate JSON format
python -m json.tool ~/.autogen/config.json

# 2. Use default configuration
mv ~/.autogen/config.json ~/.autogen/config.json.backup
autogen-framework --workspace . --help-examples

# 3. Recreate the configuration
cat > ~/.autogen/config.json << EOF
{
  "llm_base_url": "http://your-endpoint/v1",
  "llm_model": "your-model",
  "llm_api_key": "your-key"
}
EOF
```

### Issue 2: Path Issues

**Symptoms**:
```
Error: Workspace directory not found
FileNotFoundError: No such file or directory
```

**Solutions**:
```bash
# 1. Check the path
ls -la /path/to/workspace

# 2. Create the directory
mkdir -p /path/to/workspace
cd /path/to/workspace

# 3. Use an absolute path
autogen-framework --workspace $(pwd) --request "your request"

# 4. Check permissions
ls -la $(dirname /path/to/workspace)
```

## 
 Advanced Debugging Techniques

### 1. Enable Debug Mode

```bash
# Set debug environment variables
export AUTOGEN_DEBUG=true
export AUTOGEN_LOG_LEVEL=DEBUG

# Run the framework
autogen-framework --workspace . --verbose --log-file debug.log --request "your request"
```

### 2. Use a Python Debugger

```python
# Add a breakpoint in the code
import pdb; pdb.set_trace()

# Or use ipdb
import ipdb; ipdb.set_trace()
```

### 3. Network Debugging

```bash
# Use tcpdump to monitor network traffic
sudo tcpdump -i any -w network.pcap host your-llm-endpoint

# Use wireshark to analyze
wireshark network.pcap
```

### 4. Performance Profiling

```python
# Use cProfile to analyze performance
python -m cProfile -o profile.stats -m autogen_framework.main --workspace . --request "your request"

# Analyze the results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

## 
 Getting Help

### 1. Community Support

- **GitHub Issues**: Report bugs and feature requests.
- **Discussion Forums**: Technical discussions and experience sharing.
- **Documentation**: Check the latest documentation and tutorials.

### 2. Submitting a Bug Report

When creating a bug report, please include:

```markdown
## Bug Description
A brief description of the problem encountered.

## Steps to Reproduce
1.  Execute the command `autogen-framework --workspace . --request "..."`
2.  The error observed.

## Expected Behavior
Describe what should have happened.

## Environment Information
- OS: macOS 14.0
- Python: 3.11.5
- Framework Version: 1.0.0
- LLM Endpoint: http://example.com/v1

## Log Files
```
[Paste relevant log content]
```

## Additional Context
Any other information that helps to understand the problem.
```

### 3. Emergency Support

For urgent issues in a production environment:

1.  Check the [Status Page](https://status.autogen.dev).
2.  Check [Known Issues](https://github.com/your-org/autogen-framework/issues).
3.  Contact technical support (if you have commercial support).

## 
 Troubleshooting Checklist

Before contacting support, please complete the following checks:

- [ ] Check network connection.
- [ ] Verify LLM endpoint and API key.
- [ ] Confirm Python version (3.11+).
- [ ] Check disk space.
- [ ] Review the latest log files.
- [ ] Try resetting the session.
- [ ] Test with a simple request.
- [ ] Check file permissions.
- [ ] Validate configuration file format.
- [ ] Consult the relevant documentation.

---

**Maintainer**: AutoGen Framework Support Team  
**Last Updated**: 2025-08-01  
**Version**: 1.0.0

If this guide does not resolve your issue, please create an issue on GitHub or contact our support team.

## 🤖 Autonomous Execution Enhancement Issues

### TaskDecomposer Issues

#### Symptoms
- Task breakdown failures
- Invalid shell command generation
- Execution plans with missing steps
- Context analysis errors

#### Diagnostic Steps
```bash
# Check TaskDecomposer logs
grep "TaskDecomposer" logs/framework.log

# Verify task complexity analysis
autogen-framework --debug-task-decomposition --task "task_description"

# Check context availability
ls -la requirements.md design.md tasks.md
```

#### Common Causes and Solutions

**1. Insufficient Context**
```bash
# Problem: Missing requirements.md or design.md
# Solution: Ensure all context files exist
ls -la requirements.md design.md

# If missing, regenerate them
autogen-framework --request "regenerate missing documents"
```

**2. Task Complexity Too High**
```python
# Problem: Task is too complex for single decomposition
# Solution: Break down manually or simplify task description

# Check task complexity in logs
grep "complexity.*high" logs/framework.log

# Simplify task description
# Instead of: "Implement complete authentication system with OAuth, JWT, and user management"
# Use: "Create basic user login functionality"
```

**3. LLM Context Limits**
```bash
# Problem: Context exceeds token limits
# Solution: Enable context compression
export ENABLE_CONTEXT_COMPRESSION=true
autogen-framework --continue-workflow
```

### Error Handling Issues

#### Symptoms
- Phase failures without clear guidance
- Workflow stuck after errors
- Unclear error messages

#### Diagnostic Steps
```bash
# Check workflow error logs
grep "failed:" logs/framework.log

# View error guidance messages
grep "To fix this issue, use:" logs/framework.log

# Check workflow state after errors
autogen-framework --status
```

#### Common Causes and Solutions

**1. Phase Failure Without Clear Guidance**
```bash
# Problem: Phase fails but no clear next steps
# Solution: Check error logs for --revise guidance

# View recent phase failures
grep "phase failed:" logs/framework.log

# Look for revision guidance
grep "autogen-framework --revise" logs/framework.log
```

**2. Task-Level Error Recovery Issues**
```bash
# Problem: Individual tasks failing repeatedly
# Solution: Check ImplementAgent's ErrorRecovery component

# Check task execution logs
grep "ImplementAgent\|TaskDecomposer\|ErrorRecovery" logs/framework.log

# View task-level error recovery attempts
grep "recovery.*attempt\|strategy.*applied" logs/framework.log
```

**3. Manual Revision Not Working**
```bash
# Problem: --revise command not fixing issues
# Solution: Check revision feedback and workflow state

# Test revision command
autogen-framework --revise "requirements:Please provide more specific requirements"

# Check if revision was applied
autogen-framework --status
```

### Context Management Issues

#### Symptoms
- Missing project context
- Context compression errors
- Agent-specific context failures
- Memory pattern retrieval issues

#### Diagnostic Steps
```bash
# Check ContextManager status
grep "ContextManager" logs/framework.log

# Verify context files
ls -la requirements.md design.md tasks.md

# Check memory patterns
ls -la memory/projects/ memory/global/

# Test context compression
grep "compress_if_needed" logs/framework.log
```

#### Common Causes and Solutions

**1. Context Files Missing**
```bash
# Problem: Required context files don't exist
# Solution: Regenerate missing files

# Check which files are missing
ls -la requirements.md design.md tasks.md

# Regenerate missing files
autogen-framework --regenerate requirements
autogen-framework --regenerate design
```

**2. Context Compression Failures**
```python
# Problem: Context compression not working
# Solution: Check ContextCompressor configuration

# Verify compression settings
grep "context_compressor" autogen_framework/context_manager.py

# Check token limits
grep "token.*threshold" logs/framework.log

# Disable compression temporarily if needed
export DISABLE_CONTEXT_COMPRESSION=true
```

**3. Memory Pattern Issues**
```bash
# Problem: Memory patterns not being retrieved
# Solution: Check MemoryManager integration

# Verify memory directory structure
ls -la memory/projects/ memory/global/

# Check memory pattern retrieval
grep "get_relevant_patterns" logs/framework.log

# Reset memory if corrupted
rm -rf memory/projects/current_project/
```

### Quality Measurement Issues

#### Symptoms
- Quality assessment failures
- Missing baseline measurements
- Quality gate failures
- Metric calculation errors

#### Diagnostic Steps
```bash
# Check quality measurement logs
grep "QualityMetrics\|quality_score" logs/framework.log

# Verify baseline files
ls -la artifacts/quality-reports/baseline/

# Check quality gate status
grep "quality_gate" logs/framework.log

# Test quality measurement
autogen-framework --run-quality-assessment
```

#### Common Causes and Solutions

**1. Missing Baseline**
```bash
# Problem: No baseline measurements for comparison
# Solution: Establish baseline measurements

# Create baseline directory
mkdir -p artifacts/quality-reports/baseline/

# Run baseline establishment
python -m autogen_framework.quality_metrics --establish-baseline

# Verify baseline files
ls -la artifacts/quality-reports/baseline/
```

**2. Quality Metric Failures**
```python
# Problem: Individual quality metrics failing
# Solution: Check metric implementations

# Test individual metrics
python -m autogen_framework.quality_metrics --test-metric functionality
python -m autogen_framework.quality_metrics --test-metric maintainability

# Check metric configuration
grep "quality_metrics" autogen_framework/quality_metrics.py
```

**3. Quality Gate Configuration**
```bash
# Problem: Quality gates too strict or too lenient
# Solution: Adjust quality gate thresholds

# Check current thresholds
grep "gate_requirements" autogen_framework/quality_metrics.py

# Adjust thresholds in configuration
# Edit quality gate requirements based on project needs
```

## 🔧 Advanced Troubleshooting

### Component Integration Issues

#### Symptoms
- Components not communicating properly
- Integration test failures
- Workflow interruptions

#### Diagnostic Approach
```bash
# Test component integration
python -m pytest tests/integration/test_enhanced_execution_flow.py -v

# Check component initialization
grep "initialize\|setup" logs/framework.log

# Verify component dependencies
python -c "
import autogen_framework
from autogen_framework.agents.task_decomposer import TaskDecomposer
from autogen_framework.agents.error_recovery import ErrorRecovery
from autogen_framework.context_manager import ContextManager
print('All components imported successfully')
"
```

### Performance Optimization

#### Memory Usage Issues
```bash
# Monitor memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Clear context cache
rm -rf memory/cache/
mkdir -p memory/cache/

# Optimize context compression
export CONTEXT_COMPRESSION_LEVEL=high
```

#### Token Limit Issues
```bash
# Check token usage
grep "token.*count\|token.*limit" logs/framework.log

# Enable aggressive compression
export ENABLE_AGGRESSIVE_COMPRESSION=true

# Use smaller context windows
export MAX_CONTEXT_SIZE=8000
```

### Recovery Procedures

#### Complete System Reset
```bash
# Full framework reset (use with caution)
autogen-framework --reset-session
rm -rf memory/session_state.json
rm -rf logs/framework.log
rm -rf artifacts/quality-reports/current/

# Reinitialize framework
autogen-framework --workspace . --initialize
```

#### Partial Recovery
```bash
# Reset only specific components
rm -rf memory/projects/current_project/context_cache/
rm -rf artifacts/quality-reports/current/

# Restart workflow from specific phase
autogen-framework --restart-from design
```

## 📞 Getting Help

### Log Collection for Support
```bash
# Collect comprehensive logs
mkdir -p support_logs/
cp logs/framework.log support_logs/
cp memory/session_state.json support_logs/
cp -r artifacts/quality-reports/ support_logs/
tar -czf support_logs.tar.gz support_logs/

# Include system information
uv --version > support_logs/system_info.txt
python --version >> support_logs/system_info.txt
ls -la requirements.md design.md tasks.md >> support_logs/system_info.txt
```

### Reporting Issues
When reporting issues, please include:
1. Framework version and configuration
2. Complete error messages and stack traces
3. Steps to reproduce the issue
4. System information (OS, Python version, uv version)
5. Relevant log files (use log collection script above)
6. Quality measurement results if applicable

### Community Resources
- GitHub Issues: Report bugs and feature requests
- Documentation: Check latest documentation for updates
- Examples: Review example implementations in `autogen_framework/examples/`