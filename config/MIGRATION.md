# Configuration Migration Guide

This guide helps you migrate from the old environment variable-based configuration to the new three-tier configuration system.

> **📖 For comprehensive migration instructions, see [docs/migration-guide.md](../docs/migration-guide.md)**
> 
> This file provides a quick overview. The comprehensive guide includes:
> - Pre-migration assessment tools
> - Step-by-step migration procedures  
> - Environment-specific configurations
> - Troubleshooting and rollback procedures
> - Validation checklists

## Migration Overview

The new configuration system separates concerns into three distinct layers:

1. **Command Arguments**: Execution control (how the current command behaves)
2. **Environment Variables**: Connection & environment settings (where to connect, what credentials to use)
3. **Config Files**: Behavior & model properties (how the framework behaves, model-specific settings)

## Step-by-Step Migration

### Step 1: Identify Current Configuration

First, examine your current `.env` file to identify which settings need to be migrated:

```bash
# Example current .env file
LLM_BASE_URL=http://localhost:8888/openai/v1    # Keep in .env
LLM_MODEL=models/gemini-2.0-flash              # Keep in .env
LLM_API_KEY=sk-123456                          # Keep in .env
LLM_TEMPERATURE=0.7                            # Move to config/framework.json
LLM_MAX_OUTPUT_TOKENS=4096                     # Move to config/framework.json
LLM_TIMEOUT_SECONDS=60                         # Move to config/framework.json
WORKSPACE_PATH=artifacts/                      # Keep in .env
LOG_LEVEL=DEBUG                                # Keep in .env
SHELL_TIMEOUT_SECONDS=30                       # Move to config/framework.json
SHELL_MAX_RETRIES=2                            # Move to config/framework.json
```

### Step 2: Update Environment Variables

**Keep these in your `.env` file** (connection & environment settings):
```bash
# Connection settings - KEEP IN .env
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-123456

# Environment settings - KEEP IN .env
WORKSPACE_PATH=artifacts/
LOG_LEVEL=DEBUG

# Optional: Override config file locations
CONFIG_DIR=./config
MODELS_CONFIG_FILE=./config/models.json
FRAMEWORK_CONFIG_FILE=./config/framework.json
```

### Step 3: Create Framework Configuration File

**Move these to `config/framework.json`** (behavior settings):
```json
{
  "llm": {
    "temperature": 0.7,
    "max_output_tokens": 4096,
    "timeout_seconds": 60
  },
  "shell": {
    "timeout_seconds": 30,
    "max_retries": 2
  },
  "context": {
    "compression_threshold": 0.9,
    "context_size_ratio": 0.8
  },
  "workflow": {
    "auto_approve": false,
    "verbose": false,
    "reset_session_on_start": false
  },
  "memory": {
    "max_history_entries": 100,
    "compression_enabled": true,
    "pattern_learning_enabled": true
  }
}
```

### Step 4: Verify Model Configuration

Ensure your model is properly configured in `config/models.json`. The framework comes with built-in configurations for common models, but you can customize as needed.

### Step 5: Update Command Usage

**Before** (using environment variables for execution control):
```bash
export AUTO_APPROVE=true
export VERBOSE=true
autogen-framework --request "Create a hello world script"
```

**After** (using command arguments for execution control):
```bash
autogen-framework --request "Create a hello world script" --auto-approve --verbose
```

### Step 6: Test the Migration

1. **Backup your current configuration**:
   ```bash
   cp .env .env.backup
   ```

2. **Update your `.env` file** to remove migrated settings:
   ```bash
   # Remove these lines from .env:
   # LLM_TEMPERATURE=0.7
   # LLM_MAX_OUTPUT_TOKENS=4096
   # LLM_TIMEOUT_SECONDS=60
   # SHELL_TIMEOUT_SECONDS=30
   # SHELL_MAX_RETRIES=2
   ```

3. **Test basic functionality**:
   ```bash
   autogen-framework --status
   ```

4. **Test with your typical workflow**:
   ```bash
   autogen-framework --request "Simple test task" --verbose
   ```

## Migration Examples

### Example 1: Basic Development Setup

**Before (.env)**:
```bash
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-123456
LLM_TEMPERATURE=0.5
LLM_MAX_OUTPUT_TOKENS=2048
WORKSPACE_PATH=./workspace
LOG_LEVEL=INFO
```

**After (.env)**:
```bash
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-123456
WORKSPACE_PATH=./workspace
LOG_LEVEL=INFO
```

**After (config/framework.json)**:
```json
{
  "llm": {
    "temperature": 0.5,
    "max_output_tokens": 2048,
    "timeout_seconds": 60
  },
  "shell": {
    "timeout_seconds": 30,
    "max_retries": 2
  }
}
```

### Example 2: Production Setup

**Before (.env)**:
```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-real-api-key
LLM_TEMPERATURE=0.3
LLM_MAX_OUTPUT_TOKENS=8192
LLM_TIMEOUT_SECONDS=120
WORKSPACE_PATH=/var/lib/autogen
LOG_LEVEL=WARNING
SHELL_TIMEOUT_SECONDS=60
SHELL_MAX_RETRIES=3
```

**After (.env)**:
```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-real-api-key
WORKSPACE_PATH=/var/lib/autogen
LOG_LEVEL=WARNING
```

**After (config/framework.json)**:
```json
{
  "llm": {
    "temperature": 0.3,
    "max_output_tokens": 8192,
    "timeout_seconds": 120
  },
  "shell": {
    "timeout_seconds": 60,
    "max_retries": 3
  },
  "workflow": {
    "auto_approve": false,
    "verbose": false
  }
}
```

### Example 3: Custom Model Setup

**Before (.env)**:
```bash
LLM_BASE_URL=http://custom-provider:8080/v1
LLM_MODEL=custom-model:7b
LLM_API_KEY=custom-key
LLM_TEMPERATURE=0.8
```

**After (.env)**:
```bash
LLM_BASE_URL=http://custom-provider:8080/v1
LLM_MODEL=custom-model:7b
LLM_API_KEY=custom-key
```

**After (config/models.json)** - Add custom model:
```json
{
  "models": {
    "custom-model:7b": {
      "family": "GPT_4",
      "token_limit": 32000,
      "capabilities": {
        "vision": false,
        "function_calling": true,
        "streaming": true
      }
    }
  }
}
```

**After (config/framework.json)**:
```json
{
  "llm": {
    "temperature": 0.8,
    "max_output_tokens": 4096,
    "timeout_seconds": 60
  }
}
```

## Troubleshooting Migration Issues

### Issue 1: Configuration Not Found
**Error**: `Config file ./config/framework.json not found, using built-in defaults`

**Solution**: Create the config file or set the path:
```bash
export FRAMEWORK_CONFIG_FILE=/path/to/your/framework.json
```

### Issue 2: Model Not Recognized
**Error**: `Unknown model family for custom-model, using defaults`

**Solution**: Add your model to `config/models.json` or create a pattern match.

### Issue 3: Settings Not Applied
**Error**: Settings from config files are not being used

**Solution**: Check configuration precedence. Environment variables override config files:
```bash
# Remove these from .env if you want to use config files:
unset LLM_TEMPERATURE
unset LLM_MAX_OUTPUT_TOKENS
```

### Issue 4: Invalid Configuration Format
**Error**: `Invalid config format in config/framework.json`

**Solution**: Validate your JSON syntax:
```bash
python -m json.tool config/framework.json
```

## Validation Checklist

After migration, verify:

- [ ] Framework starts without errors
- [ ] Model detection works correctly
- [ ] LLM settings are applied (temperature, tokens, timeout)
- [ ] Shell execution works with new timeout/retry settings
- [ ] Command arguments work for execution control
- [ ] Environment variables still control connection settings
- [ ] Config files control behavior settings
- [ ] Custom models (if any) are properly configured

## Rollback Plan

If you encounter issues, you can quickly rollback:

1. **Restore original .env file**:
   ```bash
   cp .env.backup .env
   ```

2. **Remove config files** (framework will use environment variables):
   ```bash
   rm config/framework.json
   # Keep config/models.json for model detection
   ```

3. **Test functionality**:
   ```bash
   autogen-framework --status
   ```

## Getting Help

If you encounter issues during migration:

1. **Check logs** for detailed error messages:
   ```bash
   tail -f logs/framework.log
   ```

2. **Use verbose mode** for debugging:
   ```bash
   autogen-framework --verbose --status
   ```

3. **Validate configuration** files:
   ```bash
   python -m json.tool config/models.json
   python -m json.tool config/framework.json
   ```

4. **Test with minimal configuration** to isolate issues

The migration is designed to be backward compatible, so existing deployments should continue working even without immediate migration.