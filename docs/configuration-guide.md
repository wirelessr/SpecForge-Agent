# Dynamic Model Configuration System Guide

## Overview

The AutoGen Multi-Agent Framework uses a sophisticated three-tier configuration system that provides flexibility, maintainability, and clear separation of concerns. This guide explains how to configure and use the dynamic model configuration system effectively.

## Three-Tier Configuration Architecture

The framework implements a hierarchical configuration system with clear responsibilities:

```
Command Arguments > Environment Variables > Config Files > Built-in Defaults
```

### 1. Command-Line Arguments (Execution Control)
**Purpose**: Control how the current execution behaves  
**Scope**: Single command execution  
**Persistence**: Not persisted, affects only current command  

**Available Arguments**:
- `--verbose`: Enable detailed logging output
- `--auto-approve`: Skip user approval prompts for automated execution
- `--reset-session`: Reset session state before execution
- `--config-dir /path/to/config`: Override configuration directory path
- `--models-config /path/to/models.json`: Override models configuration file
- `--framework-config /path/to/framework.json`: Override framework configuration file

**Example Usage**:
```bash
# Run with verbose output and auto-approval
autogen-framework --request "Create a hello world script" --verbose --auto-approve

# Use custom configuration directory
autogen-framework --config-dir /custom/config --request "Build a web app"

# Override specific config files
autogen-framework --models-config /path/to/custom-models.json --request "Task"
```

### 2. Environment Variables (Connection & Environment)
**Purpose**: Define connection endpoints and execution environment  
**Scope**: Session/deployment level  
**Persistence**: Stored in `.env` files or system environment  

**Connection Settings**:
- `LLM_BASE_URL`: LLM service endpoint (e.g., `http://localhost:8888/openai/v1`)
- `LLM_MODEL`: Model name to use (e.g., `models/gemini-2.0-flash`)
- `LLM_API_KEY`: Authentication key for LLM service

**Environment Settings**:
- `LOG_LEVEL`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `WORKSPACE_PATH`: Working directory path (default: `artifacts/`)
- `CONFIG_DIR`: Configuration directory path (default: `./config`)
- `MODELS_CONFIG_FILE`: Path to models configuration file
- `FRAMEWORK_CONFIG_FILE`: Path to framework configuration file

**Example .env file**:
```bash
# Connection settings
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-your-api-key

# Environment settings
LOG_LEVEL=INFO
WORKSPACE_PATH=./workspace
CONFIG_DIR=./config
```

### 3. Config Files (Behavior & Model Properties)
**Purpose**: Define behavior settings and model-specific properties  
**Scope**: Project/framework level  
**Persistence**: JSON files in configuration directory  

**Configuration Files**:
- `config/models.json`: Model families, token limits, capabilities
- `config/framework.json`: Timeouts, retry counts, behavior settings

### 4. Built-in Defaults (Fallback)
**Purpose**: Ensure system always has working values  
**Scope**: Code-level constants  
**Usage**: Automatic fallback when no other configuration is available  

## Model Configuration System

### Supported Model Families

The framework supports the following model families with built-in configurations:

#### Google Gemini Models
- **GEMINI_2_0_FLASH**: Latest Gemini 2.0 Flash models
  - Token Limit: 1,048,576 (1M tokens)
  - Capabilities: Function calling, streaming
  - Example: `models/gemini-2.0-flash`

- **GEMINI_1_5_PRO**: Gemini 1.5 Pro models
  - Token Limit: 2,097,152 (2M tokens)
  - Capabilities: Vision, function calling, streaming
  - Example: `models/gemini-1.5-pro`

- **GEMINI_1_5_FLASH**: Gemini 1.5 Flash models
  - Token Limit: 1,048,576 (1M tokens)
  - Capabilities: Vision, function calling, streaming
  - Example: `models/gemini-1.5-flash`

#### OpenAI GPT Models
- **GPT_4**: GPT-4 family models
  - Token Limit: 8,192 (standard) or 128,000 (turbo)
  - Capabilities: Function calling, streaming (vision for turbo)
  - Examples: `gpt-4`, `gpt-4-turbo`

#### Anthropic Claude Models
- **CLAUDE_3_OPUS**: Claude 3 Opus models
  - Token Limit: 200,000
  - Capabilities: Vision, function calling, streaming
  - Example: `claude-3-opus`

#### Custom Provider Models
- **Custom Format**: `provider:model` format support
  - Example: `gpt-oss:20b` (maps to GPT_4 family with 128K tokens)

### Model Detection Methods

The framework uses three methods to detect model properties:

#### 1. Exact Model Matching
Direct model name to configuration mapping for precise control.

```json
{
  "models": {
    "models/gemini-2.0-flash": {
      "family": "GEMINI_2_0_FLASH",
      "token_limit": 1048576,
      "capabilities": {
        "vision": false,
        "function_calling": true,
        "streaming": true
      }
    }
  }
}
```

#### 2. Pattern Matching
Regular expression patterns for dynamic model detection.

```json
{
  "patterns": [
    {
      "pattern": "^models/gemini-2\\.0",
      "family": "GEMINI_2_0_FLASH",
      "token_limit": 1048576,
      "capabilities": {
        "vision": false,
        "function_calling": true,
        "streaming": true
      }
    }
  ]
}
```

#### 3. Default Fallback
Safe defaults for unrecognized models.

```json
{
  "defaults": {
    "family": "GPT_4",
    "token_limit": 8192,
    "capabilities": {
      "vision": false,
      "function_calling": false,
      "streaming": false
    }
  }
}
```

### Token Limit and Context Management

The framework uses intelligent token management:

- **token_limit**: Model's maximum supported tokens (from provider)
- **context_size_ratio**: Percentage of tokens used for context (default: 0.8)
- **Calculated context size**: `token_limit * context_size_ratio`

This reserves 20% of tokens for model output while maximizing context usage.

## Framework Configuration System

### Configuration Categories

#### LLM Settings
Control how the framework interacts with language models:

```json
{
  "llm": {
    "temperature": 0.7,           // Creativity level (0.0-1.0)
    "max_output_tokens": 4096,    // Maximum response length
    "timeout_seconds": 60         // Request timeout
  }
}
```

#### Shell Execution Settings
Control command execution behavior:

```json
{
  "shell": {
    "timeout_seconds": 30,        // Command timeout
    "max_retries": 2              // Retry attempts on failure
  }
}
```

#### Context Management Settings
Control context compression and sizing:

```json
{
  "context": {
    "compression_threshold": 0.9,  // When to compress context (90% full)
    "context_size_ratio": 0.8      // Percentage of tokens for context
  }
}
```

#### Workflow Control Settings
Control framework behavior:

```json
{
  "workflow": {
    "auto_approve": false,         // Skip approval prompts
    "verbose": false,              // Enable verbose logging
    "reset_session_on_start": false // Reset session automatically
  }
}
```

#### Memory Management Settings
Control memory and learning behavior:

```json
{
  "memory": {
    "max_history_entries": 100,    // Maximum history entries
    "compression_enabled": true,   // Enable memory compression
    "pattern_learning_enabled": true // Enable pattern learning
  }
}
```

## Configuration Path Resolution

The framework resolves configuration paths using the following precedence:

1. **Command-line arguments** (highest precedence)
   ```bash
   --config-dir /custom/config
   --models-config /path/to/models.json
   --framework-config /path/to/framework.json
   ```

2. **Environment variables**
   ```bash
   CONFIG_DIR=/custom/config
   MODELS_CONFIG_FILE=/path/to/models.json
   FRAMEWORK_CONFIG_FILE=/path/to/framework.json
   ```

3. **Default paths**
   ```
   ./config/models.json
   ./config/framework.json
   ```

4. **Built-in defaults** (lowest precedence)
   - Used when no configuration files are found
   - Ensures framework always has working values

## Error Handling and Logging

### Configuration Loading Errors

The framework handles configuration errors gracefully:

- **Invalid JSON**: Logs error, uses built-in defaults
- **Missing files**: Uses built-in defaults, logs info message
- **Partial configuration**: Merges with defaults, warns about missing values
- **Unknown models**: Uses pattern matching, falls back to defaults

### Comprehensive Logging

The framework provides detailed logging for troubleshooting:

```
INFO: Model family for models/gemini-2.0-flash: GEMINI_2_0_FLASH (source: exact_match)
INFO: Token limit for models/gemini-2.0-flash: 1048576 (source: models.json)
INFO: Loaded framework configuration from ./config/framework.json
WARNING: Config file ./config/custom.json not found, using built-in defaults
ERROR: Invalid config format in ./config/models.json: Invalid JSON syntax
```

### Migration Warnings

When deprecated environment variables are detected:

```
WARNING: Found deprecated environment variables: LLM_TEMPERATURE, LLM_MAX_OUTPUT_TOKENS
INFO: Consider moving these settings to config/framework.json for better organization
INFO: See docs/migration-guide.md for step-by-step migration instructions
```

## Best Practices

### 1. Configuration Organization
- **Keep connection settings in environment variables** (`.env` files)
- **Use config files for behavior settings** (`config/*.json`)
- **Use command arguments for execution control** (temporary overrides)
- **Document custom configurations** for team collaboration

### 2. Model Configuration
- **Use exact mappings** for known models to ensure precision
- **Use pattern matching** for model families or provider formats
- **Test custom configurations** thoroughly before deployment
- **Keep defaults conservative** for unknown model compatibility

### 3. Environment Management
- **Use different `.env` files** for different environments (`.env.development`, `.env.production`)
- **Version control config files** but not `.env` files with secrets
- **Validate configurations** before deployment
- **Monitor logs** for configuration issues

### 4. Security Considerations
- **Never commit API keys** to version control
- **Use environment variables** for sensitive information
- **Restrict config file permissions** in production
- **Rotate API keys** regularly

### 5. Performance Optimization
- **Adjust token limits** based on your use case
- **Tune timeout values** for your network conditions
- **Enable compression** for large contexts
- **Monitor memory usage** and adjust limits accordingly

## Troubleshooting

### Common Issues

#### Model Not Recognized
**Symptom**: `Model 'custom-model' not recognized by any pattern`  
**Solution**: Add model to `config/models.json` or create pattern match

#### Configuration Not Applied
**Symptom**: Settings from config files not being used  
**Solution**: Check environment variables aren't overriding config files

#### Invalid Configuration Format
**Symptom**: `Invalid config format in config/framework.json`  
**Solution**: Validate JSON syntax with `python -m json.tool config/framework.json`

#### Connection Issues
**Symptom**: Cannot connect to LLM service  
**Solution**: Verify `LLM_BASE_URL`, `LLM_API_KEY`, and network connectivity

### Diagnostic Commands

```bash
# Check configuration status
autogen-framework --status

# Validate JSON configuration files
python -m json.tool config/models.json
python -m json.tool config/framework.json

# Test with verbose logging
autogen-framework --verbose --request "Simple test"

# Check environment variables
env | grep LLM_
env | grep CONFIG_
```

### Recovery Procedures

#### Reset to Defaults
```bash
# Backup current configuration
cp config/framework.json config/framework.json.backup

# Remove config files to use built-in defaults
rm config/framework.json config/models.json

# Test with defaults
autogen-framework --status
```

#### Minimal Configuration
```bash
# Create minimal .env file
cat > .env << EOF
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-your-key
EOF

# Test basic functionality
autogen-framework --status
```

This configuration system provides maximum flexibility while maintaining simplicity and reliability. The three-tier approach ensures that you can easily manage different aspects of the framework's behavior while keeping sensitive information secure and maintaining clear separation of concerns.