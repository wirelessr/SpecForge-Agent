# Configuration Guide

This directory contains configuration files for the AutoGen Multi-Agent Framework's dynamic model configuration system.

## Configuration Files

### models.json
Contains model-specific configurations including:
- **Exact model mappings**: Direct model name to configuration mappings
- **Pattern matching**: Regex patterns for dynamic model detection
- **Default fallbacks**: Default values when no match is found

### framework.json
Contains framework behavior settings including:
- **LLM settings**: Temperature, token limits, timeouts
- **Shell execution**: Command timeouts and retry policies
- **Context management**: Compression and context size settings
- **Workflow control**: Auto-approval and verbosity settings
- **Memory management**: History limits and learning settings

## Configuration Source Hierarchy

The framework uses a three-tier configuration system with clear separation of concerns:

```
Command Args > Environment Variables > Config Files > Built-in Defaults
```

### 1. Command-Line Arguments (Execution Control)
**Purpose**: Control how the current execution behaves
**Scope**: Single command execution
**Examples**:
- `--verbose`: Enable verbose output
- `--auto-approve`: Skip approval prompts
- `--reset-session`: Reset session state
- `--config-dir /path/to/config`: Override config directory
- `--models-config /path/to/models.json`: Override models config file
- `--framework-config /path/to/framework.json`: Override framework config file

### 2. Environment Variables (Connection & Environment)
**Purpose**: Define connection endpoints and execution environment
**Scope**: Session/deployment level
**Examples**:
- `LLM_BASE_URL`: LLM service endpoint
- `LLM_MODEL`: Model name to use
- `LLM_API_KEY`: Authentication key
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `WORKSPACE_PATH`: Working directory path
- `CONFIG_DIR`: Configuration directory path
- `MODELS_CONFIG_FILE`: Path to models configuration file
- `FRAMEWORK_CONFIG_FILE`: Path to framework configuration file

### 3. Config Files (Behavior & Model Properties)
**Purpose**: Define behavior settings and model-specific properties
**Scope**: Project/framework level
**Storage**: `config/models.json`, `config/framework.json`

### 4. Built-in Defaults (Fallback)
**Purpose**: Ensure system always has working values
**Scope**: Code-level constants

## Model Configuration Schema

### Exact Model Mappings
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

### Pattern Matching
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

### Default Fallbacks
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

## Framework Configuration Schema

### LLM Settings
```json
{
  "llm": {
    "temperature": 0.7,
    "max_output_tokens": 4096,
    "timeout_seconds": 60
  }
}
```

### Shell Execution Settings
```json
{
  "shell": {
    "timeout_seconds": 30,
    "max_retries": 2
  }
}
```

### Context Management Settings
```json
{
  "context": {
    "compression_threshold": 0.9,
    "context_size_ratio": 0.8
  }
}
```

### Workflow Control Settings
```json
{
  "workflow": {
    "auto_approve": false,
    "verbose": false,
    "reset_session_on_start": false
  }
}
```

### Memory Management Settings
```json
{
  "memory": {
    "max_history_entries": 100,
    "compression_enabled": true,
    "pattern_learning_enabled": true
  }
}
```

## Supported Model Families

The framework supports the following model families:

- **GEMINI_2_0_FLASH**: Google's Gemini 2.0 Flash models
- **GEMINI_1_5_PRO**: Google's Gemini 1.5 Pro models
- **GEMINI_1_5_FLASH**: Google's Gemini 1.5 Flash models
- **GPT_4**: OpenAI's GPT-4 family models
- **CLAUDE_3**: Anthropic's Claude 3 family models

## Built-in Model Configurations

### Google Gemini Models
- `models/gemini-2.0-flash`: 1M tokens, function calling
- `models/gemini-1.5-pro`: 2M tokens, vision + function calling
- `models/gemini-1.5-flash`: 1M tokens, vision + function calling

### OpenAI GPT Models
- `gpt-4-turbo`: 128K tokens, vision + function calling
- `gpt-4`: 8K tokens, function calling

### Anthropic Claude Models
- `claude-3-opus`: 200K tokens, vision + function calling

### Custom Provider Models
- `gpt-oss:20b`: Custom provider format, 128K tokens

## Token Limit and Context Size Calculation

The framework uses the following approach for token management:

- **token_limit**: Model's maximum supported tokens (hard limit from model provider)
- **context_size_ratio**: Ratio of token_limit to use for context (default: 0.8)
- **Calculated max_context_size**: `token_limit * context_size_ratio`

This ensures that 80% of the token limit is used for context, reserving 20% for model output.

## Configuration Path Resolution

The framework resolves configuration paths in the following order:

1. **Command-line argument**: `--config-dir /path/to/config`
2. **Environment variable**: `CONFIG_DIR=/path/to/config`
3. **Default**: `./config/` (current directory)
4. **Fallback**: Built-in defaults if no config files found

Individual files can be overridden:
- `--models-config /path/to/models.json`
- `--framework-config /path/to/framework.json`
- Environment variables: `MODELS_CONFIG_FILE`, `FRAMEWORK_CONFIG_FILE`

## Error Handling

### Configuration Loading Errors
- **Invalid Config File**: Log error, use built-in defaults
- **Missing Config File**: Use built-in defaults, log info message
- **Partial Config**: Merge with defaults, log warnings for missing values
- **Unknown Model**: Use pattern matching, fallback to defaults if no match

### Logging
The framework provides comprehensive logging for configuration:
```
INFO: Model family for models/gemini-2.0-flash: GEMINI_2_0_FLASH (source: exact_match)
INFO: Token limit for models/gemini-2.0-flash: 1048576 (source: models.json)
WARNING: Config file /path/to/config/framework.json not found, using built-in defaults
ERROR: Invalid config format in /path/to/config/models.json: Invalid JSON syntax
```

## Custom Model Configuration

You can add custom model configurations by editing `config/models.json`:

### Adding Exact Model Mappings
```json
{
  "models": {
    "your-custom-model": {
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

### Adding Pattern Matching
```json
{
  "patterns": [
    {
      "pattern": "^your-provider:",
      "family": "GPT_4",
      "token_limit": 16000,
      "capabilities": {
        "vision": false,
        "function_calling": true,
        "streaming": false
      }
    }
  ]
}
```

## Best Practices

1. **Use exact mappings** for known models to ensure precise configuration
2. **Use pattern matching** for model families or provider-specific formats
3. **Test custom configurations** with your specific models before deployment
4. **Keep defaults conservative** to ensure compatibility with unknown models
5. **Document custom configurations** for team members and future reference
6. **Version control config files** to track configuration changes over time