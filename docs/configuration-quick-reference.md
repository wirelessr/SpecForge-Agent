# Configuration Quick Reference

## Three-Tier System Overview

```
Command Arguments > Environment Variables > Config Files > Built-in Defaults
```

## Command Arguments (Execution Control)

```bash
# Execution behavior
--verbose                    # Enable detailed logging
--auto-approve              # Skip approval prompts  
--reset-session             # Reset session state

# Configuration overrides
--config-dir /path/to/config
--models-config /path/to/models.json
--framework-config /path/to/framework.json
```

## Environment Variables (Connection & Environment)

```bash
# Connection settings (keep in .env)
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-your-api-key

# Environment settings (keep in .env)
WORKSPACE_PATH=./workspace
LOG_LEVEL=INFO
CONFIG_DIR=./config
```

## Config Files (Behavior & Model Properties)

### Framework Configuration (`config/framework.json`)

```json
{
  "llm": {
    "temperature": 0.7,           // Creativity (0.0-1.0)
    "max_output_tokens": 4096,    // Max response length
    "timeout_seconds": 60         // Request timeout
  },
  "shell": {
    "timeout_seconds": 30,        // Command timeout
    "max_retries": 2              // Retry attempts
  },
  "context": {
    "compression_threshold": 0.9,  // When to compress (90% full)
    "context_size_ratio": 0.8      // Context vs output ratio
  },
  "workflow": {
    "auto_approve": false,         // Skip approval prompts
    "verbose": false,              // Enable verbose logging
    "reset_session_on_start": false
  },
  "memory": {
    "max_history_entries": 100,    // Max history entries
    "compression_enabled": true,   // Enable compression
    "pattern_learning_enabled": true
  }
}
```

### Model Configuration (`config/models.json`)

```json
{
  "models": {
    "your-model-name": {
      "family": "GPT_4",
      "token_limit": 32000,
      "capabilities": {
        "vision": false,
        "function_calling": true,
        "streaming": true
      }
    }
  },
  "patterns": [
    {
      "pattern": "^your-provider:",
      "family": "GPT_4", 
      "token_limit": 16000,
      "capabilities": { ... }
    }
  ],
  "defaults": {
    "family": "GPT_4",
    "token_limit": 8192,
    "capabilities": { ... }
  }
}
```

## Supported Model Families

| Family | Token Limit | Capabilities | Example Models |
|--------|-------------|--------------|----------------|
| GEMINI_2_0_FLASH | 1M | Function calling, streaming | models/gemini-2.0-flash |
| GEMINI_1_5_PRO | 2M | Vision, function calling, streaming | models/gemini-1.5-pro |
| GEMINI_1_5_FLASH | 1M | Vision, function calling, streaming | models/gemini-1.5-flash |
| GPT_4 | 8K-128K | Function calling, streaming, vision (turbo) | gpt-4, gpt-4-turbo |
| CLAUDE_3_OPUS | 200K | Vision, function calling, streaming | claude-3-opus |

## Common Configuration Patterns

### Development Setup
```bash
# .env
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-dev-key

# Use development config
autogen-framework --framework-config config/examples/development.json
```

### Production Setup
```bash
# .env
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-prod-key

# Use production config
autogen-framework --framework-config config/examples/production.json --auto-approve
```

### Custom Model Setup
```json
// Add to config/models.json
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

## Configuration Path Resolution

1. **Command arguments** (highest precedence)
   ```bash
   --config-dir /custom/config
   ```

2. **Environment variables**
   ```bash
   CONFIG_DIR=/custom/config
   ```

3. **Default paths**
   ```
   ./config/models.json
   ./config/framework.json
   ```

4. **Built-in defaults** (lowest precedence)

## Troubleshooting

### Check Configuration Status
```bash
autogen-framework --verbose --status
```

### Validate JSON Files
```bash
python -m json.tool config/framework.json
python -m json.tool config/models.json
```

### Test Model Detection
```python
from autogen_framework.config_manager import ConfigManager
config_manager = ConfigManager()
model_info = config_manager.get_model_info('your-model')
print(model_info)
```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Config not found | `Config file not found, using defaults` | Check file paths and permissions |
| Model not recognized | `Model not recognized by any pattern` | Add model to config/models.json |
| Settings not applied | Config values ignored | Check env var overrides |
| Invalid JSON | `Invalid config format` | Validate JSON syntax |

## Migration from Old System

1. **Keep in .env**: Connection settings (URL, model, API key)
2. **Move to config files**: Behavior settings (temperature, timeouts)
3. **Use command args**: Execution control (verbose, auto-approve)

See [Migration Guide](migration-guide.md) for detailed steps.

## Quick Setup Commands

```bash
# Create minimal .env
cat > .env << EOF
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-your-key
EOF

# Use default configs (already included)
autogen-framework --status

# Test with verbose output
autogen-framework --verbose --request "Simple test"
```