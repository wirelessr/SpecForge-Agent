# ContextManager Token Configuration

## Overview

The ContextManager now uses global token configuration from the ConfigManager instead of hardcoded values. This provides a unified, configurable token threshold for all agents.

## Configuration Options

### Global Token Configuration

The following environment variables control token management:

```bash
# Unified token limit for all agents (default: 8192)
DEFAULT_TOKEN_LIMIT=8192

# Token compression threshold (0.0-1.0, default: 0.9)
TOKEN_COMPRESSION_THRESHOLD=0.9

# Enable/disable context compression (default: true)
CONTEXT_COMPRESSION_ENABLED=true

# Target compression ratio (0.0-1.0, default: 0.5)
COMPRESSION_TARGET_RATIO=0.5

# Enable verbose token logging (default: false)
VERBOSE_TOKEN_LOGGING=false
```

### Unified Token Threshold

All agents (PlanAgent, DesignAgent, TasksAgent, ImplementAgent) use the same token threshold defined by `DEFAULT_TOKEN_LIMIT`. This ensures consistent behavior across all agent types.

## Example Configuration

### Development Environment (.env.development)
```bash
DEFAULT_TOKEN_LIMIT=4096
TOKEN_COMPRESSION_THRESHOLD=0.8
VERBOSE_TOKEN_LOGGING=true
```

### Production Environment (.env.production)
```bash
DEFAULT_TOKEN_LIMIT=8192
TOKEN_COMPRESSION_THRESHOLD=0.9
CONTEXT_COMPRESSION_ENABLED=true
COMPRESSION_TARGET_RATIO=0.6
```

### High-Performance Environment (.env.high-performance)
```bash
DEFAULT_TOKEN_LIMIT=16384
TOKEN_COMPRESSION_THRESHOLD=0.95
```

## Usage in Code

### ContextManager Initialization

```python
from autogen_framework.config_manager import ConfigManager
from autogen_framework.context_manager import ContextManager

# With explicit config manager
config_manager = ConfigManager()
context_manager = ContextManager(
    work_dir="/path/to/project",
    memory_manager=memory_manager,
    context_compressor=context_compressor,
    config_manager=config_manager
)

# With default config manager (loads from environment)
context_manager = ContextManager(
    work_dir="/path/to/project",
    memory_manager=memory_manager,
    context_compressor=context_compressor
)
```

### Checking Current Configuration

```python
# Get current token configuration
token_config = context_manager.token_config
print(f"Default token limit: {token_config['default_token_limit']}")
print(f"Compression enabled: {token_config['compression_enabled']}")

# Get current threshold
threshold = context_manager.token_threshold
print(f"Token threshold: {threshold}")
```

## Benefits

1. **Flexibility**: Different environments can have different token limits
2. **Performance Tuning**: Adjust threshold based on available resources
3. **Cost Control**: Lower limits in development, higher in production
4. **Consistency**: All agents use the same token threshold for predictable behavior
5. **Runtime Configuration**: No code changes needed for threshold adjustments

## Migration from Hardcoded Values

The old hardcoded `TOKEN_THRESHOLDS` dictionary has been replaced with unified configuration:

```python
# OLD (hardcoded, agent-specific)
TOKEN_THRESHOLDS = {
    "plan": 4000,
    "design": 6000,
    "tasks": 5000,
    "implementation": 8000
}

# NEW (configurable, unified)
self.token_threshold = self.token_config.get('default_token_limit', 8192)
```

## Troubleshooting

### Common Issues

1. **Invalid Token Threshold**: If an environment variable contains an invalid integer, the system falls back to the default multiplier
2. **Missing Configuration**: If no configuration is provided, the system uses sensible defaults
3. **Compression Issues**: If compression fails, the original context is used without compression

### Debugging

Enable verbose logging to see token threshold decisions:

```bash
VERBOSE_TOKEN_LOGGING=true
```

This will log:
- Token threshold calculations
- Custom threshold usage
- Compression decisions
- Context size estimations

### Validation

The ConfigManager validates all token configuration:
- Token limits must be positive integers
- Compression thresholds must be between 0.0 and 1.0
- Invalid values are logged and replaced with defaults