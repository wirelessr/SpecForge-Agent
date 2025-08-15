# Configuration Best Practices Guide

## Overview

This guide provides comprehensive best practices for using the AutoGen Multi-Agent Framework's dynamic configuration system effectively. Following these practices will help you maintain secure, performant, and maintainable configurations across different environments and use cases.

## Configuration Architecture Best Practices

### 1. Separation of Concerns

Follow the three-tier architecture strictly to maintain clear separation:

#### Command Arguments (Execution Control)
**Use for**: Temporary overrides and execution-specific behavior
```bash
# ✅ Good - Execution control
autogen-framework --verbose --auto-approve --request "Task"

# ❌ Bad - Don't put persistent settings in arguments
autogen-framework --temperature 0.7 --request "Task"  # Not supported
```

#### Environment Variables (Connection & Environment)
**Use for**: Connection endpoints, credentials, and environment-specific paths
```bash
# ✅ Good - Connection and environment settings
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-your-api-key
WORKSPACE_PATH=./workspace
LOG_LEVEL=INFO

# ❌ Bad - Behavior settings (use config files instead)
LLM_TEMPERATURE=0.7
LLM_MAX_OUTPUT_TOKENS=4096
```

#### Config Files (Behavior & Model Properties)
**Use for**: Framework behavior, model properties, and team-shared settings
```json
{
  "llm": {
    "temperature": 0.7,
    "max_output_tokens": 4096,
    "timeout_seconds": 60
  }
}
```

### 2. Configuration Precedence Management

Understand and leverage the configuration precedence hierarchy:

```
Command Args > Environment Variables > Config Files > Built-in Defaults
```

**Best Practice**: Use higher precedence levels sparingly for overrides only:

```bash
# ✅ Good - Use config files for standard settings
# config/framework.json contains temperature: 0.7

# Use environment variable only for environment-specific override
export LLM_TEMPERATURE=0.5  # Only for this specific environment

# Use command argument only for one-time override
autogen-framework --verbose --request "Task"  # Only for this execution
```

## Security Best Practices

### 1. Credential Management

**Never commit credentials to version control**:

```bash
# ✅ Good - Keep credentials in environment variables
# .env (not committed)
LLM_API_KEY=sk-real-api-key-here

# .env.example (committed as template)
LLM_API_KEY=sk-your-api-key-here

# ❌ Bad - Never put credentials in config files
# config/framework.json (committed)
{
  "llm": {
    "api_key": "sk-real-api-key-here"  // Never do this!
  }
}
```

**Use different credentials for different environments**:

```bash
# Development (.env.development)
LLM_API_KEY=sk-dev-key

# Production (.env.production)
LLM_API_KEY=sk-prod-key

# Testing (.env.test)
LLM_API_KEY=sk-test-key
```

### 2. File Permissions

Set appropriate permissions for configuration files:

```bash
# Config files (can be world-readable)
chmod 644 config/*.json

# Environment files (should be restricted)
chmod 600 .env*

# Config directory
chmod 755 config/
```

### 3. Credential Rotation

Implement regular credential rotation:

```bash
# Script for credential rotation
#!/bin/bash
# rotate_credentials.sh

OLD_KEY=$(grep LLM_API_KEY .env | cut -d'=' -f2)
NEW_KEY="sk-new-api-key"

# Update .env file
sed -i "s/LLM_API_KEY=$OLD_KEY/LLM_API_KEY=$NEW_KEY/" .env

# Test new credentials
autogen-framework --status

echo "Credentials rotated successfully"
```

## Environment Management Best Practices

### 1. Environment-Specific Configurations

Create separate configurations for different environments:

#### Development Environment
```json
// config/framework.development.json
{
  "llm": {
    "temperature": 0.5,        // More deterministic for testing
    "max_output_tokens": 2048, // Smaller for faster iteration
    "timeout_seconds": 30      // Shorter for quick feedback
  },
  "shell": {
    "timeout_seconds": 15,     // Fast feedback
    "max_retries": 1           // Fail fast for debugging
  },
  "workflow": {
    "verbose": true,           // More logging for debugging
    "auto_approve": false      // Manual control for development
  },
  "memory": {
    "max_history_entries": 50  // Smaller for faster startup
  }
}
```

#### Production Environment
```json
// config/framework.production.json
{
  "llm": {
    "temperature": 0.3,        // Consistent results
    "max_output_tokens": 8192, // Larger for complex tasks
    "timeout_seconds": 120     // Longer for reliability
  },
  "shell": {
    "timeout_seconds": 60,     // Longer for complex operations
    "max_retries": 3           // More retries for reliability
  },
  "workflow": {
    "verbose": false,          // Cleaner logs
    "auto_approve": true       // Automated execution
  },
  "memory": {
    "max_history_entries": 200 // Better context retention
  }
}
```

#### Testing Environment
```json
// config/framework.testing.json
{
  "llm": {
    "temperature": 0.1,        // Very deterministic
    "max_output_tokens": 1024, // Small for fast tests
    "timeout_seconds": 30      // Quick timeout for tests
  },
  "shell": {
    "timeout_seconds": 10,     // Very fast timeout
    "max_retries": 1           // Fail fast
  },
  "workflow": {
    "auto_approve": true,      // No manual intervention
    "verbose": false,          // Clean test output
    "reset_session_on_start": true // Clean state
  },
  "memory": {
    "max_history_entries": 10  // Minimal memory
  }
}
```

### 2. Environment Selection

Use environment variables to select configurations:

```bash
# Set environment type
export ENVIRONMENT=development
export FRAMEWORK_CONFIG_FILE=config/framework.$ENVIRONMENT.json

# Or use command line
autogen-framework --framework-config config/framework.production.json
```

### 3. Configuration Validation

Validate configurations for each environment:

```bash
#!/bin/bash
# validate_configs.sh

ENVIRONMENTS=("development" "production" "testing")

for env in "${ENVIRONMENTS[@]}"; do
    echo "Validating $env configuration..."
    
    # Validate JSON syntax
    if ! python -m json.tool config/framework.$env.json > /dev/null; then
        echo "❌ Invalid JSON in framework.$env.json"
        exit 1
    fi
    
    # Test configuration loading
    if ! FRAMEWORK_CONFIG_FILE=config/framework.$env.json autogen-framework --status > /dev/null; then
        echo "❌ Configuration loading failed for $env"
        exit 1
    fi
    
    echo "✅ $env configuration valid"
done

echo "All configurations validated successfully"
```

## Model Configuration Best Practices

### 1. Model Organization

Organize model configurations logically:

```json
{
  "models": {
    // Group by provider
    "models/gemini-2.0-flash": { "family": "GEMINI_2_0_FLASH", ... },
    "models/gemini-1.5-pro": { "family": "GEMINI_1_5_PRO", ... },
    
    "gpt-4-turbo": { "family": "GPT_4", ... },
    "gpt-4": { "family": "GPT_4", ... },
    
    "claude-3-opus": { "family": "CLAUDE_3_OPUS", ... },
    
    // Custom models
    "custom-provider:7b": { "family": "GPT_4", ... },
    "local-model:13b": { "family": "GPT_4", ... }
  },
  "patterns": [
    // Order patterns from most specific to least specific
    { "pattern": "^models/gemini-2\\.0", ... },
    { "pattern": "^models/gemini-1\\.5-pro", ... },
    { "pattern": "^models/gemini-1\\.5", ... },
    { "pattern": "^custom-provider:", ... },
    { "pattern": "gpt-4", ... }
  ]
}
```

### 2. Token Limit Management

Set appropriate token limits based on your use case:

```json
{
  "models": {
    "high-capacity-model": {
      "token_limit": 200000,    // Large context for complex tasks
      "capabilities": { ... }
    },
    "fast-model": {
      "token_limit": 8192,      // Smaller context for quick tasks
      "capabilities": { ... }
    }
  }
}
```

**Consider context size ratio**:
```json
{
  "context": {
    "context_size_ratio": 0.8   // 80% for context, 20% for output
  }
}
```

### 3. Capability Management

Accurately define model capabilities:

```json
{
  "models": {
    "vision-model": {
      "capabilities": {
        "vision": true,           // Can process images
        "function_calling": true, // Can call functions
        "streaming": true         // Supports streaming responses
      }
    },
    "text-only-model": {
      "capabilities": {
        "vision": false,          // Text only
        "function_calling": true, // Can call functions
        "streaming": false        // No streaming support
      }
    }
  }
}
```

### 4. Custom Model Integration

Add custom models systematically:

```bash
#!/bin/bash
# add_custom_model.sh

MODEL_NAME="$1"
FAMILY="$2"
TOKEN_LIMIT="$3"

if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_name> <family> <token_limit>"
    echo "Example: $0 'custom-model:7b' 'GPT_4' 32000"
    exit 1
fi

# Add model to configuration
python3 << EOF
import json

with open('config/models.json', 'r') as f:
    config = json.load(f)

# Add new model
config['models']['$MODEL_NAME'] = {
    "family": "$FAMILY",
    "token_limit": $TOKEN_LIMIT,
    "capabilities": {
        "vision": False,
        "function_calling": True,
        "streaming": True
    }
}

with open('config/models.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Model '$MODEL_NAME' added successfully")
EOF

# Test the new model
echo "Testing model configuration..."
python3 << EOF
import sys
sys.path.append('.')
from autogen_framework.config_manager import ConfigManager

config_manager = ConfigManager()
model_info = config_manager.get_model_info('$MODEL_NAME')
print(f"Model: $MODEL_NAME")
print(f"Family: {model_info['family']}")
print(f"Token Limit: {model_info['token_limit']}")
print(f"Capabilities: {model_info['capabilities']}")
EOF
```

## Performance Optimization Best Practices

### 1. Token Management

Optimize token usage for your workload:

```json
{
  "llm": {
    // Adjust based on task complexity
    "max_output_tokens": 4096,  // Increase for complex tasks
    "timeout_seconds": 60       // Increase for large contexts
  },
  "context": {
    "compression_threshold": 0.9, // Compress when 90% full
    "context_size_ratio": 0.8     // 80% context, 20% output
  }
}
```

### 2. Timeout Configuration

Set timeouts based on your environment:

```json
{
  "llm": {
    // Network conditions
    "timeout_seconds": 60       // Local: 30s, Remote: 60s, Slow: 120s
  },
  "shell": {
    // Command complexity
    "timeout_seconds": 30,      // Simple: 15s, Complex: 60s
    "max_retries": 2            // Fast fail: 1, Reliable: 3
  }
}
```

### 3. Memory Management

Configure memory settings for optimal performance:

```json
{
  "memory": {
    "max_history_entries": 100,     // Adjust based on available memory
    "compression_enabled": true,    // Enable for large contexts
    "pattern_learning_enabled": true // Enable for improved performance
  }
}
```

### 4. Caching Strategy

Implement configuration caching:

```python
# Example: Configuration caching
class CachedConfigManager:
    def __init__(self):
        self._config_cache = {}
        self._cache_timestamp = {}
    
    def get_config_with_cache(self, config_path):
        # Check if config file has been modified
        mtime = os.path.getmtime(config_path)
        
        if (config_path not in self._cache_timestamp or 
            self._cache_timestamp[config_path] < mtime):
            
            # Reload configuration
            with open(config_path) as f:
                self._config_cache[config_path] = json.load(f)
            self._cache_timestamp[config_path] = mtime
        
        return self._config_cache[config_path]
```

## Team Collaboration Best Practices

### 1. Configuration Versioning

Version control configuration files properly:

```bash
# ✅ Commit these files
git add config/models.json
git add config/framework.json
git add config/examples/
git add .env.example

# ❌ Never commit these files
echo ".env*" >> .gitignore
echo "!.env.example" >> .gitignore
```

### 2. Configuration Documentation

Document configuration decisions:

```json
// config/framework.json
{
  "llm": {
    // Reduced temperature for more consistent code generation
    "temperature": 0.3,
    
    // Increased token limit for complex architectural tasks
    "max_output_tokens": 8192,
    
    // Extended timeout for large model responses
    "timeout_seconds": 120
  }
}
```

### 3. Configuration Reviews

Implement configuration review process:

```bash
#!/bin/bash
# config_review.sh

echo "Configuration Review Checklist:"
echo "================================"

# Check for sensitive information
if grep -r "sk-" config/; then
    echo "❌ Found potential API keys in config files"
else
    echo "✅ No API keys found in config files"
fi

# Validate JSON syntax
for file in config/*.json; do
    if python -m json.tool "$file" > /dev/null; then
        echo "✅ $file has valid JSON syntax"
    else
        echo "❌ $file has invalid JSON syntax"
    fi
done

# Check for reasonable values
python3 << 'EOF'
import json

with open('config/framework.json') as f:
    config = json.load(f)

# Check temperature range
temp = config.get('llm', {}).get('temperature', 0.7)
if 0.0 <= temp <= 1.0:
    print(f"✅ Temperature {temp} is in valid range")
else:
    print(f"❌ Temperature {temp} is outside valid range (0.0-1.0)")

# Check token limits
tokens = config.get('llm', {}).get('max_output_tokens', 4096)
if 100 <= tokens <= 32000:
    print(f"✅ Max output tokens {tokens} is reasonable")
else:
    print(f"⚠ Max output tokens {tokens} may be too high/low")

# Check timeouts
timeout = config.get('llm', {}).get('timeout_seconds', 60)
if 10 <= timeout <= 300:
    print(f"✅ Timeout {timeout}s is reasonable")
else:
    print(f"⚠ Timeout {timeout}s may be too short/long")
EOF
```

### 4. Environment Synchronization

Keep team environments synchronized:

```bash
#!/bin/bash
# sync_config.sh

echo "Synchronizing configuration across environments..."

# Pull latest config changes
git pull origin main

# Validate configurations
./validate_configs.sh

# Update local environment template
cp .env.example .env.template

echo "Configuration synchronized. Please update your .env file if needed."
```

## Monitoring and Maintenance Best Practices

### 1. Configuration Monitoring

Monitor configuration usage and performance:

```python
# Example: Configuration monitoring
import logging
import time
from functools import wraps

def monitor_config_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        logging.info(f"Config operation {func.__name__} took {duration:.2f}s")
        return result
    return wrapper

class MonitoredConfigManager(ConfigManager):
    @monitor_config_usage
    def get_model_info(self, model_name):
        return super().get_model_info(model_name)
```

### 2. Configuration Validation

Implement automated configuration validation:

```bash
#!/bin/bash
# validate_production_config.sh

echo "Validating production configuration..."

# Check required environment variables
REQUIRED_VARS=("LLM_BASE_URL" "LLM_MODEL" "LLM_API_KEY")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set"
        exit 1
    else
        echo "✅ $var is set"
    fi
done

# Test API connectivity
if curl -s "$LLM_BASE_URL/models" > /dev/null; then
    echo "✅ API endpoint is accessible"
else
    echo "❌ Cannot reach API endpoint"
    exit 1
fi

# Validate model configuration
python3 << 'EOF'
import sys
sys.path.append('.')
from autogen_framework.config_manager import ConfigManager
import os

config_manager = ConfigManager()
model_name = os.getenv('LLM_MODEL')

try:
    model_info = config_manager.get_model_info(model_name)
    print(f"✅ Model {model_name} configuration valid")
    print(f"   Family: {model_info['family']}")
    print(f"   Token Limit: {model_info['token_limit']}")
except Exception as e:
    print(f"❌ Model configuration error: {e}")
    sys.exit(1)
EOF

echo "Production configuration validation complete"
```

### 3. Performance Monitoring

Monitor configuration impact on performance:

```python
# Example: Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name):
        if name in self.metrics:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0
    
    def report(self):
        for name, values in self.metrics.items():
            avg = self.get_average(name)
            print(f"{name}: avg={avg:.2f}, count={len(values)}")

# Usage in ConfigManager
monitor = PerformanceMonitor()

def timed_operation(operation_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            monitor.record_metric(operation_name, duration)
            return result
        return wrapper
    return decorator
```

### 4. Configuration Backup and Recovery

Implement configuration backup strategies:

```bash
#!/bin/bash
# backup_config.sh

BACKUP_DIR="config_backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration files
cp config/*.json "$BACKUP_DIR/"
cp .env.example "$BACKUP_DIR/"

# Create backup manifest
cat > "$BACKUP_DIR/manifest.txt" << EOF
Backup created: $(date)
Git commit: $(git rev-parse HEAD)
Configuration files:
$(ls -la config/*.json)
EOF

echo "Configuration backed up to $BACKUP_DIR"

# Keep only last 10 backups
ls -dt config_backups/*/ | tail -n +11 | xargs rm -rf

echo "Old backups cleaned up"
```

## Troubleshooting Best Practices

### 1. Diagnostic Tools

Create diagnostic tools for common issues:

```bash
#!/bin/bash
# diagnose_config.sh

echo "AutoGen Framework Configuration Diagnostics"
echo "==========================================="

# Check environment
echo "Environment Variables:"
env | grep -E "^(LLM_|CONFIG_|WORKSPACE_|LOG_)" | sort

echo -e "\nConfiguration Files:"
find config/ -name "*.json" -exec echo "  {}" \; -exec python -m json.tool {} > /dev/null 2>&1 && echo "    ✅ Valid JSON" || echo "    ❌ Invalid JSON" \;

echo -e "\nModel Detection Test:"
python3 << 'EOF'
import sys
sys.path.append('.')
from autogen_framework.config_manager import ConfigManager
import os

config_manager = ConfigManager()
model_name = os.getenv('LLM_MODEL', 'unknown')

try:
    model_info = config_manager.get_model_info(model_name)
    print(f"  Model: {model_name}")
    print(f"  Family: {model_info['family']}")
    print(f"  Token Limit: {model_info['token_limit']}")
    print(f"  ✅ Model detection successful")
except Exception as e:
    print(f"  ❌ Model detection failed: {e}")
EOF

echo -e "\nFramework Status:"
autogen-framework --status 2>&1 | head -10
```

### 2. Error Recovery

Implement error recovery procedures:

```bash
#!/bin/bash
# recover_config.sh

echo "Configuration Recovery Tool"
echo "=========================="

# Check if backup exists
if [ -d "config_backups" ]; then
    LATEST_BACKUP=$(ls -dt config_backups/*/ | head -1)
    echo "Latest backup found: $LATEST_BACKUP"
    
    read -p "Restore from backup? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp "$LATEST_BACKUP"/*.json config/
        echo "Configuration restored from backup"
    fi
else
    echo "No backups found, creating minimal configuration..."
    
    # Create minimal framework config
    cat > config/framework.json << 'EOF'
{
  "llm": {
    "temperature": 0.7,
    "max_output_tokens": 4096,
    "timeout_seconds": 60
  },
  "shell": {
    "timeout_seconds": 30,
    "max_retries": 2
  }
}
EOF
    
    echo "Minimal configuration created"
fi

# Test configuration
echo "Testing configuration..."
if autogen-framework --status > /dev/null 2>&1; then
    echo "✅ Configuration recovery successful"
else
    echo "❌ Configuration recovery failed"
    exit 1
fi
```

Following these best practices will help you maintain a robust, secure, and performant configuration system that scales with your team and use cases. Regular review and updates of these practices ensure continued effectiveness as your usage patterns evolve.