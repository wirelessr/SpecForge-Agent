# Configuration Migration Guide

## Overview

This guide provides step-by-step instructions for migrating from the old environment variable-based configuration system to the new three-tier dynamic configuration system. The migration is designed to be backward compatible, so existing deployments will continue working during the transition.

## Migration Benefits

The new configuration system provides:

- **Clear separation of concerns**: Connection, behavior, and execution settings are properly separated
- **Better maintainability**: Configuration is organized in logical files rather than scattered environment variables
- **Enhanced flexibility**: Support for custom models and dynamic model detection
- **Improved security**: Sensitive information stays in environment variables while behavior settings move to version-controlled config files
- **Better team collaboration**: Shared configuration files with individual environment overrides

## Pre-Migration Assessment

Before starting the migration, assess your current configuration:

### 1. Inventory Current Configuration

List all environment variables currently in use:

```bash
# Check your current .env file
cat .env | grep -E "^(LLM_|SHELL_|WORKSPACE_|LOG_|CONFIG_)"

# Check system environment variables
env | grep -E "^(LLM_|SHELL_|WORKSPACE_|LOG_|CONFIG_)"
```

### 2. Categorize Settings

Identify which settings belong in each configuration tier:

**Keep in Environment Variables** (Connection & Environment):

- `LLM_BASE_URL` - LLM service endpoint
- `LLM_MODEL` - Model name to use
- `LLM_API_KEY` - Authentication key
- `LOG_LEVEL` - Logging verbosity
- `WORKSPACE_PATH` - Working directory
- `CONFIG_DIR` - Configuration directory (optional)
- `MODELS_CONFIG_FILE` - Models config file path (optional)
- `FRAMEWORK_CONFIG_FILE` - Framework config file path (optional)

**Move to Config Files** (Behavior Settings):

- `LLM_TEMPERATURE` → `config/framework.json`
- `LLM_MAX_OUTPUT_TOKENS` → `config/framework.json`
- `LLM_TIMEOUT_SECONDS` → `config/framework.json`
- `SHELL_TIMEOUT_SECONDS` → `config/framework.json`
- `SHELL_MAX_RETRIES` → `config/framework.json`

**Use Command Arguments** (Execution Control):

- `AUTO_APPROVE` → `--auto-approve`
- `VERBOSE` → `--verbose`
- `RESET_SESSION` → `--reset-session`

## Step-by-Step Migration

### Step 1: Backup Current Configuration

Create backups of your current configuration:

```bash
# Backup environment files
cp .env .env.backup
cp .env.integration .env.integration.backup 2>/dev/null || true
cp .env.test .env.test.backup 2>/dev/null || true

# Create migration log
echo "Migration started: $(date)" > migration.log
echo "Original .env contents:" >> migration.log
cat .env >> migration.log
```

### Step 2: Create Configuration Directory

Set up the configuration directory structure:

```bash
# Create config directory if it doesn't exist
mkdir -p config/examples

# Verify the framework's built-in config files exist
ls -la config/
```

You should see:

```
config/
├── examples/
├── framework.json
├── models.json
├── MIGRATION.md
└── README.md
```

### Step 3: Extract Behavior Settings

Create your framework configuration file by extracting behavior settings from environment variables:

```bash
# Create framework.json from current environment variables
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
EOF
```

**Customize the values** based on your current environment variables:

```bash
# If you have custom values, update the config file
# For example, if LLM_TEMPERATURE=0.5 in your .env:
python3 << 'EOF'
import json

# Read current config
with open('config/framework.json', 'r') as f:
    config = json.load(f)

# Update with your custom values (example)
# config['llm']['temperature'] = 0.5  # Replace with your actual values
# config['llm']['max_output_tokens'] = 8192  # Replace with your actual values

# Write updated config
with open('config/framework.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Framework configuration updated")
EOF
```

### Step 4: Update Environment Variables

Clean up your `.env` file to remove behavior settings that are now in config files:

```bash
# Create new .env file with only connection and environment settings
cat > .env.new << 'EOF'
# Connection settings - KEEP THESE
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-your-api-key

# Environment settings - KEEP THESE
WORKSPACE_PATH=artifacts/
LOG_LEVEL=INFO

# Optional: Configuration file overrides
# CONFIG_DIR=./config
# MODELS_CONFIG_FILE=./config/models.json
# FRAMEWORK_CONFIG_FILE=./config/framework.json
EOF

# Replace with your actual values
sed -i 's/sk-your-api-key/your-actual-api-key/' .env.new
sed -i 's|http://localhost:8888/openai/v1|your-actual-base-url|' .env.new
sed -i 's/models\/gemini-2.0-flash/your-actual-model/' .env.new

# Backup old .env and use new one
mv .env .env.old
mv .env.new .env

echo "Environment variables updated" >> migration.log
echo "New .env contents:" >> migration.log
cat .env >> migration.log
```

### Step 5: Verify Model Configuration

Check if your model is supported by the built-in model configurations:

```bash
# Test model detection
python3 << 'EOF'
import json
import re

# Read your model name from .env
with open('.env', 'r') as f:
    env_content = f.read()

model_name = None
for line in env_content.split('\n'):
    if line.startswith('LLM_MODEL='):
        model_name = line.split('=', 1)[1].strip()
        break

if not model_name:
    print("No LLM_MODEL found in .env")
    exit(1)

print(f"Checking model: {model_name}")

# Read models.json
with open('config/models.json', 'r') as f:
    models_config = json.load(f)

# Check exact match
if model_name in models_config['models']:
    print(f"✓ Model '{model_name}' found in exact matches")
    config = models_config['models'][model_name]
    print(f"  Family: {config['family']}")
    print(f"  Token Limit: {config['token_limit']}")
    print(f"  Capabilities: {config['capabilities']}")
else:
    # Check pattern matches
    matched = False
    for pattern_config in models_config['patterns']:
        pattern = pattern_config['pattern']
        if re.match(pattern, model_name, re.IGNORECASE):
            print(f"✓ Model '{model_name}' matches pattern: {pattern}")
            print(f"  Family: {pattern_config['family']}")
            print(f"  Token Limit: {pattern_config['token_limit']}")
            print(f"  Capabilities: {pattern_config['capabilities']}")
            matched = True
            break

    if not matched:
        print(f"⚠ Model '{model_name}' not recognized")
        print("  Will use default configuration:")
        defaults = models_config['defaults']
        print(f"  Family: {defaults['family']}")
        print(f"  Token Limit: {defaults['token_limit']}")
        print(f"  Capabilities: {defaults['capabilities']}")
        print("\nConsider adding your model to config/models.json")
EOF
```

### Step 6: Update Command Usage

Update how you invoke the framework to use command arguments instead of environment variables for execution control:

**Before Migration**:

```bash
# Old way - using environment variables
export AUTO_APPROVE=true
export VERBOSE=true
autogen-framework --request "Create a hello world script"
```

**After Migration**:

```bash
# New way - using command arguments
autogen-framework --request "Create a hello world script" --auto-approve --verbose
```

Update any scripts or automation:

```bash
# Find scripts that might need updating
grep -r "AUTO_APPROVE\|VERBOSE" . --include="*.sh" --include="*.py" --include="*.md"

# Example script update
cat > run_framework.sh << 'EOF'
#!/bin/bash
# Updated script using new command arguments

# Load environment variables
source .env

# Run framework with command arguments instead of env vars
autogen-framework \
  --request "$1" \
  --auto-approve \
  --verbose \
  --config-dir ./config
EOF

chmod +x run_framework.sh
```

### Step 7: Test the Migration

Perform comprehensive testing to ensure the migration was successful:

#### Basic Functionality Test

```bash
# Test framework status
autogen-framework --status

# Test with verbose output to see configuration loading
autogen-framework --verbose --status
```

#### Configuration Loading Test

```bash
# Test that config files are being loaded
autogen-framework --verbose --status 2>&1 | grep -E "(Loaded.*configuration|Using.*from)"
```

#### Model Detection Test

```bash
# Test model detection with your specific model
python3 << 'EOF'
import sys
sys.path.append('.')
from autogen_framework.config_manager import ConfigManager

config_manager = ConfigManager()
model_name = "your-model-name"  # Replace with your actual model

try:
    model_info = config_manager.get_model_info(model_name)
    print(f"✓ Model detection successful for '{model_name}':")
    print(f"  Family: {model_info['family']}")
    print(f"  Token Limit: {model_info['token_limit']}")
    print(f"  Capabilities: {model_info['capabilities']}")
except Exception as e:
    print(f"✗ Model detection failed: {e}")
EOF
```

#### End-to-End Test

```bash
# Test a simple workflow
autogen-framework --request "Create a simple hello world Python script" --verbose
```

### Step 8: Environment-Specific Configuration

Set up configuration for different environments:

#### Development Environment

```bash
# Create development-specific framework config
cp config/framework.json config/framework.development.json

# Customize for development (higher verbosity, shorter timeouts)
python3 << 'EOF'
import json

with open('config/framework.development.json', 'r') as f:
    config = json.load(f)

# Development-friendly settings
config['llm']['temperature'] = 0.5  # More deterministic for testing
config['shell']['timeout_seconds'] = 15  # Shorter timeouts for faster feedback
config['workflow']['verbose'] = True  # More verbose by default
config['memory']['max_history_entries'] = 50  # Smaller memory for faster startup

with open('config/framework.development.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Development configuration created")
EOF

# Use development config
export FRAMEWORK_CONFIG_FILE=config/framework.development.json
```

#### Production Environment

```bash
# Create production-specific framework config
cp config/framework.json config/framework.production.json

# Customize for production (stability, performance)
python3 << 'EOF'
import json

with open('config/framework.production.json', 'r') as f:
    config = json.load(f)

# Production-optimized settings
config['llm']['temperature'] = 0.3  # More consistent results
config['llm']['timeout_seconds'] = 120  # Longer timeouts for reliability
config['shell']['timeout_seconds'] = 60  # Longer timeouts for complex operations
config['shell']['max_retries'] = 3  # More retries for reliability
config['workflow']['verbose'] = False  # Cleaner logs
config['memory']['max_history_entries'] = 200  # Larger memory for better context

with open('config/framework.production.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Production configuration created")
EOF
```

#### Testing Environment

```bash
# Create testing-specific framework config
cp config/framework.json config/framework.testing.json

# Customize for testing (speed, determinism)
python3 << 'EOF'
import json

with open('config/framework.testing.json', 'r') as f:
    config = json.load(f)

# Testing-optimized settings
config['llm']['temperature'] = 0.1  # Very deterministic
config['llm']['max_output_tokens'] = 1024  # Smaller outputs for faster tests
config['llm']['timeout_seconds'] = 30  # Shorter timeouts for faster tests
config['shell']['timeout_seconds'] = 10  # Very short timeouts
config['workflow']['auto_approve'] = True  # No manual intervention
config['workflow']['reset_session_on_start'] = True  # Clean state for each test
config['memory']['max_history_entries'] = 10  # Minimal memory

with open('config/framework.testing.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Testing configuration created")
EOF
```

## Migration Examples

### Example 1: Simple Development Setup

**Before Migration (.env)**:

```bash
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-123456
LLM_TEMPERATURE=0.5
LLM_MAX_OUTPUT_TOKENS=2048
WORKSPACE_PATH=./workspace
LOG_LEVEL=INFO
AUTO_APPROVE=false
VERBOSE=true
```

**After Migration (.env)**:

```bash
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-123456
WORKSPACE_PATH=./workspace
LOG_LEVEL=INFO
```

**After Migration (config/framework.json)**:

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
  },
  "workflow": {
    "auto_approve": false,
    "verbose": true
  }
}
```

**Command Usage**:

```bash
# Before
export AUTO_APPROVE=false VERBOSE=true
autogen-framework --request "Task"

# After
autogen-framework --request "Task" --verbose
```

### Example 2: Custom Model Setup

**Before Migration (.env)**:

```bash
LLM_BASE_URL=http://custom-provider:8080/v1
LLM_MODEL=custom-model:7b
LLM_API_KEY=custom-key
LLM_TEMPERATURE=0.8
LLM_MAX_OUTPUT_TOKENS=8192
```

**After Migration (.env)**:

```bash
LLM_BASE_URL=http://custom-provider:8080/v1
LLM_MODEL=custom-model:7b
LLM_API_KEY=custom-key
```

**After Migration (config/models.json)** - Add custom model:

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

**After Migration (config/framework.json)**:

```json
{
  "llm": {
    "temperature": 0.8,
    "max_output_tokens": 8192,
    "timeout_seconds": 60
  }
}
```

### Example 3: Production Deployment

**Before Migration (.env)**:

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
AUTO_APPROVE=true
VERBOSE=false
```

**After Migration (.env)**:

```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4-turbo
LLM_API_KEY=sk-real-api-key
WORKSPACE_PATH=/var/lib/autogen
LOG_LEVEL=WARNING
```

**After Migration (config/framework.json)**:

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
    "auto_approve": true,
    "verbose": false
  }
}
```

**Deployment Script**:

```bash
#!/bin/bash
# Production deployment script

# Load environment
source .env

# Run with production config
autogen-framework \
  --framework-config config/framework.production.json \
  --request "$1" \
  --auto-approve
```

## Troubleshooting Migration Issues

### Issue 1: Configuration Not Found

**Symptom**:

```
WARNING: Config file ./config/framework.json not found, using built-in defaults
```

**Diagnosis**:

```bash
# Check if config files exist
ls -la config/

# Check CONFIG_DIR environment variable
echo $CONFIG_DIR

# Check current working directory
pwd
```

**Solutions**:

```bash
# Option 1: Create missing config file
cp config/examples/development.json config/framework.json

# Option 2: Set CONFIG_DIR environment variable
export CONFIG_DIR=/path/to/your/config

# Option 3: Use command line override
autogen-framework --config-dir /path/to/config --request "Task"
```

### Issue 2: Model Not Recognized

**Symptom**:

```
WARNING: Model 'custom-model' not recognized by any pattern. Using default configuration
```

**Diagnosis**:

```bash
# Check your model name
grep LLM_MODEL .env

# Check available model configurations
python3 -c "
import json
with open('config/models.json') as f:
    config = json.load(f)
print('Exact matches:', list(config['models'].keys()))
print('Patterns:', [p['pattern'] for p in config['patterns']])
"
```

**Solutions**:

```bash
# Option 1: Add exact model match
python3 << 'EOF'
import json

with open('config/models.json', 'r') as f:
    config = json.load(f)

# Add your custom model
config['models']['your-custom-model'] = {
    "family": "GPT_4",
    "token_limit": 32000,
    "capabilities": {
        "vision": false,
        "function_calling": true,
        "streaming": true
    }
}

with open('config/models.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Custom model added")
EOF

# Option 2: Add pattern match
python3 << 'EOF'
import json

with open('config/models.json', 'r') as f:
    config = json.load(f)

# Add pattern for your model family
config['patterns'].append({
    "pattern": "^your-provider:",
    "family": "GPT_4",
    "token_limit": 16000,
    "capabilities": {
        "vision": false,
        "function_calling": true,
        "streaming": false
    }
})

with open('config/models.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Pattern added")
EOF
```

### Issue 3: Settings Not Applied

**Symptom**: Configuration from config files is not being used

**Diagnosis**:

```bash
# Check for environment variable overrides
env | grep -E "^LLM_(TEMPERATURE|MAX_OUTPUT_TOKENS|TIMEOUT)"

# Check configuration precedence
autogen-framework --verbose --status 2>&1 | grep -E "(Using.*from|source:)"
```

**Solutions**:

```bash
# Remove environment variable overrides
unset LLM_TEMPERATURE
unset LLM_MAX_OUTPUT_TOKENS
unset LLM_TIMEOUT_SECONDS

# Or update .env file to remove these lines
sed -i '/^LLM_TEMPERATURE=/d' .env
sed -i '/^LLM_MAX_OUTPUT_TOKENS=/d' .env
sed -i '/^LLM_TIMEOUT_SECONDS=/d' .env
```

### Issue 4: Invalid Configuration Format

**Symptom**:

```
ERROR: Invalid config format in config/framework.json: Expecting ',' delimiter
```

**Diagnosis**:

```bash
# Validate JSON syntax
python -m json.tool config/framework.json
```

**Solutions**:

```bash
# Fix JSON syntax errors
# Common issues:
# - Missing commas between properties
# - Trailing commas
# - Unquoted property names
# - Single quotes instead of double quotes

# Use a JSON formatter/validator
python3 << 'EOF'
import json

try:
    with open('config/framework.json', 'r') as f:
        config = json.load(f)

    # Rewrite with proper formatting
    with open('config/framework.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("JSON formatting fixed")
except json.JSONDecodeError as e:
    print(f"JSON error: {e}")
    print("Please fix the syntax error manually")
EOF
```

### Issue 5: Permission Errors

**Symptom**:

```
ERROR: Permission denied: config/framework.json
```

**Solutions**:

```bash
# Fix file permissions
chmod 644 config/*.json

# Fix directory permissions
chmod 755 config/

# Check ownership
ls -la config/
```

## Rollback Procedures

If you encounter issues during migration, you can quickly rollback:

### Quick Rollback

```bash
# Restore original .env file
cp .env.backup .env

# Remove config files to force environment variable usage
mv config/framework.json config/framework.json.disabled
mv config/models.json config/models.json.disabled

# Test functionality
autogen-framework --status
```

### Partial Rollback

```bash
# Keep model configuration but rollback framework config
mv config/framework.json config/framework.json.disabled

# Framework will use environment variables for behavior settings
autogen-framework --status
```

### Gradual Migration

```bash
# Migrate one setting at a time
# Start with just temperature
python3 << 'EOF'
import json

config = {
    "llm": {
        "temperature": 0.7
    }
}

with open('config/framework.json', 'w') as f:
    json.dump(config, f, indent=2)
EOF

# Remove only LLM_TEMPERATURE from .env
sed -i '/^LLM_TEMPERATURE=/d' .env

# Test this single change
autogen-framework --verbose --status
```

## Post-Migration Validation

After completing the migration, perform these validation steps:

### 1. Configuration Loading Validation

```bash
# Verify all config files load correctly
autogen-framework --verbose --status 2>&1 | grep -E "(Loaded.*configuration|ERROR|WARNING)"
```

### 2. Model Detection Validation

```bash
# Test model detection for your specific model
python3 << 'EOF'
import sys
sys.path.append('.')
from autogen_framework.config_manager import ConfigManager

config_manager = ConfigManager()

# Get your model from .env
import os
from dotenv import load_dotenv
load_dotenv()

model_name = os.getenv('LLM_MODEL')
if model_name:
    model_info = config_manager.get_model_info(model_name)
    print(f"Model: {model_name}")
    print(f"Family: {model_info['family']}")
    print(f"Token Limit: {model_info['token_limit']}")
    print(f"Capabilities: {model_info['capabilities']}")
else:
    print("No LLM_MODEL found in environment")
EOF
```

### 3. Functionality Validation

```bash
# Test basic framework functionality
autogen-framework --request "Create a simple test file" --verbose

# Verify the request completes successfully
echo "Migration validation complete: $(date)" >> migration.log
```

### 4. Performance Validation

```bash
# Compare performance before and after migration
time autogen-framework --status

# Check memory usage
ps aux | grep autogen-framework
```

## Migration Checklist

Use this checklist to ensure complete migration:

- [ ] **Pre-migration assessment completed**

  - [ ] Current configuration inventoried
  - [ ] Settings categorized by configuration tier
  - [ ] Backup files created

- [ ] **Configuration files created**

  - [ ] `config/framework.json` created with custom values
  - [ ] Custom models added to `config/models.json` (if needed)
  - [ ] Environment-specific configs created (if needed)

- [ ] **Environment variables updated**

  - [ ] Behavior settings removed from `.env`
  - [ ] Connection settings retained in `.env`
  - [ ] Deprecated variables identified and removed

- [ ] **Command usage updated**

  - [ ] Scripts updated to use command arguments
  - [ ] Automation updated for new argument format
  - [ ] Documentation updated with new usage patterns

- [ ] **Testing completed**

  - [ ] Basic functionality test passed
  - [ ] Configuration loading test passed
  - [ ] Model detection test passed
  - [ ] End-to-end workflow test passed

- [ ] **Validation completed**

  - [ ] Configuration loading validated
  - [ ] Model detection validated
  - [ ] Functionality validated
  - [ ] Performance validated

- [ ] **Documentation updated**

  - [ ] Team documentation updated
  - [ ] Deployment procedures updated
  - [ ] Troubleshooting guides updated

- [ ] **Cleanup completed**
  - [ ] Backup files organized
  - [ ] Migration logs saved
  - [ ] Temporary files removed

The migration to the new configuration system provides significant benefits in terms of maintainability, flexibility, and team collaboration. Take your time with each step and validate thoroughly to ensure a smooth transition.
