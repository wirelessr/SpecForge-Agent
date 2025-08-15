# Dynamic Model Configuration System Overview

## Introduction

The AutoGen Multi-Agent Framework now features a sophisticated dynamic model configuration system that provides automatic model detection, flexible configuration management, and clear separation of concerns. This document provides an overview of the complete system and guides you to the appropriate documentation for your needs.

## System Architecture

The configuration system is built on a **three-tier architecture** with clear separation of responsibilities:

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Command Args    │    │ Environment Vars    │    │ Config Files    │    │ Built-in        │
│ (Execution      │────│ (Connection &       │────│ (Behavior &     │────│ Defaults        │
│  Control)       │    │  Environment)       │    │  Model Props)   │    │ (Fallback)      │
└─────────────────┘    └─────────────────────┘    └─────────────────┘    └─────────────────┘
     Highest                                                                    Lowest
    Precedence                                                                 Precedence
```

### Key Features

- **🤖 Automatic Model Detection**: Recognizes model families and properties from model names
- **⚙️ Dynamic Configuration**: Supports custom models and configurations without code changes
- **🔒 Secure by Design**: Keeps credentials in environment variables, behavior in config files
- **📊 Comprehensive Logging**: Detailed logging for troubleshooting and transparency
- **🔄 Backward Compatible**: Existing deployments continue working during migration
- **🎯 Clear Separation**: Each configuration tier has well-defined responsibilities

## Documentation Guide

### 🚀 Getting Started

**New to the configuration system?** Start here:

1. **[Configuration Quick Reference](configuration-quick-reference.md)** - Essential commands and patterns
2. **[Configuration Guide](configuration-guide.md)** - Complete system overview and usage

### 🔄 Migration

**Migrating from the old system?** Follow this path:

1. **[Migration Guide](migration-guide.md)** - Step-by-step migration instructions
2. **[Configuration Best Practices](configuration-best-practices.md)** - Security and maintenance practices

### 📚 Reference Documentation

**Need detailed information?** Explore these resources:

- **[Configuration Guide](configuration-guide.md)** - Complete configuration system documentation
- **[Configuration Best Practices](configuration-best-practices.md)** - Security, performance, and team practices
- **[Config Directory](../config/README.md)** - Configuration files and examples
- **[Framework README](../autogen_framework/README.md)** - Updated framework documentation

## Supported Models

The system includes built-in support for major LLM providers:

| Provider | Models | Token Limits | Capabilities |
|----------|--------|--------------|--------------|
| **Google Gemini** | 2.0 Flash, 1.5 Pro, 1.5 Flash | 1M - 2M tokens | Vision, Function calling, Streaming |
| **OpenAI GPT** | GPT-4, GPT-4 Turbo | 8K - 128K tokens | Function calling, Vision (turbo), Streaming |
| **Anthropic Claude** | Claude 3 Opus | 200K tokens | Vision, Function calling, Streaming |
| **Custom Providers** | Any model | Configurable | Configurable |

## Configuration Examples

### Quick Setup
```bash
# 1. Create environment file
cat > .env << EOF
LLM_BASE_URL=http://localhost:8888/openai/v1
LLM_MODEL=models/gemini-2.0-flash
LLM_API_KEY=sk-your-key
EOF

# 2. Use default configurations (included)
autogen-framework --status

# 3. Run with custom settings
autogen-framework --verbose --request "Create a hello world script"
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

### Environment-Specific Configuration
```bash
# Development
autogen-framework --framework-config config/examples/development.json

# Production  
autogen-framework --framework-config config/examples/production.json --auto-approve

# Testing
autogen-framework --framework-config config/examples/testing.json
```

## Key Benefits

### For Developers
- **Automatic model detection** - No manual configuration for supported models
- **Easy model switching** - Change `LLM_MODEL` environment variable
- **Custom model support** - Add new models via configuration files
- **Clear error messages** - Detailed logging and troubleshooting guidance

### For System Administrators
- **Centralized configuration** - All settings in logical, version-controlled files
- **Security best practices** - Credentials separate from behavior settings
- **Environment management** - Different configs for dev/staging/production
- **Migration support** - Smooth transition from old configuration system

### For Teams
- **Shared configurations** - Team settings in version-controlled config files
- **Individual environments** - Personal `.env` files for local development
- **Clear documentation** - Comprehensive guides and examples
- **Consistent behavior** - Same configuration system across all environments

## Migration Status

The new configuration system is **fully backward compatible**. Existing deployments will continue working without changes, while new features are available immediately.

### Migration Path
1. **Phase 1**: Continue using existing environment variables (works now)
2. **Phase 2**: Gradually move behavior settings to config files (optional)
3. **Phase 3**: Adopt new command argument patterns (recommended)

See the [Migration Guide](migration-guide.md) for detailed instructions.

## Troubleshooting

### Common Issues

| Issue | Quick Fix | Documentation |
|-------|-----------|---------------|
| Model not recognized | Add to `config/models.json` | [Configuration Guide](configuration-guide.md#model-configuration-system) |
| Config not loading | Check file paths and JSON syntax | [Quick Reference](configuration-quick-reference.md#troubleshooting) |
| Settings ignored | Check environment variable overrides | [Best Practices](configuration-best-practices.md#troubleshooting-best-practices) |
| Migration questions | Follow step-by-step guide | [Migration Guide](migration-guide.md) |

### Diagnostic Commands
```bash
# Check system status
autogen-framework --verbose --status

# Validate configuration files
python -m json.tool config/framework.json
python -m json.tool config/models.json

# Test model detection
python3 -c "
from autogen_framework.config_manager import ConfigManager
cm = ConfigManager()
print(cm.get_model_info('your-model-name'))
"
```

## Getting Help

- **📖 Documentation**: Start with the [Configuration Quick Reference](configuration-quick-reference.md)
- **🔧 Issues**: Check the [troubleshooting sections](configuration-best-practices.md#troubleshooting-best-practices) in each guide
- **💬 Questions**: Review the comprehensive [Configuration Guide](configuration-guide.md)
- **🚀 Migration**: Follow the detailed [Migration Guide](migration-guide.md)

## What's Next

The dynamic model configuration system provides a solid foundation for:

- **Enhanced model support** - Easy addition of new LLM providers
- **Advanced capabilities** - Model-specific feature detection and optimization
- **Team collaboration** - Shared configurations with individual overrides
- **Operational excellence** - Monitoring, validation, and automated management

Explore the documentation to get the most out of your AutoGen Multi-Agent Framework configuration!