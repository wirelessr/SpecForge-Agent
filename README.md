# AutoGen Multi-Agent Framework

A sophisticated multi-agent collaboration framework built on AutoGen that automates the complete software development lifecycle from requirements analysis to code implementation.

## 🌟 Overview

This framework implements an intelligent multi-agent system that follows a structured workflow: **Requirements → Design → Tasks → Implementation**. Each phase is handled by specialized agents with built-in approval checkpoints and revision capabilities.

### Key Features

- **🤖 Intelligent Agent Collaboration**: Four specialized agents (Plan, Design, Tasks, Implement) work together
- **🔄 Complete Development Workflow**: End-to-end automation with human oversight
- **🛠️ Advanced Architecture**: Modular design with session management and context retention
- **📊 Quality-First Approach**: Built-in quality metrics and testing standards
- **🔧 Autonomous Execution**: Enhanced implementation with intelligent error recovery

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- uv (Python package manager)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd autogen-framework

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your LLM configuration
```

### Basic Usage

```bash
# Submit a development request
autogen-framework --request "Create a REST API for user management"

# Review and approve each phase
autogen-framework --status
autogen-framework --approve requirements
autogen-framework --approve design
autogen-framework --approve tasks

# Execute implementation
autogen-framework --execute-tasks
```

## 🏗️ Architecture

### Core Components

- **MainController**: Thin I/O layer for user interactions
- **WorkflowManager**: Central orchestrator managing workflow state
- **SessionManager**: Handles session persistence across commands
- **AgentManager**: Factory for creating and managing agents

### Specialized Agents

- **PlanAgent**: Analyzes requirements and generates requirements.md
- **DesignAgent**: Creates technical design from requirements
- **TasksAgent**: Decomposes design into actionable tasks
- **ImplementAgent**: Executes tasks with intelligent error recovery

### Support Systems

- **MemoryManager**: Context retention and learning capabilities
- **ShellExecutor**: Command execution with error handling
- **TokenManager**: Context compression and optimization
- **ContextManager**: Project context integration

## 📖 Documentation

- **[Framework Documentation](autogen_framework/README.md)** - Detailed usage guide
- **[Configuration Guide](docs/configuration-guide.md)** - Setup and configuration
- **[Testing Standards](tests/TESTING_STANDARDS_SUMMARY.md)** - Testing guidelines
- **[Development Guide](.kiro/steering/workflow-execution-guide.md)** - Development workflow

## 🧪 Testing

The framework follows a comprehensive testing strategy:

```bash
# Fast unit tests
pytest tests/unit/ -x --tb=short -q

# Integration tests
pytest tests/integration/ -x --tb=short -q

# End-to-end workflow tests
./tests/e2e/workflow_test.sh
```

### Test Structure

- **Unit Tests**: Fast, isolated tests with mocked dependencies
- **Integration Tests**: Real service interactions
- **E2E Tests**: Complete workflow validation
- **Quality Tests**: Metrics and regression testing

## 🔧 Configuration

The framework uses a three-tier configuration system:

1. **Environment Variables** (.env file)
2. **Configuration Files** (config/ directory)
3. **Command Arguments** (runtime overrides)

See the [Configuration Guide](docs/configuration-guide.md) for details.

## 🎯 Development Principles

### User Requirements First

- Trust user requirements and configuration
- No unauthorized feature additions
- Transparent error reporting
- Real configuration testing

### Quality-First Development

- Comprehensive testing at all levels
- Quality metrics and baselines
- Continuous improvement tracking
- Error recovery and learning

### Autonomous Execution

- Intelligent task decomposition
- Multi-strategy error recovery
- Context-aware execution
- Learning from successful patterns

## 🤝 Contributing

We welcome contributions! Please see our development guidelines:

1. Follow the [User Requirements First](.kiro/steering/user-requirements-first.md) principle
2. Use the [Testing Standards](tests/TESTING_STANDARDS_SUMMARY.md)
3. Follow the [Development Workflow](.kiro/steering/workflow-execution-guide.md)

### Development Setup

```bash
# Install development dependencies
uv sync --group dev

# Run tests
pytest tests/ -v

# Code formatting
black autogen_framework/
```

## 📊 Project Status

- **Version**: 0.1.0
- **Python**: 3.11+
- **Status**: Active Development
- **License**: [Add License]

## 🙏 Acknowledgments

Built on the excellent AutoGen framework. Thanks to all contributors and the open-source community.

---

**Maintained by**: Kiro AI Assistant  
**Last Updated**: 2025-08-16
