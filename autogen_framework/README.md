# AutoGen Multi-Agent Framework

A multi-agent software development framework based on AutoGen that supports end-to-end automated development processes from requirements analysis to code implementation.

## 🌟 Features

### 🤖 Intelligent Multi-Agent Collaboration
- **PlanAgent**: Requirements analysis and document generation
- **DesignAgent**: Technical design and architecture planning
- **TasksAgent**: Task decomposition and planning
- **ImplementAgent**: Pure task execution and code implementation

### 🔄 Complete Development Workflow
- A complete automated process of **Requirements → Design → Tasks → Execution**
- Phased review and approval mechanism
- Revision and feedback loop support

### 🛠️ Advanced Features
- **Modular Architecture**: Dedicated components for session management, workflow orchestration, and agent coordination
- **Patch-First Strategy**: Intelligent file modification, minimizing changes
- **Memory System**: Context retention and learning capabilities
- **Shell Integration**: Real system operations and command execution
- **Session Management**: Dedicated SessionManager for state persistence across command invocations

## 🚀 Quick Start

### Installation Requirements

```bash
# Python 3.11+
# uv (Python package manager)

# Install dependencies
uv sync
```

### Basic Configuration

The framework uses a three-tier configuration system for maximum flexibility:

1.  **Environment Variables** (Connection & Environment)
    ```bash
    # Create .env file
    cat > .env << EOF
    LLM_BASE_URL=http://your-llm-endpoint/v1
    LLM_MODEL=your-model-name
    LLM_API_KEY=your-api-key
    WORKSPACE_PATH=./workspace
    LOG_LEVEL=INFO
    EOF
    ```

2.  **Configuration Files** (Behavior Settings)
    ```bash
    # Framework behavior is configured in config/framework.json
    # Model properties are configured in config/models.json
    # These files are included with sensible defaults
    ```

3.  **Command Arguments** (Execution Control)
    ```bash
    # Use command arguments for execution-specific overrides
    autogen-framework --verbose --auto-approve --request "Task"
    ```

For detailed configuration information, see:
- **[Configuration Guide](../docs/configuration-guide.md)** - Complete configuration system guide
- **[Migration Guide](../docs/migration-guide.md)** - Migrating from old configuration
- **[Configuration Best Practices](../docs/configuration-best-practices.md)** - Best practices and security

4.  **Create a Work Directory**
    ```bash
    mkdir my-project
    cd my-project
    ```

### Basic Usage

#### 1. Submit a Development Request
```bash
autogen-framework --workspace . --request "Create a REST API for user management"
```

#### 2. Review and Approve Each Phase
```bash
# Check the current status
autogen-framework --workspace . --status

# Approve requirements
autogen-framework --workspace . --approve requirements

# Approve design
autogen-framework --workspace . --approve design

# Approve task list
autogen-framework --workspace . --approve tasks
```

#### 3. Execute Tasks
```bash
# Execute all tasks
autogen-framework --workspace . --execute-tasks

# Or execute a specific task
autogen-framework --workspace . --execute-task "Create the Data Access Layer"
```

#### 4. Revise and Provide Feedback
```bash
# Revise a specific phase
autogen-framework --workspace . --revise "requirements:Add more security details"
autogen-framework --workspace . --revise "design:Include database schema diagrams"
```

## 📋 Command-Line Options

### Basic Operations
```bash
# Submit a request
--request "Describe your requirements"

# Check status
--status

# Continue an existing workflow
--continue-workflow

# Reset the session
--reset-session
```

### Phase Management
```bash
# Approve a phase
--approve [requirements|design|tasks]

# Reject a phase
--reject [requirements|design|tasks]

# Revise a phase
--revise "phase:feedback"
```

### Task Execution
```bash
# Execute all tasks
--execute-tasks

# Execute a specific task
--execute-task "Task title"
```

### Configuration Options

#### Execution Control (Command Arguments)
```bash
# Execution behavior
--verbose                    # Enable detailed logging
--auto-approve              # Skip approval prompts
--reset-session             # Reset session state

# Configuration overrides
--config-dir /path/to/config           # Override config directory
--models-config /path/to/models.json   # Override models config
--framework-config /path/to/framework.json # Override framework config
```

#### Connection & Environment (Environment Variables)
```bash
# Set in .env file or environment
LLM_BASE_URL=http://your-endpoint/v1    # LLM service endpoint
LLM_MODEL=your-model-name               # Model to use
LLM_API_KEY=your-api-key               # Authentication key
WORKSPACE_PATH=/path/to/workspace       # Work directory
LOG_LEVEL=INFO                         # Logging level
```

#### Behavior Settings (Config Files)
```bash
# Configured in config/framework.json
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
```

**Note**: The framework automatically detects model families and token limits. See the [Configuration Guide](../docs/configuration-guide.md) for supported models and custom configuration.

## 🏗️ Architecture Overview

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  MainController │────│ WorkflowManager  │────│ SessionManager  │
│  (Thin I/O)     │    │ (Orchestration)  │    │ (Persistence)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  AgentManager    │
                       │  (Agent Factory) │
                       └──────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  DependencyContainer    │
                    │  (Dependency Injection) │
                    └─────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │  PlanAgent   │ │  DesignAgent │ │  TasksAgent  │
        │ (Requirements)│ │  (Design)    │ │ (Task Gen)   │
        └──────────────┘ └──────────────┘ └──────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │ ImplementAgent│
                                          │ (Execution)  │
                                          └──────────────┘
```

### Dependency Injection System

The framework uses a clean dependency injection system that simplifies agent initialization:

```python
from autogen_framework.dependency_container import DependencyContainer
from autogen_framework.agents.plan_agent import PlanAgent

# Simple container-based initialization
container = DependencyContainer.create_production(work_dir, llm_config)
agent = PlanAgent(
    name="PlanAgent",
    llm_config=llm_config,
    system_message="Generate requirements",
    container=container
)
```

For detailed information, see the **[Dependency Injection Guide](../docs/dependency-injection-guide.md)**.

### Support Systems

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MemoryManager   │    │ ShellExecutor   │    │ TokenManager    │
│                 │    │                 │    │                 │
│ • Context Retention │    │ • Command Execution │    │ • Context Compression │
│ • Learning Log    │    │ • Error Handling │    │ • Token Optimization │
│ • Knowledge Accumulation │    │ • Retry Mechanism │    │ • Memory Management │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📖 Usage Examples

### Example 1: Create a Web API

```bash
# 1. Submit a request
autogen-framework --workspace ./my-api --request "
Create a FastAPI web service for a book management system with:
- CRUD operations for books
- User authentication
- Database integration with SQLAlchemy
- API documentation with Swagger
- Unit tests with pytest
"

# 2. Review the generated requirements
autogen-framework --workspace ./my-api --status
# View the requirements.md file

# 3. Approve or revise the requirements
autogen-framework --workspace ./my-api --approve requirements
# Or
autogen-framework --workspace ./my-api --revise "requirements:Add rate limiting and caching"

# 4. Review the design
# View the design.md file
autogen-framework --workspace ./my-api --approve design

# 5. Review the task list
# View the tasks.md file
autogen-framework --workspace ./my-api --approve tasks

# 6. Execute the implementation
autogen-framework --workspace ./my-api --execute-tasks
```

### Example 2: Data Analysis Tool

```bash
# Create a data analysis tool
autogen-framework --workspace ./data-tool --request "
Build a Python data analysis tool that:
- Reads CSV and Excel files
- Performs statistical analysis
- Generates visualizations with matplotlib
- Exports reports to PDF
- Has a command-line interface
"

# Quick approval process (if you are satisfied with the generated documents)
autogen-framework --workspace ./data-tool --approve requirements
autogen-framework --workspace ./data-tool --approve design
autogen-framework --workspace ./data-tool --approve tasks
autogen-framework --workspace ./data-tool --execute-tasks
```

### Example 3: Microservices Architecture

```bash
# Complex microservices project
autogen-framework --workspace ./microservices --request "
Design and implement a microservices architecture with:
- User service (authentication and profiles)
- Product service (catalog management)
- Order service (order processing)
- API Gateway with routing
- Docker containerization
- Kubernetes deployment configs
"

# Step-by-step review and revision
autogen-framework --workspace ./microservices --status
autogen-framework --workspace ./microservices --revise "design:Add service mesh with Istio"
autogen-framework --workspace ./microservices --revise "tasks:Include monitoring with Prometheus"
```

## 🔧 Advanced Configuration

### Custom LLM Endpoint

```bash
# Use a local LLM
autogen-framework \
  --workspace ./project \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model local-model \
  --llm-api-key local-key \
  --request "Your request here"
```

### Detailed Logging

```bash
# Enable detailed logging
autogen-framework \
  --workspace ./project \
  --verbose \
  --log-file debug.log \
  --request "Your request here"
```

### Workflow Recovery

```bash
# If interrupted, you can continue
autogen-framework --workspace ./project --continue-workflow

# Or start over
autogen-framework --workspace ./project --reset-session
```

## 📁 Generated File Structure

After execution is complete, your project directory will contain:

```
my-project/
├── requirements.md          # Requirements document
├── design.md               # Technical design document
├── tasks.md                # Task list
├── [Implemented code files] # Code generated according to tasks
├── memory/                 # Memory system files
│   ├── project_context.md
│   └── learning_log.md
└── logs/                   # Execution logs
    └── execution_log.md
```

## 🎯 Best Practices

### 1. Requirement Description
- **Specific and clear**: Provide specific functional and technical requirements.
- **Include context**: Explain the project background and usage scenarios.
- **Technical preferences**: Specify preferred technology stacks and frameworks.

```bash
# Good requirement description
autogen-framework --request "
Create a Django REST API for a library management system with:
- Book CRUD operations with ISBN validation
- User authentication using JWT tokens
- PostgreSQL database with migrations
- Redis caching for frequently accessed data
- Comprehensive unit tests with >90% coverage
- Docker deployment configuration
"

# Avoid overly simple descriptions
autogen-framework --request "Make a website"
```

### 2. Phased Review
- **Review carefully**: Carefully check the generated documents at each stage.
- **Revise promptly**: If you find problems, use `--revise` to correct them immediately.
- **Proceed step by step**: Do not skip the review and execute directly.

### 3. Task Execution
- **Monitor progress**: Use `--status` to periodically check the execution status.
- **Execute in steps**: For complex projects, consider executing tasks in steps.
- **Verify results**: Verify the generated code after execution is complete.

### 4. Error Handling
- **Check logs**: When problems occur, check the log files.
- **Use backups**: The patch strategy automatically creates backups.
- **Start over**: If necessary, use `--reset-session` to start over.

## 🐛 Troubleshooting

### Common Issues

#### 1. LLM Connection Failed
```bash
# Check if the endpoint is accessible
curl -X POST http://your-llm-endpoint/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"test"}]}'

# Confirm the configuration is correct
autogen-framework --workspace . --status
```

#### 2. Workflow Stuck
```bash
# Check the current status
autogen-framework --workspace . --status

# Reset the session
autogen-framework --workspace . --reset-session

# Continue the workflow
autogen-framework --workspace . --continue-workflow
```

#### 3. Task Execution Failed
```bash
# View detailed logs
autogen-framework --workspace . --verbose --log-file debug.log --status

# Check the generated files
ls -la
cat tasks.md

# Manually execute a specific task
autogen-framework --workspace . --execute-task "Specific task name"
```

#### 4. File Modification Issues
```bash
# The patch strategy automatically creates backups
ls *.backup_*

# If you need to restore
cp filename.backup_timestamp filename
```

### Log Analysis

The framework generates detailed execution logs:

```bash
# View execution logs
tail -f logs/execution_log.md

# View error logs
grep -i error logs/execution_log.md

# View logs for a specific agent
grep "PlanAgent\|DesignAgent\|ImplementAgent" logs/execution_log.md
```

## 🔌 Extension Development

### Adding a New Agent

```python
from autogen_framework.agents.base_agent import BaseLLMAgent

class CustomAgent(BaseLLMAgent):
    def __init__(self, name: str, llm_config: LLMConfig, system_message: str):
        super().__init__(name, llm_config, system_message)
    
    async def custom_method(self, input_data):
        # Implement custom logic
        pass
```

### Customizing the Workflow

```python
from autogen_framework.workflow import WorkflowPhase

# Add a new workflow phase
class CustomWorkflowPhase(WorkflowPhase):
    CUSTOM_PHASE = "custom_phase"
```

### Integrating New Tools

```python
from autogen_framework.shell_executor import ShellExecutor

class CustomExecutor(ShellExecutor):
    async def execute_custom_command(self, command: str):
        # Implement custom command execution
        pass
```

## 📊 Performance Optimization

### Configuration Recommendations
- **Concurrency settings**: Adjust the number of concurrent tasks based on system resources.
- **Memory management**: Periodically clean up old memory files.
- **Log rotation**: Configure log file size limits.

### Monitoring Metrics
- Task execution time
- Number of LLM API calls
- File modification success rate
- Number of error retries

## 🤝 Contribution Guide

Contributions, bug reports, and improvement suggestions are welcome!

### Development Environment Setup

```bash
# Clone the project
git clone <repository-url>
cd autogen-multi-agent-framework

# Install development dependencies
uv sync --dev

# Run tests
python -m pytest tests/ -v

# Code formatting
black autogen_framework/
isort autogen_framework/
```

### Submission Guidelines
- Follow PEP 8 code style.
- Add appropriate test cases.
- Update relevant documentation.
- Provide clear commit messages.

## 📄 License

[Add license information here]

## 🙏 Acknowledgments

Thanks to the AutoGen team for providing an excellent base framework, and to all contributors for their efforts.

---

**Version**: 1.0.0  
**Last Updated**: 2025-08-01  
**Maintainer**: Kiro AI Assistant
