# AutoGen Multi-Agent Framework

[![Run Unit Tests](https://github.com/wirelessr/SpecForge-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/wirelessr/SpecForge-Agent/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/wirelessr/SpecForge-Agent/graph/badge.svg)](https://codecov.io/gh/wirelessr/SpecForge-Agent)

A Python-based multi-agent collaboration framework using AutoGen for project development, debugging, and code review tasks.

## Features

- **Multi-Agent Collaboration**: Plan Agent, Design Agent, Tasks Agent, and Implement Agent working together
- **Modular Architecture**: Dedicated components for session management, workflow orchestration, and task execution
- **Shell Execution**: Complete shell command execution capabilities
- **Memory System**: Persistent memory storage in workspace/memory folder
- **Pure Text Interface**: Command-line interface for user interaction
- **Patch-First Strategy**: Intelligent file editing using diff/patch commands

## Installation

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Install development dependencies
uv sync --dev
```

## Usage

```bash
# Run the framework
uv run autogen-framework --workspace /path/to/workspace

# Run with verbose output
uv run autogen-framework --workspace /path/to/workspace --verbose
```

## Project Structure

```
autogen_framework/
├── agents/          # AI agent implementations
│   ├── plan_agent.py      # Requirements analysis
│   ├── design_agent.py    # Technical design
│   ├── tasks_agent.py     # Task generation
│   └── implement_agent.py # Task execution
├── models/          # Data models and schemas
├── utils/           # Utility functions
├── session_manager.py     # Session persistence
├── workflow_manager.py    # Workflow orchestration
├── agent_manager.py       # Agent coordination
├── main_controller.py     # Main entry point
├── tests/           # Test files
└── main.py          # CLI entry point
```

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run black .

# Type checking
uv run mypy autogen_framework

# Linting
uv run flake8 autogen_framework
```

## Requirements

- Python 3.11+
- uv package manager
- AutoGen framework
- Custom LLM API endpoint