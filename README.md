# AutoGen Multi-Agent Framework

A Python-based multi-agent collaboration framework using AutoGen for project development, debugging, and code review tasks.

## Features

- **Multi-Agent Collaboration**: Plan Agent, Design Agent, and Implement Agent working together
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
├── models/          # Data models and schemas
├── utils/           # Utility functions
├── tests/           # Test files
└── main.py          # Main entry point
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