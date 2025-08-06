"""
Unit tests for ContextManager.

Tests the core functionality of context loading, parsing, and agent-specific
context generation without external dependencies.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from autogen_framework.context_manager import (
    ContextManager, ProjectStructure, RequirementsDocument, DesignDocument,
    TasksDocument, MemoryPattern, PlanContext, DesignContext, TasksContext,
    ImplementationContext
)
from autogen_framework.models import TaskDefinition, ExecutionResult


class TestContextManager:
    """Test cases for ContextManager class."""
    
    @pytest.fixture
    def temp_work_dir(self):
        """Create a temporary work directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock MemoryManager."""
        mock = Mock()
        mock.search_memory = Mock(return_value=[
            {
                "category": "global",
                "file": "test_patterns.md",
                "content": "Test memory pattern content",
                "relevance_score": 1.0
            }
        ])
        return mock
    
    @pytest.fixture
    def mock_context_compressor(self):
        """Create a mock ContextCompressor."""
        mock = Mock()
        mock.compress_context = AsyncMock(return_value=Mock(
            success=True,
            compressed_content={"compressed": "content"},
            compression_ratio=0.7,
            error=None
        ))
        return mock
    
    @pytest.fixture
    def context_manager(self, temp_work_dir, mock_memory_manager, mock_context_compressor):
        """Create a ContextManager instance for testing."""
        return ContextManager(temp_work_dir, mock_memory_manager, mock_context_compressor)
    
    @pytest.fixture
    def sample_requirements_content(self):
        """Sample requirements.md content for testing."""
        return """# Requirements Document

## Introduction

This is a test requirements document.

## Requirements

### Requirement 1: Test Feature

**User Story:** As a user, I want to test features, so that I can verify functionality.

#### Acceptance Criteria

1. WHEN a test is run THEN the system SHALL execute correctly
2. WHEN validation occurs THEN the system SHALL provide feedback
3. WHEN errors happen THEN the system SHALL handle them gracefully

### Requirement 2: Another Feature

**User Story:** As a developer, I want to implement features, so that users can benefit.

#### Acceptance Criteria

1. WHEN implementation starts THEN the system SHALL follow best practices
2. WHEN code is written THEN the system SHALL maintain quality standards
"""
    
    @pytest.fixture
    def sample_design_content(self):
        """Sample design.md content for testing."""
        return """# Design Document

## Overview

This is a test design document that outlines the architecture.

## Architecture

The system follows a modular architecture with clear separation of concerns.

## Components

- Component A: Handles user input
- Component B: Processes data
- Component C: Manages output
"""
    
    @pytest.fixture
    def sample_tasks_content(self):
        """Sample tasks.md content for testing."""
        return """# Implementation Plan

- [ ] 1. Create basic structure
  - Set up project directories
  - Initialize configuration files
  - _Requirements: 1.1_

- [x] 2. Implement core functionality
  - Write main logic
  - Add error handling
  - _Requirements: 1.2, 2.1_

- [ ] 3. Add testing
  - Write unit tests
  - Add integration tests
  - _Requirements: 2.2_
"""
    
    def test_context_manager_initialization(self, context_manager, temp_work_dir):
        """Test ContextManager initialization."""
        assert context_manager.work_dir == Path(temp_work_dir)
        assert context_manager.requirements is None
        assert context_manager.design is None
        assert context_manager.tasks is None
        assert context_manager.execution_history == []
        assert context_manager.project_structure is None
    
    @pytest.mark.asyncio
    async def test_load_requirements(self, context_manager, temp_work_dir, sample_requirements_content):
        """Test loading and parsing requirements.md."""
        # Create requirements.md file
        requirements_path = Path(temp_work_dir) / "requirements.md"
        requirements_path.write_text(sample_requirements_content, encoding='utf-8')
        
        # Load requirements
        await context_manager._load_requirements()
        
        # Verify loading
        assert context_manager.requirements is not None
        assert context_manager.requirements.content == sample_requirements_content
        assert len(context_manager.requirements.requirements) == 2
        
        # Verify first requirement parsing
        req1 = context_manager.requirements.requirements[0]
        assert "Test Feature" in req1['title']
        assert "As a user, I want to test features" in req1['user_story']
        assert len(req1['acceptance_criteria']) == 3
    
    @pytest.mark.asyncio
    async def test_load_design(self, context_manager, temp_work_dir, sample_design_content):
        """Test loading design.md."""
        # Create design.md file
        design_path = Path(temp_work_dir) / "design.md"
        design_path.write_text(sample_design_content, encoding='utf-8')
        
        # Load design
        await context_manager._load_design()
        
        # Verify loading
        assert context_manager.design is not None
        assert context_manager.design.content == sample_design_content
        assert context_manager.design.file_path == str(design_path)
    
    @pytest.mark.asyncio
    async def test_load_tasks(self, context_manager, temp_work_dir, sample_tasks_content):
        """Test loading and parsing tasks.md."""
        # Create tasks.md file
        tasks_path = Path(temp_work_dir) / "tasks.md"
        tasks_path.write_text(sample_tasks_content, encoding='utf-8')
        
        # Load tasks
        await context_manager._load_tasks()
        
        # Verify loading
        assert context_manager.tasks is not None
        assert context_manager.tasks.content == sample_tasks_content
        assert len(context_manager.tasks.tasks) == 3
        
        # Verify task parsing
        task1 = context_manager.tasks.tasks[0]
        assert task1.title == "1. Create basic structure"
        assert not task1.completed
        
        task2 = context_manager.tasks.tasks[1]
        assert task2.title == "2. Implement core functionality"
        assert task2.completed  # Marked as [x]
    
    @pytest.mark.asyncio
    async def test_analyze_project_structure(self, context_manager, temp_work_dir):
        """Test project structure analysis."""
        # Create some test files and directories
        test_dir = Path(temp_work_dir)
        (test_dir / "src").mkdir()
        (test_dir / "tests").mkdir()
        (test_dir / "src" / "main.py").write_text("# Main file")
        (test_dir / "tests" / "test_main.py").write_text("# Test file")
        (test_dir / "README.md").write_text("# Test Project")
        (test_dir / "pyproject.toml").write_text("[tool.poetry]")
        
        # Analyze structure
        await context_manager._analyze_project_structure()
        
        # Verify analysis
        assert context_manager.project_structure is not None
        assert context_manager.project_structure.total_files == 4
        assert context_manager.project_structure.total_directories == 2
        assert "src" in context_manager.project_structure.directories
        assert "tests" in context_manager.project_structure.directories
        assert "src/main.py" in context_manager.project_structure.files
        assert "README.md" in context_manager.project_structure.key_files
        assert "pyproject.toml" in context_manager.project_structure.key_files
    
    @pytest.mark.asyncio
    async def test_initialize(self, context_manager, temp_work_dir, sample_requirements_content, 
                             sample_design_content, sample_tasks_content):
        """Test full initialization process."""
        # Create all project files
        test_dir = Path(temp_work_dir)
        (test_dir / "requirements.md").write_text(sample_requirements_content)
        (test_dir / "design.md").write_text(sample_design_content)
        (test_dir / "tasks.md").write_text(sample_tasks_content)
        (test_dir / "src").mkdir()
        (test_dir / "src" / "main.py").write_text("# Main file")
        
        # Initialize
        await context_manager.initialize()
        
        # Verify all components are loaded
        assert context_manager.requirements is not None
        assert context_manager.design is not None
        assert context_manager.tasks is not None
        assert context_manager.project_structure is not None
        assert len(context_manager.requirements.requirements) == 2
        assert len(context_manager.tasks.tasks) == 3
        assert context_manager.project_structure.total_files == 4
    
    @pytest.mark.asyncio
    async def test_get_plan_context(self, context_manager, temp_work_dir):
        """Test getting PlanContext."""
        # Set up project structure
        test_dir = Path(temp_work_dir)
        (test_dir / "src").mkdir()
        (test_dir / "README.md").write_text("# Test Project")
        
        # Initialize and get context
        await context_manager.initialize()
        context = await context_manager.get_plan_context("Create a new feature")
        
        # Verify context
        assert isinstance(context, PlanContext)
        assert context.user_request == "Create a new feature"
        assert context.project_structure is not None
        assert len(context.memory_patterns) > 0
        assert not context.compressed  # Should not be compressed for small content
    
    @pytest.mark.asyncio
    async def test_get_design_context(self, context_manager, temp_work_dir, sample_requirements_content):
        """Test getting DesignContext."""
        # Create requirements file
        test_dir = Path(temp_work_dir)
        (test_dir / "requirements.md").write_text(sample_requirements_content)
        (test_dir / "src").mkdir()
        
        # Initialize and get context
        await context_manager.initialize()
        context = await context_manager.get_design_context("Design the system")
        
        # Verify context
        assert isinstance(context, DesignContext)
        assert context.user_request == "Design the system"
        assert context.requirements is not None
        assert len(context.requirements.requirements) == 2
        assert context.project_structure is not None
        assert len(context.memory_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_get_tasks_context(self, context_manager, temp_work_dir, 
                                   sample_requirements_content, sample_design_content):
        """Test getting TasksContext."""
        # Create project files
        test_dir = Path(temp_work_dir)
        (test_dir / "requirements.md").write_text(sample_requirements_content)
        (test_dir / "design.md").write_text(sample_design_content)
        
        # Initialize and get context
        await context_manager.initialize()
        context = await context_manager.get_tasks_context("Create implementation tasks")
        
        # Verify context
        assert isinstance(context, TasksContext)
        assert context.user_request == "Create implementation tasks"
        assert context.requirements is not None
        assert context.design is not None
        assert len(context.memory_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_get_implementation_context(self, context_manager, temp_work_dir,
                                            sample_requirements_content, sample_design_content,
                                            sample_tasks_content):
        """Test getting ImplementationContext."""
        # Create all project files
        test_dir = Path(temp_work_dir)
        (test_dir / "requirements.md").write_text(sample_requirements_content)
        (test_dir / "design.md").write_text(sample_design_content)
        (test_dir / "tasks.md").write_text(sample_tasks_content)
        (test_dir / "src").mkdir()
        
        # Initialize
        await context_manager.initialize()
        
        # Create a test task
        test_task = TaskDefinition(
            id="test_task",
            title="Test Task",
            description="A test task for implementation",
            steps=["Step 1", "Step 2"],
            requirements_ref=["1.1"]
        )
        
        # Get context
        context = await context_manager.get_implementation_context(test_task)
        
        # Verify context
        assert isinstance(context, ImplementationContext)
        assert context.task == test_task
        assert context.requirements is not None
        assert context.design is not None
        assert context.tasks is not None
        assert context.project_structure is not None
        assert len(context.memory_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_update_execution_history(self, context_manager, temp_work_dir):
        """Test updating execution history."""
        # Create execution result
        result = ExecutionResult.create_success(
            command="echo 'test'",
            stdout="test",
            execution_time=0.1,
            working_directory=temp_work_dir,
            approach_used="direct"
        )
        
        # Update history
        await context_manager.update_execution_history(result)
        
        # Verify update
        assert len(context_manager.execution_history) == 1
        assert context_manager.execution_history[0] == result
        
        # Verify persistence file was created
        history_file = Path(temp_work_dir) / "execution_history.json"
        assert history_file.exists()
        
        # Verify file content
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        assert len(history_data) == 1
        assert history_data[0]['command'] == "echo 'test'"
        assert history_data[0]['success'] is True
    
    @pytest.mark.asyncio
    async def test_refresh_project_structure(self, context_manager, temp_work_dir):
        """Test refreshing project structure."""
        # Initial setup
        test_dir = Path(temp_work_dir)
        (test_dir / "initial.py").write_text("# Initial file")
        
        await context_manager.initialize()
        initial_file_count = context_manager.project_structure.total_files
        
        # Add new file
        (test_dir / "new_file.py").write_text("# New file")
        
        # Refresh structure
        await context_manager.refresh_project_structure()
        
        # Verify refresh
        assert context_manager.project_structure.total_files == initial_file_count + 1
        assert "new_file.py" in context_manager.project_structure.files
    
    def test_find_related_tasks(self, context_manager):
        """Test finding related tasks."""
        # Create test tasks with dependencies
        task1 = TaskDefinition(id="task1", title="Task 1", description="", steps=[], requirements_ref=[], dependencies=["dep1"])
        task2 = TaskDefinition(id="task2", title="Task 2", description="", steps=[], requirements_ref=[], dependencies=["dep1", "dep2"])
        task3 = TaskDefinition(id="task3", title="Task 3", description="", steps=[], requirements_ref=[], dependencies=["task1"])
        task4 = TaskDefinition(id="task4", title="Task 4", description="", steps=[], requirements_ref=[], dependencies=["dep3"])
        
        # Mock tasks document
        context_manager.tasks = Mock()
        context_manager.tasks.tasks = [task1, task2, task3, task4]
        
        # Find related tasks for task1
        related = context_manager._find_related_tasks(task1)
        
        # Verify relationships
        related_ids = [t.id for t in related]
        assert "task2" in related_ids  # Shared dependency "dep1"
        assert "task3" in related_ids  # task3 depends on task1
        assert "task4" not in related_ids  # No shared dependencies
    
    def test_get_relevant_history(self, context_manager):
        """Test getting relevant execution history."""
        # Create test execution history
        results = []
        for i in range(15):
            result = ExecutionResult.create_success(
                command=f"command_{i}",
                stdout=f"output_{i}",
                execution_time=0.1,
                working_directory="/test",
                approach_used="direct"
            )
            results.append(result)
        
        context_manager.execution_history = results
        
        # Create test task
        test_task = TaskDefinition(id="test", title="Test", description="", steps=[], requirements_ref=[])
        
        # Get relevant history
        relevant = context_manager._get_relevant_history(test_task)
        
        # Should return last 10 results
        assert len(relevant) == 10
        assert relevant[0].command == "command_5"  # Last 10 starting from index 5
        assert relevant[-1].command == "command_14"
    
    @pytest.mark.asyncio
    async def test_memory_pattern_retrieval(self, context_manager, mock_memory_manager):
        """Test memory pattern retrieval for different agent types."""
        # Test plan patterns
        plan_patterns = await context_manager._get_plan_memory_patterns("test request")
        assert len(plan_patterns) > 0
        assert isinstance(plan_patterns[0], MemoryPattern)
        mock_memory_manager.search_memory.assert_called_with("planning requirements project")
        
        # Test design patterns
        design_patterns = await context_manager._get_design_memory_patterns()
        assert len(design_patterns) > 0
        mock_memory_manager.search_memory.assert_called_with("design architecture patterns")
        
        # Test tasks patterns
        tasks_patterns = await context_manager._get_tasks_memory_patterns()
        assert len(tasks_patterns) > 0
        mock_memory_manager.search_memory.assert_called_with("tasks implementation breakdown")
        
        # Test implementation patterns
        test_task = TaskDefinition(id="test", title="Test Task", description="", steps=[], requirements_ref=[])
        impl_patterns = await context_manager._get_implementation_memory_patterns(test_task)
        assert len(impl_patterns) > 0
        mock_memory_manager.search_memory.assert_called_with("implementation Test Task coding")
    
    def test_context_to_string(self, context_manager):
        """Test context to string conversion for token estimation."""
        # Create test context
        context = PlanContext(
            user_request="Test request",
            project_structure=ProjectStructure(
                root_path="/test",
                files=["file1.py", "file2.py"],
                directories=["dir1"]
            ),
            memory_patterns=[
                MemoryPattern(category="test", content="Test pattern content", source="test.md")
            ]
        )
        
        # Convert to string
        context_str = context_manager._context_to_string(context)
        
        # Verify content is included
        assert "Test request" in context_str
        assert "Files: 2, Dirs: 1" in context_str
        assert "Test pattern content" in context_str
    
    @pytest.mark.asyncio
    async def test_context_compression(self, context_manager, mock_context_compressor):
        """Test context compression when token threshold is exceeded."""
        # Create large context that should trigger compression
        large_content = "x" * 20000  # Large content to exceed threshold
        context = PlanContext(
            user_request=large_content,
            memory_patterns=[]
        )
        
        # Test compression
        compressed_context = await context_manager._compress_if_needed(context, "plan")
        
        # Verify compression was attempted
        mock_context_compressor.compress_context.assert_called_once()
        
        # For this test, the mock returns success, so context should be marked as compressed
        # Note: In the actual implementation, we'd need to properly handle the compressed content
        assert compressed_context is not None
    
    def test_file_refresh_logic(self, context_manager, temp_work_dir):
        """Test file refresh logic based on modification times."""
        # Create test file
        test_file = Path(temp_work_dir) / "requirements.md"
        test_file.write_text("Initial content")
        
        # Initially should refresh (no load time recorded)
        assert context_manager._should_refresh_requirements()
        
        # After loading, should not refresh immediately
        context_manager._last_requirements_load = datetime.now()
        assert not context_manager._should_refresh_requirements()
        
        # After file modification, should refresh
        import time
        time.sleep(0.1)  # Ensure different timestamp
        test_file.write_text("Modified content")
        assert context_manager._should_refresh_requirements()


class TestDataStructures:
    """Test the data structure classes used by ContextManager."""
    
    def test_requirements_document_parsing(self):
        """Test RequirementsDocument parsing functionality."""
        content = """# Requirements

### Requirement 1: Test Feature

**User Story:** As a user, I want to test, so that I can verify.

#### Acceptance Criteria

1. WHEN testing THEN system SHALL work
2. WHEN validating THEN system SHALL respond

### Requirement 2: Another Feature

**User Story:** As a developer, I want to code, so that users benefit.

#### Acceptance Criteria

1. WHEN coding THEN system SHALL follow standards
"""
        
        doc = RequirementsDocument(content=content)
        
        assert len(doc.requirements) == 2
        assert "Test Feature" in doc.requirements[0]['title']
        assert "As a user, I want to test" in doc.requirements[0]['user_story']
        assert len(doc.requirements[0]['acceptance_criteria']) == 2
    
    def test_tasks_document_parsing(self):
        """Test TasksDocument parsing functionality."""
        content = """# Tasks

- [ ] 1. First task
  - Sub-item 1
  - Sub-item 2

- [x] 2. Completed task
  - This was done

- [ ] 3. Another task
"""
        
        doc = TasksDocument(content=content)
        
        assert len(doc.tasks) == 3
        assert doc.tasks[0].title == "1. First task"
        assert not doc.tasks[0].completed
        assert doc.tasks[1].title == "2. Completed task"
        assert doc.tasks[1].completed
        assert len(doc.tasks[0].steps) == 2
    
    def test_project_structure_initialization(self):
        """Test ProjectStructure initialization and post_init."""
        structure = ProjectStructure(
            root_path="/test",
            files=["file1.py", "file2.py"],
            directories=["dir1", "dir2"]
        )
        
        assert structure.root_path == "/test"
        assert len(structure.files) == 2
        assert len(structure.directories) == 2
        assert structure.analysis_timestamp  # Should be set by post_init
    
    def test_memory_pattern_creation(self):
        """Test MemoryPattern data structure."""
        pattern = MemoryPattern(
            category="global",
            content="Test pattern content",
            relevance_score=0.8,
            source="test.md"
        )
        
        assert pattern.category == "global"
        assert pattern.content == "Test pattern content"
        assert pattern.relevance_score == 0.8
        assert pattern.source == "test.md"
    
    def test_context_structures(self):
        """Test all context data structures."""
        # Test PlanContext
        plan_ctx = PlanContext(user_request="Plan request")
        assert plan_ctx.user_request == "Plan request"
        assert not plan_ctx.compressed
        
        # Test DesignContext
        design_ctx = DesignContext(user_request="Design request")
        assert design_ctx.user_request == "Design request"
        assert design_ctx.memory_patterns == []
        
        # Test TasksContext
        tasks_ctx = TasksContext(user_request="Tasks request")
        assert tasks_ctx.user_request == "Tasks request"
        
        # Test ImplementationContext
        test_task = TaskDefinition(id="test", title="Test", description="", steps=[], requirements_ref=[])
        impl_ctx = ImplementationContext(task=test_task)
        assert impl_ctx.task == test_task
        assert impl_ctx.execution_history == []
        assert impl_ctx.related_tasks == []