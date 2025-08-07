"""
Integration tests for ContextManager.

Tests the ContextManager with real MemoryManager and ContextCompressor instances
to verify proper integration and context generation for different agent types.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from autogen_framework.context_manager import ContextManager
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.context_compressor import ContextCompressor
from autogen_framework.models import LLMConfig, TaskDefinition


class TestContextManagerIntegration:
    """Integration tests for ContextManager with real dependencies."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create workspace structure
            workspace = Path(temp_dir)
            memory_dir = workspace / "memory"
            memory_dir.mkdir()
            (memory_dir / "global").mkdir()
            (memory_dir / "projects").mkdir()
            
            # Create some test memory content
            global_memory = memory_dir / "global" / "test_patterns.md"
            global_memory.write_text("""# Test Patterns

## Planning Patterns
- Always start with requirements analysis
- Consider user needs first

## Design Patterns
- Use modular architecture
- Separate concerns clearly

## Implementation Patterns
- Write tests first
- Use meaningful variable names
""")
            
            yield temp_dir
    
    @pytest.fixture
    def real_memory_manager(self, temp_workspace):
        """Create a real MemoryManager instance."""
        return MemoryManager(temp_workspace)
    
    @pytest.fixture
    def real_context_compressor(self, test_llm_config):
        """Create a real ContextCompressor instance."""
        return ContextCompressor(test_llm_config)
    
    @pytest.fixture
    def real_token_manager(self):
        """Create a real TokenManager instance."""
        from autogen_framework.token_manager import TokenManager
        from autogen_framework.config_manager import ConfigManager
        config_manager = ConfigManager()
        return TokenManager(config_manager)
    
    @pytest.fixture
    def real_config_manager(self):
        """Create a real ConfigManager instance."""
        from autogen_framework.config_manager import ConfigManager
        return ConfigManager()
    
    @pytest.fixture
    def work_dir(self, temp_workspace):
        """Create a work directory with test project files."""
        work_dir = Path(temp_workspace) / "test_project"
        work_dir.mkdir()
        
        # Create requirements.md
        requirements_content = """# Requirements Document

## Introduction

This is a test project for context manager integration testing.

## Requirements

### Requirement 1: User Authentication

**User Story:** As a user, I want to authenticate securely, so that my data is protected.

#### Acceptance Criteria

1. WHEN a user provides valid credentials THEN the system SHALL authenticate them
2. WHEN a user provides invalid credentials THEN the system SHALL reject access
3. WHEN authentication fails THEN the system SHALL log the attempt

### Requirement 2: Data Management

**User Story:** As a user, I want to manage my data, so that I can organize information.

#### Acceptance Criteria

1. WHEN a user creates data THEN the system SHALL store it securely
2. WHEN a user updates data THEN the system SHALL maintain version history
3. WHEN a user deletes data THEN the system SHALL confirm the action
"""
        (work_dir / "requirements.md").write_text(requirements_content)
        
        # Create design.md
        design_content = """# Design Document

## Overview

This system provides secure user authentication and data management capabilities.

## Architecture

The system follows a layered architecture:
- Presentation Layer: User interface components
- Business Layer: Authentication and data management logic
- Data Layer: Secure storage and retrieval

## Components

### Authentication Service
- Handles user login/logout
- Manages session tokens
- Validates credentials

### Data Service
- CRUD operations for user data
- Version control for data changes
- Backup and recovery mechanisms

## Security Considerations

- All passwords are hashed using bcrypt
- Session tokens expire after 24 hours
- All data is encrypted at rest
"""
        (work_dir / "design.md").write_text(design_content)
        
        # Create tasks.md
        tasks_content = """# Implementation Plan

## Phase 1: Authentication System

- [ ] 1. Set up authentication infrastructure
  - Create user model with password hashing
  - Implement login/logout endpoints
  - Set up session management
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. Implement security measures
  - Add rate limiting for login attempts
  - Implement password strength validation
  - Add audit logging for authentication events
  - _Requirements: 1.2, 1.3_

## Phase 2: Data Management

- [ ] 3. Create data models and storage
  - Design database schema for user data
  - Implement CRUD operations
  - Add data validation and sanitization
  - _Requirements: 2.1, 2.2_

- [x] 4. Implement version control
  - Add versioning to data changes
  - Create rollback functionality
  - Implement change history tracking
  - _Requirements: 2.2_

- [ ] 5. Add data security features
  - Implement encryption at rest
  - Add backup and recovery procedures
  - Create data export functionality
  - _Requirements: 2.1, 2.3_
"""
        (work_dir / "tasks.md").write_text(tasks_content)
        
        # Create project structure
        (work_dir / "src").mkdir()
        (work_dir / "src" / "auth").mkdir()
        (work_dir / "src" / "data").mkdir()
        (work_dir / "tests").mkdir()
        
        (work_dir / "src" / "auth" / "__init__.py").write_text("")
        (work_dir / "src" / "auth" / "models.py").write_text("# User authentication models")
        (work_dir / "src" / "auth" / "views.py").write_text("# Authentication views")
        
        (work_dir / "src" / "data" / "__init__.py").write_text("")
        (work_dir / "src" / "data" / "models.py").write_text("# Data models")
        (work_dir / "src" / "data" / "services.py").write_text("# Data services")
        
        (work_dir / "README.md").write_text("# Test Project\n\nA test project for integration testing.")
        (work_dir / "pyproject.toml").write_text("[tool.poetry]\nname = 'test-project'")
        
        return str(work_dir)
    
    @pytest.fixture
    def context_manager(self, work_dir, real_memory_manager, real_context_compressor, 
                       test_llm_config, real_token_manager, real_config_manager):
        """Create a ContextManager with real dependencies."""
        return ContextManager(
            work_dir=work_dir,
            memory_manager=real_memory_manager,
            context_compressor=real_context_compressor,
            llm_config=test_llm_config,
            token_manager=real_token_manager,
            config_manager=real_config_manager
        )
    
    @pytest.mark.asyncio
    async def test_full_initialization_with_real_dependencies(self, context_manager):
        """Test full initialization with real MemoryManager and ContextCompressor."""
        # Initialize the context manager
        await context_manager.initialize()
        
        # Verify all components are loaded
        assert context_manager.requirements is not None
        assert context_manager.design is not None
        assert context_manager.tasks is not None
        assert context_manager.project_structure is not None
        
        # Verify parsing worked correctly
        assert len(context_manager.requirements.requirements) == 2
        assert len(context_manager.tasks.tasks) == 5
        assert context_manager.project_structure.total_files > 0
        assert context_manager.project_structure.total_directories > 0
        
        # Verify key files were captured
        assert "README.md" in context_manager.project_structure.key_files
        assert "pyproject.toml" in context_manager.project_structure.key_files
    
    @pytest.mark.asyncio
    async def test_plan_context_with_real_memory(self, context_manager, real_memory_manager):
        """Test PlanContext generation with real MemoryManager."""
        await context_manager.initialize()
        
        # Get plan context
        context = await context_manager.get_plan_context("Create a new authentication system")
        
        # Verify context structure
        assert context.user_request == "Create a new authentication system"
        assert context.project_structure is not None
        assert context.project_structure.total_files > 0
        
        # Verify memory patterns are retrieved (may be empty if no matching patterns)
        assert isinstance(context.memory_patterns, list)
        
        # Verify project structure contains expected elements
        assert "src/auth" in context.project_structure.directories
        assert "src/data" in context.project_structure.directories
        assert "README.md" in context.project_structure.files
    
    @pytest.mark.asyncio
    async def test_design_context_with_requirements(self, context_manager):
        """Test DesignContext generation with requirements integration."""
        await context_manager.initialize()
        
        # Get design context
        context = await context_manager.get_design_context("Design the authentication system")
        
        # Verify context structure
        assert context.user_request == "Design the authentication system"
        assert context.requirements is not None
        assert context.project_structure is not None
        
        # Verify requirements are properly parsed
        assert len(context.requirements.requirements) == 2
        req1 = context.requirements.requirements[0]
        assert "User Authentication" in req1['title']
        assert "authenticate securely" in req1['user_story']
        assert len(req1['acceptance_criteria']) == 3
        
        # Verify project structure is included
        assert context.project_structure.total_files > 0
    
    @pytest.mark.asyncio
    async def test_tasks_context_with_requirements_and_design(self, context_manager):
        """Test TasksContext generation with requirements and design integration."""
        await context_manager.initialize()
        
        # Get tasks context
        context = await context_manager.get_tasks_context("Create implementation tasks")
        
        # Verify context structure
        assert context.user_request == "Create implementation tasks"
        assert context.requirements is not None
        assert context.design is not None
        
        # Verify requirements integration
        assert len(context.requirements.requirements) == 2
        
        # Verify design integration
        assert "layered architecture" in context.design.content.lower()
        assert "Authentication Service" in context.design.content
        assert "Data Service" in context.design.content
        
        # Verify memory patterns
        assert isinstance(context.memory_patterns, list)
    
    @pytest.mark.asyncio
    async def test_implementation_context_comprehensive(self, context_manager):
        """Test ImplementationContext with comprehensive project context."""
        await context_manager.initialize()
        
        # Create a test task
        test_task = TaskDefinition(
            id="auth_task",
            title="Implement user authentication",
            description="Create secure user authentication system",
            steps=["Create user model", "Implement login endpoint", "Add session management"],
            requirements_ref=["1.1", "1.2"],
            dependencies=["database_setup"]
        )
        
        # Get implementation context
        context = await context_manager.get_implementation_context(test_task)
        
        # Verify comprehensive context
        assert context.task == test_task
        assert context.requirements is not None
        assert context.design is not None
        assert context.tasks is not None
        assert context.project_structure is not None
        
        # Verify all documents are loaded
        assert len(context.requirements.requirements) == 2
        assert "Authentication Service" in context.design.content
        assert len(context.tasks.tasks) == 5
        
        # Verify project structure analysis
        assert context.project_structure.total_files > 0
        assert "src/auth/models.py" in context.project_structure.files
        assert "src/data/services.py" in context.project_structure.files
        
        # Verify execution history (should be empty initially)
        assert isinstance(context.execution_history, list)
        
        # Verify related tasks (should be empty for this test)
        assert isinstance(context.related_tasks, list)
        
        # Verify memory patterns
        assert isinstance(context.memory_patterns, list)
    
    @pytest.mark.asyncio
    async def test_context_refresh_after_file_changes(self, context_manager, work_dir):
        """Test context refresh when project files are modified."""
        await context_manager.initialize()
        
        # Get initial requirements content through tasks context (which includes requirements)
        initial_context = await context_manager.get_tasks_context("Initial tasks")
        initial_content = initial_context.requirements.content
        
        # Modify requirements.md
        requirements_path = Path(work_dir) / "requirements.md"
        modified_content = initial_content + "\n\n### Requirement 3: New Feature\n\n**User Story:** As a user, I want new features, so that I can do more.\n\n#### Acceptance Criteria\n\n1. WHEN new features are added THEN the system SHALL support them"
        requirements_path.write_text(modified_content)
        
        # Get context again - should detect file change and reload
        updated_context = await context_manager.get_tasks_context("Updated tasks")
        
        # Verify content was updated
        assert "Requirement 3: New Feature" in updated_context.requirements.content
        assert len(updated_context.requirements.content) > len(initial_content)
        assert len(updated_context.requirements.requirements) == 3  # Should now have 3 requirements
    
    @pytest.mark.asyncio
    async def test_execution_history_integration(self, context_manager):
        """Test execution history tracking and integration."""
        await context_manager.initialize()
        
        # Create test execution results
        from autogen_framework.models import ExecutionResult
        
        result1 = ExecutionResult.create_success(
            command="pip install pytest",
            stdout="Successfully installed pytest",
            execution_time=2.5,
            working_directory=context_manager.work_dir,
            approach_used="direct_install"
        )
        
        result2 = ExecutionResult.create_failure(
            command="python test_nonexistent.py",
            return_code=1,
            stderr="File not found: test_nonexistent.py",
            execution_time=0.1,
            working_directory=context_manager.work_dir,
            approach_used="direct_execution"
        )
        
        # Update execution history
        await context_manager.update_execution_history(result1)
        await context_manager.update_execution_history(result2)
        
        # Verify history is tracked
        assert len(context_manager.execution_history) == 2
        assert context_manager.execution_history[0] == result1
        assert context_manager.execution_history[1] == result2
        
        # Verify history file was created
        history_file = Path(context_manager.work_dir) / "execution_history.json"
        assert history_file.exists()
        
        # Create test task and get implementation context
        test_task = TaskDefinition(
            id="test_task",
            title="Test task",
            description="Test task for history integration",
            steps=["Step 1"],
            requirements_ref=["1.1"]
        )
        
        context = await context_manager.get_implementation_context(test_task)
        
        # Verify execution history is included in context
        assert len(context.execution_history) == 2
        assert context.execution_history[0].command == "pip install pytest"
        assert context.execution_history[1].command == "python test_nonexistent.py"
    
    @pytest.mark.asyncio
    async def test_related_tasks_identification(self, context_manager):
        """Test identification of related tasks based on dependencies."""
        await context_manager.initialize()
        
        # Create test tasks with dependencies
        task1 = TaskDefinition(
            id="setup_db",
            title="Set up database",
            description="Initialize database schema",
            steps=["Create tables", "Set up indexes"],
            requirements_ref=["2.1"],
            dependencies=[]
        )
        
        task2 = TaskDefinition(
            id="auth_system",
            title="Implement authentication",
            description="Create user authentication",
            steps=["Create user model", "Add login"],
            requirements_ref=["1.1", "1.2"],
            dependencies=["setup_db"]
        )
        
        task3 = TaskDefinition(
            id="data_management",
            title="Implement data management",
            description="Create data CRUD operations",
            steps=["Create data models", "Add CRUD endpoints"],
            requirements_ref=["2.1", "2.2"],
            dependencies=["setup_db"]
        )
        
        # Mock the tasks document to include our test tasks
        context_manager.tasks.tasks = [task1, task2, task3]
        
        # Get implementation context for auth_system task
        context = await context_manager.get_implementation_context(task2)
        
        # Verify related tasks are identified
        related_task_ids = [t.id for t in context.related_tasks]
        assert "setup_db" in related_task_ids  # task2 depends on setup_db
        assert "data_management" in related_task_ids  # both depend on setup_db
        assert "auth_system" not in related_task_ids  # shouldn't include itself
    
    @pytest.mark.asyncio
    async def test_memory_pattern_integration(self, context_manager, real_memory_manager):
        """Test memory pattern retrieval and integration."""
        await context_manager.initialize()
        
        # Ensure memory is loaded
        memory_content = real_memory_manager.load_memory()
        assert len(memory_content) > 0
        
        # Test different context types get appropriate memory patterns
        plan_context = await context_manager.get_plan_context("Plan a new feature")
        design_context = await context_manager.get_design_context("Design the system")
        tasks_context = await context_manager.get_tasks_context("Create tasks")
        
        # Verify memory patterns are included (content depends on test memory)
        assert isinstance(plan_context.memory_patterns, list)
        assert isinstance(design_context.memory_patterns, list)
        assert isinstance(tasks_context.memory_patterns, list)
        
        # If patterns are found, verify they have the correct structure
        for context in [plan_context, design_context, tasks_context]:
            for pattern in context.memory_patterns:
                assert hasattr(pattern, 'category')
                assert hasattr(pattern, 'content')
                assert hasattr(pattern, 'relevance_score')
                assert hasattr(pattern, 'source')
    
    @pytest.mark.asyncio
    async def test_project_structure_analysis_accuracy(self, context_manager):
        """Test accuracy of project structure analysis."""
        await context_manager.initialize()
        
        structure = context_manager.project_structure
        
        # Verify directory analysis
        expected_dirs = ["src", "src/auth", "src/data", "tests"]
        for expected_dir in expected_dirs:
            assert expected_dir in structure.directories
        
        # Verify file analysis
        expected_files = [
            "requirements.md", "design.md", "tasks.md", "README.md", "pyproject.toml",
            "src/auth/__init__.py", "src/auth/models.py", "src/auth/views.py",
            "src/data/__init__.py", "src/data/models.py", "src/data/services.py"
        ]
        for expected_file in expected_files:
            assert expected_file in structure.files
        
        # Verify key files content
        assert "README.md" in structure.key_files
        assert "Test Project" in structure.key_files["README.md"]
        assert "pyproject.toml" in structure.key_files
        assert "test-project" in structure.key_files["pyproject.toml"]
        
        # Verify file type analysis
        assert ".py" in structure.file_types
        assert ".md" in structure.file_types
        assert ".toml" in structure.file_types
        assert structure.file_types[".py"] >= 6  # At least 6 Python files
        assert structure.file_types[".md"] >= 3  # At least 3 Markdown files
    
    @pytest.mark.asyncio
    async def test_context_data_consistency(self, context_manager):
        """Test consistency of context data across multiple retrievals."""
        await context_manager.initialize()
        
        # Get the same context multiple times
        context1 = await context_manager.get_design_context("Test request")
        context2 = await context_manager.get_design_context("Test request")
        
        # Verify consistency
        assert context1.requirements.content == context2.requirements.content
        assert context1.project_structure.total_files == context2.project_structure.total_files
        assert context1.project_structure.total_directories == context2.project_structure.total_directories
        
        # Verify object identity (should be same objects due to caching)
        assert context1.requirements is context2.requirements
        assert context1.project_structure is context2.project_structure
    
    @pytest.mark.asyncio
    async def test_error_handling_with_missing_files(self, temp_workspace, real_memory_manager, real_context_compressor,
                                                    test_llm_config, real_token_manager, real_config_manager):
        """Test error handling when project files are missing."""
        # Create context manager with empty work directory
        empty_work_dir = Path(temp_workspace) / "empty_project"
        empty_work_dir.mkdir()
        
        context_manager = ContextManager(
            work_dir=str(empty_work_dir),
            memory_manager=real_memory_manager,
            context_compressor=real_context_compressor,
            llm_config=test_llm_config,
            token_manager=real_token_manager,
            config_manager=real_config_manager
        )
        
        # Initialize should not fail even with missing files
        await context_manager.initialize()
        
        # Verify graceful handling of missing files
        assert context_manager.requirements is None
        assert context_manager.design is None
        assert context_manager.tasks is None
        assert context_manager.project_structure is not None  # Should still analyze empty directory
        
        # Context generation should still work
        plan_context = await context_manager.get_plan_context("Test with missing files")
        assert plan_context.user_request == "Test with missing files"
        assert plan_context.project_structure is not None
        assert plan_context.project_structure.total_files == 0
        
        # Implementation context should handle missing task gracefully
        test_task = TaskDefinition(id="test", title="Test", description="", steps=[], requirements_ref=[])
        impl_context = await context_manager.get_implementation_context(test_task)
        assert impl_context.task == test_task
        assert impl_context.requirements is None
        assert impl_context.design is None
        assert impl_context.tasks is None