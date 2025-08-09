"""
Integration tests for TasksAgent coordination with ImplementAgent and workflow.

This module contains integration tests that verify TasksAgent works correctly
in coordination with ImplementAgent and the overall workflow system.
"""

import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from autogen_framework.agents.tasks_agent import TasksAgent
from autogen_framework.agent_manager import AgentManager
from autogen_framework.models import LLMConfig


class TestTasksAgentWorkflowIntegration:
    """Integration tests for TasksAgent in workflow coordination."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    # real_llm_config fixture is provided by tests/integration/conftest.py
    # It loads configuration from .env.integration file for secure testing
    
    @pytest.fixture
    def tasks_agent(self, real_llm_config):
        """Create a TasksAgent instance for testing."""
        return TasksAgent(real_llm_config)
    
    @pytest.fixture
    def work_directory(self, temp_workspace):
        """Create a work directory with test files."""
        work_dir = Path(temp_workspace) / "test_project"
        work_dir.mkdir(exist_ok=True)
        
        # Create requirements.md
        requirements_content = """# Requirements Document

## Introduction
This is a test project for integration testing.

## Requirements

### Requirement 1
**User Story:** As a developer, I want a simple calculator, so that I can perform basic math operations.

#### Acceptance Criteria
1. WHEN the user inputs two numbers THEN the system SHALL accept the input
2. WHEN the user selects an operation THEN the system SHALL perform the calculation
3. WHEN the calculation is complete THEN the system SHALL display the result
"""
        (work_dir / "requirements.md").write_text(requirements_content)
        
        # Create design.md
        design_content = """# Design Document

## Overview
Simple calculator application with basic arithmetic operations.

## Architecture
- Calculator class with methods for add, subtract, multiply, divide
- Input validation for numeric values
- Error handling for division by zero

## Components
### Calculator Class
- Methods: add(), subtract(), multiply(), divide()
- Input validation
- Error handling

## Implementation Notes
- Use Python for implementation
- Include comprehensive unit tests
- Handle edge cases (division by zero, invalid input)
"""
        (work_dir / "design.md").write_text(design_content)
        
        return work_dir
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tasks_agent_generates_real_task_list(self, tasks_agent, work_directory):
        """Test that TasksAgent generates a real task list from design and requirements."""
        # Mock the LLM response for task generation
        mock_task_content = """# Implementation Plan

- [ ] 1. Create Calculator class structure
  - Create calculator.py file
  - Define Calculator class with basic structure
  - Add docstrings and type hints
  - Requirements: 1.1, 1.2

- [ ] 2. Implement basic arithmetic operations
  - Implement add() method
  - Implement subtract() method  
  - Implement multiply() method
  - Implement divide() method with zero-division handling
  - Requirements: 1.2, 1.3

- [ ] 3. Add input validation
  - Validate numeric inputs
  - Handle invalid input types
  - Return appropriate error messages
  - Requirements: 1.1, 1.3

- [ ] 4. Create comprehensive unit tests
  - Test all arithmetic operations
  - Test input validation
  - Test error handling scenarios
  - Requirements: 1.1, 1.2, 1.3

- [ ] 5. Create main application interface
  - Create main.py file
  - Implement user input handling
  - Display results to user
  - Requirements: 1.1, 1.3
"""
        
        with patch.object(tasks_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_task_content
            
            # Generate task list
            tasks_path = await tasks_agent.generate_task_list(
                str(work_directory / "design.md"),
                str(work_directory / "requirements.md"),
                str(work_directory)
            )
            
            # Verify tasks file was created
            assert Path(tasks_path).exists()
            assert tasks_path == str(work_directory / "tasks.md")
            
            # Verify task content
            with open(tasks_path, 'r') as f:
                content = f.read()
            
            assert "# Implementation Plan" in content
            assert "- [ ] 1. Create Calculator class structure" in content
            assert "- [ ] 2. Implement basic arithmetic operations" in content
            assert "Requirements: 1.1, 1.2" in content
            
            # Verify tasks were parsed correctly
            assert len(tasks_agent.current_tasks) == 5
            assert tasks_agent.current_tasks[0].title == "1. Create Calculator class structure"
            assert "1.1" in tasks_agent.current_tasks[0].requirements_ref
            assert "1.2" in tasks_agent.current_tasks[0].requirements_ref
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tasks_agent_process_task_integration(self, tasks_agent, work_directory):
        """Test TasksAgent process_task method with real file operations."""
        # Mock the LLM response
        mock_task_content = """# Implementation Plan

- [ ] Task 1: Setup project structure
  - Create main directory
  - Initialize configuration
  - Requirements: 1.1

- [ ] Task 2: Implement core functionality  
  - Create core modules
  - Add business logic
  - Requirements: 1.2, 1.3
"""
        
        with patch.object(tasks_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_task_content
            
            # Process task generation request
            task_input = {
                "task_type": "generate_task_list",
                "design_path": str(work_directory / "design.md"),
                "requirements_path": str(work_directory / "requirements.md"),
                "work_dir": str(work_directory)
            }
            
            result = await tasks_agent.process_task(task_input)
            
            # Verify result
            assert result["success"] is True
            assert result["task_count"] == 2
            assert result["work_directory"] == str(work_directory)
            assert Path(result["tasks_file"]).exists()
            
            # Verify file content
            with open(result["tasks_file"], 'r') as f:
                content = f.read()
            assert "Task 1: Setup project structure" in content
            assert "Task 2: Implement core functionality" in content
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tasks_agent_revision_integration(self, tasks_agent, work_directory):
        """Test TasksAgent revision functionality with real file operations."""
        # First create initial tasks
        initial_tasks = """# Implementation Plan

- [ ] 1. Create basic structure
  - Setup files
  - Requirements: 1.1

- [ ] 2. Add functionality
  - Implement features
  - Requirements: 1.2
"""
        
        tasks_path = work_directory / "tasks.md"
        with open(tasks_path, 'w') as f:
            f.write(initial_tasks)
        
        # Mock the revision response
        revised_tasks = """# Implementation Plan

- [ ] 1. Create basic structure
  - Setup files
  - Add configuration
  - Requirements: 1.1

- [ ] 2. Add functionality
  - Implement features
  - Add error handling
  - Requirements: 1.2

- [ ] 3. Add comprehensive testing
  - Unit tests
  - Integration tests
  - Requirements: 1.1, 1.2, 1.3
"""
        
        with patch.object(tasks_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = revised_tasks
            
            # Process revision request
            revision_input = {
                "task_type": "revision",
                "revision_feedback": "Please add more detailed testing tasks",
                "work_directory": str(work_directory),
                "current_result": {}
            }
            
            result = await tasks_agent.process_task(revision_input)
            
            # Verify revision was successful
            assert result["success"] is True
            assert result["revision_applied"] is True
            assert result["work_directory"] == str(work_directory)
            
            # Verify file was updated
            with open(tasks_path, 'r') as f:
                content = f.read()
            assert "3. Add comprehensive testing" in content
            assert "Unit tests" in content
            assert "Integration tests" in content
    
    @pytest.mark.integration
    def test_tasks_agent_capabilities_integration(self, tasks_agent):
        """Test TasksAgent capabilities reporting for integration."""
        capabilities = tasks_agent.get_agent_capabilities()
        
        # Verify expected capabilities
        expected_capabilities = [
            "Generate task lists from design documents",
            "Decompose technical designs into implementation plans",
            "Create structured task lists with requirements references",
            "Format tasks in markdown checkbox format",
            "Maintain context access to requirements.md and design.md",
            "Ensure tasks build incrementally on each other",
            "Handle task list revisions based on user feedback"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
        
        assert len(capabilities) == len(expected_capabilities)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tasks_agent_with_agent_manager_coordination(self, temp_workspace, real_llm_config):
        """Test TasksAgent coordination through AgentManager."""
        # Create AgentManager
        agent_manager = AgentManager(temp_workspace)
        
        # Mock all agents to avoid real initialization
        with patch('autogen_framework.agent_manager.PlanAgent') as mock_plan, \
             patch('autogen_framework.agent_manager.DesignAgent') as mock_design, \
             patch('autogen_framework.agent_manager.TasksAgent') as mock_tasks, \
             patch('autogen_framework.agent_manager.ImplementAgent') as mock_implement:
            
            # Create mock instances
            mock_plan_instance = Mock()
            mock_plan_instance.initialize_autogen_agent.return_value = True
            mock_plan.return_value = mock_plan_instance
            
            mock_design_instance = Mock()
            mock_design_instance.initialize_autogen_agent.return_value = True
            mock_design.return_value = mock_design_instance
            
            mock_tasks_instance = Mock()
            mock_tasks_instance.initialize_autogen_agent.return_value = True
            mock_tasks_instance.process_task = AsyncMock(return_value={
                "success": True,
                "tasks_file": "/test/tasks.md",
                "task_count": 3
            })
            mock_tasks.return_value = mock_tasks_instance
            
            mock_implement_instance = Mock()
            mock_implement_instance.initialize_autogen_agent.return_value = True
            mock_implement.return_value = mock_implement_instance
            
            # Mock memory manager
            agent_manager.memory_manager.load_memory = Mock(return_value={})
            
            # Setup agents
            setup_result = agent_manager.setup_agents(real_llm_config)
            assert setup_result is True
            
            # Test task generation coordination
            context = {
                "task_type": "generate_task_list",
                "design_path": "/test/design.md",
                "requirements_path": "/test/requirements.md",
                "work_dir": "/test/work"
            }
            
            result = await agent_manager.coordinate_agents("task_generation", context)
            
            # Verify TasksAgent was called
            assert result["success"] is True
            assert result["tasks_file"] == "/test/tasks.md"
            assert result["task_count"] == 3
            
            # Verify TasksAgent process_task was called with correct context
            mock_tasks_instance.process_task.assert_called_once_with(context)
    
    @pytest.mark.integration
    def test_tasks_agent_file_operations_error_handling(self, tasks_agent, temp_workspace):
        """Test TasksAgent error handling with real file operations."""
        # Test reading non-existent file
        with pytest.raises(FileNotFoundError):
            asyncio.run(tasks_agent._read_file_content("/nonexistent/file.md"))
        
        # Test writing to invalid path
        with pytest.raises(IOError):
            asyncio.run(tasks_agent._write_file_content("/invalid/path/file.md", "content"))
        
        # Test successful file operations
        test_file = Path(temp_workspace) / "test.md"
        test_content = "# Test Content\nThis is a test file."
        
        # Write file
        asyncio.run(tasks_agent._write_file_content(str(test_file), test_content))
        assert test_file.exists()
        
        # Read file
        read_content = asyncio.run(tasks_agent._read_file_content(str(test_file)))
        assert read_content == test_content
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tasks_agent_context_access_integration(self, tasks_agent, work_directory):
        """Test TasksAgent context access to requirements and design documents."""
        # Mock LLM response to verify context was passed
        def mock_generate_response(prompt, context=None):
            # Verify context contains the expected documents
            if context:
                assert "design_document" in context
                assert "requirements_document" in context
                assert "Calculator class" in context["design_document"]
                assert "simple calculator" in context["requirements_document"]
            
            return """# Implementation Plan
- [ ] 1. Test task
  - Test step
  - Requirements: 1.1
"""
        
        with patch.object(tasks_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = mock_generate_response
            
            # Generate task list
            tasks_path = await tasks_agent.generate_task_list(
                str(work_directory / "design.md"),
                str(work_directory / "requirements.md"),
                str(work_directory)
            )
            
            # Verify the mock was called (which verifies context was passed correctly)
            mock_generate.assert_called_once()
            
            # Verify task file was created
            assert Path(tasks_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])