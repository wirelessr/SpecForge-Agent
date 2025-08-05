"""
Unit tests for TasksAgent.

This module contains comprehensive unit tests for the TasksAgent class,
focusing on task list generation functionality extracted from ImplementAgent.
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, AsyncMock, patch

from autogen_framework.agents.tasks_agent import TasksAgent
from autogen_framework.models import LLMConfig, TaskDefinition


class TestTasksAgent:
    """Test suite for TasksAgent basic functionality."""
    
    @pytest.fixture
    def tasks_agent(self, test_llm_config):
        """Create a TasksAgent instance for testing."""
        return TasksAgent(
            llm_config=test_llm_config,
            memory_manager=Mock()
        )
    
    @pytest.fixture
    def temp_work_dir(self):
        """Create a temporary work directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_initialization(self, test_llm_config):
        """Test TasksAgent initialization."""
        agent = TasksAgent(llm_config=test_llm_config)
        
        assert agent.name == "TasksAgent"
        assert agent.llm_config == test_llm_config
        assert agent.current_work_directory is None
        assert agent.current_tasks == []
        assert "Task generation agent" in agent.description
    
    def test_get_agent_capabilities(self, tasks_agent):
        """Test that TasksAgent returns correct capabilities."""
        capabilities = tasks_agent.get_agent_capabilities()
        
        expected_capabilities = [
            "Generate task lists from design documents",
            "Decompose technical designs into implementation plans",
            "Create structured task lists with requirements references",
            "Format tasks in markdown checkbox format",
            "Maintain context access to requirements.md and design.md",
            "Ensure tasks build incrementally on each other",
            "Handle task list revisions based on user feedback"
        ]
        
        assert capabilities == expected_capabilities
    
    @pytest.mark.asyncio
    async def test_process_task_generate_task_list(self, tasks_agent, temp_work_dir):
        """Test processing task list generation request."""
        design_path = os.path.join(temp_work_dir, "design.md")
        requirements_path = os.path.join(temp_work_dir, "requirements.md")
        
        # Mock the generate_task_list method
        tasks_agent.generate_task_list = AsyncMock()
        tasks_agent.generate_task_list.return_value = os.path.join(temp_work_dir, "tasks.md")
        tasks_agent.current_tasks = [Mock(), Mock()]  # Mock 2 tasks
        
        task_input = {
            "task_type": "generate_task_list",
            "design_path": design_path,
            "requirements_path": requirements_path,
            "work_dir": temp_work_dir
        }
        
        result = await tasks_agent.process_task(task_input)
        
        assert result["success"] is True
        assert result["task_count"] == 2
        assert result["work_directory"] == temp_work_dir
        tasks_agent.generate_task_list.assert_called_once_with(
            design_path, requirements_path, temp_work_dir
        )
    
    @pytest.mark.asyncio
    async def test_process_task_revision(self, tasks_agent, temp_work_dir):
        """Test processing task revision request."""
        # Mock the revision handler
        tasks_agent._handle_revision_task = AsyncMock()
        tasks_agent._handle_revision_task.return_value = {
            "success": True,
            "revision_applied": True
        }
        
        task_input = {
            "task_type": "revision",
            "revision_feedback": "Add more tests",
            "work_directory": temp_work_dir
        }
        
        result = await tasks_agent.process_task(task_input)
        
        assert result["success"] is True
        assert result["revision_applied"] is True
        tasks_agent._handle_revision_task.assert_called_once_with(task_input)
    
    @pytest.mark.asyncio
    async def test_process_task_unknown_type(self, tasks_agent):
        """Test processing unknown task type."""
        task_input = {"task_type": "unknown_task"}
        
        with pytest.raises(ValueError, match="Unknown task type for TasksAgent"):
            await tasks_agent.process_task(task_input)


class TestTasksAgentTaskGeneration:
    """Test suite for task list generation functionality."""
    
    @pytest.fixture
    def tasks_agent(self, test_llm_config):
        """Create a TasksAgent instance for testing."""
        return TasksAgent(llm_config=test_llm_config)
    
    @pytest.fixture
    def temp_work_dir(self):
        """Create a temporary work directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_generate_task_list_success(self, tasks_agent, temp_work_dir):
        """
        Test successful task list generation.
        
        This test verifies the complete task list generation workflow:
        1. Reading design and requirements files
        2. Generating task content via LLM
        3. Writing tasks to file
        4. Updating agent state
        """
        design_path = os.path.join(temp_work_dir, "design.md")
        requirements_path = os.path.join(temp_work_dir, "requirements.md")
        
        # Create test files
        with open(design_path, 'w') as f:
            f.write("# Design\nDesign content")
        with open(requirements_path, 'w') as f:
            f.write("# Requirements\nRequirements content")
        
        # Mock LLM response with realistic task format
        mock_task_content = """# Tasks
- [ ] Task 1
  - Step 1
  - Step 2
  - Requirements: 1.1, 2.2
- [ ] Task 2
  - Step A
  - Requirements: 3.1"""
        
        # Mock the LLM response generation
        tasks_agent.generate_response = AsyncMock()
        tasks_agent.generate_response.return_value = mock_task_content
        
        # Execute the task list generation
        result = await tasks_agent.generate_task_list(
            design_path, requirements_path, temp_work_dir
        )
        
        # Verify the expected results
        expected_tasks_path = os.path.join(temp_work_dir, "tasks.md")
        assert result == expected_tasks_path
        assert tasks_agent.current_work_directory == temp_work_dir
        assert len(tasks_agent.current_tasks) == 2
        
        # Verify the tasks.md file was created
        assert os.path.exists(expected_tasks_path)
        with open(expected_tasks_path, 'r') as f:
            content = f.read()
            assert content == mock_task_content
        
        # Verify LLM was called with correct prompt
        tasks_agent.generate_response.assert_called_once()
        call_args = tasks_agent.generate_response.call_args
        prompt = call_args[0][0]
        assert "Design content" in prompt
        assert "Requirements content" in prompt
        assert "markdown checkbox format" in prompt
    
    @pytest.mark.asyncio
    async def test_generate_task_list_file_read_error(self, tasks_agent, temp_work_dir):
        """Test task list generation with file read error."""
        design_path = "nonexistent_design.md"
        requirements_path = "nonexistent_requirements.md"
        
        with pytest.raises(FileNotFoundError):
            await tasks_agent.generate_task_list(
                design_path, requirements_path, temp_work_dir
            )
    
    @pytest.mark.asyncio
    async def test_generate_task_list_llm_error(self, tasks_agent, temp_work_dir):
        """Test task list generation with LLM error."""
        design_path = os.path.join(temp_work_dir, "design.md")
        requirements_path = os.path.join(temp_work_dir, "requirements.md")
        
        # Create test files
        with open(design_path, 'w') as f:
            f.write("# Design\nDesign content")
        with open(requirements_path, 'w') as f:
            f.write("# Requirements\nRequirements content")
        
        # Mock LLM to raise an exception
        tasks_agent.generate_response = AsyncMock()
        tasks_agent.generate_response.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception, match="LLM error"):
            await tasks_agent.generate_task_list(
                design_path, requirements_path, temp_work_dir
            )
    
    def test_parse_task_list(self, tasks_agent):
        """Test parsing task list content into TaskDefinition objects."""
        task_content = """# Implementation Tasks

- [ ] Task 1: Create base structure
  - Create directory
  - Initialize files
  - Requirements: 1.1, 2.3

- [ ] Task 2: Implement functionality
  - Write code
  - Add tests
  - Requirements: 2.1

- [x] Task 3: Completed task
  - This was done
  - Requirements: 3.1"""
        
        tasks = tasks_agent._parse_task_list(task_content)
        
        assert len(tasks) == 3
        
        # Check first task
        task1 = tasks[0]
        assert task1.id == "task_1"
        assert task1.title == "Task 1: Create base structure"
        assert task1.description == "Task 1: Create base structure"
        assert len(task1.steps) == 2
        assert "Create directory" in task1.steps
        assert "Initialize files" in task1.steps
        assert task1.requirements_ref == ["1.1", "2.3"]
        
        # Check second task
        task2 = tasks[1]
        assert task2.id == "task_2"
        assert task2.title == "Task 2: Implement functionality"
        assert len(task2.steps) == 2
        assert task2.requirements_ref == ["2.1"]
        
        # Check third task (completed)
        task3 = tasks[2]
        assert task3.id == "task_3"
        assert task3.title == "Task 3: Completed task"
        assert task3.requirements_ref == ["3.1"]
    
    def test_parse_task_list_empty(self, tasks_agent):
        """Test parsing empty task list."""
        tasks = tasks_agent._parse_task_list("")
        assert tasks == []
    
    def test_parse_task_list_no_requirements(self, tasks_agent):
        """Test parsing task list without requirements."""
        task_content = """- [ ] Simple task
  - Do something
  - Do something else"""
        
        tasks = tasks_agent._parse_task_list(task_content)
        
        assert len(tasks) == 1
        task = tasks[0]
        assert task.title == "Simple task"
        assert len(task.steps) == 2
        assert task.requirements_ref == []
    
    def test_build_task_generation_prompt(self, tasks_agent):
        """Test building task generation prompt."""
        design_content = "# Design\nSome design content"
        requirements_content = "# Requirements\nSome requirements"
        
        prompt = tasks_agent._build_task_generation_prompt(design_content, requirements_content)
        
        assert "Based on the following design document and requirements" in prompt
        assert design_content in prompt
        assert requirements_content in prompt
        assert "markdown checkbox format" in prompt
        assert "Requirements: X.Y, Z.A" in prompt
        assert "shell commands" in prompt


class TestTasksAgentFileOperations:
    """Test suite for file operations in TasksAgent."""
    
    @pytest.fixture
    def tasks_agent(self, test_llm_config):
        """Create a TasksAgent instance for testing."""
        return TasksAgent(llm_config=test_llm_config)
    
    @pytest.fixture
    def temp_work_dir(self):
        """Create a temporary work directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_read_file_content_success(self, tasks_agent, temp_work_dir):
        """Test successful file reading."""
        test_file = os.path.join(temp_work_dir, "test.md")
        test_content = "# Test Content\nThis is a test file."
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        content = await tasks_agent._read_file_content(test_file)
        assert content == test_content
    
    @pytest.mark.asyncio
    async def test_read_file_content_not_found(self, tasks_agent):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            await tasks_agent._read_file_content("nonexistent.md")
    
    @pytest.mark.asyncio
    async def test_write_file_content_success(self, tasks_agent, temp_work_dir):
        """Test successful file writing."""
        test_file = os.path.join(temp_work_dir, "output.md")
        test_content = "# Generated Content\nThis is generated content."
        
        await tasks_agent._write_file_content(test_file, test_content)
        
        assert os.path.exists(test_file)
        with open(test_file, 'r') as f:
            content = f.read()
            assert content == test_content
    
    @pytest.mark.asyncio
    async def test_write_file_content_directory_not_exists(self, tasks_agent, temp_work_dir):
        """Test writing to non-existent directory."""
        test_file = os.path.join(temp_work_dir, "subdir", "output.md")
        test_content = "Test content"
        
        with pytest.raises(IOError):
            await tasks_agent._write_file_content(test_file, test_content)


class TestTasksAgentIntegration:
    """Integration tests for TasksAgent with real functionality."""
    
    @pytest.fixture
    def tasks_agent(self, test_llm_config):
        """Create a TasksAgent instance for testing."""
        return TasksAgent(llm_config=test_llm_config)
    
    @pytest.fixture
    def temp_work_dir(self):
        """Create a temporary work directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_full_task_generation_workflow(self, tasks_agent, temp_work_dir):
        """Test the complete task generation workflow."""
        # Create realistic design and requirements files
        design_path = os.path.join(temp_work_dir, "design.md")
        requirements_path = os.path.join(temp_work_dir, "requirements.md")
        
        design_content = """# Design Document

## Architecture
- Component A: Handles user input
- Component B: Processes data
- Component C: Generates output

## Implementation Details
- Use Python for backend
- Create REST API endpoints
- Implement data validation"""
        
        requirements_content = """# Requirements

## Requirement 1: User Input Handling
- 1.1: Accept user input via API
- 1.2: Validate input format

## Requirement 2: Data Processing
- 2.1: Process user data
- 2.2: Store results"""
        
        with open(design_path, 'w') as f:
            f.write(design_content)
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        # Mock realistic LLM response
        mock_task_content = """# Implementation Tasks

- [ ] Set up project structure
  - Create main directory
  - Initialize Python project
  - Set up virtual environment
  - Requirements: 1.1

- [ ] Implement Component A
  - Create input handler module
  - Add input validation
  - Write unit tests
  - Requirements: 1.1, 1.2

- [ ] Implement Component B
  - Create data processor
  - Add business logic
  - Requirements: 2.1

- [ ] Create REST API
  - Set up Flask application
  - Define API endpoints
  - Add error handling
  - Requirements: 1.1, 2.1, 2.2"""
        
        tasks_agent.generate_response = AsyncMock()
        tasks_agent.generate_response.return_value = mock_task_content
        
        # Execute task generation
        result = await tasks_agent.generate_task_list(
            design_path, requirements_path, temp_work_dir
        )
        
        # Verify results
        assert result == os.path.join(temp_work_dir, "tasks.md")
        assert len(tasks_agent.current_tasks) == 4
        
        # Verify task parsing
        task1 = tasks_agent.current_tasks[0]
        assert "Set up project structure" in task1.title
        assert len(task1.steps) == 3
        assert task1.requirements_ref == ["1.1"]
        
        task2 = tasks_agent.current_tasks[1]
        assert "Component A" in task2.title
        assert task2.requirements_ref == ["1.1", "1.2"]
        
        # Verify file was written correctly
        with open(result, 'r') as f:
            content = f.read()
            assert content == mock_task_content


class TestTasksAgentRevision:
    """Test suite for task revision functionality."""
    
    @pytest.fixture
    def tasks_agent(self, test_llm_config):
        """Create a TasksAgent instance for testing."""
        return TasksAgent(llm_config=test_llm_config)
    
    @pytest.fixture
    def temp_work_dir(self):
        """Create a temporary work directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_handle_revision_task_success(self, tasks_agent, temp_work_dir):
        """Test successful task revision."""
        # Create initial tasks.md file
        tasks_path = os.path.join(temp_work_dir, "tasks.md")
        initial_content = """# Tasks
- [ ] Task 1
  - Step 1
  - Requirements: 1.1
- [ ] Task 2
  - Step 2
  - Requirements: 2.1"""
        
        with open(tasks_path, 'w') as f:
            f.write(initial_content)
        
        # Mock LLM response for revision
        revised_content = """# Tasks
- [ ] Task 1
  - Step 1 (updated)
  - Requirements: 1.1
- [ ] Task 2
  - Step 2
  - Requirements: 2.1
- [ ] Task 3 (new)
  - New step
  - Requirements: 3.1"""
        
        tasks_agent._apply_tasks_revision = AsyncMock()
        tasks_agent._apply_tasks_revision.return_value = revised_content
        
        task_input = {
            "revision_feedback": "Add a new task and update task 1",
            "work_directory": temp_work_dir
        }
        
        result = await tasks_agent._handle_revision_task(task_input)
        
        assert result["success"] is True
        assert result["revision_applied"] is True
        assert result["work_directory"] == temp_work_dir
        assert result["tasks_path"] == tasks_path
        
        # Verify file was updated
        with open(tasks_path, 'r') as f:
            content = f.read()
            assert content == revised_content
        
        tasks_agent._apply_tasks_revision.assert_called_once_with(
            initial_content, "Add a new task and update task 1"
        )
    
    @pytest.mark.asyncio
    async def test_handle_revision_task_missing_feedback(self, tasks_agent, temp_work_dir):
        """Test revision task with missing feedback."""
        task_input = {
            "work_directory": temp_work_dir
        }
        
        result = await tasks_agent._handle_revision_task(task_input)
        
        assert result["success"] is False
        assert "revision_feedback is required" in result["error"]
    
    @pytest.mark.asyncio
    async def test_handle_revision_task_missing_directory(self, tasks_agent):
        """Test revision task with missing work directory."""
        task_input = {
            "revision_feedback": "Some feedback"
        }
        
        result = await tasks_agent._handle_revision_task(task_input)
        
        assert result["success"] is False
        assert "work_directory is required" in result["error"]
    
    @pytest.mark.asyncio
    async def test_handle_revision_task_missing_tasks_file(self, tasks_agent, temp_work_dir):
        """Test revision task when tasks.md doesn't exist."""
        task_input = {
            "revision_feedback": "Some feedback",
            "work_directory": temp_work_dir
        }
        
        result = await tasks_agent._handle_revision_task(task_input)
        
        assert result["success"] is False
        assert "tasks.md not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_apply_tasks_revision_success(self, tasks_agent):
        """Test successful task revision application."""
        current_tasks = """# Tasks
- [ ] Task 1
  - Step 1
  - Requirements: 1.1"""
        
        revision_feedback = "Add more detailed steps to Task 1"
        
        revised_content = """# Tasks
- [ ] Task 1
  - Step 1: Initialize project
  - Step 2: Set up configuration
  - Step 3: Create basic structure
  - Requirements: 1.1"""
        
        tasks_agent.generate_response = AsyncMock()
        tasks_agent.generate_response.return_value = revised_content
        
        result = await tasks_agent._apply_tasks_revision(current_tasks, revision_feedback)
        
        assert result == revised_content
        
        # Verify LLM was called with correct prompt
        tasks_agent.generate_response.assert_called_once()
        call_args = tasks_agent.generate_response.call_args[0][0]
        assert current_tasks in call_args
        assert revision_feedback in call_args
        assert "markdown checkbox format" in call_args
    
    @pytest.mark.asyncio
    async def test_apply_tasks_revision_with_markdown_cleanup(self, tasks_agent):
        """Test revision with markdown code block cleanup."""
        current_tasks = "# Tasks\n- [ ] Task 1"
        revision_feedback = "Update task"
        
        # Mock LLM response with markdown code blocks
        llm_response = """```markdown
# Tasks
- [ ] Task 1 (updated)
  - New step
  - Requirements: 1.1
```"""
        
        tasks_agent.generate_response = AsyncMock()
        tasks_agent.generate_response.return_value = llm_response
        
        result = await tasks_agent._apply_tasks_revision(current_tasks, revision_feedback)
        
        expected = """# Tasks
- [ ] Task 1 (updated)
  - New step
  - Requirements: 1.1"""
        
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_apply_tasks_revision_llm_error(self, tasks_agent):
        """Test revision when LLM fails."""
        current_tasks = "# Tasks\n- [ ] Task 1"
        revision_feedback = "Update task"
        
        tasks_agent.generate_response = AsyncMock()
        tasks_agent.generate_response.side_effect = Exception("LLM failed")
        
        result = await tasks_agent._apply_tasks_revision(current_tasks, revision_feedback)
        
        assert current_tasks in result
        assert "Revision attempted but failed" in result
        assert "LLM failed" in result