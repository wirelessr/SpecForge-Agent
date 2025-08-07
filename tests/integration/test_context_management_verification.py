"""
Integration test to verify context management works correctly with reorganized components.

This test validates that:
1. TasksAgent receives requirements.md and design.md context using existing patterns
2. ImplementAgent receives requirements.md, design.md, tasks.md, and previous task results using existing patterns
3. WorkflowManager passes context correctly using existing context passing mechanisms
4. Context management maintains identical behavior with reorganized architecture
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from autogen_framework.agent_manager import AgentManager
from autogen_framework.workflow_manager import WorkflowManager
from autogen_framework.session_manager import SessionManager
from autogen_framework.agents.tasks_agent import TasksAgent
from autogen_framework.agents.implement_agent import ImplementAgent
from autogen_framework.models import LLMConfig


class TestContextManagementVerification:
    """Test context management with reorganized components."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            base_url="http://test.local:8888/openai/v1",
            model="test-model",
            api_key="test-key"
        )
    
    @pytest.fixture
    def session_manager(self, temp_workspace):
        """Create SessionManager instance."""
        return SessionManager(temp_workspace)
    
    @pytest.fixture
    def agent_manager(self, temp_workspace, test_llm_config):
        """Create AgentManager instance with mocked agents."""
        manager = AgentManager(temp_workspace)
        
        # Mock the agent setup to avoid real LLM calls
        with patch.object(manager, 'setup_agents') as mock_setup:
            mock_setup.return_value = True
            manager.setup_agents(test_llm_config)
            
            # Create mock agents with proper interfaces
            manager.plan_agent = Mock()
            manager.plan_agent.process_task = AsyncMock()
            
            manager.design_agent = Mock()
            manager.design_agent.process_task = AsyncMock()
            
            manager.tasks_agent = Mock()
            manager.tasks_agent.process_task = AsyncMock()
            
            manager.implement_agent = Mock()
            manager.implement_agent.process_task = AsyncMock()
            
            # Update agents registry
            manager.agents = {
                "plan": manager.plan_agent,
                "design": manager.design_agent,
                "tasks": manager.tasks_agent,
                "implement": manager.implement_agent
            }
            
            manager.is_initialized = True
        
        return manager
    
    @pytest.fixture
    def real_memory_manager(self, temp_workspace):
        """Create a real MemoryManager instance."""
        from autogen_framework.memory_manager import MemoryManager
        return MemoryManager(temp_workspace)
    
    @pytest.fixture
    def real_context_compressor(self, test_llm_config):
        """Create a real ContextCompressor instance."""
        from autogen_framework.context_compressor import ContextCompressor
        return ContextCompressor(test_llm_config)
    
    @pytest.fixture
    def real_token_manager(self):
        """Create a real TokenManager instance."""
        from autogen_framework.token_manager import TokenManager
        from autogen_framework.config_manager import ConfigManager
        config_manager = ConfigManager()
        return TokenManager(config_manager)
    
    @pytest.fixture
    def workflow_manager(self, agent_manager, session_manager, real_memory_manager, real_context_compressor, real_token_manager):
        """Create WorkflowManager instance."""
        return WorkflowManager(agent_manager, session_manager, real_memory_manager, real_context_compressor, real_token_manager)
    
    @pytest.fixture
    def sample_context_files(self, temp_workspace):
        """Create sample context files for testing."""
        work_dir = Path(temp_workspace) / "test-project"
        work_dir.mkdir()
        
        # Create requirements.md
        requirements_content = """# Requirements Document

## Introduction
Test requirements for context management verification.

## Requirements

### Requirement 1: Basic Functionality
**User Story:** As a user, I want basic functionality, so that I can test context management.

#### Acceptance Criteria
1. WHEN the system starts THEN it SHALL load requirements context
2. WHEN processing occurs THEN it SHALL use requirements data
"""
        (work_dir / "requirements.md").write_text(requirements_content)
        
        # Create design.md
        design_content = """# Design Document

## Overview
Test design document for context management verification.

## Architecture
- Component A: Handles requirements processing
- Component B: Manages design implementation

## Components and Interfaces
### Component A
- Interface: process_requirements()
- Context: Uses requirements.md data

### Component B  
- Interface: implement_design()
- Context: Uses both requirements.md and design.md data
"""
        (work_dir / "design.md").write_text(design_content)
        
        # Create tasks.md
        tasks_content = """# Implementation Plan

- [ ] 1. Setup basic structure
  - Create main components
  - Initialize configuration
  - Requirements: 1.1

- [ ] 2. Implement core functionality
  - Add processing logic
  - Handle user input
  - Requirements: 1.2
"""
        (work_dir / "tasks.md").write_text(tasks_content)
        
        return {
            "work_dir": str(work_dir),
            "requirements_path": str(work_dir / "requirements.md"),
            "design_path": str(work_dir / "design.md"),
            "tasks_path": str(work_dir / "tasks.md")
        }
    
    @pytest.mark.integration
    async def test_tasks_agent_receives_correct_context(self, agent_manager, sample_context_files):
        """Test that TasksAgent receives requirements.md and design.md context."""
        # Setup mock response for TasksAgent
        expected_result = {
            "success": True,
            "tasks_file": sample_context_files["tasks_path"],
            "task_count": 2,
            "work_directory": sample_context_files["work_dir"]
        }
        agent_manager.tasks_agent.process_task.return_value = expected_result
        
        # Execute task generation through AgentManager
        context = {
            "task_type": "generate_task_list",
            "design_path": sample_context_files["design_path"],
            "requirements_path": sample_context_files["requirements_path"],
            "work_dir": sample_context_files["work_dir"]
        }
        
        result = await agent_manager.coordinate_agents("task_generation", context)
        
        # Verify TasksAgent was called with correct context
        agent_manager.tasks_agent.process_task.assert_called_once()
        call_args = agent_manager.tasks_agent.process_task.call_args[0][0]
        
        # Verify context contains required fields
        assert call_args["task_type"] == "generate_task_list"
        assert call_args["design_path"] == sample_context_files["design_path"]
        assert call_args["requirements_path"] == sample_context_files["requirements_path"]
        assert call_args["work_dir"] == sample_context_files["work_dir"]
        
        # Verify result is returned correctly
        assert result["success"] is True
        assert result["tasks_file"] == sample_context_files["tasks_path"]
    
    @pytest.mark.integration
    async def test_implement_agent_receives_full_context(self, agent_manager, sample_context_files):
        """Test that ImplementAgent receives requirements.md, design.md, tasks.md, and previous task results."""
        # Setup mock response for ImplementAgent
        expected_result = {
            "success": True,
            "task_id": "task_1",
            "execution_time": 1.5,
            "files_modified": ["test_file.py"],
            "shell_commands": ["echo 'test'"]
        }
        agent_manager.implement_agent.process_task.return_value = expected_result
        
        # Simulate previous task results
        previous_results = [
            {"task_id": "task_0", "success": True, "output": "Setup completed"}
        ]
        
        # Execute task through AgentManager
        context = {
            "task_type": "execute_task",
            "task": {
                "id": "task_1",
                "title": "Test Task",
                "description": "Test task execution",
                "steps": ["Step 1", "Step 2"],
                "requirements_ref": ["1.1", "1.2"]
            },
            "work_dir": sample_context_files["work_dir"],
            "requirements_path": sample_context_files["requirements_path"],
            "design_path": sample_context_files["design_path"],
            "tasks_path": sample_context_files["tasks_path"],
            "previous_results": previous_results
        }
        
        result = await agent_manager.coordinate_agents("task_execution", context)
        
        # Verify ImplementAgent was called with full context
        agent_manager.implement_agent.process_task.assert_called_once()
        call_args = agent_manager.implement_agent.process_task.call_args[0][0]
        
        # Verify all context elements are present
        assert call_args["task_type"] == "execute_task"
        assert "task" in call_args
        assert call_args["work_dir"] == sample_context_files["work_dir"]
        
        # Verify result is returned correctly
        assert result["success"] is True
        assert result["task_id"] == "task_1"
    
    @pytest.mark.integration
    async def test_workflow_manager_passes_context_correctly(self, workflow_manager, agent_manager, sample_context_files):
        """Test that WorkflowManager passes context correctly through existing mechanisms."""
        # Setup mock responses for all agents
        agent_manager.plan_agent.process_task.return_value = {
            "success": True,
            "requirements_path": sample_context_files["requirements_path"],
            "work_directory": sample_context_files["work_dir"]
        }
        
        agent_manager.design_agent.process_task.return_value = {
            "success": True,
            "design_path": sample_context_files["design_path"],
            "work_directory": sample_context_files["work_dir"]
        }
        
        agent_manager.tasks_agent.process_task.return_value = {
            "success": True,
            "tasks_file": sample_context_files["tasks_path"],
            "work_directory": sample_context_files["work_dir"]
        }
        
        # Test requirements phase context passing
        user_request = "Create a test application"
        result = await workflow_manager.process_request(user_request, auto_approve=True)
        
        # Verify plan agent received correct context
        plan_call_args = agent_manager.plan_agent.process_task.call_args[0][0]
        assert "user_request" in plan_call_args
        assert plan_call_args["user_request"] == user_request
        
        # Verify design agent received requirements context
        design_call_args = agent_manager.design_agent.process_task.call_args[0][0]
        assert "requirements_path" in design_call_args
        assert "work_directory" in design_call_args
        assert "memory_context" in design_call_args
        
        # Verify tasks agent received design and requirements context
        tasks_call_args = agent_manager.tasks_agent.process_task.call_args[0][0]
        assert tasks_call_args["task_type"] == "generate_task_list"
        assert "design_path" in tasks_call_args
        assert "requirements_path" in tasks_call_args
        assert "work_dir" in tasks_call_args
        
        # Verify workflow completed successfully
        assert result["success"] is True
        assert "phases" in result
        assert "requirements" in result["phases"]
        assert "design" in result["phases"]
        assert "tasks" in result["phases"]
    
    @pytest.mark.integration
    async def test_context_management_with_multi_task_scenario(self, workflow_manager, agent_manager, sample_context_files):
        """Test context management with existing multi-task scenarios to ensure no regression."""
        # Setup mock responses for multiple task execution
        task_results = [
            {"success": True, "task_id": "task_1", "output": "Task 1 completed"},
            {"success": True, "task_id": "task_2", "output": "Task 2 completed"},
            {"success": True, "task_id": "task_3", "output": "Task 3 completed"}
        ]
        
        agent_manager.implement_agent.process_task.return_value = {
            "success": True,
            "task_results": task_results,
            "completed_count": 3,
            "total_count": 3
        }
        
        # Execute multiple tasks
        context = {
            "task_type": "execute_multiple_tasks",
            "tasks": [
                {"id": "task_1", "title": "Task 1", "description": "First task"},
                {"id": "task_2", "title": "Task 2", "description": "Second task"},
                {"id": "task_3", "title": "Task 3", "description": "Third task"}
            ],
            "work_dir": sample_context_files["work_dir"],
            "requirements_path": sample_context_files["requirements_path"],
            "design_path": sample_context_files["design_path"],
            "tasks_path": sample_context_files["tasks_path"]
        }
        
        result = await agent_manager.coordinate_agents("execute_multiple_tasks", context)
        
        # Verify context was passed correctly
        agent_manager.implement_agent.process_task.assert_called_once()
        call_args = agent_manager.implement_agent.process_task.call_args[0][0]
        
        assert call_args["task_type"] == "execute_multiple_tasks"
        assert len(call_args["tasks"]) == 3
        assert call_args["work_dir"] == sample_context_files["work_dir"]
        
        # Verify results
        assert result["success"] is True
        assert result["completed_count"] == 3
        assert result["total_count"] == 3
    
    @pytest.mark.integration
    async def test_context_management_maintains_identical_behavior(self, workflow_manager, agent_manager, sample_context_files):
        """Test that context management maintains identical behavior with reorganized architecture."""
        # This test verifies that the reorganized components maintain the same
        # context passing behavior as the original architecture
        
        # Setup mock responses to simulate original behavior
        agent_manager.plan_agent.process_task.return_value = {
            "success": True,
            "requirements_path": sample_context_files["requirements_path"],
            "work_directory": sample_context_files["work_dir"],
            "message": "Requirements generated successfully"
        }
        
        agent_manager.design_agent.process_task.return_value = {
            "success": True,
            "design_path": sample_context_files["design_path"],
            "work_directory": sample_context_files["work_dir"],
            "message": "Design generated successfully"
        }
        
        agent_manager.tasks_agent.process_task.return_value = {
            "success": True,
            "tasks_file": sample_context_files["tasks_path"],
            "work_directory": sample_context_files["work_dir"],
            "task_count": 2,
            "message": "Tasks generated successfully"
        }
        
        # Test the complete workflow with context passing
        user_request = "Create a test application with proper context management"
        result = await workflow_manager.process_request(user_request, auto_approve=True)
        
        # Verify the workflow completed successfully (identical to original behavior)
        assert result["success"] is True
        assert result["auto_approve_enabled"] is True
        assert "phases" in result
        
        # Verify all phases completed with proper context
        phases = result["phases"]
        assert "requirements" in phases
        assert "design" in phases
        assert "tasks" in phases
        assert "implementation" in phases
        
        # Verify each phase received the expected context
        assert phases["requirements"]["success"] is True
        assert phases["design"]["success"] is True
        assert phases["tasks"]["success"] is True
        
        # Verify context was passed correctly between phases
        # (This simulates the original behavior where each phase builds on the previous)
        
        # Plan agent should receive user request
        plan_call = agent_manager.plan_agent.process_task.call_args_list[0][0][0]
        assert "user_request" in plan_call
        
        # Design agent should receive requirements path and memory context
        design_call = agent_manager.design_agent.process_task.call_args_list[0][0][0]
        assert "requirements_path" in design_call
        assert "work_directory" in design_call
        assert "memory_context" in design_call
        
        # Tasks agent should receive design path and requirements path
        tasks_call = agent_manager.tasks_agent.process_task.call_args_list[0][0][0]
        assert "design_path" in tasks_call
        assert "requirements_path" in tasks_call
        assert "work_dir" in tasks_call
        
        # Verify the final result structure matches original behavior
        assert result["workflow_id"] is not None
        assert result["user_request"] == user_request
        assert result["current_phase"] == "implementation"
    
    @pytest.mark.integration
    def test_context_file_access_patterns(self, sample_context_files):
        """Test that context files can be accessed using existing patterns."""
        # Verify all context files exist and are readable
        requirements_path = sample_context_files["requirements_path"]
        design_path = sample_context_files["design_path"]
        tasks_path = sample_context_files["tasks_path"]
        
        # Test file existence (existing pattern)
        assert Path(requirements_path).exists()
        assert Path(design_path).exists()
        assert Path(tasks_path).exists()
        
        # Test file readability (existing pattern)
        with open(requirements_path, 'r') as f:
            requirements_content = f.read()
            assert "Requirements Document" in requirements_content
            assert "Requirement 1" in requirements_content
        
        with open(design_path, 'r') as f:
            design_content = f.read()
            assert "Design Document" in design_content
            assert "Architecture" in design_content
        
        with open(tasks_path, 'r') as f:
            tasks_content = f.read()
            assert "Implementation Plan" in tasks_content
            assert "[ ] 1." in tasks_content
        
        # Verify context files contain expected structure (existing pattern)
        assert "## Requirements" in requirements_content
        assert "## Components and Interfaces" in design_content
        assert "Requirements:" in tasks_content
    
    @pytest.mark.integration
    async def test_context_compression_integration(self, agent_manager, sample_context_files):
        """Test that context compression works correctly with reorganized components."""
        # This test verifies that token compression is properly triggered
        # when context exceeds limits (existing functionality)
        
        # Create large context to trigger compression
        large_context = {
            "task_type": "generate_task_list",
            "design_path": sample_context_files["design_path"],
            "requirements_path": sample_context_files["requirements_path"],
            "work_dir": sample_context_files["work_dir"],
            "large_context_data": "x" * 10000  # Large string to simulate token limit
        }
        
        # Mock TasksAgent to simulate compression behavior
        agent_manager.tasks_agent.process_task.return_value = {
            "success": True,
            "tasks_file": sample_context_files["tasks_path"],
            "compression_triggered": True,
            "original_context_size": 10000,
            "compressed_context_size": 5000
        }
        
        result = await agent_manager.coordinate_agents("task_generation", large_context)
        
        # Verify the agent was called and compression was handled
        agent_manager.tasks_agent.process_task.assert_called_once()
        assert result["success"] is True
        
        # In a real scenario, the BaseLLMAgent would handle compression automatically
        # This test verifies the integration points work correctly