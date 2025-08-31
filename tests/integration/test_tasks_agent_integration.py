"""
Integration tests for TasksAgent integration with AgentManager.

This module contains integration tests that verify TasksAgent is properly
integrated with AgentManager and can be coordinated correctly.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from autogen_framework.agent_manager import AgentManager
from autogen_framework.models import LLMConfig


class TestTasksAgentIntegration:
    """Integration tests for TasksAgent with AgentManager."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def llm_config(self):
        """Create a test LLM configuration."""
        return LLMConfig(
            base_url="http://test.local:8888/openai/v1",
            model="test-model",
            api_key="sk-test123"
        )
    
    @pytest.fixture
    def agent_manager(self, temp_workspace):
        """Create an AgentManager instance for testing."""
        return AgentManager(temp_workspace)
    
    @pytest.mark.integration
    @patch('autogen_framework.agent_manager.TasksAgent')
    @patch('autogen_framework.agent_manager.ImplementAgent')
    @patch('autogen_framework.agent_manager.DesignAgent')
    @patch('autogen_framework.agent_manager.PlanAgent')
    def test_tasks_agent_initialization_in_agent_manager(self, mock_plan, mock_design, mock_tasks, mock_implement,
                                       agent_manager, real_llm_config):
        """Test that TasksAgent is properly initialized in AgentManager."""
        # Mock agent instances
        mock_plan_instance = Mock()
        mock_plan_instance.initialize_autogen_agent.return_value = True
        mock_plan.return_value = mock_plan_instance
        
        mock_design_instance = Mock()
        mock_design_instance.initialize_autogen_agent.return_value = True
        mock_design.return_value = mock_design_instance
        
        mock_tasks_instance = Mock()
        mock_tasks_instance.initialize_autogen_agent.return_value = True
        mock_tasks.return_value = mock_tasks_instance
        
        mock_implement_instance = Mock()
        mock_implement_instance.initialize_autogen_agent.return_value = True
        mock_implement.return_value = mock_implement_instance
        
        # Mock container and memory manager
        mock_container = Mock()
        mock_memory_manager = Mock()
        mock_memory_manager.load_memory = Mock(return_value={})
        mock_container.get_memory_manager = Mock(return_value=mock_memory_manager)
        agent_manager.container = mock_container
        
        # Setup agents
        result = agent_manager.setup_agents(real_llm_config)
        
        # Verify TasksAgent is properly initialized
        assert result is True
        assert agent_manager.is_initialized is True
        assert "tasks" in agent_manager.agents
        assert agent_manager.tasks_agent is not None
        
        # Verify TasksAgent was created with correct parameters
        mock_tasks.assert_called_once()
        mock_tasks_instance.initialize_autogen_agent.assert_called_once()
        
        # Verify TasksAgent is properly integrated in agent registry
        assert len(agent_manager.agents) == 4
        assert "tasks" in agent_manager.agents
        assert agent_manager.agents["tasks"] is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_task_generation_routing_to_tasks_agent(self, agent_manager, real_llm_config):
        """Test that task generation requests are routed to TasksAgent."""
        # Create mock TasksAgent that doesn't try to read files
        mock_tasks_agent = Mock()
        mock_tasks_agent.initialize_autogen_agent.return_value = True
        mock_tasks_agent.process_task = AsyncMock(return_value={
            "success": True,
            "tasks_file": "/test/tasks.md",
            "task_count": 5
        })
        
        # Create other mock agents
        mock_plan_agent = Mock()
        mock_plan_agent.initialize_autogen_agent.return_value = True
        
        mock_design_agent = Mock()
        mock_design_agent.initialize_autogen_agent.return_value = True
        
        mock_implement_agent = Mock()
        mock_implement_agent.initialize_autogen_agent.return_value = True
        
        # Manually set up agents to avoid real initialization
        agent_manager.agents = {
            "plan": mock_plan_agent,
            "design": mock_design_agent,
            "tasks": mock_tasks_agent,
            "implement": mock_implement_agent
        }
        agent_manager.tasks_agent = mock_tasks_agent
        agent_manager.is_initialized = True
        
        # Test task generation coordination
        context = {
            "task_type": "generate_task_list",
            "design_path": "/test/design.md",
            "requirements_path": "/test/requirements.md",
            "work_dir": "/test/work"
        }
        
        result = await agent_manager.coordinate_agents("task_generation", context)
        
        # Verify TasksAgent was called for task generation
        assert result["success"] is True
        assert result["tasks_file"] == "/test/tasks.md"
        assert result["task_count"] == 5
        
        # Verify TasksAgent process_task was called with correct context
        mock_tasks_agent.process_task.assert_called_once_with(context)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_workflow_uses_tasks_agent(self, agent_manager, real_llm_config):
        """Test that full workflow uses TasksAgent for task generation phase."""
        # Create mock agents that return success without real processing
        mock_plan_agent = Mock()
        mock_plan_agent.initialize_autogen_agent.return_value = True
        mock_plan_agent.process_task = AsyncMock(return_value={
            "success": True,
            "work_directory": "/test/work",
            "requirements_path": "/test/work/requirements.md"
        })
        
        mock_design_agent = Mock()
        mock_design_agent.initialize_autogen_agent.return_value = True
        mock_design_agent.process_task = AsyncMock(return_value={
            "success": True,
            "design_path": "/test/work/design.md"
        })
        
        mock_tasks_agent = Mock()
        mock_tasks_agent.initialize_autogen_agent.return_value = True
        mock_tasks_agent.process_task = AsyncMock(return_value={
            "success": True,
            "tasks_file": "/test/work/tasks.md",
            "task_count": 5
        })
        
        mock_implement_agent = Mock()
        mock_implement_agent.initialize_autogen_agent.return_value = True
        
        # Manually set up agents to avoid real initialization
        agent_manager.agents = {
            "plan": mock_plan_agent,
            "design": mock_design_agent,
            "tasks": mock_tasks_agent,
            "implement": mock_implement_agent
        }
        agent_manager.plan_agent = mock_plan_agent
        agent_manager.design_agent = mock_design_agent
        agent_manager.tasks_agent = mock_tasks_agent
        agent_manager.implement_agent = mock_implement_agent
        agent_manager.is_initialized = True
        
        # Test full workflow
        context = {"user_request": "Create a test application"}
        result = await agent_manager.coordinate_agents("full_workflow", context)
        
        # Verify workflow completed successfully
        assert result["success"] is True
        assert result["current_phase"] == "completed"
        assert "task_generation" in result["workflow_phases"]
        
        # Verify TasksAgent was called during task generation phase
        mock_tasks_agent.process_task.assert_called_once()
        
        # Verify the task generation call had correct parameters
        call_args = mock_tasks_agent.process_task.call_args[0][0]
        assert call_args["task_type"] == "generate_task_list"
        assert "design_path" in call_args
        assert "requirements_path" in call_args
        assert "work_dir" in call_args
    
    @pytest.mark.integration
    def test_agent_registry_includes_tasks_agent(self, agent_manager, temp_workspace):
        """Test that agent registry includes TasksAgent after setup."""
        # Create mock agents
        mock_plan_agent = Mock()
        mock_plan_agent.initialize_autogen_agent.return_value = True
        
        mock_design_agent = Mock()
        mock_design_agent.initialize_autogen_agent.return_value = True
        
        mock_tasks_agent = Mock()
        mock_tasks_agent.initialize_autogen_agent.return_value = True
        
        mock_implement_agent = Mock()
        mock_implement_agent.initialize_autogen_agent.return_value = True
        
        # Manually set up agents to test registry
        agent_manager.agents = {
            "plan": mock_plan_agent,
            "design": mock_design_agent,
            "tasks": mock_tasks_agent,
            "implement": mock_implement_agent
        }
        agent_manager.is_initialized = True
        
        # Verify agent registry
        assert len(agent_manager.agents) == 4
        assert "plan" in agent_manager.agents
        assert "design" in agent_manager.agents
        assert "tasks" in agent_manager.agents
        assert "implement" in agent_manager.agents
        
        # Verify agent status includes TasksAgent
        agent_status = agent_manager.get_agent_status()
        assert "tasks" in agent_status