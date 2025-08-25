"""
Unit tests for the AgentManager class.

This module contains unit tests that focus on testing the AgentManager class
in isolation using mocked dependencies. These tests are designed to:

- Run quickly (under 1 second each)
- Use only mocked external dependencies (no real agents or services)
- Test agent coordination logic, initialization, and communication patterns
- Validate manager behavior without external service dependencies

Key test classes:
- TestAgentManager: Core functionality tests with mocked dependencies
- TestAgentManagerMocking: Comprehensive mocking tests for agent coordination

All external dependencies (individual agents, AutoGen components) are mocked to ensure
fast, reliable unit tests that can run without network access or real services.

For tests that use real components and services, see:
tests/integration/test_real_agent_manager.py
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from autogen_framework.agent_manager import AgentManager
from autogen_framework.models import LLMConfig, WorkflowPhase, WorkflowState
from autogen_framework.dependency_container import DependencyContainer

class TestAgentManager:
    """Test cases for AgentManager class."""
    # Using shared fixtures from conftest.py
    # Using shared fixtures from conftest.py
    @pytest.fixture
    def agent_manager(self, temp_workspace):
        """Create an AgentManager instance for testing."""
        return AgentManager(temp_workspace)
    
    def test_agent_manager_initialization(self, agent_manager, temp_workspace):
        """Test AgentManager initialization."""
        assert agent_manager.workspace_path == Path(temp_workspace)
        assert agent_manager.container is None  # Container not created until setup_agents
        assert agent_manager.agents == {}
        assert agent_manager.workflow_state is None
        assert agent_manager.agent_interactions == []
        assert agent_manager.coordination_log == []
        assert not agent_manager.is_initialized
    
    @patch('autogen_framework.agent_manager.DependencyContainer')
    @patch('autogen_framework.agent_manager.PlanAgent')
    @patch('autogen_framework.agent_manager.DesignAgent')
    @patch('autogen_framework.agent_manager.TasksAgent')
    @patch('autogen_framework.agent_manager.ImplementAgent')
    def test_setup_agents_success(self, mock_implement, mock_tasks, mock_design, mock_plan, 
                                 mock_container_class, agent_manager, test_llm_config):
        """Test successful agent setup using DependencyContainer."""
        # Mock the container
        mock_container = Mock()
        mock_container.get_memory_manager.return_value = Mock()
        mock_container.get_memory_manager.return_value.load_memory.return_value = {}
        mock_container.get_created_managers.return_value = {"memory_manager": "MemoryManager"}
        mock_container_class.create_production.return_value = mock_container
        
        # Mock agent instances
        mock_plan_instance = Mock()
        mock_plan_instance.initialize_autogen_agent.return_value = True
        mock_plan_instance.get_agent_status.return_value = {"name": "PlanAgent", "initialized": True}
        mock_plan.return_value = mock_plan_instance
        
        mock_design_instance = Mock()
        mock_design_instance.initialize_autogen_agent.return_value = True
        mock_design_instance.get_agent_status.return_value = {"name": "DesignAgent", "initialized": True}
        mock_design.return_value = mock_design_instance
        
        mock_tasks_instance = Mock()
        mock_tasks_instance.initialize_autogen_agent.return_value = True
        mock_tasks_instance.get_agent_status.return_value = {"name": "TasksAgent", "initialized": True}
        mock_tasks.return_value = mock_tasks_instance
        
        mock_implement_instance = Mock()
        mock_implement_instance.initialize_autogen_agent.return_value = True
        mock_implement_instance.get_agent_status.return_value = {"name": "ImplementAgent", "initialized": True}
        mock_implement.return_value = mock_implement_instance
        
        # Test agent setup
        result = agent_manager.setup_agents(test_llm_config)
        
        # Verify results
        assert result is True
        assert agent_manager.is_initialized is True
        assert len(agent_manager.agents) == 4  # plan, design, tasks, implement
        assert "plan" in agent_manager.agents
        assert "design" in agent_manager.agents
        assert "tasks" in agent_manager.agents
        assert "implement" in agent_manager.agents
        assert agent_manager.container is mock_container
        
        # Verify container was created
        mock_container_class.create_production.assert_called_once_with(
            work_dir=str(agent_manager.workspace_path),
            llm_config=test_llm_config
        )
        
        # Verify agents were created with container
        mock_plan.assert_called_once_with(
            container=mock_container,
            name="PlanAgent",
            llm_config=test_llm_config,
            system_message="Generate project requirements and create work directory structure"
        )
        mock_design.assert_called_once_with(
            container=mock_container,
            name="DesignAgent", 
            llm_config=test_llm_config,
            system_message="Generate technical design documents based on requirements"
        )
        mock_tasks.assert_called_once_with(
            container=mock_container,
            name="TasksAgent",
            llm_config=test_llm_config,
            system_message="Generate implementation task lists from design documents"
        )
        
        # Verify agent initialization was called
        mock_plan_instance.initialize_autogen_agent.assert_called_once()
        mock_design_instance.initialize_autogen_agent.assert_called_once()
        mock_tasks_instance.initialize_autogen_agent.assert_called_once()
        mock_implement_instance.initialize_autogen_agent.assert_called_once()
    
    @patch('autogen_framework.agent_manager.DependencyContainer')
    @patch('autogen_framework.agent_manager.PlanAgent')
    @patch('autogen_framework.agent_manager.DesignAgent')
    @patch('autogen_framework.agent_manager.TasksAgent')
    @patch('autogen_framework.agent_manager.ImplementAgent')
    def test_setup_agents_failure(self, mock_implement, mock_tasks, mock_design, mock_plan,
                                 mock_container_class, agent_manager, test_llm_config):
        """Test agent setup failure when agent initialization fails."""
        # Mock the container
        mock_container = Mock()
        mock_container_class.create_production.return_value = mock_container
        
        # Create mock agents where one fails to initialize
        mock_plan_instance = Mock()
        mock_plan_instance.initialize_autogen_agent.return_value = False  # This one fails
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
        
        # Test agent setup
        result = agent_manager.setup_agents(test_llm_config)
        
        # Should fail because plan agent failed to initialize
        assert result is False
        assert agent_manager.is_initialized is False
    
    def test_setup_agents_invalid_config(self, agent_manager):
        """Test agent setup with invalid LLM configuration."""
        invalid_config = LLMConfig(base_url="", model="", api_key="")
        
        result = agent_manager.setup_agents(invalid_config)
        
        assert result is False
        assert agent_manager.is_initialized is False
        assert agent_manager.container is None
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_not_initialized(self, agent_manager):
        """Test coordination when agents are not initialized."""
        with pytest.raises(RuntimeError, match="Agents not initialized"):
            await agent_manager.coordinate_agents("full_workflow", {"user_request": "test"})
    
    @pytest.mark.asyncio
    async def test_coordinate_full_workflow_success(self, agent_manager, test_llm_config):
        """Test successful full workflow coordination."""
        # Manually set up mock agents to test coordination logic
        mock_plan_agent = Mock()
        mock_design_agent = Mock()
        mock_tasks_agent = Mock()
        mock_implement_agent = Mock()
        
        # Mock agent task results
        plan_result = {
            "success": True,
            "work_directory": "/test/work",
            "requirements_path": "/test/work/requirements.md",
            "parsed_request": {"summary": "Test request"}
        }
        
        design_result = {
            "success": True,
            "design_path": "/test/work/design.md",
            "design_content": "Test design content"
        }
        
        task_gen_result = {
            "success": True,
            "tasks_file": "/test/work/tasks.md",
            "task_count": 5
        }
        
        # Mock process_task methods
        mock_plan_agent.process_task = AsyncMock(return_value=plan_result)
        mock_design_agent.process_task = AsyncMock(return_value=design_result)
        mock_tasks_agent.process_task = AsyncMock(return_value=task_gen_result)
        mock_implement_agent.process_task = AsyncMock(return_value={"success": True})
        
        # Set up the agent manager with mock container and agents
        mock_container = Mock()
        mock_memory_manager = Mock()
        mock_memory_manager.load_memory.return_value = {}
        mock_container.get_memory_manager.return_value = mock_memory_manager
        
        agent_manager.container = mock_container
        agent_manager.agents = {
            "plan": mock_plan_agent,
            "design": mock_design_agent,
            "tasks": mock_tasks_agent,
            "implement": mock_implement_agent
        }
        agent_manager.is_initialized = True
        
        # Test coordination
        context = {"user_request": "Create a test application"}
        result = await agent_manager.coordinate_agents("full_workflow", context)
        
        assert result["success"] is True
        assert result["current_phase"] == "completed"
        assert "workflow_phases" in result
        assert "planning" in result["workflow_phases"]
        assert "design" in result["workflow_phases"]
        assert "task_generation" in result["workflow_phases"]
        assert "implementation" in result["workflow_phases"]
        
        # Verify workflow state
        assert agent_manager.workflow_state.phase == WorkflowPhase.COMPLETED
        assert agent_manager.workflow_state.work_directory == "/test/work"
        
        # Verify coordination log
        assert len(agent_manager.coordination_log) > 0
        assert len(agent_manager.agent_interactions) > 0
    
    @pytest.mark.asyncio
    async def test_coordinate_full_workflow_plan_failure(self, agent_manager, test_llm_config):
        """Test full workflow coordination when planning fails."""
        # Manually set up mock agents to test coordination logic
        mock_plan_agent = Mock()
        mock_design_agent = Mock()
        mock_tasks_agent = Mock()
        mock_implement_agent = Mock()
        
        # Mock plan agent failure
        plan_result = {"success": False, "error": "Planning failed"}
        mock_plan_agent.process_task = AsyncMock(return_value=plan_result)
        
        # Set up the agent manager with mock container and agents
        mock_container = Mock()
        mock_memory_manager = Mock()
        mock_memory_manager.load_memory.return_value = {}
        mock_container.get_memory_manager.return_value = mock_memory_manager
        
        agent_manager.container = mock_container
        agent_manager.agents = {
            "plan": mock_plan_agent,
            "design": mock_design_agent,
            "tasks": mock_tasks_agent,
            "implement": mock_implement_agent
        }
        agent_manager.is_initialized = True
        
        # Test coordination
        context = {"user_request": "Create a test application"}
        result = await agent_manager.coordinate_agents("full_workflow", context)
        
        assert result["success"] is False
        assert "error" in result
        assert result["current_phase"] == "planning"
        assert "planning" in result["workflow_phases"]
        assert result["workflow_phases"]["planning"]["success"] is False
    
    @pytest.mark.asyncio
    async def test_coordinate_requirements_generation(self, agent_manager, test_llm_config):
        """Test requirements generation coordination."""
        # Manually set up a mock agent to test coordination logic
        mock_plan_agent = Mock()
        mock_plan_agent.process_task = AsyncMock(return_value={"success": True, "requirements_path": "/test/requirements.md"})
        mock_plan_agent.get_agent_status.return_value = {"name": "PlanAgent", "initialized": True}
        
        # Set up the agent manager with mock container and agents
        mock_container = Mock()
        mock_memory_manager = Mock()
        mock_memory_manager.load_memory.return_value = {}
        mock_container.get_memory_manager.return_value = mock_memory_manager
        
        agent_manager.container = mock_container
        agent_manager.agents = {"plan": mock_plan_agent}
        agent_manager.is_initialized = True
        
        # Test coordination
        context = {"user_request": "Test requirements"}
        result = await agent_manager.coordinate_agents("requirements_generation", context)
        
        assert result["success"] is True
        assert result["requirements_path"] == "/test/requirements.md"
        assert len(agent_manager.agent_interactions) > 0
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_unknown_task_type(self, agent_manager, llm_config):
        """Test coordination with unknown task type."""
        # Setup minimal agent
        agent_manager.is_initialized = True
        
        with pytest.raises(ValueError, match="Unknown task type"):
            await agent_manager.coordinate_agents("unknown_task", {})
    
    def test_get_agent_status_all_agents(self, agent_manager):
        """Test getting status for all agents."""
        # Mock agents
        mock_agent1 = Mock()
        mock_agent1.get_agent_status.return_value = {"name": "agent1", "status": "ready"}
        mock_agent2 = Mock()
        mock_agent2.get_agent_status.return_value = {"name": "agent2", "status": "busy"}
        
        agent_manager.agents = {"agent1": mock_agent1, "agent2": mock_agent2}
        
        status = agent_manager.get_agent_status()
        
        assert len(status) == 2
        assert status["agent1"]["name"] == "agent1"
        assert status["agent2"]["name"] == "agent2"
    
    def test_get_agent_status_specific_agent(self, agent_manager):
        """Test getting status for a specific agent."""
        mock_agent = Mock()
        mock_agent.get_agent_status.return_value = {"name": "test_agent", "status": "ready"}
        agent_manager.agents = {"test_agent": mock_agent}
        
        status = agent_manager.get_agent_status("test_agent")
        
        assert status["name"] == "test_agent"
        assert status["status"] == "ready"
    
    def test_get_agent_status_unknown_agent(self, agent_manager):
        """Test getting status for unknown agent."""
        with pytest.raises(ValueError, match="Unknown agent"):
            agent_manager.get_agent_status("unknown_agent")
    
    def test_get_coordination_log(self, agent_manager):
        """Test getting coordination log."""
        # Add test events
        agent_manager.coordination_log = [
            {"event": "test1", "timestamp": "2024-01-01"},
            {"event": "test2", "timestamp": "2024-01-02"}
        ]
        
        log = agent_manager.get_coordination_log()
        
        assert len(log) == 2
        assert log[0]["event"] == "test1"
        assert log[1]["event"] == "test2"
        
        # Verify it's a copy
        log.append({"event": "test3"})
        assert len(agent_manager.coordination_log) == 2
    
    def test_get_agent_interactions(self, agent_manager):
        """Test getting agent interactions."""
        # Add test interactions
        agent_manager.agent_interactions = [
            {"agent": "plan", "action": "start"},
            {"agent": "design", "action": "complete"}
        ]
        
        interactions = agent_manager.get_agent_interactions()
        
        assert len(interactions) == 2
        assert interactions[0]["agent"] == "plan"
        assert interactions[1]["agent"] == "design"
        
        # Verify it's a copy
        interactions.append({"agent": "implement", "action": "start"})
        assert len(agent_manager.agent_interactions) == 2
    
    def test_get_workflow_state(self, agent_manager):
        """Test getting workflow state."""
        # Initially None
        assert agent_manager.get_workflow_state() is None
        
        # Set workflow state
        workflow_state = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test"
        )
        agent_manager.workflow_state = workflow_state
        
        retrieved_state = agent_manager.get_workflow_state()
        assert retrieved_state == workflow_state
        assert retrieved_state.phase == WorkflowPhase.PLANNING
    
    def test_reset_coordination_state(self, agent_manager):
        """Test resetting coordination state."""
        # Setup some state
        agent_manager.workflow_state = WorkflowState(WorkflowPhase.DESIGN, "/test")
        agent_manager.coordination_log = [{"event": "test"}]
        agent_manager.agent_interactions = [{"interaction": "test"}]
        
        # Add mock agents
        mock_agent = Mock()
        agent_manager.agents = {"test": mock_agent}
        
        # Reset
        agent_manager.reset_coordination_state()
        
        assert agent_manager.workflow_state is None
        assert agent_manager.coordination_log == []
        assert agent_manager.agent_interactions == []
        mock_agent.reset_agent.assert_called_once()
    
    def test_export_coordination_report_success(self, agent_manager, temp_workspace):
        """Test successful coordination report export."""
        # Setup test data
        agent_manager.coordination_log = [{"event": "test_event"}]
        agent_manager.agent_interactions = [{"interaction": "test_interaction"}]
        agent_manager.agents = {"test": Mock()}
        agent_manager.agents["test"].get_agent_status.return_value = {"status": "ready"}
        
        output_path = Path(temp_workspace) / "report.json"
        
        result = agent_manager.export_coordination_report(str(output_path))
        
        assert result is True
        assert output_path.exists()
        
        # Verify report content
        import json
        with open(output_path, 'r') as f:
            report = json.load(f)
        
        assert "timestamp" in report
        assert "workspace_path" in report
        assert "coordination_events" in report
        assert "agent_interactions" in report
        assert "agent_status" in report
    
    def test_get_agent_capabilities(self, agent_manager):
        """Test getting agent capabilities."""
        # Mock agents with capabilities
        mock_agent1 = Mock()
        mock_agent1.get_agent_capabilities.return_value = ["capability1", "capability2"]
        mock_agent2 = Mock()
        mock_agent2.get_agent_capabilities.return_value = ["capability3"]
        
        agent_manager.agents = {"agent1": mock_agent1, "agent2": mock_agent2}
        
        capabilities = agent_manager.get_agent_capabilities()
        
        assert len(capabilities) == 2
        assert capabilities["agent1"] == ["capability1", "capability2"]
        assert capabilities["agent2"] == ["capability3"]
    
    def test_update_agent_memory(self, agent_manager):
        """Test updating agent memory."""
        # Mock agents
        mock_agent1 = Mock()
        mock_agent2 = Mock()
        agent_manager.agents = {"agent1": mock_agent1, "agent2": mock_agent2}
        
        memory_context = {"global": {"test": "content"}}
        
        agent_manager.update_agent_memory(memory_context)
        
        mock_agent1.update_memory_context.assert_called_once_with(memory_context)
        mock_agent2.update_memory_context.assert_called_once_with(memory_context)
    
    def test_get_coordination_statistics(self, agent_manager):
        """Test getting coordination statistics."""
        # Setup test data
        agent_manager.agent_interactions = [
            {"agent_name": "plan", "event_type": "start"},
            {"agent_name": "plan", "event_type": "complete"},
            {"agent_name": "design", "event_type": "start"}
        ]
        agent_manager.coordination_log = [
            {"event_type": "workflow_start"},
            {"event_type": "workflow_start"},
            {"event_type": "agent_init"}
        ]
        agent_manager.agents = {"plan": Mock(), "design": Mock()}
        agent_manager.workflow_state = WorkflowState(WorkflowPhase.DESIGN, "/test")
        
        stats = agent_manager.get_coordination_statistics()
        
        assert stats["total_interactions"] == 3
        assert stats["total_coordination_events"] == 3
        assert stats["agent_interaction_counts"]["plan"] == 2
        assert stats["agent_interaction_counts"]["design"] == 1
        assert stats["event_type_counts"]["workflow_start"] == 2
        assert stats["event_type_counts"]["agent_init"] == 1
        assert stats["agents_initialized"] == 2
        assert stats["current_workflow_phase"] == "design"
    
    # Helper methods
    
    async def _setup_mocked_agents(self, agent_manager, mock_plan, mock_design, mock_tasks, mock_implement, 
                                  llm_config, mock_container_class):
        """Helper to setup mocked agents for testing."""
        # Mock the container
        mock_container = Mock()
        mock_memory_manager = Mock()
        mock_memory_manager.load_memory.return_value = {"global": {"test": "content"}}
        mock_container.get_memory_manager.return_value = mock_memory_manager
        mock_container.get_created_managers.return_value = {"memory_manager": "MemoryManager"}
        mock_container_class.create_production.return_value = mock_container
        
        # Mock agent instances
        mock_plan_instance = Mock()
        mock_plan_instance.initialize_autogen_agent.return_value = True
        mock_plan_instance.get_agent_status.return_value = {"name": "PlanAgent", "initialized": True}
        mock_plan.return_value = mock_plan_instance
        
        mock_design_instance = Mock()
        mock_design_instance.initialize_autogen_agent.return_value = True
        mock_design_instance.get_agent_status.return_value = {"name": "DesignAgent", "initialized": True}
        mock_design.return_value = mock_design_instance
        
        mock_tasks_instance = Mock()
        mock_tasks_instance.initialize_autogen_agent.return_value = True
        mock_tasks_instance.get_agent_status.return_value = {"name": "TasksAgent", "initialized": True}
        mock_tasks.return_value = mock_tasks_instance
        
        mock_implement_instance = Mock()
        mock_implement_instance.initialize_autogen_agent.return_value = True
        mock_implement_instance.get_agent_status.return_value = {"name": "ImplementAgent", "initialized": True}
        mock_implement.return_value = mock_implement_instance
        
        # Setup agents
        agent_manager.setup_agents(llm_config)

class TestAgentManagerMocking:
    """Test suite for AgentManager with comprehensive mocking."""
    
    @pytest.fixture
    def agent_manager(self, temp_workspace):
        """Create an AgentManager instance for testing."""
        return AgentManager(temp_workspace)
    
    @patch('autogen_framework.agent_manager.DependencyContainer')
    @patch('autogen_framework.agent_manager.PlanAgent')
    @patch('autogen_framework.agent_manager.DesignAgent')
    @patch('autogen_framework.agent_manager.TasksAgent')
    @patch('autogen_framework.agent_manager.ImplementAgent')
    def test_setup_agents_with_mocked_autogen(self, mock_implement, mock_tasks, mock_design, mock_plan, 
                                            mock_container_class, agent_manager, test_llm_config):
        """Test agent setup with fully mocked AutoGen components."""
        # Mock the container
        mock_container = Mock()
        mock_memory_manager = Mock()
        mock_memory_manager.load_memory.return_value = {}
        mock_container.get_memory_manager.return_value = mock_memory_manager
        mock_container.get_created_managers.return_value = {"memory_manager": "MemoryManager"}
        mock_container_class.create_production.return_value = mock_container
        
        # Mock agent instances
        mock_plan_instance = Mock()
        mock_plan_instance.initialize_autogen_agent.return_value = True
        mock_plan_instance.get_agent_status.return_value = {"name": "PlanAgent", "initialized": True}
        mock_plan_instance.get_agent_capabilities.return_value = ["requirements_generation", "directory_creation"]
        mock_plan.return_value = mock_plan_instance
        
        mock_design_instance = Mock()
        mock_design_instance.initialize_autogen_agent.return_value = True
        mock_design_instance.get_agent_status.return_value = {"name": "DesignAgent", "initialized": True}
        mock_design_instance.get_agent_capabilities.return_value = ["design_generation", "mermaid_diagrams"]
        mock_design.return_value = mock_design_instance
        
        mock_tasks_instance = Mock()
        mock_tasks_instance.initialize_autogen_agent.return_value = True
        mock_tasks_instance.get_agent_status.return_value = {"name": "TasksAgent", "initialized": True}
        mock_tasks_instance.get_agent_capabilities.return_value = ["task_generation", "task_decomposition"]
        mock_tasks.return_value = mock_tasks_instance
        
        mock_implement_instance = Mock()
        mock_implement_instance.initialize_autogen_agent.return_value = True
        mock_implement_instance.get_agent_status.return_value = {"name": "ImplementAgent", "initialized": True}
        mock_implement_instance.get_agent_capabilities.return_value = ["task_execution", "shell_commands"]
        mock_implement.return_value = mock_implement_instance
        
        # Test agent setup
        result = agent_manager.setup_agents(test_llm_config)
        
        assert result is True
        assert agent_manager.is_initialized is True
        assert len(agent_manager.agents) == 4  # plan, design, tasks, implement
        assert agent_manager.container is mock_container
        
        # Verify container was created
        mock_container_class.create_production.assert_called_once_with(
            work_dir=str(agent_manager.workspace_path),
            llm_config=test_llm_config
        )
        
        # Verify agents were created with container
        mock_plan.assert_called_once_with(
            container=mock_container,
            name="PlanAgent",
            llm_config=test_llm_config,
            system_message="Generate project requirements and create work directory structure"
        )
        
        # Verify initialization was called for all agents
        assert mock_plan_instance.initialize_autogen_agent.called
        assert mock_design_instance.initialize_autogen_agent.called
        assert mock_tasks_instance.initialize_autogen_agent.called
        assert mock_implement_instance.initialize_autogen_agent.called
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_with_mocked_responses(self, agent_manager, test_llm_config):
        """Test agent coordination with mocked agent responses."""
        # Manually set up mock agents to test coordination logic
        mock_plan_agent = Mock()
        mock_design_agent = Mock()
        mock_tasks_agent = Mock()
        mock_implement_agent = Mock()
        
        # Mock agent task results
        plan_result = {
            "success": True,
            "work_directory": "/test/work",
            "requirements_path": "/test/work/requirements.md",
            "parsed_request": {"summary": "Test request"}
        }
        
        design_result = {
            "success": True,
            "design_path": "/test/work/design.md",
            "design_content": "Mocked design content"
        }
        
        task_gen_result = {
            "success": True,
            "tasks_file": "/test/work/tasks.md",
            "task_count": 3
        }
        
        # Mock process_task methods
        mock_plan_agent.process_task = AsyncMock(return_value=plan_result)
        mock_design_agent.process_task = AsyncMock(return_value=design_result)
        mock_tasks_agent.process_task = AsyncMock(return_value=task_gen_result)
        mock_implement_agent.process_task = AsyncMock(return_value={"success": True})
        
        # Set up the agent manager with mock container and agents
        mock_container = Mock()
        mock_memory_manager = Mock()
        mock_memory_manager.load_memory.return_value = {"global": {"test": "content"}}
        mock_container.get_memory_manager.return_value = mock_memory_manager
        
        agent_manager.container = mock_container
        agent_manager.agents = {
            "plan": mock_plan_agent,
            "design": mock_design_agent,
            "tasks": mock_tasks_agent,
            "implement": mock_implement_agent
        }
        agent_manager.is_initialized = True
        
        # Test coordination
        context = {"user_request": "Create a test application"}
        result = await agent_manager.coordinate_agents("full_workflow", context)
        
        assert result["success"] is True
        assert result["current_phase"] == "completed"
        assert "workflow_phases" in result
        
        # Verify all agents were called
        mock_plan_agent.process_task.assert_called_once()
        mock_design_agent.process_task.assert_called_once()
        mock_tasks_agent.process_task.assert_called_once()
    
    @patch('autogen_framework.agent_manager.DependencyContainer')
    @patch('autogen_framework.agent_manager.PlanAgent')
    @patch('autogen_framework.agent_manager.DesignAgent')
    @patch('autogen_framework.agent_manager.TasksAgent')
    @patch('autogen_framework.agent_manager.ImplementAgent')
    def test_agent_capabilities_with_mocks(self, mock_implement, mock_tasks, mock_design, mock_plan,
                                         mock_container_class, agent_manager, test_llm_config):
        """Test agent capabilities reporting with mocked agents."""
        # Mock the container
        mock_container = Mock()
        mock_memory_manager = Mock()
        mock_memory_manager.load_memory.return_value = {}
        mock_container.get_memory_manager.return_value = mock_memory_manager
        mock_container.get_created_managers.return_value = {"memory_manager": "MemoryManager"}
        mock_container_class.create_production.return_value = mock_container
        
        # Mock agent instances with capabilities
        mock_plan_instance = Mock()
        mock_plan_instance.initialize_autogen_agent.return_value = True
        mock_plan_instance.get_agent_capabilities.return_value = ["requirements", "planning"]
        mock_plan.return_value = mock_plan_instance
        
        mock_design_instance = Mock()
        mock_design_instance.initialize_autogen_agent.return_value = True
        mock_design_instance.get_agent_capabilities.return_value = ["design", "architecture"]
        mock_design.return_value = mock_design_instance
        
        mock_tasks_instance = Mock()
        mock_tasks_instance.initialize_autogen_agent.return_value = True
        mock_tasks_instance.get_agent_capabilities.return_value = ["task_generation", "task_decomposition"]
        mock_tasks.return_value = mock_tasks_instance
        
        mock_implement_instance = Mock()
        mock_implement_instance.initialize_autogen_agent.return_value = True
        mock_implement_instance.get_agent_capabilities.return_value = ["implementation", "testing"]
        mock_implement.return_value = mock_implement_instance
        
        # Setup agents
        agent_manager.setup_agents(test_llm_config)
        
        # Test capabilities
        capabilities = agent_manager.get_agent_capabilities()
        
        assert isinstance(capabilities, dict)
        assert len(capabilities) == 4  # plan, design, tasks, implement
        assert capabilities["plan"] == ["requirements", "planning"]
        assert capabilities["design"] == ["design", "architecture"]
        assert capabilities["tasks"] == ["task_generation", "task_decomposition"]
        assert capabilities["implement"] == ["implementation", "testing"]
    
    async def _setup_mocked_agents_for_coordination(self, agent_manager, mock_plan, mock_design, mock_tasks, mock_implement, 
                                                   llm_config, mock_container_class):
        """Helper to setup mocked agents for coordination testing."""
        # Mock the container
        mock_container = Mock()
        mock_memory_manager = Mock()
        mock_memory_manager.load_memory.return_value = {"global": {"test": "content"}}
        mock_container.get_memory_manager.return_value = mock_memory_manager
        mock_container.get_created_managers.return_value = {"memory_manager": "MemoryManager"}
        mock_container_class.create_production.return_value = mock_container
        
        # Mock agent instances
        mock_plan_instance = Mock()
        mock_plan_instance.initialize_autogen_agent.return_value = True
        mock_plan_instance.get_agent_status.return_value = {"name": "PlanAgent", "initialized": True}
        mock_plan.return_value = mock_plan_instance
        
        mock_design_instance = Mock()
        mock_design_instance.initialize_autogen_agent.return_value = True
        mock_design_instance.get_agent_status.return_value = {"name": "DesignAgent", "initialized": True}
        mock_design.return_value = mock_design_instance
        
        mock_tasks_instance = Mock()
        mock_tasks_instance.initialize_autogen_agent.return_value = True
        mock_tasks_instance.get_agent_status.return_value = {"name": "TasksAgent", "initialized": True}
        mock_tasks.return_value = mock_tasks_instance
        
        mock_implement_instance = Mock()
        mock_implement_instance.initialize_autogen_agent.return_value = True
        mock_implement_instance.get_agent_status.return_value = {"name": "ImplementAgent", "initialized": True}
        mock_implement.return_value = mock_implement_instance
        
        # Setup agents
        agent_manager.setup_agents(llm_config)
        
        # Setup agents
        agent_manager.setup_agents(llm_config)


# Integration tests have been moved to tests/integration/test_real_agent_manager.py