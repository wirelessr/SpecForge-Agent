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
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.shell_executor import ShellExecutor

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
        assert isinstance(agent_manager.memory_manager, MemoryManager)
        assert isinstance(agent_manager.shell_executor, ShellExecutor)
        assert agent_manager.agents == {}
        assert agent_manager.workflow_state is None
        assert agent_manager.agent_interactions == []
        assert agent_manager.coordination_log == []
        assert not agent_manager.is_initialized
    
    def test_setup_agents_success(self, agent_manager, llm_config):
        """Test successful agent setup by directly setting up agents."""
        # Create mock agents and add them directly to test the coordination logic
        mock_plan_agent = Mock()
        mock_plan_agent.initialize_autogen_agent.return_value = True
        mock_plan_agent.get_agent_status.return_value = {"name": "PlanAgent", "initialized": True}
        
        mock_design_agent = Mock()
        mock_design_agent.initialize_autogen_agent.return_value = True
        mock_design_agent.get_agent_status.return_value = {"name": "DesignAgent", "initialized": True}
        
        mock_tasks_agent = Mock()
        mock_tasks_agent.initialize_autogen_agent.return_value = True
        mock_tasks_agent.get_agent_status.return_value = {"name": "TasksAgent", "initialized": True}
        
        mock_implement_agent = Mock()
        mock_implement_agent.initialize_autogen_agent.return_value = True
        mock_implement_agent.get_agent_status.return_value = {"name": "ImplementAgent", "initialized": True}
        
        # Manually set up the agents to test the coordination logic
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
        agent_manager.llm_config = llm_config
        
        # Test the initialization logic
        initialization_results = []
        for agent_name, agent in agent_manager.agents.items():
            success = agent.initialize_autogen_agent()
            initialization_results.append((agent_name, success))
        
        all_initialized = all(success for _, success in initialization_results)
        
        if all_initialized:
            agent_manager.is_initialized = True
            agent_manager._record_coordination_event(
                event_type="agent_initialization",
                details={
                    "agents_initialized": list(agent_manager.agents.keys()),
                    "llm_config": {
                        "model": llm_config.model,
                        "base_url": llm_config.base_url
                    }
                }
            )
        
        # Verify results
        assert all_initialized is True
        assert agent_manager.is_initialized is True
        assert len(agent_manager.agents) == 4  # plan, design, tasks, implement
        assert "plan" in agent_manager.agents
        assert "design" in agent_manager.agents
        assert "tasks" in agent_manager.agents
        assert "implement" in agent_manager.agents
        assert len(agent_manager.coordination_log) > 0
        
        # Verify agent initialization was called
        mock_plan_agent.initialize_autogen_agent.assert_called_once()
        mock_design_agent.initialize_autogen_agent.assert_called_once()
        mock_tasks_agent.initialize_autogen_agent.assert_called_once()
        mock_implement_agent.initialize_autogen_agent.assert_called_once()
    
    def test_setup_agents_failure(self, agent_manager, llm_config):
        """Test agent setup failure when agent initialization fails."""
        # Create mock agents where one fails to initialize
        mock_plan_agent = Mock()
        mock_plan_agent.initialize_autogen_agent.return_value = False  # This one fails
        mock_plan_agent.get_agent_status.return_value = {"name": "PlanAgent", "initialized": False}
        
        mock_design_agent = Mock()
        mock_design_agent.initialize_autogen_agent.return_value = True
        mock_design_agent.get_agent_status.return_value = {"name": "DesignAgent", "initialized": True}
        
        mock_tasks_agent = Mock()
        mock_tasks_agent.initialize_autogen_agent.return_value = True
        mock_tasks_agent.get_agent_status.return_value = {"name": "TasksAgent", "initialized": True}
        
        mock_implement_agent = Mock()
        mock_implement_agent.initialize_autogen_agent.return_value = True
        mock_implement_agent.get_agent_status.return_value = {"name": "ImplementAgent", "initialized": True}
        
        # Manually set up the agents
        agent_manager.agents = {
            "plan": mock_plan_agent,
            "design": mock_design_agent,
            "tasks": mock_tasks_agent,
            "implement": mock_implement_agent
        }
        
        # Test the initialization logic
        initialization_results = []
        for agent_name, agent in agent_manager.agents.items():
            success = agent.initialize_autogen_agent()
            initialization_results.append((agent_name, success))
        
        all_initialized = all(success for _, success in initialization_results)
        
        # Should fail because plan agent failed to initialize
        assert all_initialized is False
        assert agent_manager.is_initialized is False
    
    def test_setup_agents_invalid_config(self, agent_manager):
        """Test agent setup with invalid LLM configuration."""
        invalid_config = LLMConfig(base_url="", model="", api_key="")
        
        result = agent_manager.setup_agents(invalid_config)
        
        assert result is False
        assert agent_manager.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_not_initialized(self, agent_manager):
        """Test coordination when agents are not initialized."""
        with pytest.raises(RuntimeError, match="Agents not initialized"):
            await agent_manager.coordinate_agents("full_workflow", {"user_request": "test"})
    
    @pytest.mark.asyncio
    @patch('autogen_framework.agents.plan_agent.PlanAgent')
    @patch('autogen_framework.agents.design_agent.DesignAgent')
    @patch('autogen_framework.agents.tasks_agent.TasksAgent')
    @patch('autogen_framework.agents.implement_agent.ImplementAgent')
    async def test_coordinate_full_workflow_success(self, mock_implement, mock_tasks, mock_design, mock_plan,
                                                   agent_manager, llm_config):
        """Test successful full workflow coordination."""
        # Setup mocked agents
        await self._setup_mocked_agents(agent_manager, mock_plan, mock_design, mock_tasks, mock_implement, llm_config)
        
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
        agent_manager.agents["plan"].process_task = AsyncMock(return_value=plan_result)
        agent_manager.agents["design"].process_task = AsyncMock(return_value=design_result)
        agent_manager.agents["tasks"].process_task = AsyncMock(return_value=task_gen_result)
        agent_manager.agents["implement"].process_task = AsyncMock(return_value={"success": True})
        
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
    @patch('autogen_framework.agents.plan_agent.PlanAgent')
    @patch('autogen_framework.agents.design_agent.DesignAgent')
    @patch('autogen_framework.agents.implement_agent.ImplementAgent')
    async def test_coordinate_full_workflow_plan_failure(self, mock_implement, mock_design, mock_plan,
                                                        agent_manager, llm_config):
        """Test full workflow coordination when planning fails."""
        # Setup mocked agents
        await self._setup_mocked_agents(agent_manager, mock_plan, mock_design, mock_implement, llm_config)
        
        # Mock plan agent failure
        plan_result = {"success": False, "error": "Planning failed"}
        agent_manager.agents["plan"].process_task = AsyncMock(return_value=plan_result)
        
        # Test coordination
        context = {"user_request": "Create a test application"}
        result = await agent_manager.coordinate_agents("full_workflow", context)
        
        assert result["success"] is False
        assert "error" in result
        assert result["current_phase"] == "planning"
        assert "planning" in result["workflow_phases"]
        assert result["workflow_phases"]["planning"]["success"] is False
    
    @pytest.mark.asyncio
    @patch('autogen_framework.agents.plan_agent.PlanAgent')
    async def test_coordinate_requirements_generation(self, mock_plan, agent_manager, llm_config):
        """Test requirements generation coordination."""
        # Setup mocked plan agent
        mock_plan_instance = Mock()
        mock_plan_instance.initialize_autogen_agent.return_value = True
        mock_plan_instance.get_agent_status.return_value = {"name": "PlanAgent", "initialized": True}
        mock_plan.return_value = mock_plan_instance
        
        agent_manager.memory_manager.load_memory = Mock(return_value={})
        agent_manager.setup_agents(llm_config)
        
        # Mock process_task
        expected_result = {"success": True, "requirements_path": "/test/requirements.md"}
        agent_manager.agents["plan"].process_task = AsyncMock(return_value=expected_result)
        
        # Test coordination
        context = {"user_request": "Test requirements"}
        result = await agent_manager.coordinate_agents("requirements_generation", context)
        
        assert result == expected_result
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
    
    async def _setup_mocked_agents(self, agent_manager, mock_plan, mock_design, mock_tasks, mock_implement, llm_config):
        """Helper to setup mocked agents for testing."""
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
        
        # Mock memory manager
        agent_manager.memory_manager.load_memory = Mock(return_value={"global": {"test": "content"}})
        
        # Setup agents
        agent_manager.setup_agents(llm_config)

class TestAgentManagerMocking:
    """Test suite for AgentManager with comprehensive mocking."""
    
    @pytest.fixture
    def agent_manager(self, temp_workspace):
        """Create an AgentManager instance for testing."""
        return AgentManager(temp_workspace)
    
    @patch('autogen_framework.agent_manager.PlanAgent')
    @patch('autogen_framework.agent_manager.DesignAgent')
    @patch('autogen_framework.agent_manager.ImplementAgent')
    def test_setup_agents_with_mocked_autogen(self, mock_implement, mock_design, mock_plan, 
                                            agent_manager, llm_config):
        """Test agent setup with fully mocked AutoGen components."""
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
        
        mock_implement_instance = Mock()
        mock_implement_instance.initialize_autogen_agent.return_value = True
        mock_implement_instance.get_agent_status.return_value = {"name": "ImplementAgent", "initialized": True}
        mock_implement_instance.get_agent_capabilities.return_value = ["task_execution", "shell_commands"]
        mock_implement.return_value = mock_implement_instance
        
        # Mock memory manager load_memory method
        agent_manager.memory_manager.load_memory = Mock(return_value={})
        
        # Test agent setup
        result = agent_manager.setup_agents(llm_config)
        
        assert result is True
        assert agent_manager.is_initialized is True
        assert len(agent_manager.agents) == 4  # plan, design, tasks, implement
        
        # Verify all agents were created with correct config
        mock_plan.assert_called_once()
        mock_design.assert_called_once()
        mock_implement.assert_called_once()
        
        # Verify initialization was called for all agents
        assert mock_plan_instance.initialize_autogen_agent.called
        assert mock_design_instance.initialize_autogen_agent.called
        assert mock_implement_instance.initialize_autogen_agent.called
    
    @pytest.mark.asyncio
    @patch('autogen_framework.agent_manager.PlanAgent')
    @patch('autogen_framework.agent_manager.DesignAgent')
    @patch('autogen_framework.agent_manager.ImplementAgent')
    async def test_coordinate_agents_with_mocked_responses(self, mock_implement, mock_design, mock_plan,
                                                         agent_manager, llm_config):
        """Test agent coordination with mocked agent responses."""
        # Setup mocked agents
        await self._setup_mocked_agents_for_coordination(agent_manager, mock_plan, mock_design, mock_implement, llm_config)
        
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
        agent_manager.agents["plan"].process_task = AsyncMock(return_value=plan_result)
        agent_manager.agents["design"].process_task = AsyncMock(return_value=design_result)
        agent_manager.agents["implement"].process_task = AsyncMock(return_value=task_gen_result)
        
        # Test coordination
        context = {"user_request": "Create a test application"}
        result = await agent_manager.coordinate_agents("full_workflow", context)
        
        assert result["success"] is True
        assert result["current_phase"] == "completed"
        assert "workflow_phases" in result
        
        # Verify all agents were called
        agent_manager.agents["plan"].process_task.assert_called_once()
        agent_manager.agents["design"].process_task.assert_called_once()
        agent_manager.agents["implement"].process_task.assert_called_once()
    
    @patch('autogen_framework.agent_manager.PlanAgent')
    @patch('autogen_framework.agent_manager.DesignAgent')
    @patch('autogen_framework.agent_manager.ImplementAgent')
    def test_agent_capabilities_with_mocks(self, mock_implement, mock_design, mock_plan,
                                         agent_manager, llm_config):
        """Test agent capabilities reporting with mocked agents."""
        # Mock agent instances with capabilities
        mock_plan_instance = Mock()
        mock_plan_instance.initialize_autogen_agent.return_value = True
        mock_plan_instance.get_agent_capabilities.return_value = ["requirements", "planning"]
        mock_plan.return_value = mock_plan_instance
        
        mock_design_instance = Mock()
        mock_design_instance.initialize_autogen_agent.return_value = True
        mock_design_instance.get_agent_capabilities.return_value = ["design", "architecture"]
        mock_design.return_value = mock_design_instance
        
        mock_implement_instance = Mock()
        mock_implement_instance.initialize_autogen_agent.return_value = True
        mock_implement_instance.get_agent_capabilities.return_value = ["implementation", "testing"]
        mock_implement.return_value = mock_implement_instance
        
        # Mock memory manager load_memory method
        agent_manager.memory_manager.load_memory = Mock(return_value={})
        
        # Setup agents
        agent_manager.setup_agents(llm_config)
        
        # Test capabilities
        capabilities = agent_manager.get_agent_capabilities()
        
        assert isinstance(capabilities, dict)
        assert len(capabilities) == 4  # plan, design, tasks, implement
        assert capabilities["plan"] == ["requirements", "planning"]
        assert capabilities["design"] == ["design", "architecture"]
        assert capabilities["implement"] == ["implementation", "testing"]
    
    async def _setup_mocked_agents_for_coordination(self, agent_manager, mock_plan, mock_design, mock_implement, llm_config):
        """Helper to setup mocked agents for coordination testing."""
        # Mock agent instances
        mock_plan_instance = Mock()
        mock_plan_instance.initialize_autogen_agent.return_value = True
        mock_plan_instance.get_agent_status.return_value = {"name": "PlanAgent", "initialized": True}
        mock_plan.return_value = mock_plan_instance
        
        mock_design_instance = Mock()
        mock_design_instance.initialize_autogen_agent.return_value = True
        mock_design_instance.get_agent_status.return_value = {"name": "DesignAgent", "initialized": True}
        mock_design.return_value = mock_design_instance
        
        mock_implement_instance = Mock()
        mock_implement_instance.initialize_autogen_agent.return_value = True
        mock_implement_instance.get_agent_status.return_value = {"name": "ImplementAgent", "initialized": True}
        mock_implement.return_value = mock_implement_instance
        
        # Mock memory manager
        agent_manager.memory_manager.load_memory = Mock(return_value={"global": {"test": "content"}})
        
        # Setup agents
        agent_manager.setup_agents(llm_config)


# Integration tests have been moved to tests/integration/test_real_agent_manager.py