"""
Unit tests for the MainController class.

This module contains unit tests that focus on testing the MainController class
in isolation using mocked dependencies. These tests are designed to:

- Run quickly (under 1 second each)
- Use only mocked external dependencies (no real components or services)
- Test workflow management, user request processing, and framework initialization
- Validate controller behavior without external service dependencies

Key test classes:
- TestMainController: Core functionality tests with mocked dependencies
- TestMainControllerMocking: Comprehensive mocking tests for component integration

All external dependencies (AgentManager, MemoryManager, ShellExecutor) are mocked to ensure
fast, reliable unit tests that can run without network access or real services.

For tests that use real components and services, see:
tests/integration/test_real_main_controller.py
"""

import pytest
import asyncio
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from autogen_framework.main_controller import MainController, UserApprovalStatus
from autogen_framework.models import LLMConfig, WorkflowPhase
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.agent_manager import AgentManager
from autogen_framework.shell_executor import ShellExecutor

class TestMainController:
    """Test cases for MainController class."""
    # Using shared fixtures from conftest.py
    # Using shared fixtures from conftest.py
    @pytest.fixture
    def main_controller(self, temp_workspace):
        """Create a MainController instance for testing."""
        return MainController(temp_workspace)
    
    def test_main_controller_initialization(self, main_controller, temp_workspace):
        """Test MainController initialization."""
        assert main_controller.workspace_path == Path(temp_workspace)
        assert main_controller.memory_manager is None
        assert main_controller.agent_manager is None
        assert main_controller.shell_executor is None
        assert main_controller.llm_config is None
        assert not main_controller.is_initialized
        assert main_controller.current_workflow is None
        assert main_controller.user_approval_status == {}
        assert main_controller.execution_log == []
    
    def test_initialize_framework_success(self, main_controller, llm_config):
        """Test successful framework initialization."""
        with patch('autogen_framework.main_controller.MemoryManager') as mock_memory, \
             patch('autogen_framework.main_controller.AgentManager') as mock_agent, \
             patch('autogen_framework.main_controller.ShellExecutor') as mock_shell:
            
            # Mock component instances
            mock_memory_instance = Mock()
            mock_memory_instance.load_memory.return_value = {"global": {"test": "content"}}
            mock_memory.return_value = mock_memory_instance
            
            mock_agent_instance = Mock()
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory = Mock()
            mock_agent.return_value = mock_agent_instance
            
            mock_shell_instance = Mock()
            mock_shell.return_value = mock_shell_instance
            
            # Test initialization
            result = main_controller.initialize_framework(llm_config)
            
            assert result is True
            assert main_controller.is_initialized is True
            assert main_controller.llm_config == llm_config
            assert main_controller.memory_manager is not None
            assert main_controller.agent_manager is not None
            assert main_controller.shell_executor is not None
            assert len(main_controller.execution_log) > 0
            
            # Verify component initialization
            mock_memory.assert_called_once()
            mock_agent.assert_called_once()
            mock_shell.assert_called_once()
            mock_agent_instance.setup_agents.assert_called_once_with(llm_config)
            mock_agent_instance.update_agent_memory.assert_called_once()
    
    @patch.dict(os.environ, {
        'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
        'LLM_MODEL': 'test-model',
        'LLM_API_KEY': 'test-key'
    })
    def test_initialize_framework_default_config(self, main_controller):
        """Test framework initialization with default LLM configuration."""
        with patch('autogen_framework.main_controller.MemoryManager'), \
             patch('autogen_framework.main_controller.AgentManager') as mock_agent, \
             patch('autogen_framework.main_controller.ShellExecutor'):
            
            # Mock agent manager
            mock_agent_instance = Mock()
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory = Mock()
            mock_agent.return_value = mock_agent_instance
            
            # Test initialization without config
            result = main_controller.initialize_framework()
            
            assert result is True
            assert main_controller.llm_config is not None
            # Verify LLM config was set correctly from environment
            assert main_controller.llm_config.base_url == 'http://test.local:8888/openai/v1'
            assert main_controller.llm_config.model == 'test-model'
            assert main_controller.llm_config.api_key == 'test-key'
    
    def test_initialize_framework_agent_setup_failure(self, main_controller, llm_config):
        """Test framework initialization when agent setup fails."""
        with patch('autogen_framework.main_controller.MemoryManager') as mock_memory, \
             patch('autogen_framework.main_controller.AgentManager') as mock_agent, \
             patch('autogen_framework.main_controller.ShellExecutor') as mock_shell:
            
            # Mock component instances
            mock_memory_instance = Mock()
            mock_memory_instance.load_memory.return_value = {}
            mock_memory.return_value = mock_memory_instance
            
            mock_agent_instance = Mock()
            mock_agent_instance.setup_agents.return_value = False  # Agent setup fails
            mock_agent_instance.update_agent_memory = Mock()
            mock_agent.return_value = mock_agent_instance
            
            mock_shell_instance = Mock()
            mock_shell.return_value = mock_shell_instance
            
            # Test initialization
            result = main_controller.initialize_framework(llm_config)
            
            assert result is False
            assert not main_controller.is_initialized
    
    def test_initialize_framework_invalid_config(self, main_controller):
        """Test framework initialization with invalid LLM configuration."""
        invalid_config = LLMConfig(base_url="", model="", api_key="")
        
        result = main_controller.initialize_framework(invalid_config)
        
        assert result is False
        assert not main_controller.is_initialized
    
    @pytest.mark.asyncio
    async def test_process_user_request_not_initialized(self, main_controller):
        """Test processing user request when framework is not initialized."""
        with pytest.raises(RuntimeError, match="Framework not initialized"):
            await main_controller.process_user_request("test request")
    
    @pytest.mark.asyncio
    async def test_single_workflow_restriction(self, main_controller, llm_config):
        """Test that only one workflow can be active at a time."""
        # Initialize framework
        assert main_controller.initialize_framework(llm_config) is True
        
        # Mock the agent coordination to avoid actual LLM calls
        with patch.object(main_controller.agent_manager, 'coordinate_agents') as mock_coordinate:
            mock_coordinate.return_value = {
                "success": True,
                "requirements_path": "/test/requirements.md",
                "work_directory": "/test/work"
            }
            
            # Start first workflow
            result1 = await main_controller.process_user_request("Create a calculator")
            assert result1.get("requires_user_approval") is True
            assert result1.get("approval_needed_for") == "requirements"
            
            # Try to start second workflow while first is active
            result2 = await main_controller.process_user_request("Create a web server")
            assert result2.get("success") is False
            assert "Active workflow in progress" in result2.get("error", "")
            
            # Complete the workflow
            completed = main_controller.complete_workflow()
            assert completed is True
            
            # Now should be able to start new workflow
            result3 = await main_controller.process_user_request("Create a database")
            assert result3.get("requires_user_approval") is True
            assert result3.get("approval_needed_for") == "requirements"
    
    @pytest.mark.asyncio
    async def test_process_user_request_requirements_phase(self, main_controller, llm_config):
        """Test user request processing stopping at requirements phase for approval."""
        # Initialize framework
        await self._setup_initialized_controller(main_controller, llm_config)
        
        # Mock agent manager coordination
        requirements_result = {
            "success": True,
            "work_directory": "/test/work",
            "requirements_path": "/test/work/requirements.md"
        }
        
        main_controller.agent_manager.coordinate_agents = AsyncMock(return_value=requirements_result)
        
        # Process user request
        result = await main_controller.process_user_request("Create a test application")
        
        # Should stop at requirements phase waiting for approval
        assert result["success"] is False  # Not complete yet
        assert result["requires_user_approval"] is True
        assert result["approval_needed_for"] == "requirements"
        assert result["current_phase"] == "requirements"
        assert "requirements" in result["phases"]
        assert result["phases"]["requirements"]["success"] is True
        
        # Verify workflow state
        assert main_controller.current_workflow.phase == WorkflowPhase.PLANNING
        assert main_controller.current_workflow.work_directory == "/test/work"
    
    @pytest.mark.asyncio
    async def test_process_user_request_full_workflow_with_approvals(self, main_controller, llm_config):
        """Test complete user request processing with all phases approved."""
        # Initialize framework
        await self._setup_initialized_controller(main_controller, llm_config)
        
        # Pre-approve all phases
        main_controller.user_approval_status = {
            "requirements": UserApprovalStatus.APPROVED,
            "design": UserApprovalStatus.APPROVED,
            "tasks": UserApprovalStatus.APPROVED
        }
        
        # Mock agent manager coordination for all phases
        # Each phase returns a structured result matching the actual implementation
        requirements_result = {
            "success": True,
            "work_directory": "/test/work",
            "requirements_path": "/test/work/requirements.md"
        }
        
        design_result = {
            "success": True,
            "design_path": "/test/work/design.md"
        }
        
        tasks_result = {
            "success": True,
            "tasks_file": "/test/work/tasks.md"
        }
        
        # Mock coordinate_workflow to return different results based on task type
        # This simulates the actual workflow where different phases return different data structures
        async def mock_coordinate_workflow(task_type, context):
            """
            Mock function that simulates agent coordination for different workflow phases.
            
            The actual coordinate_agents method is called with different task_type values
            depending on the workflow phase:
            - "requirements_generation" for planning phase
            - "design_generation" for design phase  
            - "task_execution" for implementation phase
            """
            if task_type == "requirements_generation":
                return requirements_result
            elif task_type == "design_generation":
                return design_result
            elif task_type == "task_execution":
                return tasks_result
            else:
                return {"success": False, "error": "Unknown task type"}
        
        # Use side_effect to make the mock return different values based on parameters
        main_controller.agent_manager.coordinate_agents = AsyncMock(side_effect=mock_coordinate_workflow)
        
        # Process user request
        result = await main_controller.process_user_request("Create a test application")
        
        # Should complete all phases
        assert result["success"] is True or result["requires_user_approval"] is True
        assert result["current_phase"] == "implementation"
        assert "requirements" in result["phases"]
        assert "design" in result["phases"]
        assert "tasks" in result["phases"]
        assert "implementation" in result["phases"]
        
        # Verify workflow was completed and cleared
        assert main_controller.current_workflow is None
    
    def test_approve_phase_success(self, main_controller):
        """Test successful phase approval."""
        # Setup a mock workflow to avoid the "no active workflow" error
        from autogen_framework.models import WorkflowState, WorkflowPhase
        main_controller.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory=""  # Empty work directory to skip file existence check
        )
        
        result = main_controller.approve_phase("requirements", True)
        
        assert result["phase"] == "requirements"
        assert result["approved"] is True
        assert result["status"] == "approved"
        assert result["can_proceed"] is True
        assert "approved" in result["message"]
        
        # Verify approval status
        assert main_controller.user_approval_status["requirements"] == UserApprovalStatus.APPROVED
        assert len(main_controller.execution_log) > 0
    
    def test_approve_phase_rejection(self, main_controller):
        """Test phase rejection."""
        # Setup a mock workflow to avoid the "no active workflow" error
        from autogen_framework.models import WorkflowState, WorkflowPhase
        main_controller.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory=""  # Empty work directory to skip file existence check
        )
        
        result = main_controller.approve_phase("design", False)
        
        assert result["phase"] == "design"
        assert result["approved"] is False
        assert result["status"] == "rejected"
        assert result["can_proceed"] is False
        assert "rejected" in result["message"]
        
        # Verify approval status
        assert main_controller.user_approval_status["design"] == UserApprovalStatus.REJECTED
    
    def test_approve_phase_invalid_phase(self, main_controller):
        """Test approval of invalid phase."""
        result = main_controller.approve_phase("invalid_phase", True)
        
        assert result["success"] is False
        assert "Invalid phase 'invalid_phase'" in result["error"]
    
    @pytest.mark.asyncio
    async def test_continue_workflow_no_active_workflow(self, main_controller):
        """Test continuing workflow when no workflow is active."""
        with pytest.raises(RuntimeError, match="No active workflow"):
            await main_controller.continue_workflow()
    
    @pytest.mark.asyncio
    async def test_continue_workflow_to_design_phase(self, main_controller, llm_config):
        """Test continuing workflow from requirements to design phase."""
        # Setup initialized controller with active workflow
        await self._setup_initialized_controller(main_controller, llm_config)
        
        # Setup workflow state
        from autogen_framework.models import WorkflowState, WorkflowPhase
        main_controller.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        
        # Approve requirements phase
        main_controller.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
        
        # Add requirements result to execution log
        main_controller._record_execution_event(
            "phase_completed",
            {
                "phase": "requirements",
                "work_directory": "/test/work",
                "requirements_path": "/test/work/requirements.md"
            }
        )
        
        # Mock design phase execution
        design_result = {"success": True, "design_path": "/test/work/design.md"}
        main_controller.agent_manager.coordinate_agents = AsyncMock(return_value=design_result)
        
        # Continue workflow
        result = await main_controller.continue_workflow()
        
        assert result["phase"] == "design"
        assert result["result"]["success"] is True
        assert result["requires_approval"] is True  # Design not pre-approved
    
    def test_get_framework_status(self, main_controller, llm_config):
        """Test getting framework status."""
        # Test uninitialized status
        status = main_controller.get_framework_status()
        
        assert status["initialized"] is False
        assert status["llm_config"] is None
        assert status["current_workflow"]["active"] is False
        assert status["approval_status"] == {}
        
        # Initialize framework
        with patch('autogen_framework.main_controller.MemoryManager'), \
             patch('autogen_framework.main_controller.AgentManager') as mock_agent, \
             patch('autogen_framework.main_controller.ShellExecutor'):
            
            mock_agent_instance = Mock()
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory = Mock()
            mock_agent_instance.get_agent_status.return_value = {"status": "ready"}
            mock_agent.return_value = mock_agent_instance
            
            main_controller.initialize_framework(llm_config)
        
        # Test initialized status
        status = main_controller.get_framework_status()
        
        assert status["initialized"] is True
        assert status["llm_config"]["model"] == llm_config.model
        assert "components" in status
        assert "agent_manager" in status["components"]
    
    def test_get_execution_log(self, main_controller):
        """Test getting execution log."""
        # Initially empty
        log = main_controller.get_execution_log()
        assert log == []
        
        # Add some events
        main_controller._record_execution_event("test_event", {"data": "test"})
        main_controller._record_execution_event("another_event", {"data": "test2"})
        
        log = main_controller.get_execution_log()
        assert len(log) == 2
        assert log[0]["event_type"] == "test_event"
        assert log[1]["event_type"] == "another_event"
        
        # Verify it's a copy
        log.append({"event_type": "modified"})
        assert len(main_controller.execution_log) == 2
    
    def test_reset_framework(self, main_controller, llm_config):
        """Test framework reset."""
        # Setup some state
        with patch('autogen_framework.main_controller.MemoryManager'), \
             patch('autogen_framework.main_controller.AgentManager') as mock_agent, \
             patch('autogen_framework.main_controller.ShellExecutor') as mock_shell:
            
            mock_agent_instance = Mock()
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory = Mock()
            mock_agent_instance.reset_coordination_state = Mock()
            mock_agent.return_value = mock_agent_instance
            
            mock_shell_instance = Mock()
            mock_shell_instance.clear_history = Mock()
            mock_shell.return_value = mock_shell_instance
            
            main_controller.initialize_framework(llm_config)
            
            # Add some state
            from autogen_framework.models import WorkflowState, WorkflowPhase
            main_controller.current_workflow = WorkflowState(WorkflowPhase.DESIGN, "/test")
            main_controller.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
            main_controller._record_execution_event("test", {})
            
            # Reset
            result = main_controller.reset_framework()
            
            assert result is True
            assert main_controller.current_workflow is None
            assert main_controller.user_approval_status == {}
            assert main_controller.execution_log == []
            
            # Verify component reset calls
            mock_agent_instance.reset_coordination_state.assert_called_once()
            mock_shell_instance.clear_history.assert_called_once()
    
    def test_export_workflow_report_success(self, main_controller, temp_workspace):
        """Test successful workflow report export."""
        # Add some execution events
        main_controller._record_execution_event("test_event", {"data": "test"})
        
        output_path = Path(temp_workspace) / "workflow_report.json"
        
        result = main_controller.export_workflow_report(str(output_path))
        
        assert result is True
        assert output_path.exists()
        
        # Verify report content
        import json
        with open(output_path, 'r') as f:
            report = json.load(f)
        
        assert "timestamp" in report
        assert "framework_status" in report
        assert "execution_log" in report
        assert "workflow_state" in report
    
    @pytest.mark.asyncio
    async def test_execute_specific_task_no_workflow(self, main_controller):
        """Test executing specific task when no workflow is active."""
        task_def = {"id": "1", "title": "Test task"}
        
        with pytest.raises(RuntimeError, match="No active workflow"):
            await main_controller.execute_specific_task(task_def)
    
    @pytest.mark.asyncio
    async def test_execute_specific_task_wrong_phase(self, main_controller):
        """Test executing specific task in wrong workflow phase."""
        from autogen_framework.models import WorkflowState, WorkflowPhase
        main_controller.current_workflow = WorkflowState(WorkflowPhase.PLANNING, "/test")
        
        task_def = {"id": "1", "title": "Test task"}
        
        with pytest.raises(RuntimeError, match="Workflow not in implementation phase"):
            await main_controller.execute_specific_task(task_def)
    
    @pytest.mark.asyncio
    async def test_execute_specific_task_success(self, main_controller, llm_config):
        """Test successful specific task execution."""
        # Setup initialized controller
        await self._setup_initialized_controller(main_controller, llm_config)
        
        # Setup workflow in implementation phase
        from autogen_framework.models import WorkflowState, WorkflowPhase
        main_controller.current_workflow = WorkflowState(WorkflowPhase.IMPLEMENTATION, "/test/work")
        
        # Mock task execution
        task_result = {"success": True, "output": "Task completed"}
        main_controller.agent_manager.coordinate_agents = AsyncMock(return_value=task_result)
        
        task_def = {"id": "1", "title": "Test task"}
        
        # Execute task
        result = await main_controller.execute_specific_task(task_def)
        
        assert result["success"] is True
        assert result["output"] == "Task completed"
        
        # Verify coordination call
        main_controller.agent_manager.coordinate_agents.assert_called_once_with(
            "task_execution",
            {
                "task_type": "execute_task",
                "task": task_def,
                "work_dir": "/test/work"
            }
        )
        
        # Verify execution log
        assert len(main_controller.execution_log) > 0
        task_events = [e for e in main_controller.execution_log if e["event_type"] == "task_executed"]
        assert len(task_events) == 1
        assert task_events[0]["details"]["task_id"] == "1"
    
    def test_get_available_commands(self, main_controller):
        """Test getting available commands."""
        commands = main_controller.get_available_commands()
        
        assert isinstance(commands, list)
        assert len(commands) > 0
        assert any("initialize_framework" in cmd for cmd in commands)
        assert any("process_user_request" in cmd for cmd in commands)
        assert any("approve_phase" in cmd for cmd in commands)
    
    # Helper methods
    
    async def _setup_initialized_controller(self, main_controller, llm_config):
        """
        Helper to setup an initialized controller for testing.
        
        This method provides a standardized way to initialize a MainController
        with all dependencies mocked for unit testing. It ensures consistent
        mock setup across multiple test methods.
        
        Mock Configuration:
        - MemoryManager: Returns test content, simulates successful memory operations
        - AgentManager: Returns successful setup, provides mock coordination methods
        - ShellExecutor: Provides basic shell execution capabilities
        
        Args:
            main_controller: MainController instance to initialize
            llm_config: LLM configuration for initialization
        """
        with patch('autogen_framework.main_controller.MemoryManager') as mock_memory, \
             patch('autogen_framework.main_controller.AgentManager') as mock_agent, \
             patch('autogen_framework.main_controller.ShellExecutor') as mock_shell:
            
            # Mock MemoryManager instance
            # Simulates successful memory loading with test content
            mock_memory_instance = Mock()
            mock_memory_instance.load_memory.return_value = {"global": {"test": "content"}}
            mock_memory.return_value = mock_memory_instance
            
            # Mock AgentManager instance  
            # Simulates successful agent setup and provides coordination methods
            mock_agent_instance = Mock()
            mock_agent_instance.setup_agents.return_value = True  # Successful agent initialization
            mock_agent_instance.update_agent_memory = Mock()     # Mock memory update method
            mock_agent.return_value = mock_agent_instance
            
            # Mock ShellExecutor instance
            # Provides basic shell execution capabilities for testing
            mock_shell_instance = Mock()
            mock_shell.return_value = mock_shell_instance
            
            # Initialize framework with mocked dependencies
            # This will create the mocked instances and set up the controller
            main_controller.initialize_framework(llm_config)

class TestMainControllerMocking:
    """Test suite for MainController with comprehensive mocking."""
    
    @pytest.fixture
    def main_controller(self, temp_workspace):
        """Create a MainController instance for testing."""
        return MainController(temp_workspace)
    
    @patch('autogen_framework.main_controller.MemoryManager')
    @patch('autogen_framework.main_controller.AgentManager')
    @patch('autogen_framework.main_controller.ShellExecutor')
    def test_initialize_framework_with_mocked_components(self, mock_shell, mock_agent, mock_memory,
                                                       main_controller, llm_config):
        """Test framework initialization with fully mocked components."""
        # Mock component instances
        mock_memory_instance = Mock()
        mock_memory_instance.load_memory.return_value = {"global": {"test": "content"}}
        mock_memory.return_value = mock_memory_instance
        
        mock_agent_instance = Mock()
        mock_agent_instance.setup_agents.return_value = True
        mock_agent_instance.update_agent_memory = Mock()
        mock_agent_instance.get_agent_status.return_value = {"status": "ready"}
        mock_agent.return_value = mock_agent_instance
        
        mock_shell_instance = Mock()
        mock_shell.return_value = mock_shell_instance
        
        # Test initialization
        result = main_controller.initialize_framework(llm_config)
        
        assert result is True
        assert main_controller.is_initialized is True
        assert main_controller.llm_config == llm_config
        
        # Verify component creation
        mock_memory.assert_called_once()
        mock_agent.assert_called_once()
        mock_shell.assert_called_once()
        
        # Verify component setup
        mock_agent_instance.setup_agents.assert_called_once_with(llm_config)
        mock_agent_instance.update_agent_memory.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('autogen_framework.main_controller.MemoryManager')
    @patch('autogen_framework.main_controller.AgentManager')
    @patch('autogen_framework.main_controller.ShellExecutor')
    async def test_process_user_request_with_mocked_agents(self, mock_shell, mock_agent, mock_memory,
                                                         main_controller, llm_config):
        """Test user request processing with mocked agent responses."""
        # Setup mocked components
        mock_memory_instance = Mock()
        mock_memory_instance.load_memory.return_value = {"global": {"test": "content"}}
        mock_memory.return_value = mock_memory_instance
        
        mock_agent_instance = Mock()
        mock_agent_instance.setup_agents.return_value = True
        mock_agent_instance.update_agent_memory = Mock()
        mock_agent_instance.coordinate_agents = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        
        mock_shell_instance = Mock()
        mock_shell.return_value = mock_shell_instance
        
        # Initialize framework
        main_controller.initialize_framework(llm_config)
        
        # Mock agent coordination response
        requirements_result = {
            "success": True,
            "work_directory": "/test/work",
            "requirements_path": "/test/work/requirements.md"
        }
        mock_agent_instance.coordinate_agents.return_value = requirements_result
        
        # Process user request
        result = await main_controller.process_user_request("Create a test application")
        
        # Should stop at requirements phase waiting for approval
        assert result["success"] is False  # Not complete yet
        assert result["requires_user_approval"] is True
        assert result["approval_needed_for"] == "requirements"
        
        # Verify agent coordination was called
        mock_agent_instance.coordinate_agents.assert_called_once_with(
            "requirements_generation",
            {"user_request": "Create a test application", "workspace_path": str(main_controller.workspace_path)}
        )
    
    @patch('autogen_framework.main_controller.MemoryManager')
    @patch('autogen_framework.main_controller.AgentManager')
    @patch('autogen_framework.main_controller.ShellExecutor')
    def test_framework_status_with_mocked_components(self, mock_shell, mock_agent, mock_memory,
                                                   main_controller, llm_config):
        """Test framework status reporting with mocked components."""
        # Setup mocked components
        mock_memory_instance = Mock()
        mock_memory_instance.load_memory.return_value = {}
        mock_memory_instance.get_memory_stats.return_value = {"total_files": 5}
        mock_memory.return_value = mock_memory_instance
        
        mock_agent_instance = Mock()
        mock_agent_instance.setup_agents.return_value = True
        mock_agent_instance.update_agent_memory = Mock()
        mock_agent_instance.get_agent_status.return_value = {"agents": 3, "initialized": True}
        mock_agent.return_value = mock_agent_instance
        
        mock_shell_instance = Mock()
        mock_shell_instance.get_execution_stats.return_value = {"commands_executed": 10}
        mock_shell.return_value = mock_shell_instance
        
        # Initialize framework
        main_controller.initialize_framework(llm_config)
        
        # Get status
        status = main_controller.get_framework_status()
        
        assert status["initialized"] is True
        assert status["llm_config"]["model"] == llm_config.model
        assert "components" in status
        assert status["components"]["agent_manager"]["agents"] == 3
        assert status["components"]["memory_manager"]["total_files"] == 5
        assert status["components"]["shell_executor"]["commands_executed"] == 10
    
    @pytest.mark.asyncio
    @patch('autogen_framework.main_controller.MemoryManager')
    @patch('autogen_framework.main_controller.AgentManager')
    @patch('autogen_framework.main_controller.ShellExecutor')
    async def test_workflow_continuation_with_mocked_agents(self, mock_shell, mock_agent, mock_memory,
                                                          main_controller, llm_config):
        """Test workflow continuation with mocked agent responses."""
        # Setup mocked components
        mock_memory_instance = Mock()
        mock_memory_instance.load_memory.return_value = {}
        mock_memory.return_value = mock_memory_instance
        
        mock_agent_instance = Mock()
        mock_agent_instance.setup_agents.return_value = True
        mock_agent_instance.update_agent_memory = Mock()
        mock_agent_instance.coordinate_agents = AsyncMock()
        mock_agent.return_value = mock_agent_instance
        
        mock_shell_instance = Mock()
        mock_shell.return_value = mock_shell_instance
        
        # Initialize framework
        main_controller.initialize_framework(llm_config)
        
        # Setup workflow state
        from autogen_framework.models import WorkflowState, WorkflowPhase
        main_controller.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        
        # Approve requirements phase
        main_controller.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
        
        # Add requirements result to phase results and execution log
        requirements_result = {
            "success": True,
            "work_directory": "/test/work",
            "requirements_path": "/test/work/requirements.md"
        }
        main_controller.phase_results["requirements"] = requirements_result
        main_controller._record_execution_event(
            "phase_completed",
            {
                "phase": "requirements",
                "work_directory": "/test/work",
                "requirements_path": "/test/work/requirements.md"
            }
        )
        
        # Mock design phase execution
        design_result = {"success": True, "design_path": "/test/work/design.md"}
        mock_agent_instance.coordinate_agents.return_value = design_result
        
        # Continue workflow
        result = await main_controller.continue_workflow()
        
        assert result["phase"] == "design"
        assert result["result"]["success"] is True
        assert result["requires_approval"] is True
        
        # Verify agent coordination was called for design
        mock_agent_instance.coordinate_agents.assert_called_once_with(
            "design_generation",
            {
                "requirements_path": "/test/work/requirements.md",
                "work_directory": "/test/work",
                "memory_context": {}
            }
        )


# Integration tests have been moved to tests/integration/test_real_main_controller.py