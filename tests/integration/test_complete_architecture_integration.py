"""
Integration tests for complete architecture with all refactored components.

This module contains integration tests that verify the complete refactored architecture
works correctly with MainController, WorkflowManager, SessionManager, and TasksAgent
all working together.
"""

import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from autogen_framework.main_controller import MainController
from autogen_framework.models import LLMConfig, WorkflowState, WorkflowPhase


class TestCompleteArchitectureIntegration:
    """Integration tests for complete refactored architecture."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def main_controller(self, temp_workspace):
        """Create a MainController instance for testing."""
        return MainController(temp_workspace)
    
    # real_llm_config fixture is provided by tests/integration/conftest.py
    # It loads configuration from .env.integration file for secure testing
    
    @pytest.mark.integration
    def test_complete_component_initialization(self, main_controller, temp_workspace, real_llm_config):
        """Test that all refactored components are properly initialized."""
        # Mock agent manager to avoid AutoGen dependencies
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            mock_agent_instance.get_agent_status.return_value = {"status": "ready"}
            
            # Initialize framework
            result = main_controller.initialize_framework(real_llm_config)
            
            # Verify initialization was successful
            assert result is True
            assert main_controller.is_initialized is True
            
            # Verify all components were created
            assert main_controller.memory_manager is not None
            assert main_controller.shell_executor is not None
            assert main_controller.session_manager is not None
            assert main_controller.workflow_manager is not None
            assert main_controller.agent_manager is not None
            
            # Verify component integration
            assert main_controller.workflow_manager.session_manager is main_controller.session_manager
            assert main_controller.workflow_manager.agent_manager is main_controller.agent_manager
            
            # Verify session was created
            session_id = main_controller.get_session_id()
            assert session_id is not None
            assert len(session_id) > 0
            
            # Verify session file exists
            session_file = Path(temp_workspace) / "memory" / "session_state.json"
            assert session_file.exists()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Skipping long-running complete workflow test")
    async def test_complete_workflow_with_all_components(self, main_controller, temp_workspace, real_llm_config):
        """Test complete workflow execution with all refactored components."""
        # Mock agent manager with realistic responses
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            mock_agent_instance.get_agent_status.return_value = {"status": "ready"}
            
            # Create test project directory and files
            test_project_dir = Path(temp_workspace) / "test_project"
            test_project_dir.mkdir(exist_ok=True)
            
            # Create tasks.md file that implementation phase expects
            (test_project_dir / "tasks.md").write_text("""# Implementation Plan

- [ ] 1. Create basic structure
  - Setup project files
  - Requirements: 1.1

- [ ] 2. Implement core functionality
  - Add main features
  - Requirements: 1.2
""")
            
            # Mock workflow phases with AsyncMock
            async def mock_coordinate_agents(action, context):
                if action == "requirements_generation":
                    return {
                        "success": True,
                        "work_directory": str(test_project_dir),
                        "requirements_path": str(test_project_dir / "requirements.md")
                    }
                elif action == "design_generation":
                    return {
                        "success": True,
                        "design_path": str(test_project_dir / "design.md")
                    }
                elif action == "task_generation":
                    return {
                        "success": True,
                        "tasks_file": str(test_project_dir / "tasks.md"),
                        "task_count": 5
                    }
                else:
                    return {
                        "success": True,
                        "implementation_complete": True
                    }
            
            mock_agent_instance.coordinate_agents = AsyncMock(side_effect=mock_coordinate_agents)
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Process request with auto-approve
            result = await main_controller.process_request(
                "Create a simple calculator application",
                auto_approve=True
            )
            
            # Verify workflow completed successfully
            assert result["success"] is True
            assert result["auto_approve_enabled"] is True
            assert "phases" in result
            assert len(result["phases"]) == 4
            
            # Verify all phases were executed
            assert "requirements" in result["phases"]
            assert "design" in result["phases"]
            assert "tasks" in result["phases"]
            assert "implementation" in result["phases"]
            
            # Verify each phase was successful
            for phase_name, phase_result in result["phases"].items():
                if not phase_result["success"]:
                    print(f"Phase {phase_name} failed: {phase_result}")
                assert phase_result["success"] is True
            
            # Verify workflow was completed and cleared
            assert main_controller.current_workflow is None
            
            # Verify session was updated
            session_data = main_controller.session_manager.get_session_data()
            assert session_data["current_workflow"] is None
            
            # Verify execution log contains workflow events
            execution_log = main_controller.get_execution_log()
            phase_events = [e for e in execution_log if e["event_type"] == "phase_completed"]
            assert len(phase_events) >= 3  # At least requirements, design, tasks
            
            completion_events = [e for e in execution_log if e["event_type"] == "workflow_completed"]
            assert len(completion_events) == 1
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_manual_approvals(self, main_controller, temp_workspace, real_llm_config):
        """Test workflow with manual approval checkpoints using all components."""
        # Mock agent manager
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Mock requirements phase only
            mock_agent_instance.coordinate_agents = AsyncMock(return_value={
                "success": True,
                "work_directory": str(Path(temp_workspace) / "test_project"),
                "requirements_path": str(Path(temp_workspace) / "test_project" / "requirements.md")
            })
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Process request without auto-approve
            result = await main_controller.process_request(
                "Create a simple calculator application",
                auto_approve=False
            )
            
            # Verify workflow is waiting for approval
            assert result["requires_user_approval"] is True
            assert result["approval_needed_for"] == "requirements"
            assert result["auto_approve_enabled"] is False
            
            # Verify workflow state is maintained
            assert main_controller.current_workflow is not None
            assert main_controller.current_workflow.phase == WorkflowPhase.PLANNING
            
            # Verify session persistence
            session_data = main_controller.session_manager.get_session_data()
            assert session_data["current_workflow"] is not None
            assert session_data["current_workflow"]["phase"] == "planning"
            
            # Create requirements file for approval
            work_dir = Path(temp_workspace) / "test_project"
            work_dir.mkdir(exist_ok=True)
            (work_dir / "requirements.md").write_text("# Requirements\nTest requirements")
            
            # Approve requirements phase
            approval_result = main_controller.approve_phase("requirements", True)
            
            # Verify approval was successful
            assert approval_result["approved"] is True
            assert approval_result["phase"] == "requirements"
            
            # Verify approval status was updated in all components
            assert main_controller.user_approval_status["requirements"].value == "approved"
            assert main_controller.workflow_manager.user_approval_status["requirements"].value == "approved"
            
            # Verify session was updated
            updated_session_data = main_controller.session_manager.get_session_data()
            assert updated_session_data["user_approval_status"]["requirements"] == "approved"
    
    @pytest.mark.integration
    def test_session_persistence_across_component_restarts(self, temp_workspace, real_llm_config):
        """Test session persistence across component restarts."""
        # Create first MainController instance
        main_controller1 = MainController(temp_workspace)
        
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize and create session
            main_controller1.initialize_framework(real_llm_config)
            session_id1 = main_controller1.get_session_id()
            
            # Set up some workflow state
            main_controller1.workflow_manager.current_workflow = WorkflowState(
                phase=WorkflowPhase.DESIGN,
                work_directory="/test/work"
            )
            from autogen_framework.models import UserApprovalStatus
            main_controller1.workflow_manager.user_approval_status["requirements"] = \
                UserApprovalStatus.APPROVED
            
            # Save session state through WorkflowManager
            main_controller1.workflow_manager._save_session_state()
        
        # Create second MainController instance
        main_controller2 = MainController(temp_workspace)
        
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize second instance
            main_controller2.initialize_framework(real_llm_config)
            session_id2 = main_controller2.get_session_id()
            
            # Sync workflow state from WorkflowManager to MainController
            main_controller2._sync_workflow_state_from_manager()
            
            # Verify session persistence
            assert session_id2 == session_id1
            
            # Verify workflow state was restored
            assert main_controller2.current_workflow is not None
            assert main_controller2.current_workflow.phase == WorkflowPhase.DESIGN
            assert main_controller2.current_workflow.work_directory == "/test/work"
            
            # Verify approval status was restored
            assert "requirements" in main_controller2.user_approval_status
            assert main_controller2.user_approval_status["requirements"].value == "approved"
            
            # Verify WorkflowManager state was restored
            assert main_controller2.workflow_manager.current_workflow is not None
            assert main_controller2.workflow_manager.current_workflow.phase == WorkflowPhase.DESIGN
            assert "requirements" in main_controller2.workflow_manager.user_approval_status
    
    @pytest.mark.integration
    def test_framework_status_with_all_components(self, main_controller, temp_workspace, real_llm_config):
        """Test framework status reporting includes all refactored components."""
        # Mock agent manager
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            mock_agent_instance.get_agent_status.return_value = {
                "plan": {"status": "ready"},
                "design": {"status": "ready"},
                "tasks": {"status": "ready"},
                "implement": {"status": "ready"}
            }
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Get framework status
            status = main_controller.get_framework_status()
            
            # Verify status includes all components
            assert status["initialized"] is True
            assert status["session_id"] is not None
            assert status["workspace_path"] == temp_workspace
            assert status["llm_config"]["model"] == real_llm_config.model
            
            # Verify component status
            assert "components" in status
            assert "agent_manager" in status["components"]
            assert "memory_manager" in status["components"]
            assert "shell_executor" in status["components"]
            
            # Verify agent status includes TasksAgent
            agent_status = status["components"]["agent_manager"]
            assert "tasks" in agent_status
            assert agent_status["tasks"]["status"] == "ready"
    
    @pytest.mark.integration
    def test_framework_reset_with_all_components(self, main_controller, temp_workspace, real_llm_config):
        """Test framework reset clears state in all components."""
        # Mock agent manager
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            mock_agent_instance.reset_coordination_state.return_value = None
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Set up some state in all components
            main_controller.workflow_manager.current_workflow = WorkflowState(
                phase=WorkflowPhase.IMPLEMENTATION,
                work_directory="/test/work"
            )
            from autogen_framework.models import UserApprovalStatus
            main_controller.workflow_manager.user_approval_status["requirements"] = \
                UserApprovalStatus.APPROVED
            main_controller.workflow_manager.execution_log.append({"event": "test"})
            
            # Sync state to MainController
            main_controller._sync_workflow_state_from_manager()
            
            # Verify state exists
            assert main_controller.current_workflow is not None
            assert len(main_controller.user_approval_status) > 0
            assert len(main_controller.execution_log) > 0
            
            # Reset session through SessionManager
            reset_result = main_controller.session_manager.reset_session()
            
            # Verify reset was successful
            assert reset_result["success"] is True
            
            # Verify SessionManager created new session
            new_session_id = main_controller.get_session_id()
            assert new_session_id is not None
            
            # Verify session file was updated
            session_file = Path(temp_workspace) / "memory" / "session_state.json"
            assert session_file.exists()
            
            import json
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            assert session_data["session_id"] == new_session_id
            assert session_data["current_workflow"] is None
            assert session_data["user_approval_status"] == {}
            
            # Verify SessionManager state was cleared
            assert main_controller.session_manager.current_workflow is None
            assert main_controller.session_manager.user_approval_status == {}
            assert main_controller.session_manager.execution_log == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])