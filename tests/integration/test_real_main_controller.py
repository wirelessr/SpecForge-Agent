"""
Integration tests for MainController with real components.

This module contains tests that use actual MemoryManager, ShellExecutor, and other
real components to verify the main controller integration.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from autogen_framework.main_controller import MainController, UserApprovalStatus
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.shell_executor import ShellExecutor


class TestMainControllerRealIntegration:
    """Integration tests for MainController with real components."""
    
    @pytest.fixture
    def main_controller(self, temp_workspace):
        """Create a MainController instance for integration testing."""
        return MainController(temp_workspace)
    
    @pytest.mark.integration
    def test_main_controller_with_real_memory_manager(self, main_controller, temp_workspace, real_llm_config):
        """Test MainController with real MemoryManager integration."""
        # Mock only the agent setup to avoid AutoGen dependencies
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize framework
            result = main_controller.initialize_framework(real_llm_config)
            
            assert result is True
            assert isinstance(main_controller.memory_manager, MemoryManager)
            assert main_controller.memory_manager.workspace_path == Path(temp_workspace)
            
            # Test that memory manager can save and load content
            main_controller.memory_manager.save_memory(
                "integration_test",
                "# Integration Test\n\nTesting MainController with real MemoryManager.",
                "global"
            )
            
            # Verify memory was saved
            memory_content = main_controller.memory_manager.load_memory()
            assert "global" in memory_content
            global_memory = memory_content["global"]
            assert any("integration_test" in key for key in global_memory.keys())
    
    @pytest.mark.integration
    def test_main_controller_with_real_shell_executor(self, main_controller, temp_workspace, real_llm_config):
        """Test MainController with real ShellExecutor integration."""
        # Mock only the agent setup to avoid AutoGen dependencies
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize framework
            result = main_controller.initialize_framework(real_llm_config)
            
            assert result is True
            assert isinstance(main_controller.shell_executor, ShellExecutor)
            assert main_controller.shell_executor.default_working_dir == temp_workspace
            
            # Test that shell executor can execute real commands
            import asyncio
            
            async def test_shell_execution():
                result = await main_controller.shell_executor.execute_command("echo 'main controller test'")
                assert result.success is True
                assert "main controller test" in result.stdout
                return result
            
            # Run the async test
            result = asyncio.run(test_shell_execution())
            assert result.success is True
    
    @pytest.mark.integration
    def test_workflow_state_transitions_with_real_components(self, main_controller, temp_workspace, real_llm_config):
        """Test workflow state transitions through phases with real components."""
        # Mock agent manager
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Create a mock workflow state with work directory
            from autogen_framework.models import WorkflowState, WorkflowPhase
            work_dir = Path(temp_workspace) / "test_workflow"
            work_dir.mkdir(exist_ok=True)
            
            # Create phase files that approve_phase expects
            (work_dir / "requirements.md").write_text("# Requirements\nTest requirements")
            (work_dir / "design.md").write_text("# Design\nTest design")
            (work_dir / "tasks.md").write_text("# Tasks\nTest tasks")
            
            # Set up workflow state
            main_controller.current_workflow = WorkflowState(
                phase=WorkflowPhase.PLANNING,
                work_directory=str(work_dir)
            )
            
            # Test phase approvals with real execution log
            result1 = main_controller.approve_phase("requirements", True)
            result2 = main_controller.approve_phase("design", True)
            result3 = main_controller.approve_phase("tasks", True)
            
            # Verify approval results are successful
            assert result1["approved"] is True
            assert result2["approved"] is True
            assert result3["approved"] is True
            
            # Verify approval status
            assert main_controller.user_approval_status["requirements"] == UserApprovalStatus.APPROVED
            assert main_controller.user_approval_status["design"] == UserApprovalStatus.APPROVED
            assert main_controller.user_approval_status["tasks"] == UserApprovalStatus.APPROVED
            
            # Verify execution log contains approval events
            approval_events = [e for e in main_controller.execution_log if e["event_type"] == "phase_approval"]
            assert len(approval_events) == 3
            
            # Verify each approval event has proper structure
            for event in approval_events:
                assert "phase" in event["details"]
                assert "approved" in event["details"]
                assert "timestamp" in event
    
    @pytest.mark.integration
    def test_framework_status_with_real_components(self, main_controller, temp_workspace, real_llm_config):
        """Test framework status reporting with real components."""
        # Test uninitialized status
        status = main_controller.get_framework_status()
        assert status["initialized"] is False
        assert status["llm_config"] is None
        
        # Mock agent manager
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            mock_agent_instance.get_agent_status.return_value = {"status": "ready"}
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Test initialized status
            status = main_controller.get_framework_status()
            
            assert status["initialized"] is True
            assert status["llm_config"]["model"] == real_llm_config.model
            assert status["llm_config"]["base_url"] == real_llm_config.base_url
            assert "components" in status
            assert "agent_manager" in status["components"]
            assert "memory_manager" in status["components"]
            assert "shell_executor" in status["components"]
            
            # Verify workspace path is correct
            assert status["workspace_path"] == temp_workspace
    
    @pytest.mark.integration
    def test_execution_log_with_real_components(self, main_controller, temp_workspace, real_llm_config):
        """Test execution log functionality with real components."""
        # Mock agent manager
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Initially should have initialization event
            log = main_controller.get_execution_log()
            assert len(log) > 0
            assert any(event["event_type"] == "framework_initialization" for event in log)
            
            # Add some test events
            main_controller._record_execution_event("test_event", {"data": "test"})
            main_controller._record_execution_event("another_event", {"data": "test2"})
            
            # Verify events were recorded
            updated_log = main_controller.get_execution_log()
            assert len(updated_log) >= 3  # initialization + 2 test events
            
            # Verify event structure
            test_events = [e for e in updated_log if e["event_type"] == "test_event"]
            assert len(test_events) == 1
            assert test_events[0]["details"]["data"] == "test"
            
            # Verify it's a copy (modifications don't affect original)
            updated_log.append({"event_type": "modified"})
            final_log = main_controller.get_execution_log()
            assert len(final_log) < len(updated_log)
    
    @pytest.mark.integration
    def test_framework_reset_with_real_components(self, main_controller, temp_workspace, real_llm_config):
        """Test framework reset with real components."""
        # Mock agent manager
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            mock_agent_instance.reset_coordination_state.return_value = None
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Add some state
            from autogen_framework.models import WorkflowState, WorkflowPhase
            main_controller.current_workflow = WorkflowState(WorkflowPhase.DESIGN, "/test")
            main_controller.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
            main_controller._record_execution_event("test", {})
            
            # Verify state exists
            assert main_controller.current_workflow is not None
            assert len(main_controller.user_approval_status) > 0
            assert len(main_controller.execution_log) > 0
            
            # Reset
            result = main_controller.reset_framework()
            
            assert result is True
            assert main_controller.current_workflow is None
            assert main_controller.user_approval_status == {}
            assert main_controller.execution_log == []
            
            # Verify component reset calls were made
            mock_agent_instance.reset_coordination_state.assert_called_once()
    
    @pytest.mark.integration
    def test_workflow_report_export_with_real_components(self, main_controller, temp_workspace, real_llm_config):
        """Test workflow report export with real components."""
        # Mock agent manager
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Create a mock workflow state with work directory for approval test
            from autogen_framework.models import WorkflowState, WorkflowPhase
            work_dir = Path(temp_workspace) / "test_workflow"
            work_dir.mkdir(exist_ok=True)
            
            # Create phase file that approve_phase expects
            (work_dir / "requirements.md").write_text("# Requirements\nTest requirements")
            
            # Set up workflow state
            main_controller.current_workflow = WorkflowState(
                phase=WorkflowPhase.PLANNING,
                work_directory=str(work_dir)
            )
            
            # Add some execution events
            main_controller._record_execution_event("test_event", {"data": "test"})
            approval_result = main_controller.approve_phase("requirements", True)
            
            # Verify approval was successful
            assert approval_result["approved"] is True
            
            # Export report
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
            
            # Verify framework status includes real component info
            framework_status = report["framework_status"]
            assert framework_status["initialized"] is True
            assert framework_status["workspace_path"] == temp_workspace
            assert "components" in framework_status
            
            # Verify execution log includes our events
            execution_log = report["execution_log"]
            assert len(execution_log) >= 2  # initialization + test_event + approval
            assert any(event["event_type"] == "test_event" for event in execution_log)
            assert any(event["event_type"] == "phase_approval" for event in execution_log)
    
    @pytest.mark.integration
    def test_session_management_with_real_components(self, main_controller, temp_workspace, real_llm_config):
        """Test session management with real components."""
        # Mock agent manager
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Get session ID
            session_id = main_controller.get_session_id()
            assert session_id is not None
            assert len(session_id) > 0
            
            # Verify session file was created
            session_file = Path(temp_workspace) / "memory" / "session_state.json"
            assert session_file.exists()
            
            # Verify session file content
            import json
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            assert session_data['session_id'] == session_id
            assert 'last_updated' in session_data
            
            # Test session persistence across instances
            main_controller2 = MainController(temp_workspace)
            main_controller2.initialize_framework(real_llm_config)
            
            # Should have the same session ID
            session_id2 = main_controller2.get_session_id()
            assert session_id2 == session_id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])