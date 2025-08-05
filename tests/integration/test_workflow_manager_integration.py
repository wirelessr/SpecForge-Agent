"""
Integration tests for WorkflowManager with real components.

This module contains integration tests that verify WorkflowManager works correctly
with real AgentManager and SessionManager instances and handles workflow orchestration.
"""

import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from autogen_framework.workflow_manager import WorkflowManager, UserApprovalStatus
from autogen_framework.session_manager import SessionManager
from autogen_framework.models import WorkflowState, WorkflowPhase


class TestWorkflowManagerIntegration:
    """Integration tests for WorkflowManager with real components."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def session_manager(self, temp_workspace):
        """Create a real SessionManager instance for testing."""
        return SessionManager(temp_workspace)
    
    @pytest.fixture
    def mock_agent_manager(self):
        """Create a mock AgentManager for testing."""
        mock_agent = Mock()
        mock_agent.coordinate_agents = AsyncMock()
        return mock_agent
    
    @pytest.fixture
    def workflow_manager(self, mock_agent_manager, session_manager):
        """Create a WorkflowManager instance for testing."""
        return WorkflowManager(mock_agent_manager, session_manager)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_manager_with_real_session_manager(self, workflow_manager, session_manager, mock_agent_manager):
        """Test WorkflowManager integration with real SessionManager."""
        # Initialize session first
        session_manager.load_or_create_session()
        
        # Setup mock agent responses
        mock_agent_manager.coordinate_agents.side_effect = [
            # Requirements phase
            {
                "success": True,
                "work_directory": "/test/work",
                "requirements_path": "/test/work/requirements.md"
            },
            # Design phase
            {
                "success": True,
                "design_path": "/test/work/design.md"
            },
            # Tasks phase
            {
                "success": True,
                "tasks_file": "/test/work/tasks.md",
                "task_count": 5
            },
            # Implementation phase
            {
                "success": True,
                "implementation_complete": True
            }
        ]
        
        # Process request with auto-approve
        result = await workflow_manager.process_request(
            "Create a test application",
            auto_approve=True
        )
        
        # Verify workflow completed successfully
        assert result["success"] is True
        assert result["auto_approve_enabled"] is True
        assert "phases" in result
        assert "requirements" in result["phases"]
        assert "design" in result["phases"]
        assert "tasks" in result["phases"]
        assert "implementation" in result["phases"]
        
        # Verify session was updated
        session_data = session_manager.get_session_data()
        assert session_data["session_id"] is not None
        
        # Verify workflow was completed and cleared
        assert workflow_manager.current_workflow is None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_state_persistence_through_session_manager(self, workflow_manager, session_manager, mock_agent_manager):
        """Test that workflow state persists through SessionManager."""
        # Initialize session first
        session_manager.load_or_create_session()
        
        # Setup mock agent response for requirements phase only
        mock_agent_manager.coordinate_agents.return_value = {
            "success": True,
            "work_directory": "/test/work",
            "requirements_path": "/test/work/requirements.md"
        }
        
        # Process request without auto-approve (should stop at requirements)
        result = await workflow_manager.process_request(
            "Create a test application",
            auto_approve=False
        )
        
        # Verify workflow is waiting for approval
        assert result["requires_user_approval"] is True
        assert result["approval_needed_for"] == "requirements"
        assert workflow_manager.current_workflow is not None
        assert workflow_manager.current_workflow.phase == WorkflowPhase.PLANNING
        
        # Verify session manager has the workflow state
        session_data = session_manager.get_session_data()
        assert session_data["current_workflow"] is not None
        assert session_data["current_workflow"]["phase"] == "planning"
        assert session_data["current_workflow"]["work_directory"] == "/test/work"
        
        # Create new WorkflowManager instance with same SessionManager
        new_workflow_manager = WorkflowManager(mock_agent_manager, session_manager)
        
        # Verify workflow state was loaded from session
        assert new_workflow_manager.current_workflow is not None
        assert new_workflow_manager.current_workflow.phase == WorkflowPhase.PLANNING
        assert new_workflow_manager.current_workflow.work_directory == "/test/work"
    
    @pytest.mark.integration
    def test_phase_approval_with_real_session_persistence(self, workflow_manager, session_manager, temp_workspace):
        """Test phase approval with real session persistence."""
        # Setup workflow state
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory=str(Path(temp_workspace) / "test_work")
        )
        
        # Create the work directory and requirements file
        work_dir = Path(temp_workspace) / "test_work"
        work_dir.mkdir(exist_ok=True)
        (work_dir / "requirements.md").write_text("# Requirements\nTest requirements")
        
        # Approve requirements phase
        result = workflow_manager.approve_phase("requirements", True)
        
        # Verify approval was successful
        assert result["approved"] is True
        assert result["phase"] == "requirements"
        assert result["can_proceed"] is True
        
        # Verify approval was persisted in session
        session_data = session_manager.get_session_data()
        assert "requirements" in session_data["user_approval_status"]
        assert session_data["user_approval_status"]["requirements"] == "approved"
        
        # Verify session file was updated on disk
        session_file = Path(temp_workspace) / "memory" / "session_state.json"
        assert session_file.exists()
        
        import json
        with open(session_file, 'r') as f:
            file_data = json.load(f)
        assert file_data["user_approval_status"]["requirements"] == "approved"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_continuation_with_session_persistence(self, workflow_manager, session_manager, mock_agent_manager, temp_workspace):
        """Test workflow continuation with session persistence."""
        # Initialize session first
        session_manager.load_or_create_session()
        
        # Setup initial workflow state
        work_dir = Path(temp_workspace) / "test_work"
        work_dir.mkdir(exist_ok=True)
        (work_dir / "requirements.md").write_text("# Requirements\nTest requirements")
        
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory=str(work_dir)
        )
        
        # Set up phase results as if requirements phase was completed
        workflow_manager.phase_results["requirements"] = {
            "success": True,
            "work_directory": str(work_dir),
            "requirements_path": str(work_dir / "requirements.md")
        }
        
        # Approve requirements phase
        workflow_manager.approve_phase("requirements", True)
        
        # Setup mock for design phase
        mock_agent_manager.coordinate_agents.return_value = {
            "success": True,
            "design_path": str(work_dir / "design.md")
        }
        
        # Continue workflow
        result = await workflow_manager.continue_workflow()
        
        # Verify design phase was executed
        assert result["phase"] == "design"
        assert result["result"]["success"] is True
        assert workflow_manager.current_workflow.phase == WorkflowPhase.DESIGN
        
        # Verify session was updated
        session_data = session_manager.get_session_data()
        assert session_data["current_workflow"]["phase"] == "design"
        
        # Verify agent was called with correct parameters
        mock_agent_manager.coordinate_agents.assert_called_once_with(
            "design_generation",
            {
                "requirements_path": str(work_dir / "requirements.md"),
                "work_directory": str(work_dir),
                "memory_context": {}
            }
        )
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_phase_revision_with_session_persistence(self, workflow_manager, session_manager, mock_agent_manager, temp_workspace):
        """Test phase revision with session persistence."""
        # Initialize session first
        session_manager.load_or_create_session()
        
        # Setup workflow state
        work_dir = Path(temp_workspace) / "test_work"
        work_dir.mkdir(exist_ok=True)
        
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory=str(work_dir)
        )
        
        # Set up existing phase result
        workflow_manager.phase_results["requirements"] = {
            "success": True,
            "requirements_path": str(work_dir / "requirements.md")
        }
        
        # Setup mock for revision
        mock_agent_manager.coordinate_agents.return_value = {
            "success": True,
            "requirements_path": str(work_dir / "requirements.md")
        }
        
        # Apply revision
        result = await workflow_manager.apply_phase_revision(
            "requirements",
            "Please add more detailed user stories"
        )
        
        # Verify revision was successful
        assert result["success"] is True
        assert result["message"] == "Requirements phase revised successfully"
        
        # Verify session was updated
        session_data = session_manager.get_session_data()
        assert len(session_data["execution_log"]) > 0
        
        # Find revision event in execution log
        revision_events = [
            event for event in session_data["execution_log"]
            if event.get("event_type") == "phase_revision"
        ]
        assert len(revision_events) == 1
        assert revision_events[0]["details"]["phase"] == "requirements"
        assert "Please add more detailed user stories" in revision_events[0]["details"]["feedback"]
        
        # Verify agent was called with correct parameters
        mock_agent_manager.coordinate_agents.assert_called_once_with(
            "requirements_revision",
            {
                "phase": "requirements",
                "revision_feedback": "Please add more detailed user stories",
                "current_result": workflow_manager.phase_results["requirements"],
                "work_directory": str(work_dir)
            }
        )
    
    @pytest.mark.integration
    def test_workflow_completion_with_session_cleanup(self, workflow_manager, session_manager):
        """Test workflow completion with proper session cleanup."""
        # Initialize session first
        session_manager.load_or_create_session()
        
        # Setup active workflow
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.IMPLEMENTATION,
            work_directory="/test/work"
        )
        workflow_manager.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
        workflow_manager.user_approval_status["design"] = UserApprovalStatus.APPROVED
        
        # Complete workflow
        result = workflow_manager.complete_workflow()
        
        # Verify completion was successful
        assert result is True
        assert workflow_manager.current_workflow is None
        assert workflow_manager.user_approval_status == {}
        
        # Verify session was updated
        session_data = session_manager.get_session_data()
        assert session_data["current_workflow"] is None
        assert session_data["user_approval_status"] == {}
        
        # Verify completion event was logged
        completion_events = [
            event for event in session_data["execution_log"]
            if event.get("event_type") == "workflow_completed"
        ]
        assert len(completion_events) == 1
        assert completion_events[0]["details"]["final_phase"] == "implementation"
    
    @pytest.mark.integration
    def test_pending_approval_detection_with_session_state(self, workflow_manager, session_manager):
        """Test pending approval detection with session state."""
        # Initialize session first
        session_manager.load_or_create_session()
        
        # Test no workflow case
        pending = workflow_manager.get_pending_approval()
        assert pending is None
        
        # Setup workflow in planning phase without approval
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        
        pending = workflow_manager.get_pending_approval()
        assert pending is not None
        assert pending["phase"] == "requirements"
        assert pending["phase_name"] == "Requirements"
        
        # Approve requirements and move to design phase
        workflow_manager.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
        workflow_manager.current_workflow.phase = WorkflowPhase.DESIGN
        
        pending = workflow_manager.get_pending_approval()
        assert pending is not None
        assert pending["phase"] == "design"
        assert pending["phase_name"] == "Design"
        
        # Approve design and move to task generation phase
        workflow_manager.user_approval_status["design"] = UserApprovalStatus.APPROVED
        workflow_manager.current_workflow.phase = WorkflowPhase.TASK_GENERATION
        
        pending = workflow_manager.get_pending_approval()
        assert pending is not None
        assert pending["phase"] == "tasks"
        assert pending["phase_name"] == "Tasks"
        
        # Approve tasks - no more pending approvals
        workflow_manager.user_approval_status["tasks"] = UserApprovalStatus.APPROVED
        
        pending = workflow_manager.get_pending_approval()
        assert pending is None
    
    @pytest.mark.integration
    def test_auto_approve_mode_with_session_tracking(self, workflow_manager, session_manager):
        """Test auto-approve mode functionality with session tracking."""
        # Initialize session first
        session_manager.load_or_create_session()
        
        # Test auto-approve disabled with no manual approval
        assert workflow_manager.should_auto_approve("requirements", False) is False
        assert workflow_manager.should_auto_approve("design", False) is False
        assert workflow_manager.should_auto_approve("tasks", False) is False
        
        # Test auto-approve enabled - should approve phases automatically
        assert workflow_manager.should_auto_approve("requirements", True) is True
        assert workflow_manager.should_auto_approve("design", True) is True
        assert workflow_manager.should_auto_approve("tasks", True) is True
        
        # Save session state to persist the auto-approvals
        workflow_manager._save_session_state()
        
        # Verify session tracking after auto-approval
        session_data = session_manager.get_session_data()
        assert session_data["user_approval_status"]["requirements"] == "approved"
        assert session_data["user_approval_status"]["design"] == "approved"
        assert session_data["user_approval_status"]["tasks"] == "approved"
        
        # Test that already approved phases return True regardless of auto_approve flag
        assert workflow_manager.should_auto_approve("requirements", False) is True
        assert workflow_manager.should_auto_approve("design", False) is True
        assert workflow_manager.should_auto_approve("tasks", False) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])