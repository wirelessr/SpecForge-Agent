"""
Unit tests for WorkflowManager.

Tests the workflow orchestration logic extracted from MainController,
including workflow state management, phase approvals, and agent coordination.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from autogen_framework.workflow_manager import WorkflowManager
from autogen_framework.models import UserApprovalStatus
from autogen_framework.models import WorkflowState, WorkflowPhase
from autogen_framework.session_manager import SessionManager


class TestWorkflowManager:
    """Test cases for WorkflowManager workflow orchestration."""
    
    @pytest.fixture
    def mock_agent_manager(self):
        """Create a mock AgentManager."""
        mock = Mock()
        mock.coordinate_agents = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock SessionManager."""
        mock = Mock(spec=SessionManager)
        mock.workspace_path = Path("/test/workspace")
        mock.session_id = "test-session-123"
        mock.current_workflow = None
        mock.user_approval_status = {}
        mock.phase_results = {}
        mock.execution_log = []
        mock.approval_log = []
        mock.error_recovery_attempts = {}
        mock.workflow_summary = {
            'phases_completed': [],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [],
            'errors_recovered': []
        }
        mock.save_session_state = Mock(return_value=True)
        return mock
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock MemoryManager."""
        mock = Mock()
        mock.search_memory = Mock(return_value=[])
        return mock
    
    @pytest.fixture
    def mock_context_compressor(self):
        """Create a mock ContextCompressor."""
        mock = Mock()
        mock.compress_context = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_token_manager(self):
        """Create a mock TokenManager."""
        mock = Mock()
        mock.get_model_limit = Mock(return_value=8192)
        mock.check_token_limit = Mock()
        mock.current_context_size = 0
        mock.usage_stats = {'compressions_performed': 0}
        return mock
    
    @pytest.fixture
    def workflow_manager(self, mock_agent_manager, mock_session_manager, mock_memory_manager, mock_context_compressor, mock_token_manager):
        """Create a WorkflowManager instance with mocked dependencies."""
        return WorkflowManager(mock_agent_manager, mock_session_manager, mock_memory_manager, mock_context_compressor, mock_token_manager)
    
    def test_initialization(self, workflow_manager, mock_agent_manager, mock_session_manager):
        """Test WorkflowManager initialization."""
        assert workflow_manager.agent_manager == mock_agent_manager
        assert workflow_manager.session_manager == mock_session_manager
        assert workflow_manager.current_workflow is None
        assert workflow_manager.user_approval_status == {}
        assert workflow_manager.phase_results == {}
        assert isinstance(workflow_manager.workflow_summary, dict)
    
    @pytest.mark.asyncio
    async def test_process_request_success_with_auto_approve(self, workflow_manager, mock_agent_manager):
        """Test successful request processing with auto-approve enabled."""
        # Mock successful agent responses for all phases
        mock_agent_manager.coordinate_agents.side_effect = [
            {"success": True, "requirements_path": "/test/requirements.md", "work_directory": "/test/work"},
            {"success": True, "design_path": "/test/design.md"},
            {"success": True, "tasks_file": "/test/tasks.md"},
            {"success": True, "execution_completed": True}
        ]
        
        # Mock task parsing to return empty list (no tasks to execute)
        with patch.object(workflow_manager, '_parse_tasks_from_file', return_value=[]):
            result = await workflow_manager.process_request("Create a test app", auto_approve=True)
        
        assert result["success"] is True
        assert result["auto_approve_enabled"] is True
        assert "workflow_id" in result
        assert len(result["phases"]) == 4  # requirements, design, tasks, implementation
        
        # Verify first 3 phases were executed, implementation phase returns early with no tasks
        assert mock_agent_manager.coordinate_agents.call_count == 3
        
        # Verify workflow was completed and cleared
        assert workflow_manager.current_workflow is None
    
    @pytest.mark.asyncio
    async def test_process_request_requires_approval(self, workflow_manager, mock_agent_manager):
        """Test request processing that requires user approval."""
        # Mock successful requirements generation
        mock_agent_manager.coordinate_agents.return_value = {
            "success": True,
            "requirements_path": "/test/requirements.md",
            "work_directory": "/test/work"
        }
        
        result = await workflow_manager.process_request("Create a test app", auto_approve=False)
        
        assert result["success"] is False
        assert result["requires_user_approval"] is True
        assert result["approval_needed_for"] == "requirements"
        assert result["requirements_path"] == "/test/requirements.md"
        
        # Verify only requirements phase was executed
        assert mock_agent_manager.coordinate_agents.call_count == 1
        
        # Verify workflow is still active
        assert workflow_manager.current_workflow is not None
        assert workflow_manager.current_workflow.phase == WorkflowPhase.PLANNING
    
    @pytest.mark.asyncio
    async def test_process_request_with_active_workflow_error(self, workflow_manager):
        """Test process_request with active workflow returns error."""
        # Set up active workflow
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        
        result = await workflow_manager.process_request("Another request")
        
        assert result["success"] is False
        assert "Active workflow in progress" in result["error"]
        assert "current_workflow" in result
    
    def test_approve_phase_success(self, workflow_manager):
        """Test successful phase approval."""
        # Set up active workflow
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            result = workflow_manager.approve_phase("requirements", True)
        
        assert result["approved"] is True
        assert result["phase"] == "requirements"
        assert result["can_proceed"] is True
        assert workflow_manager.user_approval_status["requirements"] == UserApprovalStatus.APPROVED
    
    def test_approve_phase_invalid_phase(self, workflow_manager):
        """Test phase approval with invalid phase name."""
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        
        result = workflow_manager.approve_phase("invalid_phase", True)
        
        assert result["success"] is False
        assert "Invalid phase" in result["error"]
    
    def test_approve_phase_no_active_workflow(self, workflow_manager):
        """Test phase approval with no active workflow."""
        result = workflow_manager.approve_phase("requirements", True)
        
        assert result["success"] is False
        assert "No active workflow found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_continue_workflow_to_design(self, workflow_manager, mock_agent_manager):
        """Test continuing workflow from requirements to design phase."""
        # Set up workflow in planning phase with approved requirements
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        workflow_manager.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
        workflow_manager.phase_results["requirements"] = {
            "requirements_path": "/test/requirements.md",
            "work_directory": "/test/work"
        }
        
        # Mock successful design generation
        mock_agent_manager.coordinate_agents.return_value = {
            "success": True,
            "design_path": "/test/design.md"
        }
        
        result = await workflow_manager.continue_workflow()
        
        assert result["phase"] == "design"
        assert result["result"]["success"] is True
        assert workflow_manager.current_workflow.phase == WorkflowPhase.DESIGN
    
    @pytest.mark.asyncio
    async def test_continue_workflow_no_active_workflow(self, workflow_manager):
        """Test continue_workflow with no active workflow."""
        with pytest.raises(RuntimeError, match="No active workflow to continue"):
            await workflow_manager.continue_workflow()
    
    @pytest.mark.asyncio
    async def test_apply_phase_revision_success(self, workflow_manager, mock_agent_manager):
        """Test successful phase revision."""
        # Set up active workflow with phase result
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        workflow_manager.phase_results["requirements"] = {
            "requirements_path": "/test/requirements.md"
        }
        
        # Mock successful revision
        mock_agent_manager.coordinate_agents.return_value = {
            "success": True,
            "requirements_path": "/test/requirements_revised.md"
        }
        
        result = await workflow_manager.apply_phase_revision("requirements", "Please add more details")
        
        assert result["success"] is True
        assert "revised successfully" in result["message"]
        
        # Verify approval status was reset
        assert "requirements" not in workflow_manager.user_approval_status
    
    @pytest.mark.asyncio
    async def test_apply_phase_revision_invalid_phase(self, workflow_manager):
        """Test phase revision with invalid phase."""
        result = await workflow_manager.apply_phase_revision("invalid", "feedback")
        
        assert result["success"] is False
        assert "Invalid phase" in result["error"]
    
    def test_get_pending_approval_requirements(self, workflow_manager):
        """Test getting pending approval for requirements phase."""
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        
        result = workflow_manager.get_pending_approval()
        
        assert result is not None
        assert result["phase"] == "requirements"
        assert result["phase_name"] == "Requirements"
    
    def test_get_pending_approval_no_workflow(self, workflow_manager):
        """Test getting pending approval with no active workflow."""
        result = workflow_manager.get_pending_approval()
        assert result is None
    
    def test_complete_workflow_success(self, workflow_manager):
        """Test successful workflow completion."""
        # Set up active workflow
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.COMPLETED,
            work_directory="/test/work"
        )
        workflow_manager.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
        
        result = workflow_manager.complete_workflow()
        
        assert result is True
        assert workflow_manager.current_workflow is None
        assert len(workflow_manager.user_approval_status) == 0
    
    def test_complete_workflow_no_active_workflow(self, workflow_manager):
        """Test workflow completion with no active workflow."""
        result = workflow_manager.complete_workflow()
        assert result is False
    
    def test_get_workflow_state(self, workflow_manager):
        """Test getting current workflow state."""
        # Test with no workflow
        assert workflow_manager.get_workflow_state() is None
        
        # Test with active workflow
        workflow_state = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        workflow_manager.current_workflow = workflow_state
        
        result = workflow_manager.get_workflow_state()
        assert result == workflow_state
    
    def test_should_auto_approve_enabled(self, workflow_manager):
        """Test auto-approve logic when enabled."""
        result = workflow_manager.should_auto_approve("requirements", True)
        
        assert result is True
        assert workflow_manager.user_approval_status["requirements"] == UserApprovalStatus.APPROVED
        assert len(workflow_manager.workflow_summary['phases_completed']) == 1
    
    def test_should_auto_approve_disabled(self, workflow_manager):
        """Test auto-approve logic when disabled."""
        result = workflow_manager.should_auto_approve("requirements", False)
        
        assert result is False
        assert "requirements" not in workflow_manager.user_approval_status
    
    def test_should_auto_approve_already_approved(self, workflow_manager):
        """Test auto-approve with already approved phase."""
        workflow_manager.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
        
        result = workflow_manager.should_auto_approve("requirements", False)
        assert result is True
    
    @patch.dict('os.environ', {'AUTO_APPROVE_CRITICAL_CHECKPOINTS': 'requirements,design'})
    def test_should_auto_approve_critical_checkpoint(self, workflow_manager):
        """Test auto-approve with critical checkpoint."""
        result = workflow_manager.should_auto_approve("requirements", True)
        
        assert result is False
        assert "requirements" not in workflow_manager.user_approval_status
    
    def test_log_auto_approval(self, workflow_manager):
        """Test auto-approval logging."""
        workflow_manager.log_auto_approval("requirements", True, "Test reason")
        
        assert len(workflow_manager.approval_log) == 1
        assert workflow_manager.approval_log[0]["phase"] == "requirements"
        assert workflow_manager.approval_log[0]["decision"] is True
        assert workflow_manager.approval_log[0]["reason"] == "Test reason"
    
    def test_handle_error_recovery_max_attempts_exceeded(self, workflow_manager):
        """Test error recovery when max attempts exceeded."""
        workflow_manager.error_recovery_attempts["requirements"] = 5
        
        result = workflow_manager.handle_error_recovery(
            Exception("Test error"), "requirements", {}
        )
        
        assert result is False
    
    def test_handle_error_recovery_simplified(self, workflow_manager):
        """Test simplified error recovery - always returns False and provides guidance."""
        result = workflow_manager.handle_error_recovery(
            Exception("Test error"), "requirements", {}
        )
        
        # Simplified error recovery always returns False (no automatic recovery)
        assert result is False
        
        # Test that guidance method works
        guidance = workflow_manager._get_phase_error_guidance("requirements")
        assert "--revise" in guidance
        assert "requirements" in guidance
    
    def test_parse_tasks_from_file_success(self, workflow_manager, tmp_path):
        """Test successful task parsing from file."""
        tasks_file = tmp_path / "tasks.md"
        tasks_file.write_text("""
# Tasks

- [ ] Task 1: Create basic structure
  - Step 1: Initialize project
  - Requirements: 1.1, 1.2

- [x] Task 2: Setup configuration
  - Step 1: Create config file
  - Requirements: 2.1
        """)
        
        tasks = workflow_manager._parse_tasks_from_file(str(tasks_file))
        
        # Should only return uncompleted tasks
        assert len(tasks) == 1
        assert tasks[0].title == "Task 1: Create basic structure"
        assert not tasks[0].completed
    
    def test_parse_tasks_from_file_not_found(self, workflow_manager):
        """Test task parsing with non-existent file."""
        tasks = workflow_manager._parse_tasks_from_file("/nonexistent/tasks.md")
        assert tasks == []
    
    def test_update_tasks_file_with_completion(self, workflow_manager, tmp_path):
        """Test updating tasks file with completion status."""
        tasks_file = tmp_path / "tasks.md"
        tasks_file.write_text("""
# Tasks

- [ ] Task 1: Create basic structure
- [ ] Task 2: Setup configuration
        """)
        
        task_results = [
            {"success": True},
            {"success": False}
        ]
        
        workflow_manager._update_tasks_file_with_completion(str(tasks_file), task_results)
        
        content = tasks_file.read_text()
        assert "- [x] Task 1: Create basic structure" in content
        assert "- [ ] Task 2: Setup configuration" in content
    
    def test_session_state_sync(self, workflow_manager, mock_session_manager):
        """Test session state synchronization."""
        # Set up workflow state
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        workflow_manager.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
        
        # Call save session state
        workflow_manager._save_session_state()
        
        # Verify SessionManager was updated
        assert mock_session_manager.current_workflow == workflow_manager.current_workflow
        assert mock_session_manager.user_approval_status["requirements"] == "approved"
        mock_session_manager.save_session_state.assert_called_once()


class TestWorkflowManagerIntegration:
    """Integration tests for WorkflowManager with real SessionManager."""
    
    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace directory."""
        workspace = tmp_path / "test_workspace"
        workspace.mkdir()
        return workspace
    
    @pytest.fixture
    def real_session_manager(self, temp_workspace):
        """Create a real SessionManager instance."""
        return SessionManager(str(temp_workspace))
    
    @pytest.fixture
    def mock_agent_manager(self):
        """Create a mock AgentManager."""
        mock = Mock()
        mock.coordinate_agents = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock MemoryManager."""
        mock = Mock()
        mock.search_memory = Mock(return_value=[])
        return mock
    
    @pytest.fixture
    def mock_context_compressor(self):
        """Create a mock ContextCompressor."""
        mock = Mock()
        mock.compress_context = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_token_manager(self):
        """Create a mock TokenManager."""
        mock = Mock()
        mock.get_model_limit = Mock(return_value=8192)
        mock.check_token_limit = Mock()
        mock.current_context_size = 0
        mock.usage_stats = {'compressions_performed': 0}
        return mock
    
    @pytest.fixture
    def workflow_manager_with_real_session(self, mock_agent_manager, real_session_manager, mock_memory_manager, mock_context_compressor, mock_token_manager):
        """Create WorkflowManager with real SessionManager."""
        return WorkflowManager(mock_agent_manager, real_session_manager, mock_memory_manager, mock_context_compressor, mock_token_manager)
    
    def test_session_persistence(self, workflow_manager_with_real_session):
        """Test that workflow state persists across sessions."""
        # Set up workflow state
        workflow_state = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/work"
        )
        workflow_manager_with_real_session.current_workflow = workflow_state
        workflow_manager_with_real_session.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
        
        # Save session state
        result = workflow_manager_with_real_session._save_session_state()
        assert result is True
        
        # Create new WorkflowManager with same SessionManager
        new_workflow_manager = WorkflowManager(
            Mock(), workflow_manager_with_real_session.session_manager, Mock(), Mock(), Mock()
        )
        
        # Verify state was loaded
        assert new_workflow_manager.current_workflow is not None
        assert new_workflow_manager.current_workflow.phase == WorkflowPhase.PLANNING
        assert new_workflow_manager.user_approval_status["requirements"] == UserApprovalStatus.APPROVED
    
    @pytest.mark.asyncio
    async def test_workflow_state_consistency(self, workflow_manager_with_real_session, mock_agent_manager):
        """Test workflow state remains consistent during operations."""
        # Mock successful requirements generation
        mock_agent_manager.coordinate_agents.return_value = {
            "success": True,
            "requirements_path": "/test/requirements.md",
            "work_directory": "/test/work"
        }
        
        # Process request
        result = await workflow_manager_with_real_session.process_request("Test request")
        
        # Verify workflow state is consistent
        assert workflow_manager_with_real_session.current_workflow is not None
        assert workflow_manager_with_real_session.current_workflow.phase == WorkflowPhase.PLANNING
        assert "requirements" in workflow_manager_with_real_session.phase_results
        
        # Verify session was saved
        session_data = workflow_manager_with_real_session.session_manager.get_session_data()
        assert session_data["current_workflow"] is not None
        assert session_data["phase_results"]["requirements"] is not None