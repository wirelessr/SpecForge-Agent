"""
Integration tests for SessionManager with real components.

This module contains integration tests that verify SessionManager works correctly
with real file system operations and integrates properly with other components.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch

from autogen_framework.session_manager import SessionManager
from autogen_framework.models import WorkflowState, WorkflowPhase


class TestSessionManagerIntegration:
    """Integration tests for SessionManager with real file operations."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def session_manager(self, temp_workspace):
        """Create a SessionManager instance for testing."""
        return SessionManager(temp_workspace)
    
    @pytest.mark.integration
    def test_session_creation_with_real_filesystem(self, session_manager, temp_workspace):
        """Test session creation with real file system operations."""
        # Load or create session (should create new)
        session_data = session_manager.load_or_create_session()
        
        # Verify session was created
        assert session_data["session_id"] is not None
        assert len(session_data["session_id"]) > 0
        assert session_data["current_workflow"] is None
        assert session_data["user_approval_status"] == {}
        assert session_data["save_success"] is True
        
        # Verify session file was created on disk
        session_file = Path(temp_workspace) / "memory" / "session_state.json"
        assert session_file.exists()
        
        # Verify file content
        with open(session_file, 'r') as f:
            file_data = json.load(f)
        assert file_data["session_id"] == session_data["session_id"]
        assert "last_updated" in file_data
    
    @pytest.mark.integration
    def test_session_persistence_across_instances(self, temp_workspace):
        """Test that session data persists across SessionManager instances."""
        # Create first instance and session
        session_manager1 = SessionManager(temp_workspace)
        session_data1 = session_manager1.load_or_create_session()
        session_id1 = session_data1["session_id"]
        
        # Add some workflow state
        session_manager1.current_workflow = WorkflowState(
            phase=WorkflowPhase.DESIGN,
            work_directory="/test/work"
        )
        session_manager1.user_approval_status["requirements"] = "approved"
        session_manager1.execution_log.append({
            "event_type": "test_event",
            "details": {"data": "test"}
        })
        
        # Save state
        save_result = session_manager1.save_session_state()
        assert save_result is True
        
        # Create second instance and load session
        session_manager2 = SessionManager(temp_workspace)
        session_data2 = session_manager2.load_or_create_session()
        
        # Verify session persistence
        assert session_data2["session_id"] == session_id1
        assert session_manager2.current_workflow is not None
        assert session_manager2.current_workflow.phase == WorkflowPhase.DESIGN
        assert session_manager2.current_workflow.work_directory == "/test/work"
        assert session_manager2.user_approval_status["requirements"] == "approved"
        assert len(session_manager2.execution_log) == 1
        assert session_manager2.execution_log[0]["event_type"] == "test_event"
    
    @pytest.mark.integration
    def test_session_reset_with_real_filesystem(self, session_manager, temp_workspace):
        """Test session reset with real file system operations."""
        # Create initial session with data
        session_data = session_manager.load_or_create_session()
        original_session_id = session_data["session_id"]
        
        # Add some state
        session_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.IMPLEMENTATION,
            work_directory="/test/work"
        )
        session_manager.user_approval_status["design"] = "approved"
        session_manager.execution_log.append({"event": "test"})
        session_manager.save_session_state()
        
        # Verify session file exists
        session_file = Path(temp_workspace) / "memory" / "session_state.json"
        assert session_file.exists()
        
        # Reset session
        reset_result = session_manager.reset_session()
        
        # Verify reset was successful
        assert reset_result["success"] is True
        assert reset_result["session_id"] != original_session_id
        assert session_manager.current_workflow is None
        assert session_manager.user_approval_status == {}
        assert session_manager.execution_log == []
        
        # Verify new session file was created
        assert session_file.exists()
        with open(session_file, 'r') as f:
            new_session_data = json.load(f)
        assert new_session_data["session_id"] == reset_result["session_id"]
        assert new_session_data["current_workflow"] is None
    
    @pytest.mark.integration
    def test_session_data_integrity_with_complex_workflow(self, session_manager):
        """Test session data integrity with complex workflow state."""
        # Create session
        session_manager.load_or_create_session()
        
        # Set up complex workflow state
        session_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.TASK_GENERATION,
            work_directory="/complex/work/dir"
        )
        
        session_manager.user_approval_status = {
            "requirements": "approved",
            "design": "pending",
            "tasks": "needs_revision"
        }
        
        session_manager.phase_results = {
            "requirements": {
                "success": True,
                "requirements_path": "/work/requirements.md"
            },
            "design": {
                "success": True,
                "design_path": "/work/design.md"
            }
        }
        
        session_manager.execution_log = [
            {
                "event_type": "phase_completed",
                "details": {"phase": "requirements"},
                "timestamp": "2024-01-01T10:00:00"
            },
            {
                "event_type": "phase_approval",
                "details": {"phase": "requirements", "approved": True},
                "timestamp": "2024-01-01T10:05:00"
            }
        ]
        
        session_manager.workflow_summary = {
            "phases_completed": ["requirements", "design"],
            "tasks_completed": [],
            "token_usage": {"total": 1500},
            "compression_events": [],
            "auto_approvals": ["requirements"],
            "errors_recovered": []
        }
        
        # Save and reload
        save_result = session_manager.save_session_state()
        assert save_result is True
        
        # Create new instance and verify data integrity
        new_session_manager = SessionManager(session_manager.workspace_path)
        session_data = new_session_manager.load_or_create_session()
        
        # Verify workflow state
        assert new_session_manager.current_workflow.phase == WorkflowPhase.TASK_GENERATION
        assert new_session_manager.current_workflow.work_directory == "/complex/work/dir"
        
        # Verify approval status
        assert new_session_manager.user_approval_status["requirements"] == "approved"
        assert new_session_manager.user_approval_status["design"] == "pending"
        assert new_session_manager.user_approval_status["tasks"] == "needs_revision"
        
        # Verify phase results
        assert "requirements" in new_session_manager.phase_results
        assert new_session_manager.phase_results["requirements"]["success"] is True
        assert "design" in new_session_manager.phase_results
        
        # Verify execution log
        assert len(new_session_manager.execution_log) == 2
        assert new_session_manager.execution_log[0]["event_type"] == "phase_completed"
        assert new_session_manager.execution_log[1]["event_type"] == "phase_approval"
        
        # Verify workflow summary
        assert new_session_manager.workflow_summary["phases_completed"] == ["requirements", "design"]
        assert new_session_manager.workflow_summary["token_usage"]["total"] == 1500
        assert new_session_manager.workflow_summary["auto_approvals"] == ["requirements"]
    
    @pytest.mark.integration
    def test_session_error_recovery_with_corrupted_file(self, session_manager, temp_workspace):
        """Test session error recovery when session file is corrupted."""
        # Create session file with invalid JSON
        session_file = Path(temp_workspace) / "memory" / "session_state.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(session_file, 'w') as f:
            f.write("invalid json content {")
        
        # Attempt to load session (should create new one)
        session_data = session_manager.load_or_create_session()
        
        # Verify new session was created
        assert session_data["session_id"] is not None
        assert session_data["current_workflow"] is None
        assert session_data["user_approval_status"] == {}
        
        # Verify session file was overwritten with valid content
        with open(session_file, 'r') as f:
            file_data = json.load(f)
        assert file_data["session_id"] == session_data["session_id"]
    
    @pytest.mark.integration
    def test_session_directory_creation(self, temp_workspace):
        """Test that session manager creates necessary directories."""
        # Remove memory directory if it exists
        memory_dir = Path(temp_workspace) / "memory"
        if memory_dir.exists():
            shutil.rmtree(memory_dir)
        
        # Create session manager and load session
        session_manager = SessionManager(temp_workspace)
        session_data = session_manager.load_or_create_session()
        
        # Verify directory was created
        assert memory_dir.exists()
        assert memory_dir.is_dir()
        
        # Verify session file was created
        session_file = memory_dir / "session_state.json"
        assert session_file.exists()
        assert session_file.is_file()
    
    @pytest.mark.integration
    def test_get_session_data_completeness(self, session_manager):
        """Test that get_session_data returns complete session information."""
        # Create session with data
        session_manager.load_or_create_session()
        
        # Add comprehensive test data
        session_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.DESIGN,
            work_directory="/test/work"
        )
        session_manager.user_approval_status["requirements"] = "approved"
        session_manager.phase_results["requirements"] = {"success": True}
        session_manager.execution_log.append({"event": "test"})
        session_manager.approval_log.append({"approval": "test"})
        session_manager.error_recovery_attempts["phase1"] = 2
        session_manager.workflow_summary["phases_completed"] = ["requirements"]
        
        # Get session data
        session_data = session_manager.get_session_data()
        
        # Verify completeness
        assert "session_id" in session_data
        assert "current_workflow" in session_data
        assert "user_approval_status" in session_data
        assert "phase_results" in session_data
        assert "execution_log" in session_data
        assert "approval_log" in session_data
        assert "error_recovery_attempts" in session_data
        assert "workflow_summary" in session_data
        
        # Verify workflow data structure
        workflow_data = session_data["current_workflow"]
        assert workflow_data["phase"] == "design"
        assert workflow_data["work_directory"] == "/test/work"
        
        # Verify other data
        assert session_data["user_approval_status"]["requirements"] == "approved"
        assert session_data["phase_results"]["requirements"]["success"] is True
        assert len(session_data["execution_log"]) == 1
        assert len(session_data["approval_log"]) == 1
        assert session_data["error_recovery_attempts"]["phase1"] == 2
        assert session_data["workflow_summary"]["phases_completed"] == ["requirements"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])