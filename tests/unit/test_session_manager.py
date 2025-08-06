"""
Unit tests for the SessionManager class.

This module contains unit tests that focus on testing the SessionManager class
in isolation. These tests verify session persistence, state management,
and file I/O operations while maintaining compatibility with existing
session logic and file formats.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock

from autogen_framework.session_manager import SessionManager
from autogen_framework.models import WorkflowState, WorkflowPhase


class TestSessionManager:
    """Test cases for SessionManager class."""
    
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
    
    def test_session_manager_initialization(self, session_manager, temp_workspace):
        """Test SessionManager initialization."""
        assert session_manager.workspace_path == Path(temp_workspace)
        assert session_manager.session_id is None
        assert session_manager.session_file == Path(temp_workspace) / "memory" / "session_state.json"
        assert session_manager.current_workflow is None
        assert session_manager.user_approval_status == {}
        assert session_manager.phase_results == {}
        assert session_manager.execution_log == []
        assert session_manager.approval_log == []
        assert session_manager.error_recovery_attempts == {}
        assert isinstance(session_manager.workflow_summary, dict)
    
    def test_create_new_session(self, session_manager):
        """Test creating a new session."""
        session_data = session_manager._create_new_session()
        
        # Verify session was created
        assert session_manager.session_id is not None
        assert len(session_manager.session_id) > 0
        assert session_manager.current_workflow is None
        assert session_manager.user_approval_status == {}
        assert session_manager.phase_results == {}
        assert session_manager.execution_log == []
        
        # Verify session data structure
        assert session_data['session_id'] == session_manager.session_id
        assert session_data['current_workflow'] is None
        assert session_data['user_approval_status'] == {}
        assert session_data['phase_results'] == {}
        assert session_data['execution_log'] == []
        assert 'last_updated' in session_data
        assert 'save_success' in session_data
    
    def test_save_session_state(self, session_manager):
        """Test saving session state to disk."""
        # Create a new session first
        session_manager._create_new_session()
        
        # Add some test data
        session_manager.phase_results = {"test": {"data": "value"}}
        session_manager.execution_log = [{"event": "test"}]
        
        # Save session state
        result = session_manager.save_session_state()
        
        assert result is True
        assert session_manager.session_file.exists()
        
        # Verify file content
        with open(session_manager.session_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert saved_data['session_id'] == session_manager.session_id
        assert saved_data['phase_results'] == {"test": {"data": "value"}}
        assert saved_data['execution_log'] == [{"event": "test"}]
        assert 'last_updated' in saved_data
    
    def test_load_existing_session(self, session_manager, temp_workspace):
        """Test loading an existing session from disk."""
        # Create session file manually
        session_file = Path(temp_workspace) / "memory" / "session_state.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_session_data = {
            'session_id': 'test-session-123',
            'current_workflow': {
                'phase': 'planning',
                'work_directory': '/test/work'
            },
            'user_approval_status': {'requirements': 'approved'},
            'phase_results': {'test': {'data': 'value'}},
            'execution_log': [{'event': 'test'}],
            'approval_log': [],
            'error_recovery_attempts': {},
            'workflow_summary': {
                'phases_completed': [],
                'tasks_completed': [],
                'token_usage': {},
                'compression_events': [],
                'auto_approvals': [],
                'errors_recovered': []
            },
            'last_updated': '2024-01-01T00:00:00'
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(test_session_data, f)
        
        # Load session
        loaded_data = session_manager.load_or_create_session()
        
        # Verify session was loaded correctly
        assert session_manager.session_id == 'test-session-123'
        assert session_manager.current_workflow is not None
        assert session_manager.current_workflow.phase == WorkflowPhase.PLANNING
        assert session_manager.current_workflow.work_directory == '/test/work'
        assert session_manager.user_approval_status == {'requirements': 'approved'}
        assert session_manager.phase_results == {'test': {'data': 'value'}}
        assert session_manager.execution_log == [{'event': 'test'}]
        
        # Verify returned data
        assert loaded_data['session_id'] == 'test-session-123'
    
    def test_load_or_create_session_no_existing_file(self, session_manager):
        """Test loading session when no existing file exists."""
        # Ensure no session file exists
        assert not session_manager.session_file.exists()
        
        # Load or create session
        session_data = session_manager.load_or_create_session()
        
        # Should create new session
        assert session_manager.session_id is not None
        assert len(session_manager.session_id) > 0
        assert session_manager.current_workflow is None
        assert session_data['session_id'] == session_manager.session_id
    
    def test_load_or_create_session_corrupted_file(self, session_manager, temp_workspace):
        """Test loading session when file is corrupted."""
        # Create corrupted session file
        session_file = Path(temp_workspace) / "memory" / "session_state.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(session_file, 'w') as f:
            f.write("invalid json content")
        
        # Load or create session - should handle corruption gracefully
        session_data = session_manager.load_or_create_session()
        
        # Should create new session due to corruption
        assert session_manager.session_id is not None
        assert len(session_manager.session_id) > 0
        assert session_data['session_id'] == session_manager.session_id
    
    def test_get_session_id(self, session_manager):
        """Test getting session ID."""
        # Initially None
        assert session_manager.get_session_id() is None
        
        # After creating session
        session_manager._create_new_session()
        session_id = session_manager.get_session_id()
        assert session_id is not None
        assert session_id == session_manager.session_id
    
    def test_get_session_data(self, session_manager):
        """Test getting complete session data."""
        # Create session with test data
        session_manager._create_new_session()
        session_manager.phase_results = {"test": {"data": "value"}}
        session_manager.execution_log = [{"event": "test"}]
        
        # Get session data
        session_data = session_manager.get_session_data()
        
        # Verify data structure
        assert session_data['session_id'] == session_manager.session_id
        assert session_data['current_workflow'] is None
        assert session_data['user_approval_status'] == {}
        assert session_data['phase_results'] == {"test": {"data": "value"}}
        assert session_data['execution_log'] == [{"event": "test"}]
        assert session_data['approval_log'] == []
        assert session_data['error_recovery_attempts'] == {}
        assert 'workflow_summary' in session_data
    
    def test_reset_session(self, session_manager):
        """Test resetting session."""
        # Create session with test data
        session_manager._create_new_session()
        original_session_id = session_manager.session_id
        
        # Add some state
        session_manager.phase_results = {"test": {"data": "value"}}
        session_manager.execution_log = [{"event": "test"}]
        session_manager.user_approval_status = {"requirements": "approved"}
        
        # Reset session
        result = session_manager.reset_session()
        
        # Verify reset was successful
        assert result["success"] is True
        assert "Session reset successfully" in result["message"]
        assert result["session_id"] != original_session_id
        
        # Verify state was cleared
        assert session_manager.session_id != original_session_id
        assert session_manager.current_workflow is None
        assert session_manager.user_approval_status == {}
        assert session_manager.phase_results == {}
        assert session_manager.execution_log == []
        assert session_manager.approval_log == []
        assert session_manager.error_recovery_attempts == {}
        
        # Verify old session file was removed and new one created
        assert session_manager.session_file.exists()
    
    def test_reset_session_file_removal_error(self, session_manager):
        """Test reset session when file removal fails."""
        # Create session
        session_manager._create_new_session()
        
        # Mock Path.unlink to raise an exception
        with patch('pathlib.Path.unlink', side_effect=OSError("Permission denied")):
            result = session_manager.reset_session()
            
            # Should still succeed but log the error
            assert result["success"] is False
            assert "Failed to reset session" in result["error"]
    
    def test_workflow_state_serialization(self, session_manager):
        """Test that WorkflowState objects are properly serialized and deserialized."""
        # Create session with workflow state
        session_manager._create_new_session()
        session_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.DESIGN,
            work_directory="/test/work"
        )
        
        # Save and reload
        session_manager.save_session_state()
        
        # Create new session manager and load
        new_session_manager = SessionManager(str(session_manager.workspace_path))
        new_session_manager.load_or_create_session()
        
        # Verify workflow state was preserved
        assert new_session_manager.current_workflow is not None
        assert new_session_manager.current_workflow.phase == WorkflowPhase.DESIGN
        assert new_session_manager.current_workflow.work_directory == "/test/work"
    
    def test_session_file_directory_creation(self, session_manager):
        """Test that session file directory is created if it doesn't exist."""
        # Ensure memory directory doesn't exist
        memory_dir = session_manager.session_file.parent
        if memory_dir.exists():
            shutil.rmtree(memory_dir)
        
        # Create and save session
        session_manager._create_new_session()
        
        # Verify directory was created
        assert memory_dir.exists()
        assert session_manager.session_file.exists()
    
    def test_save_session_state_error_handling(self, session_manager):
        """Test error handling in save_session_state."""
        # Create session
        session_manager._create_new_session()
        
        # Mock open to raise an exception
        with patch('builtins.open', side_effect=OSError("Disk full")):
            result = session_manager.save_session_state()
            
            # Should return False on error
            assert result is False
    
    def test_session_data_types_preservation(self, session_manager):
        """Test that different data types are preserved correctly."""
        # Create session with various data types
        session_manager._create_new_session()
        
        test_data = {
            'string': 'test_value',
            'number': 42,
            'float': 3.14,
            'boolean': True,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'},
            'null': None
        }
        
        session_manager.phase_results = test_data
        session_manager.error_recovery_attempts = {'phase1': 2, 'phase2': 0}
        
        # Save and reload
        session_manager.save_session_state()
        
        new_session_manager = SessionManager(str(session_manager.workspace_path))
        new_session_manager.load_or_create_session()
        
        # Verify data types were preserved
        assert new_session_manager.phase_results == test_data
        assert new_session_manager.error_recovery_attempts == {'phase1': 2, 'phase2': 0}
        assert isinstance(new_session_manager.error_recovery_attempts['phase1'], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])