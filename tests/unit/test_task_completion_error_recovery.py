"""
Unit tests for task completion error recovery and configuration methods.

Tests the individual methods and configuration options for task completion
error recovery and fallback mechanisms.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from autogen_framework.workflow_manager import WorkflowManager
from autogen_framework.models import WorkflowState, WorkflowPhase, TaskDefinition, TaskFileMapping
from autogen_framework.session_manager import SessionManager
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.context_compressor import ContextCompressor


class TestTaskCompletionErrorRecoveryUnit:
    """Unit tests for task completion error recovery methods."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def workflow_manager(self, temp_workspace, test_llm_config):
        """Create WorkflowManager instance for testing."""
        mock_agent_manager = MagicMock()
        mock_agent_manager.llm_config = test_llm_config
        
        session_manager = SessionManager(workspace_path=temp_workspace)
        memory_manager = MemoryManager(workspace_path=str(temp_workspace))
        context_compressor = ContextCompressor(llm_config=test_llm_config)
        token_manager = MagicMock()
        
        workflow_manager = WorkflowManager(
            agent_manager=mock_agent_manager,
            session_manager=session_manager,
            memory_manager=memory_manager,
            context_compressor=context_compressor,
            token_manager=token_manager
        )
        
        # Set up workflow state
        work_dir = temp_workspace / "test_project"
        work_dir.mkdir()
        workflow_manager.current_workflow = WorkflowState(
            phase=WorkflowPhase.IMPLEMENTATION,
            work_directory=str(work_dir)
        )
        
        return workflow_manager
    
    def test_load_task_completion_config_default_values(self, workflow_manager):
        """Test loading task completion configuration with default values."""
        # Clear environment variables
        env_vars_to_clear = [
            'TASK_REAL_TIME_UPDATES_ENABLED',
            'TASK_FALLBACK_TO_BATCH_ENABLED',
            'TASK_MAX_INDIVIDUAL_UPDATE_RETRIES',
            'TASK_INDIVIDUAL_UPDATE_RETRY_DELAY',
            'TASK_FILE_LOCK_TIMEOUT',
            'TASK_DETAILED_LOGGING_ENABLED',
            'TASK_RECOVERY_MECHANISM_ENABLED'
        ]
        
        with patch.dict(os.environ, {}, clear=True):
            config = workflow_manager._load_task_completion_config()
            
            # Verify default values
            assert config['real_time_updates_enabled'] is True
            assert config['fallback_to_batch_enabled'] is True
            assert config['max_individual_update_retries'] == 3
            assert config['individual_update_retry_delay'] == 0.1
            assert config['file_lock_timeout'] == 5
            assert config['detailed_logging_enabled'] is True
            assert config['recovery_mechanism_enabled'] is True
    
    def test_load_task_completion_config_custom_values(self, workflow_manager):
        """Test loading task completion configuration with custom environment values."""
        custom_env = {
            'TASK_REAL_TIME_UPDATES_ENABLED': 'false',
            'TASK_FALLBACK_TO_BATCH_ENABLED': 'false',
            'TASK_MAX_INDIVIDUAL_UPDATE_RETRIES': '5',
            'TASK_INDIVIDUAL_UPDATE_RETRY_DELAY': '0.2',
            'TASK_FILE_LOCK_TIMEOUT': '10',
            'TASK_DETAILED_LOGGING_ENABLED': 'false',
            'TASK_RECOVERY_MECHANISM_ENABLED': 'false'
        }
        
        with patch.dict(os.environ, custom_env):
            config = workflow_manager._load_task_completion_config()
            
            # Verify custom values
            assert config['real_time_updates_enabled'] is False
            assert config['fallback_to_batch_enabled'] is False
            assert config['max_individual_update_retries'] == 5
            assert config['individual_update_retry_delay'] == 0.2
            assert config['file_lock_timeout'] == 10
            assert config['detailed_logging_enabled'] is False
            assert config['recovery_mechanism_enabled'] is False
    
    def test_record_task_update_stats(self, workflow_manager):
        """Test recording task update statistics."""
        # Initialize stats
        workflow_manager.task_update_stats = {
            'individual_updates_attempted': 0,
            'individual_updates_successful': 0,
            'fallback_updates_used': 0,
            'file_access_errors': 0,
            'task_identification_errors': 0,
            'partial_update_recoveries': 0
        }
        
        # Record some statistics
        workflow_manager._record_task_update_stats('individual_updates_attempted', 3)
        workflow_manager._record_task_update_stats('individual_updates_successful', 2)
        workflow_manager._record_task_update_stats('file_access_errors', 1)
        
        # Verify statistics were recorded
        assert workflow_manager.task_update_stats['individual_updates_attempted'] == 3
        assert workflow_manager.task_update_stats['individual_updates_successful'] == 2
        assert workflow_manager.task_update_stats['file_access_errors'] == 1
        assert workflow_manager.task_update_stats['fallback_updates_used'] == 0
    
    def test_log_task_completion_debug_enabled(self, workflow_manager):
        """Test debug logging when detailed logging is enabled."""
        workflow_manager.task_completion_config = {'detailed_logging_enabled': True}
        
        with patch.object(workflow_manager.logger, 'debug') as mock_debug:
            workflow_manager._log_task_completion_debug("Test message", {"key": "value"})
            
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args[0][0]
            assert "[TaskCompletion]" in call_args
            assert "Test message" in call_args
    
    def test_log_task_completion_debug_disabled(self, workflow_manager):
        """Test debug logging when detailed logging is disabled."""
        workflow_manager.task_completion_config = {'detailed_logging_enabled': False}
        
        with patch.object(workflow_manager.logger, 'debug') as mock_debug:
            workflow_manager._log_task_completion_debug("Test message")
            
            mock_debug.assert_not_called()
    
    def test_attempt_task_mapping_recovery_success(self, workflow_manager, temp_workspace):
        """Test successful task mapping recovery."""
        # Create a tasks file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_content = """# Tasks
- [ ] 1. First task
- [ ] 2. Second task
"""
        tasks_file.write_text(tasks_content)
        
        # Clear existing mappings
        workflow_manager.task_file_mappings = {}
        
        # Mock _parse_tasks_from_file to simulate successful parsing
        def mock_parse_tasks(file_path):
            workflow_manager.task_file_mappings = {
                "1": TaskFileMapping("1", 1, "- [ ] 1. First task", 0),
                "2": TaskFileMapping("2", 2, "- [ ] 2. Second task", 0)
            }
            return []
        
        workflow_manager._parse_tasks_from_file = mock_parse_tasks
        
        # Attempt recovery
        result = workflow_manager._attempt_task_mapping_recovery(str(tasks_file), "1")
        
        # Verify recovery was successful
        assert result is True
        assert "1" in workflow_manager.task_file_mappings
    
    def test_attempt_task_mapping_recovery_failure(self, workflow_manager, temp_workspace):
        """Test failed task mapping recovery."""
        # Create a tasks file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text("# Empty tasks file")
        
        # Mock _parse_tasks_from_file to simulate failed parsing
        def mock_parse_tasks(file_path):
            workflow_manager.task_file_mappings = {}
            return []
        
        workflow_manager._parse_tasks_from_file = mock_parse_tasks
        
        # Attempt recovery
        result = workflow_manager._attempt_task_mapping_recovery(str(tasks_file), "1")
        
        # Verify recovery failed
        assert result is False
    
    def test_attempt_line_number_recovery_success(self, workflow_manager):
        """Test successful line number recovery."""
        lines = [
            "# Tasks",
            "- [ ] 1. First task",
            "- [ ] 2. Second task",
            "- [ ] 3. Third task"
        ]
        
        # Create a mapping with incorrect line number
        mapping = TaskFileMapping("2", 999, "- [ ] 2. Second task", 0)
        
        # Attempt recovery
        result = workflow_manager._attempt_line_number_recovery(lines, "2", mapping)
        
        # Verify recovery was successful
        assert result is True
        assert mapping.line_number == 2
        assert "2. Second task" in mapping.original_line
    
    def test_attempt_line_number_recovery_failure(self, workflow_manager):
        """Test failed line number recovery."""
        lines = [
            "# Tasks",
            "- [ ] 1. First task",
            "- [ ] 3. Third task"
        ]
        
        # Create a mapping for non-existent task
        mapping = TaskFileMapping("2", 999, "- [ ] 2. Second task", 0)
        
        # Attempt recovery
        result = workflow_manager._attempt_line_number_recovery(lines, "2", mapping)
        
        # Verify recovery failed
        assert result is False
        assert mapping.line_number == 999  # Unchanged
    
    def test_attempt_task_line_recovery_success(self, workflow_manager):
        """Test successful task line recovery."""
        lines = [
            "# Tasks",
            "- [ ] 1. First task",
            "- [ ] 2. Second task",
            "- [ ] 3. Third task"
        ]
        
        # Create a mapping with incorrect line number but within search window
        mapping = TaskFileMapping("2", 1, "wrong line", 0)
        
        # Mock _verify_task_line to return True for correct line
        def mock_verify_task_line(line, task_id):
            return task_id == "2" and "2. Second task" in line
        
        workflow_manager._verify_task_line = mock_verify_task_line
        
        # Attempt recovery
        result = workflow_manager._attempt_task_line_recovery(lines, "2", mapping)
        
        # Verify recovery was successful
        assert result is True
        assert mapping.line_number == 2
        assert "2. Second task" in mapping.original_line
    
    def test_attempt_task_line_recovery_failure(self, workflow_manager):
        """Test failed task line recovery."""
        lines = [
            "# Tasks",
            "- [ ] 1. First task",
            "- [ ] 3. Third task"
        ]
        
        # Create a mapping for non-existent task
        mapping = TaskFileMapping("2", 1, "wrong line", 0)
        
        # Mock _verify_task_line to always return False
        workflow_manager._verify_task_line = lambda line, task_id: False
        
        # Attempt recovery
        result = workflow_manager._attempt_task_line_recovery(lines, "2", mapping)
        
        # Verify recovery failed
        assert result is False
        assert mapping.line_number == 1  # Unchanged
    
    def test_log_task_update_statistics(self, workflow_manager):
        """Test logging of task update statistics."""
        # Set up statistics
        workflow_manager.task_update_stats = {
            'individual_updates_attempted': 10,
            'individual_updates_successful': 8,
            'fallback_updates_used': 2,
            'file_access_errors': 1,
            'task_identification_errors': 1,
            'partial_update_recoveries': 0
        }
        
        with patch.object(workflow_manager.logger, 'info') as mock_info:
            workflow_manager._log_task_update_statistics(
                total_tasks=10,
                failed_updates=2,
                fallback_used=True,
                recovery_successful=True
            )
            
            # Verify comprehensive logging occurred
            assert mock_info.call_count >= 8  # At least 8 info messages
            
            # Check for specific log messages
            info_messages = [call.args[0] for call in mock_info.call_args_list]
            assert any("Task completion update statistics:" in msg for msg in info_messages)
            assert any("Total tasks processed: 10" in msg for msg in info_messages)
            assert any("Individual updates attempted: 10" in msg for msg in info_messages)
            assert any("Individual updates successful: 8" in msg for msg in info_messages)
            assert any("Individual update success rate: 80.0%" in msg for msg in info_messages)
    
    def test_ensure_no_tasks_lost_all_tasks_found(self, workflow_manager, temp_workspace):
        """Test task loss prevention when all tasks are properly accounted for."""
        # Create tasks file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_content = """# Tasks
- [x] 1. First task
- [ ] 2. Second task
- [x] 3. Third task
"""
        tasks_file.write_text(tasks_content)
        
        # Create expected tasks
        expected_tasks = [
            TaskDefinition("1", "First task", "First task", [], [], completed=False),
            TaskDefinition("2", "Second task", "Second task", [], [], completed=False),
            TaskDefinition("3", "Third task", "Third task", [], [], completed=False)
        ]
        
        # Create task results
        task_results = [
            {"task_id": "1", "success": True},
            {"task_id": "2", "success": False},
            {"task_id": "3", "success": True}
        ]
        
        # Test task loss prevention
        result = workflow_manager._ensure_no_tasks_lost(str(tasks_file), expected_tasks, task_results)
        
        # Verify no tasks were lost
        assert result is True
    
    def test_ensure_no_tasks_lost_with_recovery_disabled(self, workflow_manager, temp_workspace):
        """Test task loss prevention when recovery mechanism is disabled."""
        workflow_manager.task_completion_config = {'recovery_mechanism_enabled': False}
        
        # Create tasks file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text("# Empty")
        
        # Test task loss prevention
        result = workflow_manager._ensure_no_tasks_lost(str(tasks_file), [], [])
        
        # Verify it returns True when disabled (no-op)
        assert result is True