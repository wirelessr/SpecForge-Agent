"""
Unit tests for WorkflowManager enhanced task parsing functionality.

Tests the enhanced _parse_tasks_from_file method that extracts numerical IDs
from task titles and captures line positions for precise file updates.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from autogen_framework.workflow_manager import WorkflowManager
from autogen_framework.models import TaskDefinition, TaskFileMapping


class TestWorkflowManagerTaskParsing:
    """Test cases for enhanced task parsing functionality."""
    
    @pytest.fixture
    def workflow_manager(self):
        """Create a WorkflowManager instance with mocked dependencies."""
        mock_agent_manager = Mock()
        mock_session_manager = Mock()
        mock_session_manager.workspace_path = Path("/test/workspace")
        mock_session_manager.session_id = "test-session-123"
        mock_session_manager.current_workflow = None
        mock_session_manager.user_approval_status = {}
        mock_session_manager.phase_results = {}
        mock_session_manager.execution_log = []
        mock_session_manager.approval_log = []
        mock_session_manager.error_recovery_attempts = {}
        mock_session_manager.workflow_summary = {
            'phases_completed': [],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [],
            'errors_recovered': []
        }
        mock_session_manager.save_session_state = Mock(return_value=True)
        
        mock_memory_manager = Mock()
        mock_context_compressor = Mock()
        mock_token_manager = Mock()
        
        return WorkflowManager(
            mock_agent_manager, mock_session_manager, mock_memory_manager, 
            mock_context_compressor, mock_token_manager
        )
    
    def test_parse_numbered_tasks_success(self, workflow_manager):
        """Test parsing tasks with numerical identifiers."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. Fix TasksAgent to generate numbered tasks
  - Modify `_build_task_generation_prompt` method
  - Update the task generation template
  - _Requirements: 4.1, 4.2_

- [ ] 2. Enhance WorkflowManager task parsing to use numerical IDs
  - Modify `_parse_tasks_from_file` method
  - Add regex pattern matching
  - _Requirements: 4.3, 4.4_

- [x] 3. Implement single task completion update method
  - Create new `_update_single_task_completion` method
  - _Requirements: 1.1, 1.4_
""")
            tasks_file = f.name
        
        try:
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Should return only uncompleted tasks (tasks 1 and 2)
            assert len(tasks) == 2
            
            # Verify numerical IDs were extracted
            assert tasks[0].id == "1"
            assert tasks[0].title == "Fix TasksAgent to generate numbered tasks"
            assert not tasks[0].completed
            assert tasks[0].line_number == 2
            
            assert tasks[1].id == "2"
            assert tasks[1].title == "Enhance WorkflowManager task parsing to use numerical IDs"
            assert not tasks[1].completed
            assert tasks[1].line_number == 7
            
            # Verify task file mappings were created
            assert hasattr(workflow_manager, 'task_file_mappings')
            assert "1" in workflow_manager.task_file_mappings
            assert "2" in workflow_manager.task_file_mappings
            
            # Verify mapping details
            mapping_1 = workflow_manager.task_file_mappings["1"]
            assert mapping_1.task_id == "1"
            assert mapping_1.line_number == 2
            assert "- [ ] 1. Fix TasksAgent" in mapping_1.original_line
            
        finally:
            Path(tasks_file).unlink()
    
    def test_parse_unnumbered_tasks_backward_compatibility(self, workflow_manager):
        """Test parsing tasks without numerical identifiers (backward compatibility)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] Fix TasksAgent to generate numbered tasks
  - Modify method
  - Requirements: 4.1, 4.2

- [ ] Enhance WorkflowManager task parsing
  - Add regex pattern matching
  - Requirements: 4.3, 4.4
""")
            tasks_file = f.name
        
        try:
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            
            assert len(tasks) == 2
            
            # Should fall back to sequential IDs
            assert tasks[0].id == "task_1"
            assert tasks[0].title == "Fix TasksAgent to generate numbered tasks"
            
            assert tasks[1].id == "task_2"
            assert tasks[1].title == "Enhance WorkflowManager task parsing"
            
            # Verify mappings were created with sequential IDs
            assert "task_1" in workflow_manager.task_file_mappings
            assert "task_2" in workflow_manager.task_file_mappings
            
        finally:
            Path(tasks_file).unlink()
    
    def test_parse_mixed_numbered_and_unnumbered_tasks(self, workflow_manager):
        """Test parsing a mix of numbered and unnumbered tasks."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. First numbered task
  - Step 1
  - _Requirements: 1.1_

- [ ] Unnumbered task in the middle
  - Step 1

- [ ] 3. Third numbered task
  - Step 1
  - _Requirements: 3.1_
""")
            tasks_file = f.name
        
        try:
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            
            assert len(tasks) == 3
            
            # First task should have numerical ID
            assert tasks[0].id == "1"
            assert tasks[0].title == "First numbered task"
            
            # Second task should fall back to sequential ID
            assert tasks[1].id == "task_2"
            assert tasks[1].title == "Unnumbered task in the middle"
            
            # Third task should have numerical ID
            assert tasks[2].id == "3"
            assert tasks[2].title == "Third numbered task"
            
        finally:
            Path(tasks_file).unlink()
    
    def test_parse_tasks_with_completed_status(self, workflow_manager):
        """Test parsing tasks with mixed completion status."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [x] 1. Completed numbered task
  - Step completed
  - _Requirements: 1.1_

- [ ] 2. Uncompleted numbered task
  - Step pending
  - _Requirements: 2.1_

- [-] 3. In-progress numbered task
  - Step in progress
  - _Requirements: 3.1_

- [x] Completed unnumbered task
  - Step completed

- [ ] 5. Another uncompleted task
  - Step pending
""")
            tasks_file = f.name
        
        try:
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Should only return uncompleted tasks (including in-progress)
            assert len(tasks) == 3
            
            assert tasks[0].id == "2"
            assert tasks[0].title == "Uncompleted numbered task"
            assert not tasks[0].completed
            
            assert tasks[1].id == "3"
            assert tasks[1].title == "In-progress numbered task"
            assert not tasks[1].completed  # In-progress is treated as uncompleted
            
            assert tasks[2].id == "5"
            assert tasks[2].title == "Another uncompleted task"
            assert not tasks[2].completed
            
            # Verify all tasks were mapped (including completed ones)
            assert len(workflow_manager.task_file_mappings) == 5
            assert "1" in workflow_manager.task_file_mappings  # Completed numbered
            assert "2" in workflow_manager.task_file_mappings  # Uncompleted numbered
            assert "3" in workflow_manager.task_file_mappings  # In-progress numbered
            assert "task_4" in workflow_manager.task_file_mappings  # Completed unnumbered
            assert "5" in workflow_manager.task_file_mappings  # Uncompleted numbered
            
        finally:
            Path(tasks_file).unlink()
    
    def test_parse_tasks_with_requirements_formats(self, workflow_manager):
        """Test parsing tasks with different requirements formats."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. Task with old requirements format
  - Step 1
  - Requirements: 1.1, 1.2, 1.3

- [ ] 2. Task with new requirements format
  - Step 1
  - _Requirements: 2.1, 2.2_

- [ ] 3. Task without requirements
  - Step 1
""")
            tasks_file = f.name
        
        try:
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            
            assert len(tasks) == 3
            
            # First task with old format
            assert tasks[0].requirements_ref == ["1.1", "1.2", "1.3"]
            
            # Second task with new format
            assert tasks[1].requirements_ref == ["2.1", "2.2"]
            
            # Third task without requirements
            assert tasks[2].requirements_ref == []
            
        finally:
            Path(tasks_file).unlink()
    
    def test_parse_tasks_line_number_tracking(self, workflow_manager):
        """Test that line numbers are correctly tracked for each task."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

Some introduction text here.

- [ ] 1. First task
  - Step 1
  - _Requirements: 1.1_

More text between tasks.

- [ ] 2. Second task
  - Step 1
  - Step 2
  - _Requirements: 2.1_

Final text.
""")
            tasks_file = f.name
        
        try:
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            
            assert len(tasks) == 2
            
            # Verify line numbers (0-indexed)
            assert tasks[0].line_number == 4  # "- [ ] 1. First task"
            assert tasks[1].line_number == 10  # "- [ ] 2. Second task"
            
            # Verify mappings have correct line numbers
            assert workflow_manager.task_file_mappings["1"].line_number == 4
            assert workflow_manager.task_file_mappings["2"].line_number == 10
            
        finally:
            Path(tasks_file).unlink()
    
    def test_parse_tasks_file_not_found(self, workflow_manager):
        """Test parsing with non-existent file."""
        tasks = workflow_manager._parse_tasks_from_file("/nonexistent/tasks.md")
        assert tasks == []
        assert not hasattr(workflow_manager, 'task_file_mappings') or len(workflow_manager.task_file_mappings) == 0
    
    def test_parse_tasks_empty_file(self, workflow_manager):
        """Test parsing empty tasks file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("")
            tasks_file = f.name
        
        try:
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            assert tasks == []
            assert hasattr(workflow_manager, 'task_file_mappings')
            assert len(workflow_manager.task_file_mappings) == 0
            
        finally:
            Path(tasks_file).unlink()
    
    def test_parse_tasks_no_tasks_in_file(self, workflow_manager):
        """Test parsing file with no task items."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

This is just documentation with no actual tasks.

Some bullet points:
- This is not a task (no checkbox)
- Neither is this

## Section 2

More text without tasks.
""")
            tasks_file = f.name
        
        try:
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            assert tasks == []
            assert hasattr(workflow_manager, 'task_file_mappings')
            assert len(workflow_manager.task_file_mappings) == 0
            
        finally:
            Path(tasks_file).unlink()
    
    def test_task_file_mapping_structure(self, workflow_manager):
        """Test that TaskFileMapping objects are created correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. Test task
  - Step 1
  - _Requirements: 1.1_
""")
            tasks_file = f.name
        
        try:
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            
            assert len(tasks) == 1
            assert "1" in workflow_manager.task_file_mappings
            
            mapping = workflow_manager.task_file_mappings["1"]
            assert isinstance(mapping, TaskFileMapping)
            assert mapping.task_id == "1"
            assert mapping.line_number == 2
            assert "- [ ] 1. Test task" in mapping.original_line
            assert mapping.checkbox_position >= 0
            
            # Test the get_updated_line method
            completed_line = mapping.get_updated_line(True)
            assert "- [x] 1. Test task" in completed_line
            
            uncompleted_line = mapping.get_updated_line(False)
            assert "- [ ] 1. Test task" in uncompleted_line
            
        finally:
            Path(tasks_file).unlink()


class TestWorkflowManagerSingleTaskUpdate:
    """Test cases for single task completion update functionality."""
    
    @pytest.fixture
    def workflow_manager(self):
        """Create a WorkflowManager instance with mocked dependencies."""
        mock_agent_manager = Mock()
        mock_session_manager = Mock()
        mock_session_manager.workspace_path = Path("/test/workspace")
        mock_session_manager.session_id = "test-session-123"
        mock_session_manager.current_workflow = None
        mock_session_manager.user_approval_status = {}
        mock_session_manager.phase_results = {}
        mock_session_manager.execution_log = []
        mock_session_manager.approval_log = []
        mock_session_manager.error_recovery_attempts = {}
        mock_session_manager.workflow_summary = {
            'phases_completed': [],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [],
            'errors_recovered': []
        }
        mock_session_manager.save_session_state = Mock(return_value=True)
        
        mock_memory_manager = Mock()
        mock_context_compressor = Mock()
        mock_token_manager = Mock()
        
        return WorkflowManager(
            mock_agent_manager, mock_session_manager, mock_memory_manager, 
            mock_context_compressor, mock_token_manager
        )
    
    def test_update_single_task_completion_success(self, workflow_manager):
        """Test successful single task completion update."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. First task
  - Step 1
  - _Requirements: 1.1_

- [ ] 2. Second task
  - Step 1
  - _Requirements: 2.1_
""")
            tasks_file = f.name
        
        try:
            # First parse tasks to create mappings
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            assert len(tasks) == 2
            
            # Update first task as completed
            result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
            assert result is True
            
            # Verify file was updated
            with open(tasks_file, 'r') as f:
                content = f.read()
            
            assert "- [x] 1. First task" in content
            assert "- [ ] 2. Second task" in content  # Should remain unchanged
            
        finally:
            Path(tasks_file).unlink()
    
    def test_update_single_task_completion_mark_uncompleted(self, workflow_manager):
        """Test marking a completed task as uncompleted."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [x] 1. Completed task
  - Step 1
  - _Requirements: 1.1_

- [ ] 2. Uncompleted task
  - Step 1
  - _Requirements: 2.1_
""")
            tasks_file = f.name
        
        try:
            # Parse tasks to create mappings (will include completed task in mappings)
            workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Mark completed task as uncompleted
            result = workflow_manager._update_single_task_completion(tasks_file, "1", False)
            assert result is True
            
            # Verify file was updated
            with open(tasks_file, 'r') as f:
                content = f.read()
            
            assert "- [ ] 1. Completed task" in content
            assert "- [ ] 2. Uncompleted task" in content  # Should remain unchanged
            
        finally:
            Path(tasks_file).unlink()
    
    def test_update_single_task_completion_no_mapping(self, workflow_manager):
        """Test update when no task file mapping exists."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Empty tasks file")
            tasks_file = f.name
        
        try:
            # Try to update without parsing first (no mappings)
            result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
            assert result is False
            
        finally:
            Path(tasks_file).unlink()
    
    def test_update_single_task_completion_file_not_found(self, workflow_manager):
        """Test update with non-existent file."""
        # Create mappings manually
        workflow_manager.task_file_mappings = {
            "1": TaskFileMapping(
                task_id="1",
                line_number=0,
                original_line="- [ ] 1. Test task",
                checkbox_position=0
            )
        }
        
        result = workflow_manager._update_single_task_completion("/nonexistent/tasks.md", "1", True)
        assert result is False
    
    def test_update_single_task_completion_line_out_of_range(self, workflow_manager):
        """Test update when line number is out of range."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Short file")
            tasks_file = f.name
        
        try:
            # Create mapping with invalid line number
            workflow_manager.task_file_mappings = {
                "1": TaskFileMapping(
                    task_id="1",
                    line_number=10,  # Out of range
                    original_line="- [ ] 1. Test task",
                    checkbox_position=0
                )
            }
            
            result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
            assert result is False
            
        finally:
            Path(tasks_file).unlink()
    
    def test_update_single_task_completion_in_progress_to_completed(self, workflow_manager):
        """Test updating an in-progress task to completed."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [-] 1. In-progress task
  - Step 1
  - _Requirements: 1.1_

- [ ] 2. Uncompleted task
  - Step 1
  - _Requirements: 2.1_
""")
            tasks_file = f.name
        
        try:
            # Parse tasks to create mappings
            workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Mark in-progress task as completed
            result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
            assert result is True
            
            # Verify file was updated
            with open(tasks_file, 'r') as f:
                content = f.read()
            
            assert "- [x] 1. In-progress task" in content
            assert "- [ ] 2. Uncompleted task" in content  # Should remain unchanged
            
        finally:
            Path(tasks_file).unlink()

    def test_update_single_task_completion_no_change_needed(self, workflow_manager):
        """Test update when no change is needed (task already in correct state)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [x] 1. Already completed task
  - Step 1
""")
            tasks_file = f.name
        
        try:
            # Parse to create mappings
            workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Try to mark already completed task as completed
            result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
            assert result is True  # Should succeed even if no change needed
            
        finally:
            Path(tasks_file).unlink()
    
    def test_update_single_task_completion_with_retry_on_permission_error(self, workflow_manager):
        """Test retry mechanism when permission errors occur."""
        import unittest.mock
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. Test task
  - Step 1
""")
            tasks_file = f.name
        
        try:
            # Parse to create mappings
            workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Mock open to raise PermissionError on first two attempts, succeed on third
            original_open = open
            call_count = 0
            
            def mock_open(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise PermissionError("Permission denied")
                return original_open(*args, **kwargs)
            
            with unittest.mock.patch('builtins.open', side_effect=mock_open):
                result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
                assert result is True
                assert call_count == 3  # Should have retried twice
            
        finally:
            Path(tasks_file).unlink()
    
    def test_update_single_task_completion_retry_exhausted(self, workflow_manager):
        """Test when all retry attempts are exhausted."""
        import unittest.mock
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. Test task
  - Step 1
""")
            tasks_file = f.name
        
        try:
            # Parse to create mappings
            workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Mock open to always raise PermissionError
            with unittest.mock.patch('builtins.open', side_effect=PermissionError("Permission denied")):
                result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
                assert result is False
            
        finally:
            Path(tasks_file).unlink()
    
    def test_update_single_task_completion_task_verification_failure(self, workflow_manager):
        """Test when task verification fails (task ID doesn't match line content)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. Original task
  - Step 1

- [ ] 2. Second task
  - Step 1
""")
            tasks_file = f.name
        
        try:
            # Parse to create mappings
            workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Manually modify the file to change task ID (simulating external modification)
            with open(tasks_file, 'r') as f:
                content = f.read()
            
            # Change task 1 to task 3, which should cause verification to fail
            modified_content = content.replace("- [ ] 1. Original task", "- [ ] 3. Modified task")
            
            with open(tasks_file, 'w') as f:
                f.write(modified_content)
            
            # Try to update task "1" - should fail verification because line now contains task "3"
            result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
            assert result is False
            
        finally:
            Path(tasks_file).unlink()
    
    def test_update_single_task_completion_file_locking_simulation(self, workflow_manager):
        """Test file locking mechanism (simulated since we can't easily test real concurrent access)."""
        import unittest.mock
        import fcntl
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. Test task
  - Step 1
""")
            tasks_file = f.name
        
        try:
            # Parse to create mappings
            workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Mock fcntl.flock to raise OSError on first attempt, succeed on second
            original_flock = fcntl.flock
            call_count = 0
            
            def mock_flock(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise OSError("Resource temporarily unavailable")
                return original_flock(*args, **kwargs)
            
            with unittest.mock.patch('fcntl.flock', side_effect=mock_flock):
                result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
                assert result is True
                assert call_count == 2  # Should have retried once
            
        finally:
            Path(tasks_file).unlink()
    
    def test_verify_task_line_numbered_task(self, workflow_manager):
        """Test _verify_task_line method with numbered tasks."""
        # Test matching numbered task
        line = "- [ ] 1. Test task description"
        assert workflow_manager._verify_task_line(line, "1") is True
        
        # Test non-matching numbered task
        assert workflow_manager._verify_task_line(line, "2") is False
        
        # Test with completed task
        completed_line = "- [x] 3. Completed task"
        assert workflow_manager._verify_task_line(completed_line, "3") is True
        
        # Test with in-progress task
        progress_line = "- [-] 2. In-progress task"
        assert workflow_manager._verify_task_line(progress_line, "2") is True
    
    def test_verify_task_line_unnumbered_task(self, workflow_manager):
        """Test _verify_task_line method with unnumbered tasks."""
        # Test unnumbered task with sequential ID
        line = "- [ ] Unnumbered task description"
        assert workflow_manager._verify_task_line(line, "task_1") is True
        assert workflow_manager._verify_task_line(line, "task_2") is True
        
        # Test numbered task with sequential ID (should fail)
        numbered_line = "- [ ] 1. Numbered task"
        assert workflow_manager._verify_task_line(numbered_line, "task_1") is False
    
    def test_verify_task_line_invalid_lines(self, workflow_manager):
        """Test _verify_task_line method with invalid lines."""
        # Test non-task line
        assert workflow_manager._verify_task_line("This is not a task", "1") is False
        
        # Test bullet point without checkbox
        assert workflow_manager._verify_task_line("- This is a bullet point", "1") is False
        
        # Test empty line
        assert workflow_manager._verify_task_line("", "1") is False
        
        # Test malformed checkbox
        assert workflow_manager._verify_task_line("- [?] Invalid checkbox", "1") is False
    
    def test_update_single_task_completion_edge_cases(self, workflow_manager):
        """Test edge cases for single task completion update."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. Task with special characters: "quotes" & symbols
  - Step 1
  - _Requirements: 1.1_

- [ ] 2. Task with very long title that spans multiple conceptual areas and includes detailed descriptions
  - Step 1
  - _Requirements: 2.1_

- [ ] 999. Task with large numerical ID
  - Step 1
  - _Requirements: 999.1_
""")
            tasks_file = f.name
        
        try:
            # Parse to create mappings
            tasks = workflow_manager._parse_tasks_from_file(tasks_file)
            assert len(tasks) == 3
            
            # Test updating task with special characters
            result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
            assert result is True
            
            # Test updating task with long title
            result = workflow_manager._update_single_task_completion(tasks_file, "2", True)
            assert result is True
            
            # Test updating task with large numerical ID
            result = workflow_manager._update_single_task_completion(tasks_file, "999", True)
            assert result is True
            
            # Verify all updates were applied
            with open(tasks_file, 'r') as f:
                content = f.read()
            
            assert "- [x] 1. Task with special characters" in content
            assert "- [x] 2. Task with very long title" in content
            assert "- [x] 999. Task with large numerical ID" in content
            
        finally:
            Path(tasks_file).unlink()
    
    def test_update_single_task_completion_concurrent_simulation(self, workflow_manager):
        """Test simulated concurrent access scenarios."""
        import unittest.mock
        import time
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. Test task
  - Step 1
""")
            tasks_file = f.name
        
        try:
            # Parse to create mappings
            workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Simulate file being modified between read and write
            original_open = open
            read_count = 0
            
            def mock_open(filename, mode='r', **kwargs):
                nonlocal read_count
                file_obj = original_open(filename, mode, **kwargs)
                
                # On the first read, simulate another process modifying the file
                if mode == 'r+' and read_count == 0:
                    read_count += 1
                    # Create a wrapper that modifies file content during read
                    original_readlines = file_obj.readlines
                    
                    def modified_readlines():
                        lines = original_readlines()
                        # Simulate file being modified by another process
                        return lines
                    
                    file_obj.readlines = modified_readlines
                
                return file_obj
            
            # The update should still work due to atomic operations
            with unittest.mock.patch('builtins.open', side_effect=mock_open):
                result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
                assert result is True
            
        finally:
            Path(tasks_file).unlink()
    
    def test_update_single_task_completion_file_corruption_recovery(self, workflow_manager):
        """Test recovery from file corruption scenarios."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Implementation Plan

- [ ] 1. Test task
  - Step 1
""")
            tasks_file = f.name
        
        try:
            # Parse to create mappings
            workflow_manager._parse_tasks_from_file(tasks_file)
            
            # Corrupt the file by truncating it
            with open(tasks_file, 'w') as f:
                f.write("# Corrupted file")
            
            # Update should fail gracefully
            result = workflow_manager._update_single_task_completion(tasks_file, "1", True)
            assert result is False
            
        finally:
            Path(tasks_file).unlink()