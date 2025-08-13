"""
Integration tests for individual task execution flow with real-time completion marking.

Tests the modified _execute_implementation_phase method to ensure tasks are
executed individually and marked as completed incrementally.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from autogen_framework.workflow_manager import WorkflowManager
from autogen_framework.models import WorkflowState, WorkflowPhase, TaskDefinition, TaskFileMapping
from autogen_framework.session_manager import SessionManager
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.context_compressor import ContextCompressor


class TestIndividualTaskExecutionFlow:
    """Test individual task execution with real-time completion marking."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_tasks_md_content(self):
        """Sample tasks.md content with numbered tasks."""
        return """# Implementation Plan

- [ ] 1. Set up project structure
  - Create directory structure for models, services, repositories
  - Define interfaces that establish system boundaries
  - _Requirements: 1.1_

- [ ] 2. Implement data models and validation
  - Write TypeScript interfaces for all data models
  - Implement validation functions for data integrity
  - _Requirements: 2.1, 3.3, 1.2_

- [ ] 3. Create storage mechanism
  - Write connection management code
  - Create error handling utilities for database operations
  - _Requirements: 2.1, 3.3, 1.2_
"""
    
    @pytest.fixture
    def mock_agent_manager(self):
        """Mock agent manager for testing."""
        agent_manager = MagicMock()
        agent_manager.llm_config = MagicMock()
        
        # Mock individual task execution results
        async def mock_coordinate_agents(action, context):
            if action == "execute_task":
                task = context.get("task")
                # Simulate success for tasks 1 and 3, failure for task 2
                if task.id == "2":
                    return {
                        "success": False,
                        "error": "Simulated task failure",
                        "execution_details": "Task 2 failed for testing"
                    }
                else:
                    return {
                        "success": True,
                        "message": f"Task {task.id} completed successfully",
                        "execution_details": f"Successfully executed task {task.id}"
                    }
            return {"success": True}
        
        agent_manager.coordinate_agents = AsyncMock(side_effect=mock_coordinate_agents)
        return agent_manager
    
    @pytest.fixture
    def workflow_manager(self, temp_workspace, mock_agent_manager):
        """Create WorkflowManager instance for testing."""
        session_manager = SessionManager(workspace_path=temp_workspace)
        memory_manager = MemoryManager(workspace_path=str(temp_workspace))
        
        # Create mock LLM config for ContextCompressor
        from autogen_framework.models import LLMConfig
        mock_llm_config = LLMConfig(
            base_url="http://test.local:8888/openai/v1",
            model="test-model",
            api_key="test-key"
        )
        
        context_compressor = ContextCompressor(llm_config=mock_llm_config)
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
    
    @pytest.mark.integration
    async def test_individual_task_execution_with_completion_marking(
        self, workflow_manager, temp_workspace, sample_tasks_md_content
    ):
        """Test that tasks are executed individually and marked as completed incrementally."""
        # Create tasks.md file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Prepare tasks_result
        tasks_result = {
            "tasks_file": str(tasks_file),
            "success": True
        }
        
        # Execute implementation phase
        result = await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify overall result structure
        assert result["success"] is True  # Overall success despite one task failure
        assert result["total_tasks"] == 3
        assert result["completed_tasks"] == 2  # Tasks 1 and 3 should succeed
        assert result["individual_execution"] is True
        assert "task_results" in result
        
        # Verify individual task results
        task_results = result["task_results"]
        assert len(task_results) == 3
        
        # Check task 1 (should succeed)
        task1_result = next(r for r in task_results if r["task_id"] == "1")
        assert task1_result["success"] is True
        assert task1_result["task_title"] == "Set up project structure"
        
        # Check task 2 (should fail)
        task2_result = next(r for r in task_results if r["task_id"] == "2")
        assert task2_result["success"] is False
        assert task2_result["task_title"] == "Implement data models and validation"
        
        # Check task 3 (should succeed)
        task3_result = next(r for r in task_results if r["task_id"] == "3")
        assert task3_result["success"] is True
        assert task3_result["task_title"] == "Create storage mechanism"
        
        # Verify tasks.md file was updated with completion status
        updated_content = tasks_file.read_text()
        lines = updated_content.split('\n')
        
        # Task 1 should be marked as completed
        task1_line = next(line for line in lines if "1. Set up project structure" in line)
        assert task1_line.startswith("- [x]"), f"Task 1 should be completed, but line is: {task1_line}"
        
        # Task 2 should remain uncompleted
        task2_line = next(line for line in lines if "2. Implement data models and validation" in line)
        assert task2_line.startswith("- [ ]"), f"Task 2 should be uncompleted, but line is: {task2_line}"
        
        # Task 3 should be marked as completed
        task3_line = next(line for line in lines if "3. Create storage mechanism" in line)
        assert task3_line.startswith("- [x]"), f"Task 3 should be completed, but line is: {task3_line}"
    
    @pytest.mark.integration
    async def test_individual_task_execution_calls_agent_manager_correctly(
        self, workflow_manager, temp_workspace, sample_tasks_md_content
    ):
        """Test that individual tasks are executed via separate agent_manager calls."""
        # Create tasks.md file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Prepare tasks_result
        tasks_result = {
            "tasks_file": str(tasks_file),
            "success": True
        }
        
        # Execute implementation phase
        await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify agent_manager.coordinate_agents was called for each task individually
        agent_manager = workflow_manager.agent_manager
        assert agent_manager.coordinate_agents.call_count == 3
        
        # Verify each call was for execute_task (not execute_multiple_tasks)
        calls = agent_manager.coordinate_agents.call_args_list
        for call in calls:
            action, context = call[0]
            assert action == "execute_task"
            assert context["task_type"] == "execute_task"
            assert "task" in context
            assert context["work_dir"] == workflow_manager.current_workflow.work_directory
    
    @pytest.mark.integration
    async def test_fallback_to_batch_update_on_individual_update_failure(
        self, workflow_manager, temp_workspace, sample_tasks_md_content
    ):
        """Test fallback to batch update when individual task updates fail."""
        # Create tasks.md file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Mock _update_single_task_completion to fail for some tasks
        original_update_method = workflow_manager._update_single_task_completion
        
        def mock_update_single_task(tasks_file_path, task_id, success):
            # Fail update for task 2, succeed for others
            if task_id == "2":
                return False
            return original_update_method(tasks_file_path, task_id, success)
        
        workflow_manager._update_single_task_completion = mock_update_single_task
        
        # Mock batch update method to track if it was called
        batch_update_called = False
        original_batch_update = workflow_manager._update_tasks_file_with_completion
        
        def mock_batch_update(tasks_file_path, task_results):
            nonlocal batch_update_called
            batch_update_called = True
            return original_batch_update(tasks_file_path, task_results)
        
        workflow_manager._update_tasks_file_with_completion = mock_batch_update
        
        # Prepare tasks_result
        tasks_result = {
            "tasks_file": str(tasks_file),
            "success": True
        }
        
        # Execute implementation phase
        result = await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify fallback batch update was called
        assert batch_update_called, "Batch update should have been called as fallback"
        
        # Verify result indicates some individual updates failed
        task_results = result["task_results"]
        failed_individual_updates = [r for r in task_results if not r.get("completion_updated", False)]
        assert len(failed_individual_updates) == 1
        assert failed_individual_updates[0]["task_id"] == "2"
    
    @pytest.mark.integration
    async def test_execution_log_records_individual_task_completions(
        self, workflow_manager, temp_workspace, sample_tasks_md_content
    ):
        """Test that execution log records individual task completions."""
        # Create tasks.md file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Prepare tasks_result
        tasks_result = {
            "tasks_file": str(tasks_file),
            "success": True
        }
        
        # Clear execution log
        workflow_manager.execution_log = []
        
        # Execute implementation phase
        await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify execution log contains individual task completion events
        task_completion_events = [
            event for event in workflow_manager.execution_log
            if event.get("event_type") == "task_completed"
        ]
        
        assert len(task_completion_events) == 3, "Should have 3 task completion events"
        
        # Verify each task completion event has correct structure
        for i, event in enumerate(task_completion_events):
            details = event["details"]
            assert "task_id" in details
            assert "task_title" in details
            assert "success" in details
            assert "workflow_id" in details
            assert details["task_index"] == i + 1
            assert details["total_tasks"] == 3
        
        # Verify phase completion event
        phase_completion_events = [
            event for event in workflow_manager.execution_log
            if event.get("event_type") == "phase_completed"
        ]
        
        assert len(phase_completion_events) == 1
        phase_event = phase_completion_events[0]
        assert phase_event["details"]["status"] == "tasks_executed_individually"
        assert phase_event["details"]["individual_updates"] >= 0
        assert phase_event["details"]["fallback_updates"] >= 0
    
    @pytest.mark.integration
    async def test_backward_compatibility_with_existing_interfaces(
        self, workflow_manager, temp_workspace, sample_tasks_md_content
    ):
        """Test that the new implementation maintains backward compatibility."""
        # Create tasks.md file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Prepare tasks_result
        tasks_result = {
            "tasks_file": str(tasks_file),
            "success": True
        }
        
        # Execute implementation phase
        result = await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify all expected fields are present for backward compatibility
        expected_fields = [
            "success", "message", "tasks_file", "work_directory",
            "total_tasks", "completed_tasks", "task_results", "execution_completed"
        ]
        
        for field in expected_fields:
            assert field in result, f"Missing expected field: {field}"
        
        # Verify task_results structure matches expected format
        task_results = result["task_results"]
        for task_result in task_results:
            assert "task_id" in task_result
            assert "task_title" in task_result
            assert "success" in task_result
            assert "execution_result" in task_result
        
        # Verify message format is consistent
        assert "Implementation phase completed" in result["message"]
        assert f"{result['completed_tasks']}/{result['total_tasks']}" in result["message"]
    
    @pytest.mark.integration
    async def test_error_handling_preserves_existing_behavior(
        self, workflow_manager, temp_workspace
    ):
        """Test that error handling preserves existing logging behavior."""
        # Create invalid tasks_result (missing tasks file)
        tasks_result = {
            "tasks_file": "/nonexistent/tasks.md",
            "success": True
        }
        
        # Execute implementation phase
        result = await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify error handling
        assert result["success"] is False
        assert "Tasks file not found" in result["error"]
        
        # Test with empty tasks file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        empty_tasks_file = work_dir / "empty_tasks.md"
        empty_tasks_file.write_text("# Empty Tasks File\n\nNo tasks here.")
        
        tasks_result = {
            "tasks_file": str(empty_tasks_file),
            "success": True
        }
        
        result = await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify error handling for empty tasks
        assert result["success"] is False
        assert "No tasks found in tasks.md file" in result["error"]