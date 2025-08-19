"""
Integration tests for task completion error recovery and fallback mechanisms.

Tests the comprehensive error recovery system including fallback to batch updates,
file access error handling, task identification failures, and recovery mechanisms.
"""

import pytest
import asyncio
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from autogen_framework.workflow_manager import WorkflowManager
from autogen_framework.models import WorkflowState, WorkflowPhase, TaskDefinition, TaskFileMapping
from autogen_framework.session_manager import SessionManager
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.context_compressor import ContextCompressor


class TestTaskCompletionErrorRecovery:
    """Test comprehensive error recovery and fallback mechanisms for task completion."""
    
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

- [ ] 4. Implement API endpoints
  - Create REST API endpoints for all operations
  - Add proper error handling and validation
  - _Requirements: 4.1, 4.2_
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
                # Simulate success for all tasks
                return {
                    "success": True,
                    "message": f"Task {task.id} completed successfully",
                    "execution_details": f"Successfully executed task {task.id}"
                }
            return {"success": True}
        
        agent_manager.coordinate_agents = AsyncMock(side_effect=mock_coordinate_agents)
        return agent_manager
    
    @pytest.fixture
    def workflow_manager_with_config(self, temp_workspace, mock_agent_manager):
        """Create WorkflowManager instance with custom configuration for testing."""
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
    async def test_fallback_to_batch_update_on_individual_failures(
        self, workflow_manager_with_config, temp_workspace, sample_tasks_md_content
    ):
        """Test fallback to batch update when individual task updates fail."""
        workflow_manager = workflow_manager_with_config
        
        # Create tasks.md file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Mock _update_single_task_completion to fail for some tasks
        original_update_method = workflow_manager._update_single_task_completion
        
        def mock_update_single_task(tasks_file_path, task_id, success):
            # Fail update for tasks 2 and 4, succeed for others
            if task_id in ["2", "4"]:
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
        
        # Verify error recovery statistics
        error_recovery = result["error_recovery"]
        assert error_recovery["individual_updates_attempted"] == 4
        assert error_recovery["individual_updates_successful"] == 2
        assert error_recovery["fallback_updates_used"] > 0
        assert error_recovery["fallback_used"] is True
        assert error_recovery["failed_individual_updates"] == 2
        
        # Verify task update statistics were recorded
        assert workflow_manager.task_update_stats["individual_updates_attempted"] == 4
        assert workflow_manager.task_update_stats["individual_updates_successful"] == 2
        assert workflow_manager.task_update_stats["fallback_updates_used"] > 0
    
    @pytest.mark.integration
    async def test_file_access_error_recovery_with_retries(
        self, workflow_manager_with_config, temp_workspace, sample_tasks_md_content
    ):
        """Test file access error recovery with retry mechanisms."""
        workflow_manager = workflow_manager_with_config
        
        # Create tasks.md file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Mock _update_single_task_completion to simulate file access errors
        def mock_update_single_task(tasks_file_path, task_id, success):
            # Simulate file access errors for some tasks
            if task_id in ["2", "3"]:
                raise PermissionError("Simulated file access error")
            return True  # Success for other tasks
        
        workflow_manager._update_single_task_completion = mock_update_single_task
        
        # Prepare tasks_result
        tasks_result = {
            "tasks_file": str(tasks_file),
            "success": True
        }
        
        # Execute implementation phase
        result = await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify overall success despite file access errors
        assert result["success"] is True
        
        # Verify error recovery statistics show file access errors
        error_recovery = result["error_recovery"]
        assert error_recovery["file_access_errors"] > 0
        
        # Verify that fallback mechanisms were used
        assert error_recovery["fallback_used"] is True or error_recovery["recovery_successful"] is True
    
    @pytest.mark.integration
    async def test_task_identification_error_recovery(
        self, workflow_manager_with_config, temp_workspace, sample_tasks_md_content
    ):
        """Test recovery from task identification failures."""
        workflow_manager = workflow_manager_with_config
        
        # Create tasks.md file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Mock _update_single_task_completion to simulate task identification errors
        def mock_update_single_task(tasks_file_path, task_id, success):
            # Simulate task identification errors for some tasks
            if task_id in ["2", "3"]:
                raise ValueError("Task identification error")
            return True  # Success for other tasks
        
        workflow_manager._update_single_task_completion = mock_update_single_task
        
        # Prepare tasks_result
        tasks_result = {
            "tasks_file": str(tasks_file),
            "success": True
        }
        
        # Execute implementation phase
        result = await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify overall success despite task identification errors
        assert result["success"] is True
        
        # Verify error recovery statistics show task identification errors
        error_recovery = result["error_recovery"]
        assert error_recovery["task_identification_errors"] > 0
        
        # Verify recovery mechanisms were attempted
        assert error_recovery["recovery_successful"] is True or error_recovery["fallback_used"] is True
    
    @pytest.mark.integration
    async def test_task_loss_prevention_mechanism(
        self, workflow_manager_with_config, temp_workspace, sample_tasks_md_content
    ):
        """Test the task loss prevention mechanism."""
        workflow_manager = workflow_manager_with_config
        
        # Create tasks.md file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Mock _update_single_task_completion to fail for all tasks
        def mock_update_single_task(tasks_file_path, task_id, success):
            return False  # Always fail individual updates
        
        workflow_manager._update_single_task_completion = mock_update_single_task
        
        # Mock _ensure_no_tasks_lost to track if it was called
        recovery_called = False
        original_ensure_no_tasks_lost = workflow_manager._ensure_no_tasks_lost
        
        def mock_ensure_no_tasks_lost(tasks_file_path, expected_tasks, task_results):
            nonlocal recovery_called
            recovery_called = True
            return original_ensure_no_tasks_lost(tasks_file_path, expected_tasks, task_results)
        
        workflow_manager._ensure_no_tasks_lost = mock_ensure_no_tasks_lost
        
        # Prepare tasks_result
        tasks_result = {
            "tasks_file": str(tasks_file),
            "success": True
        }
        
        # Execute implementation phase
        result = await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify task loss prevention was called
        assert recovery_called, "Task loss prevention mechanism should have been called"
        
        # Verify error recovery statistics
        error_recovery = result["error_recovery"]
        assert error_recovery["individual_updates_attempted"] == 4
        assert error_recovery["individual_updates_successful"] == 0
        assert error_recovery["fallback_used"] is True
        assert error_recovery["recovery_successful"] is True
    
    @pytest.mark.integration
    async def test_configuration_options_disable_features(
        self, workflow_manager_with_config, temp_workspace, sample_tasks_md_content
    ):
        """Test that configuration options can disable real-time updates and recovery features."""
        workflow_manager = workflow_manager_with_config
        
        # Disable real-time updates and recovery mechanisms
        workflow_manager.task_completion_config = {
            'real_time_updates_enabled': False,
            'fallback_to_batch_enabled': False,
            'recovery_mechanism_enabled': False,
            'detailed_logging_enabled': False,
            'max_individual_update_retries': 1,
            'individual_update_retry_delay': 0.01,
            'file_lock_timeout': 1
        }
        
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
        
        # Verify overall success
        assert result["success"] is True
        
        # Verify that individual updates were not attempted due to configuration
        error_recovery = result["error_recovery"]
        assert error_recovery["individual_updates_attempted"] == 0
        assert error_recovery["fallback_used"] is False
        assert error_recovery["recovery_successful"] is False
    
    @pytest.mark.integration
    async def test_partial_update_recovery_scenarios(
        self, workflow_manager_with_config, temp_workspace, sample_tasks_md_content
    ):
        """Test recovery from partial update scenarios."""
        workflow_manager = workflow_manager_with_config
        
        # Create tasks.md file with some tasks already completed
        modified_content = sample_tasks_md_content.replace("- [ ] 2. Implement", "- [x] 2. Implement")
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(modified_content)
        
        # Mock _update_single_task_completion to fail intermittently
        update_attempts = 0
        
        def mock_update_single_task(tasks_file_path, task_id, success):
            nonlocal update_attempts
            update_attempts += 1
            # Fail every other attempt
            return update_attempts % 2 == 0
        
        workflow_manager._update_single_task_completion = mock_update_single_task
        
        # Prepare tasks_result
        tasks_result = {
            "tasks_file": str(tasks_file),
            "success": True
        }
        
        # Execute implementation phase
        result = await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
        
        # Verify overall success
        assert result["success"] is True
        
        # Verify error recovery statistics show partial recoveries
        error_recovery = result["error_recovery"]
        assert error_recovery["individual_updates_attempted"] > 0
        assert error_recovery["partial_update_recoveries"] >= 0
        
        # Verify that some form of recovery was used
        assert (error_recovery["fallback_used"] is True or 
                error_recovery["recovery_successful"] is True or
                error_recovery["individual_updates_successful"] > 0)
    
    @pytest.mark.integration
    async def test_comprehensive_error_logging_and_monitoring(
        self, workflow_manager_with_config, temp_workspace, sample_tasks_md_content
    ):
        """Test comprehensive error logging and monitoring capabilities."""
        workflow_manager = workflow_manager_with_config
        
        # Enable detailed logging
        workflow_manager.task_completion_config['detailed_logging_enabled'] = True
        
        # Create tasks.md file
        work_dir = Path(workflow_manager.current_workflow.work_directory)
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Mock various error conditions
        def mock_update_single_task(tasks_file_path, task_id, success):
            # Simulate different types of failures
            if task_id == "1":
                return True  # Success
            elif task_id == "2":
                return False  # Individual update failure
            elif task_id == "3":
                raise PermissionError("File access error")
            else:
                return False  # Task identification error
        
        workflow_manager._update_single_task_completion = mock_update_single_task
        
        # Prepare tasks_result
        tasks_result = {
            "tasks_file": str(tasks_file),
            "success": True
        }
        
        # Capture log messages
        with patch.object(workflow_manager.logger, 'info') as mock_info, \
             patch.object(workflow_manager.logger, 'warning') as mock_warning, \
             patch.object(workflow_manager.logger, 'error') as mock_error:
            
            # Execute implementation phase
            result = await workflow_manager._execute_implementation_phase(tasks_result, "test_workflow")
            
            # Verify comprehensive logging occurred
            assert mock_info.call_count > 0, "Should have info log messages"
            assert mock_warning.call_count > 0, "Should have warning log messages"
            
            # Check for specific log messages about statistics
            info_messages = [call.args[0] for call in mock_info.call_args_list]
            assert any("Task completion update statistics:" in msg for msg in info_messages)
            assert any("Individual updates attempted:" in msg for msg in info_messages)
            assert any("Individual update success rate:" in msg for msg in info_messages)
        
        # Verify comprehensive error recovery statistics
        error_recovery = result["error_recovery"]
        assert "individual_updates_attempted" in error_recovery
        assert "individual_updates_successful" in error_recovery
        assert "fallback_updates_used" in error_recovery
        assert "file_access_errors" in error_recovery
        assert "task_identification_errors" in error_recovery
        assert "partial_update_recoveries" in error_recovery
        assert "fallback_used" in error_recovery
        assert "recovery_successful" in error_recovery
        assert "failed_individual_updates" in error_recovery
        
        # Verify overall success despite various errors
        assert result["success"] is True