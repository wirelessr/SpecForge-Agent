"""
Unit tests for workflow continuation and task execution.

This test module specifically addresses the issue found in tests/e2e/workflow_test.sh
where tasks were not being executed after approval.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from autogen_framework.main_controller import MainController
from autogen_framework.models import LLMConfig, WorkflowPhase


class TestWorkflowContinuation:
    """Test workflow continuation and task execution."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def main_controller(self, temp_workspace):
        """Create a MainController instance for testing."""
        return MainController(str(temp_workspace))
    
    @pytest.mark.asyncio
    async def test_workflow_continuation_after_tasks_approval(self, main_controller, test_llm_config):
        """
        Test that the workflow continues to implementation phase after tasks approval.
        
        This test reproduces and verifies the fix for the issue where tasks were
        not being executed after approval.
        """
        # Initialize the controller
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            # Mock the agent manager setup
            mock_agent_instance = Mock()
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory = Mock()
            mock_agent.return_value = mock_agent_instance
            
            # Mock the container's memory manager
            with patch('autogen_framework.dependency_container.DependencyContainer.get_memory_manager') as mock_memory_manager:
                mock_memory_manager.return_value.load_memory.return_value = {}
                
                # Initialize framework - this will create the container with managers
                success = main_controller.initialize_framework(test_llm_config)
                assert success, "Framework initialization should succeed"
                
                # Mock the agent coordination for each phase
                mock_agent_instance.coordinate_agents = AsyncMock()
                    
                # Mock requirements phase
                mock_agent_instance.coordinate_agents.side_effect = [
                    # Requirements generation
                    {
                        "success": True,
                        "work_directory": "/test/work/dir",
                        "requirements_path": "/test/work/dir/requirements.md"
                    },
                    # Design generation
                    {
                        "success": True,
                        "design_path": "/test/work/dir/design.md"
                    },
                    # Tasks generation
                    {
                        "success": True,
                        "tasks_file": "/test/work/dir/tasks.md"
                    },
                    # Implementation execution (this is the key part!)
                    {
                        "success": True,
                        "completed_count": 5,
                        "task_results": [
                            {"success": True, "task_id": "task_1"},
                            {"success": True, "task_id": "task_2"},
                            {"success": True, "task_id": "task_3"},
                            {"success": True, "task_id": "task_4"},
                            {"success": True, "task_id": "task_5"},
                        ]
                    }
                ]
                
                # Mock the task parsing to return some test tasks
                with patch.object(main_controller, '_parse_tasks_from_file') as mock_parse_tasks:
                    from autogen_framework.models import TaskDefinition
                    
                    mock_tasks = [
                        TaskDefinition(
                            id="task_1",
                            title="Create calculator.py",
                            description="Create the main calculator file",
                            steps=["Step 1: Create file", "Step 2: Add functions"],
                            requirements_ref=["1.1", "2.1"],
                            completed=False
                        ),
                        TaskDefinition(
                            id="task_2", 
                            title="Add validation",
                            description="Add input validation",
                            steps=["Step 1: Validate inputs"],
                            requirements_ref=["1.2"],
                            completed=False
                        ),
                        TaskDefinition(
                            id="task_3",
                            title="Add tests",
                            description="Add unit tests",
                            steps=["Step 1: Create test file"],
                            requirements_ref=["3.1"],
                            completed=False
                        ),
                        TaskDefinition(
                            id="task_4",
                            title="Add documentation",
                            description="Add code documentation",
                            steps=["Step 1: Add docstrings"],
                            requirements_ref=["4.1"],
                            completed=False
                        ),
                        TaskDefinition(
                            id="task_5",
                            title="Final integration",
                            description="Integrate all components",
                            steps=["Step 1: Test integration"],
                            requirements_ref=["5.1"],
                            completed=False
                        )
                    ]
                    mock_parse_tasks.return_value = mock_tasks
                    
                    # Mock the tasks file update
                    with patch.object(main_controller, '_update_tasks_file_with_completion') as mock_update_tasks:
                        # Mock file existence checks for approval
                        with patch('pathlib.Path.exists', return_value=True):
                            # Process a user request
                            user_request = "Create a simple calculator"
                            result = await main_controller.process_user_request(user_request)
                            
                            # Should require approval for requirements
                            assert result.get("requires_user_approval") == True
                            assert result.get("approval_needed_for") == "requirements"
                            
                            # Approve requirements
                            approval_result = main_controller.approve_phase("requirements", True)
                            assert approval_result.get("approved") == True
                            
                            # Continue workflow (should go to design)
                            continue_result = await main_controller.continue_workflow()
                            assert continue_result.get("phase") == "design"
                            assert continue_result.get("requires_approval") == True
                            
                            # Approve design
                            approval_result = main_controller.approve_phase("design", True)
                            assert approval_result.get("approved") == True
                            
                            # Continue workflow (should go to tasks)
                            continue_result = await main_controller.continue_workflow()
                            assert continue_result.get("phase") == "tasks"
                            assert continue_result.get("requires_approval") == True
                            
                            # Approve tasks
                            approval_result = main_controller.approve_phase("tasks", True)
                            assert approval_result.get("approved") == True
                            
                            # Continue workflow (should go to implementation and complete!)
                            continue_result = await main_controller.continue_workflow()
                            
                            # This is the key assertion - the workflow should complete
                            assert continue_result.get("phase") == "implementation"
                            assert continue_result.get("workflow_completed") == True
                            
                            # Verify that the agent coordination was called for the 3 phases:
                            # 1. requirements_generation
                            # 2. design_generation  
                            # 3. task_generation (via tasks_generation)
                            # Note: implementation phase no longer calls agent coordination
                            assert mock_agent_instance.coordinate_agents.call_count == 3
                            
                            # Check the last call was for task generation
                            last_call = mock_agent_instance.coordinate_agents.call_args_list[-1]
                            assert last_call[0][0] == "task_generation"
                            
                            # Verify workflow is completed and cleared
                            assert main_controller.current_workflow is None
    
    @pytest.mark.asyncio
    async def test_workflow_without_continue_does_not_execute_tasks(self, main_controller, test_llm_config):
        """
        Test that tasks are NOT executed if continue_workflow is not called.
        
        This test verifies the original issue - that just approving phases
        without calling continue_workflow doesn't execute tasks.
        """
        # Initialize the controller
        with patch.object(main_controller, 'agent_manager') as mock_agent_manager:
            mock_agent_manager.setup_agents.return_value = True
            
            # Mock the container's memory manager instead of direct memory_manager
            with patch('autogen_framework.dependency_container.DependencyContainer.get_memory_manager') as mock_memory_manager:
                mock_memory_manager.return_value.load_memory.return_value = {}
                
                success = main_controller.initialize_framework(test_llm_config)
                assert success
                
                # Mock agent coordination for first 3 phases only
                mock_agent_manager.coordinate_agents = AsyncMock()
                mock_agent_manager.coordinate_agents.side_effect = [
                    # Requirements
                    {
                        "success": True,
                        "work_directory": "/test/work/dir",
                        "requirements_path": "/test/work/dir/requirements.md"
                    },
                    # Design  
                    {
                        "success": True,
                        "design_path": "/test/work/dir/design.md"
                    },
                    # Tasks
                    {
                        "success": True,
                        "tasks_file": "/test/work/dir/tasks.md"
                    }
                ]
                
                # Process request
                result = await main_controller.process_user_request("Create a calculator")
                assert result.get("requires_user_approval") == True
                
                # Approve all phases but DON'T call continue_workflow
                main_controller.approve_phase("requirements", True)
                main_controller.approve_phase("design", True) 
                main_controller.approve_phase("tasks", True)
                
                # Verify that only 1 agent call was made (requirements)
                # because we never called continue_workflow
                assert mock_agent_manager.coordinate_agents.call_count == 1
                
                # Verify workflow is still active and not completed
                assert main_controller.current_workflow is not None
                assert main_controller.current_workflow.phase == WorkflowPhase.PLANNING
    
    @pytest.mark.asyncio
    async def test_task_execution_failure_handling(self, main_controller, test_llm_config):
        """Test that task execution failures are handled properly."""
        with patch.object(main_controller, 'agent_manager') as mock_agent_manager:
            mock_agent_manager.setup_agents.return_value = True
            
            # Mock the container's memory manager instead of direct memory_manager
            with patch('autogen_framework.dependency_container.DependencyContainer.get_memory_manager') as mock_memory_manager:
                mock_memory_manager.return_value.load_memory.return_value = {}
                
                success = main_controller.initialize_framework(test_llm_config)
                assert success
                
                # Mock successful phases but failed implementation
                mock_agent_manager.coordinate_agents = AsyncMock()
                mock_agent_manager.coordinate_agents.side_effect = [
                    {"success": True, "work_directory": "/test", "requirements_path": "/test/req.md"},
                    {"success": True, "design_path": "/test/design.md"},
                    {"success": True, "tasks_file": "/test/tasks.md"},
                    # Implementation fails
                    {"success": False, "error": "Task execution failed", "completed_count": 2}
                ]
                
                with patch.object(main_controller, '_parse_tasks_from_file') as mock_parse:
                    from autogen_framework.models import TaskDefinition
                    mock_parse.return_value = [
                        TaskDefinition("task_1", "Test task", "Test", [], [], False)
                    ]
                    
                    with patch.object(main_controller, '_update_tasks_file_with_completion'):
                        # Mock file existence for approvals
                        with patch('pathlib.Path.exists', return_value=True):
                            # Go through full workflow
                            await main_controller.process_user_request("Test request")
                            main_controller.approve_phase("requirements", True)
                            await main_controller.continue_workflow()
                            main_controller.approve_phase("design", True)
                            await main_controller.continue_workflow()
                            main_controller.approve_phase("tasks", True)
                            
                            # This should handle the implementation failure gracefully
                            continue_result = await main_controller.continue_workflow()
                            
                            assert continue_result.get("phase") == "implementation"
                            assert continue_result.get("workflow_completed") == True
                            
                            # Even with failures, workflow should complete
                            assert main_controller.current_workflow is None