"""
Integration tests for MainController auto-approve functionality with real components.

This module contains integration tests that use real MemoryManager, ShellExecutor,
and other components to verify the auto-approve functionality works correctly
in a real environment.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

from autogen_framework.main_controller import MainController, UserApprovalStatus
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.shell_executor import ShellExecutor


class TestMainControllerAutoApproveIntegration:
    """Integration tests for MainController auto-approve functionality."""
    
    @pytest.fixture
    def main_controller(self, temp_workspace):
        """Create a MainController instance for integration testing."""
        return MainController(temp_workspace)
    
    @pytest.mark.integration
    def test_auto_approve_with_real_memory_manager(self, main_controller, temp_workspace, real_llm_config):
        """Test auto-approve functionality with real MemoryManager integration."""
        # Mock only the agent setup to avoid AutoGen dependencies
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize framework
            result = main_controller.initialize_framework(real_llm_config)
            assert result is True
            
            # Test auto-approve functionality
            assert main_controller.should_auto_approve("requirements", True) is True
            assert main_controller.should_auto_approve("design", True) is True
            assert main_controller.should_auto_approve("tasks", True) is True
            
            # Verify approval logging
            assert len(main_controller.approval_log) == 3
            assert len(main_controller.workflow_summary['auto_approvals']) == 3
            
            # Verify memory manager can save approval log
            main_controller.memory_manager.save_memory(
                "auto_approve_test",
                f"Auto-approve test completed with {len(main_controller.approval_log)} approvals",
                "global"
            )
            
            # Verify memory was saved
            memory_content = main_controller.memory_manager.load_memory()
            assert "global" in memory_content
            global_memory = memory_content["global"]
            assert any("auto_approve_test" in key for key in global_memory.keys())
    
    @pytest.mark.integration
    def test_comprehensive_summary_with_real_components(self, main_controller, temp_workspace, real_llm_config):
        """Test comprehensive summary generation with real components."""
        # Mock only the agent setup to avoid AutoGen dependencies
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Simulate some auto-approve activity
            main_controller.should_auto_approve("requirements", True)
            main_controller.should_auto_approve("design", True)
            
            # Simulate error recovery
            main_controller.error_recovery_attempts["design"] = 1
            main_controller.workflow_summary['errors_recovered'].append({
                'phase': 'design',
                'error': 'Test error',
                'strategy': 'retry',
                'attempt': 1
            })
            
            # Get comprehensive summary
            summary = main_controller.get_comprehensive_summary()
            
            # Verify summary structure and content
            assert 'workflow_summary' in summary
            assert 'approval_log' in summary
            assert 'error_recovery_attempts' in summary
            assert 'total_phases_completed' in summary
            assert 'total_auto_approvals' in summary
            assert 'total_errors_recovered' in summary
            assert 'session_id' in summary
            assert 'timestamp' in summary
            
            # Verify counts
            assert summary['total_auto_approvals'] == 2
            assert summary['total_errors_recovered'] == 1
            assert summary['session_id'] == main_controller.session_id
            
            # Verify real components are accessible
            assert isinstance(main_controller.memory_manager, MemoryManager)
            assert isinstance(main_controller.shell_executor, ShellExecutor)
    
    @pytest.mark.integration
    def test_session_persistence_auto_approve_data(self, temp_workspace, real_llm_config):
        """Test that auto-approve data persists across sessions with real components."""
        # Create first controller and add auto-approve data
        controller1 = MainController(temp_workspace)
        
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            controller1.initialize_framework(real_llm_config)
            
            # Add auto-approve data
            controller1.should_auto_approve("requirements", True)
            controller1.error_recovery_attempts["design"] = 2
            controller1.workflow_summary['errors_recovered'].append({
                'phase': 'design',
                'error': 'Test persistence error',
                'strategy': 'fallback'
            })
            
            # Save session state
            controller1._save_session_state()
            session_id = controller1.session_id
        
        # Create second controller and load session
        controller2 = MainController(temp_workspace)
        controller2._load_or_create_session()
        
        # Verify auto-approve data was persisted
        assert controller2.session_id == session_id
        assert len(controller2.approval_log) == 1
        assert controller2.approval_log[0]['phase'] == 'requirements'
        assert controller2.error_recovery_attempts == {'design': 2}
        assert len(controller2.workflow_summary['errors_recovered']) == 1
        assert controller2.workflow_summary['errors_recovered'][0]['phase'] == 'design'
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_auto_approve_workflow_simulation(self, main_controller, temp_workspace, real_llm_config):
        """Test a simulated auto-approve workflow with real components."""
        # Mock only the agent coordination to avoid AutoGen dependencies
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Mock agent coordination to simulate workflow phases
            async def mock_coordinate_workflow(task_type, context):
                work_path = Path(temp_workspace) / "test_work"
                if task_type == "requirements_generation":
                    return {
                        "success": True,
                        "work_directory": str(work_path),
                        "requirements_path": str(work_path / "requirements.md")
                    }
                elif task_type == "design_generation":
                    return {
                        "success": True,
                        "design_path": str(work_path / "design.md")
                    }
                elif task_type == "task_execution":
                    return {
                        "success": True,
                        "tasks_file": str(work_path / "tasks.md")
                    }
                else:
                    return {"success": False, "error": "Unknown task type"}
            
            mock_agent_instance.coordinate_agents = AsyncMock(side_effect=mock_coordinate_workflow)
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Create work directory to avoid file not found errors
            work_dir = Path(temp_workspace) / "test_work"
            work_dir.mkdir(exist_ok=True)
            
            # Mock task parsing and file operations
            with patch.object(main_controller, '_parse_tasks_from_file') as mock_parse:
                mock_parse.return_value = []  # No tasks to execute
                
                with patch.object(main_controller, '_update_tasks_file_with_completion'):
                    # Process request with auto_approve enabled
                    result = await main_controller.process_request(
                        "Create a test application with auto-approve",
                        auto_approve=True
                    )
            
            # Verify the workflow completed successfully
            assert result["success"] is True
            assert result["auto_approve_enabled"] is True
            
            # Verify auto-approvals were logged
            assert len(main_controller.workflow_summary['auto_approvals']) >= 3
            
            # Verify real components were used
            assert isinstance(main_controller.memory_manager, MemoryManager)
            assert isinstance(main_controller.shell_executor, ShellExecutor)
            
            # Verify memory manager can save workflow results
            main_controller.memory_manager.save_memory(
                "auto_approve_workflow_test",
                f"Auto-approve workflow completed successfully: {result['workflow_id']}",
                "global"
            )
            
            # Verify memory was saved
            memory_content = main_controller.memory_manager.load_memory()
            assert "global" in memory_content
            global_memory = memory_content["global"]
            assert any("auto_approve_workflow_test" in key for key in global_memory.keys())
    
    @pytest.mark.integration
    def test_error_recovery_with_real_components(self, main_controller, temp_workspace, real_llm_config):
        """Test error recovery functionality with real components."""
        # Mock only the agent setup to avoid AutoGen dependencies
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            # Initialize framework
            main_controller.initialize_framework(real_llm_config)
            
            # Test error recovery with different error types
            timeout_error = Exception("Connection timeout occurred")
            memory_error = Exception("Memory limit exceeded")
            format_error = Exception("Parse error in format")
            
            # Test parameter modification for different error types
            timeout_context = main_controller._modify_parameters_for_retry(
                timeout_error, "requirements", {"timeout": 30}
            )
            assert timeout_context["timeout"] == 60
            assert timeout_context["max_retries"] == 3
            
            memory_context = main_controller._modify_parameters_for_retry(
                memory_error, "design", {"complexity": "high"}
            )
            assert memory_context["max_complexity"] == "low"
            assert memory_context["simplified_mode"] is True
            
            format_context = main_controller._modify_parameters_for_retry(
                format_error, "tasks", {"format": "flexible"}
            )
            assert format_context["strict_format"] is True
            assert format_context["use_templates"] is True
            
            # Test non-critical step identification
            req_steps = main_controller._identify_non_critical_steps("requirements", timeout_error)
            assert len(req_steps) > 0
            assert "detailed_examples" in req_steps
            
            design_steps = main_controller._identify_non_critical_steps("design", memory_error)
            assert len(design_steps) > 0
            assert "detailed_diagrams" in design_steps
            
            # Test simplified context creation
            simplified_context = main_controller._create_simplified_context(
                {"original": "value"}, ["step1", "step2"]
            )
            assert simplified_context["skip_steps"] == ["step1", "step2"]
            assert simplified_context["simplified_execution"] is True
            
            # Test fallback implementations
            assert main_controller._fallback_requirements_generation({"test": "context"}) is True
            assert main_controller._fallback_design_generation({"test": "context"}) is True
            assert main_controller._fallback_tasks_generation({"test": "context"}) is True
            assert main_controller._fallback_implementation_execution({"test": "context"}) is True
            
            # Verify real components are still accessible and functional
            assert isinstance(main_controller.memory_manager, MemoryManager)
            assert isinstance(main_controller.shell_executor, ShellExecutor)
            
            # Test that memory manager can save error recovery information
            main_controller.memory_manager.save_memory(
                "error_recovery_test",
                f"Error recovery test completed with {len(req_steps)} non-critical steps identified",
                "global"
            )
            
            # Verify memory was saved
            memory_content = main_controller.memory_manager.load_memory()
            assert "global" in memory_content
            global_memory = memory_content["global"]
            assert any("error_recovery_test" in key for key in global_memory.keys())
    
    @pytest.mark.integration
    def test_error_recovery_attempt_tracking_with_persistence(self, temp_workspace, real_llm_config):
        """Test error recovery attempt tracking with session persistence."""
        # Create first controller
        controller1 = MainController(temp_workspace)
        
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent:
            mock_agent_instance = mock_agent.return_value
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory.return_value = None
            
            controller1.initialize_framework(real_llm_config)
            
            # Simulate error recovery attempts
            with patch.object(controller1, '_get_error_recovery_max_attempts', return_value=3):
                with patch.object(controller1, '_retry_with_modified_parameters', return_value=True):
                    error = Exception("Test integration error")
                    
                    # First recovery attempt
                    result1 = controller1.handle_error_recovery(error, "requirements", {"test": "context"})
                    assert result1 is True
                    assert controller1.error_recovery_attempts["requirements"] == 1
                    
                    # Second recovery attempt
                    result2 = controller1.handle_error_recovery(error, "design", {"test": "context"})
                    assert result2 is True
                    assert controller1.error_recovery_attempts["design"] == 1
                    
                    # Verify recovery logging
                    assert len(controller1.workflow_summary['errors_recovered']) == 2
            
            # Save session state
            controller1._save_session_state()
            session_id = controller1.session_id
        
        # Create second controller and load session
        controller2 = MainController(temp_workspace)
        
        # Initialize the second controller to set up components
        with patch('autogen_framework.main_controller.AgentManager') as mock_agent2:
            mock_agent_instance2 = mock_agent2.return_value
            mock_agent_instance2.setup_agents.return_value = True
            mock_agent_instance2.update_agent_memory.return_value = None
            
            controller2.initialize_framework(real_llm_config)
        
        # Verify error recovery data was persisted
        assert controller2.session_id == session_id
        assert controller2.error_recovery_attempts == {"requirements": 1, "design": 1}
        assert len(controller2.workflow_summary['errors_recovered']) == 2
        
        # Verify real components are functional after session reload
        assert isinstance(controller2.memory_manager, MemoryManager)
        assert isinstance(controller2.shell_executor, ShellExecutor)
        
        # Test that we can continue error recovery from persisted state
        with patch.object(controller2, '_get_error_recovery_max_attempts', return_value=3):
            with patch.object(controller2, '_skip_non_critical_steps', return_value=True):
                error = Exception("Another test error")
                
                # Third recovery attempt (should increment from persisted state)
                result3 = controller2.handle_error_recovery(error, "requirements", {"test": "context"})
                assert result3 is True
                assert controller2.error_recovery_attempts["requirements"] == 2  # Incremented from 1
                
                # Verify total recovery count
                assert len(controller2.workflow_summary['errors_recovered']) == 3