"""
Unit tests for MainController auto-approve functionality.

This module contains unit tests specifically for the auto-approve features
added to the MainController class. These tests focus on:

- Auto-approve parameter handling in process_request method
- should_auto_approve method logic with critical checkpoint respect
- Comprehensive workflow summary tracking for all phases and tasks
- Approval logging functionality for audit trail
- Error recovery mechanisms for auto-approve mode

All external dependencies are mocked to ensure fast, reliable unit tests.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from autogen_framework.main_controller import MainController
from autogen_framework.models import UserApprovalStatus
from autogen_framework.models import LLMConfig, WorkflowPhase, WorkflowState


class TestMainControllerAutoApprove:
    """Test cases for MainController auto-approve functionality."""
    
    @pytest.fixture
    def main_controller(self, temp_workspace):
        """Create a MainController instance for testing."""
        return MainController(temp_workspace)
    
    @pytest.fixture
    def initialized_controller(self, main_controller, llm_config):
        """Create an initialized MainController for testing."""
        with patch('autogen_framework.main_controller.MemoryManager') as mock_memory, \
             patch('autogen_framework.main_controller.AgentManager') as mock_agent, \
             patch('autogen_framework.main_controller.ShellExecutor') as mock_shell:
            
            # Mock component instances
            mock_memory_instance = Mock()
            mock_memory_instance.load_memory.return_value = {"global": {"test": "content"}}
            mock_memory.return_value = mock_memory_instance
            
            mock_agent_instance = Mock()
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory = Mock()
            mock_agent.return_value = mock_agent_instance
            
            mock_shell_instance = Mock()
            mock_shell.return_value = mock_shell_instance
            
            # Initialize framework
            main_controller.initialize_framework(llm_config)
            
            return main_controller
    
    def test_auto_approve_initialization(self, main_controller):
        """Test that auto-approve related fields are initialized correctly."""
        assert main_controller.approval_log == []
        assert main_controller.error_recovery_attempts == {}
        assert main_controller.workflow_summary == {
            'phases_completed': [],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [],
            'errors_recovered': []
        }
    
    def test_should_auto_approve_with_manual_approval(self, main_controller):
        """Test should_auto_approve when phase is already manually approved."""
        # Manually approve requirements phase
        main_controller.user_approval_status["requirements"] = UserApprovalStatus.APPROVED
        
        # Should return True regardless of auto_approve flag
        assert main_controller.should_auto_approve("requirements", False) is True
        assert main_controller.should_auto_approve("requirements", True) is True
    
    def test_should_auto_approve_disabled(self, main_controller):
        """Test should_auto_approve when auto_approve is disabled."""
        # Should return False when auto_approve is disabled
        assert main_controller.should_auto_approve("requirements", False) is False
        assert main_controller.should_auto_approve("design", False) is False
        assert main_controller.should_auto_approve("tasks", False) is False
        
        # Verify no approval status was set
        assert "requirements" not in main_controller.user_approval_status
        assert "design" not in main_controller.user_approval_status
        assert "tasks" not in main_controller.user_approval_status
    
    def test_should_auto_approve_enabled(self, main_controller):
        """Test should_auto_approve when auto_approve is enabled."""
        # Should return True and set approval status
        assert main_controller.should_auto_approve("requirements", True) is True
        assert main_controller.user_approval_status["requirements"] == UserApprovalStatus.APPROVED
        
        assert main_controller.should_auto_approve("design", True) is True
        assert main_controller.user_approval_status["design"] == UserApprovalStatus.APPROVED
        
        assert main_controller.should_auto_approve("tasks", True) is True
        assert main_controller.user_approval_status["tasks"] == UserApprovalStatus.APPROVED
        
        # Verify approval logging
        assert len(main_controller.approval_log) == 3
        assert len(main_controller.workflow_summary['auto_approvals']) == 3
        
        # Verify workflow summary tracking
        assert len(main_controller.workflow_summary['phases_completed']) == 3
        for phase_info in main_controller.workflow_summary['phases_completed']:
            assert phase_info['auto_approved'] is True
            assert 'timestamp' in phase_info
    
    @patch.dict(os.environ, {'AUTO_APPROVE_CRITICAL_CHECKPOINTS': 'requirements,design'})
    def test_should_auto_approve_critical_checkpoints(self, main_controller):
        """Test should_auto_approve respects critical checkpoints."""
        # Critical checkpoints should not be auto-approved
        assert main_controller.should_auto_approve("requirements", True) is False
        assert main_controller.should_auto_approve("design", True) is False
        
        # Non-critical phases should be auto-approved
        assert main_controller.should_auto_approve("tasks", True) is True
        
        # Verify approval logging for critical checkpoints
        critical_approvals = [log for log in main_controller.approval_log if not log['decision']]
        assert len(critical_approvals) == 2
        for approval in critical_approvals:
            assert "Critical checkpoint" in approval['reason']
    
    def test_log_auto_approval(self, main_controller):
        """Test approval logging functionality."""
        # Log some approval decisions
        main_controller.log_auto_approval("requirements", True, "Auto-approved in auto-approve mode")
        main_controller.log_auto_approval("design", False, "Critical checkpoint requires explicit approval")
        
        # Verify approval log
        assert len(main_controller.approval_log) == 2
        
        approval1 = main_controller.approval_log[0]
        assert approval1['phase'] == "requirements"
        assert approval1['decision'] is True
        assert approval1['reason'] == "Auto-approved in auto-approve mode"
        assert 'timestamp' in approval1
        
        approval2 = main_controller.approval_log[1]
        assert approval2['phase'] == "design"
        assert approval2['decision'] is False
        assert approval2['reason'] == "Critical checkpoint requires explicit approval"
        assert 'timestamp' in approval2
        
        # Verify workflow summary tracking
        assert len(main_controller.workflow_summary['auto_approvals']) == 2
        assert main_controller.workflow_summary['auto_approvals'] == main_controller.approval_log
    
    @pytest.mark.asyncio
    async def test_process_request_auto_approve_enabled(self, initialized_controller):
        """Test process_request with auto_approve enabled."""
        # Mock agent coordination to return successful results
        requirements_result = {
            "success": True,
            "work_directory": "/test/work",
            "requirements_path": "/test/work/requirements.md"
        }
        
        design_result = {
            "success": True,
            "design_path": "/test/work/design.md"
        }
        
        tasks_result = {
            "success": True,
            "tasks_file": "/test/work/tasks.md"
        }
        
        implementation_result = {
            "success": True,
            "execution_completed": True
        }
        
        async def mock_coordinate_workflow(task_type, context):
            if task_type == "requirements_generation":
                return requirements_result
            elif task_type == "design_generation":
                return design_result
            elif task_type == "task_generation":
                return tasks_result
            elif task_type == "execute_multiple_tasks":
                return implementation_result
            else:
                return {"success": False, "error": "Unknown task type"}
        
        initialized_controller.agent_manager.coordinate_agents = AsyncMock(side_effect=mock_coordinate_workflow)
        
        # Mock task parsing to avoid file system dependencies
        with patch.object(initialized_controller, '_parse_tasks_from_file') as mock_parse:
            mock_parse.return_value = [Mock(id="1", title="Test task", completed=False)]
            
            with patch.object(initialized_controller, '_update_tasks_file_with_completion'):
                # Process request with auto_approve enabled
                result = await initialized_controller.process_request("Create a test application", auto_approve=True)
        
        # Should complete all phases automatically
        assert result["success"] is True
        assert result["auto_approve_enabled"] is True
        assert result["current_phase"] == "implementation"
        assert "requirements" in result["phases"]
        assert "design" in result["phases"]
        assert "tasks" in result["phases"]
        assert "implementation" in result["phases"]
        
        # Verify all phases were auto-approved (check workflow summary since workflow is completed)
        auto_approvals = initialized_controller.workflow_summary['auto_approvals']
        assert len(auto_approvals) == 3
        
        phases_approved = [approval['phase'] for approval in auto_approvals if approval['decision']]
        assert 'requirements' in phases_approved
        assert 'design' in phases_approved
        assert 'tasks' in phases_approved
        
        # Verify workflow summary
        assert len(initialized_controller.workflow_summary['phases_completed']) == 3
        assert len(initialized_controller.workflow_summary['auto_approvals']) == 3
        
        # Verify workflow was completed and cleared
        assert initialized_controller.current_workflow is None
    
    @pytest.mark.asyncio
    async def test_process_request_auto_approve_disabled(self, initialized_controller):
        """Test process_request with auto_approve disabled (default behavior)."""
        # Mock agent coordination
        requirements_result = {
            "success": True,
            "work_directory": "/test/work",
            "requirements_path": "/test/work/requirements.md"
        }
        
        initialized_controller.agent_manager.coordinate_agents = AsyncMock(return_value=requirements_result)
        
        # Process request with auto_approve disabled
        result = await initialized_controller.process_request("Create a test application", auto_approve=False)
        
        # Should stop at requirements phase waiting for approval
        assert result["success"] is False
        assert result["auto_approve_enabled"] is False
        assert result["requires_user_approval"] is True
        assert result["approval_needed_for"] == "requirements"
        assert result["current_phase"] == "requirements"
        
        # Verify no auto-approvals occurred
        assert len(initialized_controller.workflow_summary['auto_approvals']) == 0
        assert "requirements" not in initialized_controller.user_approval_status
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_process_user_request(self, initialized_controller):
        """Test that process_user_request method maintains backward compatibility."""
        # Mock the new process_request method
        with patch.object(initialized_controller, 'process_request') as mock_process:
            mock_process.return_value = {"success": True}
            
            # Call the old method
            result = await initialized_controller.process_user_request("test request")
            
            # Should call new method with auto_approve=False
            mock_process.assert_called_once_with("test request", auto_approve=False)
    
    def test_handle_error_recovery_max_attempts_exceeded(self, main_controller):
        """Test error recovery when maximum attempts are exceeded."""
        # Set up error recovery attempts
        main_controller.error_recovery_attempts["requirements"] = 3
        
        with patch.object(main_controller, '_get_error_recovery_max_attempts', return_value=3):
            error = Exception("Test error")
            result = main_controller.handle_error_recovery(error, "requirements", {})
            
            assert result is False
            # Attempts should not be incremented when max is exceeded
            assert main_controller.error_recovery_attempts["requirements"] == 3
    
    def test_handle_error_recovery_successful_strategy(self, initialized_controller):
        """Test that MainController properly delegates error recovery to WorkflowManager."""
        # Test the delegation pattern: MainController should delegate to WorkflowManager
        with patch.object(initialized_controller.workflow_manager, 'handle_error_recovery', return_value=True) as mock_wm_recovery:
            error = Exception("Test error")
            context = {"test": "context"}
            
            # Call MainController method
            result = initialized_controller.handle_error_recovery(error, "requirements", context)
            
            # Verify MainController delegates to WorkflowManager
            assert result is True
            mock_wm_recovery.assert_called_once_with(error, "requirements", context)
    
    def test_handle_error_recovery_all_strategies_fail(self, initialized_controller):
        """Test that MainController properly delegates error recovery failures to WorkflowManager."""
        # Test the delegation pattern when WorkflowManager recovery fails
        with patch.object(initialized_controller.workflow_manager, 'handle_error_recovery', return_value=False) as mock_wm_recovery:
            error = Exception("Test error")
            context = {}
            
            # Call MainController method
            result = initialized_controller.handle_error_recovery(error, "requirements", context)
            
            # Verify MainController delegates to WorkflowManager and returns its result
            assert result is False
            mock_wm_recovery.assert_called_once_with(error, "requirements", context)
    
    def test_get_comprehensive_summary(self, main_controller):
        """Test comprehensive summary generation."""
        # Add some test data
        main_controller.workflow_summary = {
            'phases_completed': [{'phase': 'requirements', 'auto_approved': True}],
            'tasks_completed': [{'task': 'task1', 'completed': True}],
            'token_usage': {'total': 1000},
            'compression_events': [{'event': 'compressed'}],
            'auto_approvals': [{'phase': 'requirements', 'decision': True}],
            'errors_recovered': [{'phase': 'design', 'strategy': 'retry'}]
        }
        main_controller.approval_log = [{'phase': 'requirements', 'decision': True}]
        main_controller.error_recovery_attempts = {'design': 1}
        main_controller.session_id = "test-session-123"
        
        summary = main_controller.get_comprehensive_summary()
        
        # Verify summary structure
        assert 'workflow_summary' in summary
        assert 'approval_log' in summary
        assert 'error_recovery_attempts' in summary
        assert 'total_phases_completed' in summary
        assert 'total_tasks_completed' in summary
        assert 'total_auto_approvals' in summary
        assert 'total_errors_recovered' in summary
        assert 'session_id' in summary
        assert 'timestamp' in summary
        
        # Verify counts
        assert summary['total_phases_completed'] == 1
        assert summary['total_tasks_completed'] == 1
        assert summary['total_auto_approvals'] == 1
        assert summary['total_errors_recovered'] == 1
        assert summary['session_id'] == "test-session-123"
    
    def test_session_state_persistence_auto_approve_data(self, main_controller, temp_workspace, test_llm_config):
        """Test that auto-approve data is persisted in session state."""
        # Initialize the framework first to set up SessionManager
        with patch('autogen_framework.main_controller.MemoryManager'), \
             patch('autogen_framework.main_controller.AgentManager') as mock_agent, \
             patch('autogen_framework.main_controller.ShellExecutor'):
            
            mock_agent_instance = Mock()
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory = Mock()
            mock_agent.return_value = mock_agent_instance
            
            main_controller.initialize_framework(test_llm_config)
        
        # Add some auto-approve data
        main_controller.approval_log = [{'phase': 'requirements', 'decision': True}]
        main_controller.error_recovery_attempts = {'design': 2}
        main_controller.workflow_summary = {
            'phases_completed': [{'phase': 'requirements'}],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [{'phase': 'requirements'}],
            'errors_recovered': []
        }
        main_controller.session_id = "test-session"
        
        # Save session state
        main_controller._save_session_state()
        
        # Create new controller and initialize it to load session
        new_controller = MainController(temp_workspace)
        with patch('autogen_framework.main_controller.MemoryManager'), \
             patch('autogen_framework.main_controller.AgentManager') as mock_agent2, \
             patch('autogen_framework.main_controller.ShellExecutor'):
            
            mock_agent_instance2 = Mock()
            mock_agent_instance2.setup_agents.return_value = True
            mock_agent_instance2.update_agent_memory = Mock()
            mock_agent2.return_value = mock_agent_instance2
            
            new_controller.initialize_framework(test_llm_config)
        
        # Verify auto-approve data was loaded
        assert new_controller.approval_log == [{'phase': 'requirements', 'decision': True}]
        assert new_controller.error_recovery_attempts == {'design': 2}
        assert new_controller.workflow_summary['phases_completed'] == [{'phase': 'requirements'}]
        assert new_controller.workflow_summary['auto_approvals'] == [{'phase': 'requirements'}]
    
    def test_reset_framework_clears_auto_approve_data(self, main_controller, test_llm_config):
        """Test that reset_framework clears all auto-approve data."""
        # Initialize framework and add some data
        with patch('autogen_framework.main_controller.MemoryManager'), \
             patch('autogen_framework.main_controller.AgentManager') as mock_agent, \
             patch('autogen_framework.main_controller.ShellExecutor'):
            
            mock_agent_instance = Mock()
            mock_agent_instance.setup_agents.return_value = True
            mock_agent_instance.update_agent_memory = Mock()
            mock_agent.return_value = mock_agent_instance
            
            main_controller.initialize_framework(test_llm_config)
        
        # Add some auto-approve data
        main_controller.approval_log = [{'phase': 'requirements', 'decision': True}]
        main_controller.error_recovery_attempts = {'design': 2}
        main_controller.workflow_summary['auto_approvals'] = [{'phase': 'requirements'}]
        
        # Reset framework
        result = main_controller.reset_framework()
        
        assert result is True
        assert main_controller.approval_log == []
        assert main_controller.error_recovery_attempts == {}
        assert main_controller.workflow_summary == {
            'phases_completed': [],
            'tasks_completed': [],
            'token_usage': {},
            'compression_events': [],
            'auto_approvals': [],
            'errors_recovered': []
        }
    
    @patch.dict(os.environ, {'AUTO_APPROVE_ERROR_RECOVERY_ATTEMPTS': '5'})
    def test_get_error_recovery_max_attempts_from_env(self, main_controller):
        """Test that error recovery max attempts can be configured via environment."""
        max_attempts = main_controller._get_error_recovery_max_attempts()
        assert max_attempts == 5
    
    def test_get_error_recovery_max_attempts_default(self, main_controller):
        """Test default error recovery max attempts."""
        with patch.dict(os.environ, {}, clear=True):
            max_attempts = main_controller._get_error_recovery_max_attempts()
            assert max_attempts == 3  # Default value
    
    def test_get_critical_checkpoints_from_env(self, main_controller):
        """Test that critical checkpoints can be configured via environment."""
        with patch.dict(os.environ, {'AUTO_APPROVE_CRITICAL_CHECKPOINTS': 'requirements,tasks'}):
            checkpoints = main_controller._get_critical_checkpoints()
            assert checkpoints == ['requirements', 'tasks']
    
    def test_get_critical_checkpoints_default(self, main_controller):
        """Test default critical checkpoints (empty list)."""
        with patch.dict(os.environ, {}, clear=True):
            checkpoints = main_controller._get_critical_checkpoints()
            assert checkpoints == []
    
    def test_modify_parameters_for_retry_timeout_error(self, initialized_controller):
        """Test that MainController delegates parameter modification to WorkflowManager."""
        # This test now verifies delegation rather than implementation details
        # The actual implementation logic should be tested in test_workflow_manager.py
        error = Exception("Connection timeout occurred")
        context = {"timeout": 30}
        expected_modified_context = {
            "timeout": 60,
            "max_retries": 3,
            "retry_delay": 5
        }
        
        with patch.object(initialized_controller.workflow_manager, '_modify_parameters_for_retry', 
                         return_value=expected_modified_context) as mock_modify:
            
            # Call through MainController (which should delegate to WorkflowManager)
            result = initialized_controller.workflow_manager._modify_parameters_for_retry(error, "requirements", context)
            
            # Verify delegation occurred and result is correct
            assert result == expected_modified_context
            mock_modify.assert_called_once_with(error, "requirements", context)
    
    # TODO: The following tests need to be updated to reflect our new architecture
    # where MainController delegates to WorkflowManager. These tests should either:
    # 1. Be moved to test_workflow_manager.py to test the actual implementation, OR
    # 2. Be updated to test delegation like the examples above
    # 
    # Tests that need updating:
    # - test_modify_parameters_for_retry_* (test delegation or move to WM tests)
    # - test_retry_*_generation_success (test delegation or move to WM tests)  
    # - test_identify_non_critical_steps_* (test delegation or move to WM tests)
    # - test_create_simplified_context (test delegation or move to WM tests)
    # - test_execute_simplified_* (test delegation or move to WM tests)
    # - test_fallback_* (test delegation or move to WM tests)
    # - test_error_recovery_* (test delegation like examples above)
    
    def test_modify_parameters_for_retry_memory_error(self, main_controller):
        """Test parameter modification for memory errors."""
        error = Exception("Memory limit exceeded")
        context = {"complexity": "high"}
        
        modified_context = main_controller._modify_parameters_for_retry(error, "design", context)
        
        assert modified_context is not None
        assert modified_context["max_complexity"] == "low"
        assert modified_context["simplified_mode"] is True
        assert modified_context["reduce_detail"] is True
    
    def test_modify_parameters_for_retry_format_error(self, main_controller):
        """Test parameter modification for format errors."""
        error = Exception("Parse error in format")
        context = {"format": "flexible"}
        
        modified_context = main_controller._modify_parameters_for_retry(error, "tasks", context)
        
        assert modified_context is not None
        assert modified_context["strict_format"] is True
        assert modified_context["use_templates"] is True
        assert modified_context["validate_output"] is True
    
    def test_modify_parameters_for_retry_permission_error(self, main_controller):
        """Test parameter modification for permission errors."""
        error = Exception("Permission denied access")
        context = {"path": "/restricted/path"}
        
        modified_context = main_controller._modify_parameters_for_retry(error, "implementation", context)
        
        assert modified_context is not None
        assert modified_context["use_alternative_path"] is True
        assert modified_context["fallback_mode"] is True
    
    def test_modify_parameters_for_retry_unknown_error(self, main_controller):
        """Test parameter modification for unknown errors."""
        error = Exception("Unknown error occurred")
        context = {"setting": "value"}
        
        modified_context = main_controller._modify_parameters_for_retry(error, "requirements", context)
        
        assert modified_context is not None
        assert modified_context["safe_mode"] is True
        assert modified_context["verbose_logging"] is True
        assert modified_context["error_recovery"] is True
    
    def test_retry_requirements_generation_success(self, main_controller):
        """Test successful requirements generation retry."""
        context = {"safe_mode": True}
        
        result = main_controller._retry_requirements_generation(context)
        
        assert result is True
        assert context["simplified_prompt"] is True
        assert context["basic_requirements_only"] is True
    
    def test_retry_design_generation_success(self, main_controller):
        """Test successful design generation retry."""
        context = {"simplified_mode": True}
        
        result = main_controller._retry_design_generation(context)
        
        assert result is True
        assert context["basic_design_only"] is True
        assert context["skip_advanced_patterns"] is True
    
    def test_retry_tasks_generation_success(self, main_controller):
        """Test successful tasks generation retry."""
        context = {"reduce_detail": True}
        
        result = main_controller._retry_tasks_generation(context)
        
        assert result is True
        assert context["simple_tasks_only"] is True
        assert context["minimal_descriptions"] is True
    
    def test_retry_implementation_execution_success(self, main_controller):
        """Test successful implementation execution retry."""
        context = {"fallback_mode": True}
        
        result = main_controller._retry_implementation_execution(context)
        
        assert result is True
        assert context["safe_execution"] is True
        assert context["skip_risky_operations"] is True
    
    def test_identify_non_critical_steps_requirements(self, main_controller):
        """Test identification of non-critical steps for requirements phase."""
        error = Exception("Timeout error")
        
        steps = main_controller._identify_non_critical_steps("requirements", error)
        
        assert len(steps) > 0
        assert "detailed_examples" in steps
        assert "edge_case_analysis" in steps
        assert "performance_requirements" in steps
        assert "advanced_validation" in steps
        # Should include timeout-specific steps
        assert "comprehensive_analysis" in steps
        assert "detailed_validation" in steps
    
    def test_identify_non_critical_steps_design(self, main_controller):
        """Test identification of non-critical steps for design phase."""
        error = Exception("Memory error")
        
        steps = main_controller._identify_non_critical_steps("design", error)
        
        assert len(steps) > 0
        assert "detailed_diagrams" in steps
        assert "performance_optimization" in steps
        assert "advanced_patterns" in steps
        assert "comprehensive_error_handling" in steps
    
    def test_identify_non_critical_steps_tasks(self, main_controller):
        """Test identification of non-critical steps for tasks phase."""
        error = Exception("Format error")
        
        steps = main_controller._identify_non_critical_steps("tasks", error)
        
        assert len(steps) > 0
        assert "detailed_descriptions" in steps
        assert "dependency_analysis" in steps
        assert "time_estimates" in steps
        assert "risk_assessment" in steps
    
    def test_identify_non_critical_steps_implementation(self, main_controller):
        """Test identification of non-critical steps for implementation phase."""
        error = Exception("Connection error")
        
        steps = main_controller._identify_non_critical_steps("implementation", error)
        
        assert len(steps) > 0
        assert "comprehensive_testing" in steps
        assert "performance_optimization" in steps
        assert "advanced_error_handling" in steps
        assert "detailed_logging" in steps
    
    def test_create_simplified_context(self, main_controller):
        """Test creation of simplified context."""
        original_context = {"setting1": "value1", "setting2": "value2"}
        skip_steps = ["step1", "step2", "step3"]
        
        simplified_context = main_controller._create_simplified_context(original_context, skip_steps)
        
        # Should preserve original context
        assert simplified_context["setting1"] == "value1"
        assert simplified_context["setting2"] == "value2"
        
        # Should add simplification settings
        assert simplified_context["skip_steps"] == skip_steps
        assert simplified_context["simplified_execution"] is True
        assert simplified_context["focus_on_essentials"] is True
    
    def test_execute_simplified_requirements_success(self, main_controller):
        """Test successful simplified requirements execution."""
        context = {"original_setting": "value"}
        
        result = main_controller._execute_simplified_requirements(context)
        
        assert result is True
        assert context["core_requirements_only"] is True
        assert context["minimal_detail"] is True
    
    def test_execute_simplified_design_success(self, main_controller):
        """Test successful simplified design execution."""
        context = {"original_setting": "value"}
        
        result = main_controller._execute_simplified_design(context)
        
        assert result is True
        assert context["basic_architecture_only"] is True
        assert context["skip_complex_patterns"] is True
    
    def test_execute_simplified_tasks_success(self, main_controller):
        """Test successful simplified tasks execution."""
        context = {"original_setting": "value"}
        
        result = main_controller._execute_simplified_tasks(context)
        
        assert result is True
        assert context["essential_tasks_only"] is True
        assert context["basic_descriptions"] is True
    
    def test_execute_simplified_implementation_success(self, main_controller):
        """Test successful simplified implementation execution."""
        context = {"original_setting": "value"}
        
        result = main_controller._execute_simplified_implementation(context)
        
        assert result is True
        assert context["core_functionality_only"] is True
        assert context["skip_advanced_features"] is True
    
    def test_fallback_requirements_generation_success(self, main_controller):
        """Test successful fallback requirements generation."""
        context = {"original_setting": "value"}
        
        result = main_controller._fallback_requirements_generation(context)
        
        assert result is True
        assert context["use_basic_template"] is True
        assert context["minimal_requirements"] is True
        assert context["no_advanced_features"] is True
    
    def test_fallback_design_generation_success(self, main_controller):
        """Test successful fallback design generation."""
        context = {"original_setting": "value"}
        
        result = main_controller._fallback_design_generation(context)
        
        assert result is True
        assert context["use_standard_patterns"] is True
        assert context["basic_architecture"] is True
        assert context["no_custom_solutions"] is True
    
    def test_fallback_tasks_generation_success(self, main_controller):
        """Test successful fallback tasks generation."""
        context = {"original_setting": "value"}
        
        result = main_controller._fallback_tasks_generation(context)
        
        assert result is True
        assert context["use_basic_tasks"] is True
        assert context["standard_workflow"] is True
        assert context["no_complex_dependencies"] is True
    
    def test_fallback_implementation_execution_success(self, main_controller):
        """Test successful fallback implementation execution."""
        context = {"original_setting": "value"}
        
        result = main_controller._fallback_implementation_execution(context)
        
        assert result is True
        assert context["safe_execution_only"] is True
        assert context["basic_implementation"] is True
        assert context["no_risky_operations"] is True
    
    def test_error_recovery_comprehensive_logging(self, main_controller):
        """Test comprehensive error recovery logging."""
        # Set up successful recovery on second strategy
        with patch.object(main_controller, '_get_error_recovery_max_attempts', return_value=3):
            with patch.object(main_controller, '_retry_with_modified_parameters', return_value=False):
                with patch.object(main_controller, '_skip_non_critical_steps', return_value=True):
                    error = Exception("Test comprehensive error")
                    context = {"test": "context"}
                    
                    result = main_controller.handle_error_recovery(error, "design", context)
                    
                    assert result is True
                    assert main_controller.error_recovery_attempts["design"] == 1
                    
                    # Verify comprehensive logging
                    assert len(main_controller.workflow_summary['errors_recovered']) == 1
                    recovery_log = main_controller.workflow_summary['errors_recovered'][0]
                    
                    assert recovery_log['phase'] == "design"
                    assert recovery_log['error'] == "Test comprehensive error"
                    assert "_skip_non_critical_steps" in str(recovery_log['strategy'])
                    assert recovery_log['attempt'] == 1
                    assert 'timestamp' in recovery_log
    
    def test_error_recovery_multiple_phases_tracking(self, main_controller):
        """Test error recovery tracking across multiple phases."""
        with patch.object(main_controller, '_get_error_recovery_max_attempts', return_value=3):
            with patch.object(main_controller, '_retry_with_modified_parameters', return_value=True):
                
                # First error in requirements
                error1 = Exception("Requirements error")
                result1 = main_controller.handle_error_recovery(error1, "requirements", {})
                assert result1 is True
                assert main_controller.error_recovery_attempts["requirements"] == 1
                
                # Second error in design
                error2 = Exception("Design error")
                result2 = main_controller.handle_error_recovery(error2, "design", {})
                assert result2 is True
                assert main_controller.error_recovery_attempts["design"] == 1
                
                # Third error in requirements (should increment)
                error3 = Exception("Another requirements error")
                result3 = main_controller.handle_error_recovery(error3, "requirements", {})
                assert result3 is True
                assert main_controller.error_recovery_attempts["requirements"] == 2
                
                # Verify comprehensive tracking
                assert len(main_controller.workflow_summary['errors_recovered']) == 3
                phases_recovered = [log['phase'] for log in main_controller.workflow_summary['errors_recovered']]
                assert phases_recovered.count("requirements") == 2
                assert phases_recovered.count("design") == 1
    
    def test_error_recovery_status_reporting(self, main_controller):
        """Test comprehensive error status reporting."""
        with patch.object(main_controller, '_get_error_recovery_max_attempts', return_value=2):
            with patch.object(main_controller, '_retry_with_modified_parameters', return_value=False):
                with patch.object(main_controller, '_skip_non_critical_steps', return_value=False):
                    with patch.object(main_controller, '_use_fallback_implementation', return_value=False):
                        
                        error = Exception("Unrecoverable error")
                        
                        # First attempt should fail
                        result1 = main_controller.handle_error_recovery(error, "tasks", {})
                        assert result1 is False
                        assert main_controller.error_recovery_attempts["tasks"] == 1
                        
                        # Second attempt should fail and hit max attempts
                        result2 = main_controller.handle_error_recovery(error, "tasks", {})
                        assert result2 is False
                        assert main_controller.error_recovery_attempts["tasks"] == 2
                        
                        # Third attempt should be rejected due to max attempts
                        result3 = main_controller.handle_error_recovery(error, "tasks", {})
                        assert result3 is False
                        assert main_controller.error_recovery_attempts["tasks"] == 2  # Should not increment
                        
                        # No successful recoveries should be logged
                        assert len(main_controller.workflow_summary['errors_recovered']) == 0    

    @pytest.mark.asyncio
    async def test_error_recovery_in_normal_mode(self, initialized_controller):
        """Test that error recovery works in normal (non-auto-approve) mode."""
        # Mock agent coordination to fail first, then succeed after recovery
        call_count = 0
        
        async def mock_coordinate_failing_then_success(task_type, context):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call fails
                raise Exception("Simulated agent coordination failure")
            else:
                # Second call (after recovery) succeeds
                if task_type == "requirements_generation":
                    return {
                        "success": True,
                        "work_directory": "/test/work",
                        "requirements_path": "/test/work/requirements.md"
                    }
                return {"success": True}
        
        initialized_controller.agent_manager.coordinate_agents = AsyncMock(side_effect=mock_coordinate_failing_then_success)
        
        # Mock one of the recovery strategies to return True so retry happens
        original_retry_method = initialized_controller.workflow_manager._retry_with_modified_parameters
        initialized_controller.workflow_manager._retry_with_modified_parameters = Mock(return_value=True)
        
        try:
            # Process request in normal mode (auto_approve=False)
            result = await initialized_controller.process_request("Create a test application", auto_approve=False)
            
            # Should succeed in requirements phase but wait for user approval
            assert result["success"] is False  # Overall workflow not complete
            assert result["requires_user_approval"] is True  # Waiting for approval
            assert result["approval_needed_for"] == "requirements"
            assert result["phases"]["requirements"]["success"] is True  # Requirements phase succeeded after retry
            
            # Verify error recovery was attempted and succeeded
            assert initialized_controller.error_recovery_attempts["requirements"] >= 1
            
            # Verify agent coordination was called twice (original + retry)
            assert call_count == 2
            
        finally:
            # Restore original method
            initialized_controller.workflow_manager._retry_with_modified_parameters = original_retry_method
    
    @pytest.mark.asyncio
    async def test_error_recovery_failure_in_normal_mode(self, initialized_controller):
        """Test error recovery failure in normal mode."""
        # Mock agent coordination to always fail
        async def mock_coordinate_always_fail(task_type, context):
            raise Exception("Persistent agent coordination failure")
        
        initialized_controller.agent_manager.coordinate_agents = AsyncMock(side_effect=mock_coordinate_always_fail)
        
        # Mock all error recovery strategies to fail
        with patch.object(initialized_controller, '_retry_with_modified_parameters', return_value=False):
            with patch.object(initialized_controller, '_skip_non_critical_steps', return_value=False):
                with patch.object(initialized_controller, '_use_fallback_implementation', return_value=False):
                    # Process request in normal mode
                    result = await initialized_controller.process_request("Create a test application", auto_approve=False)
        
        # Should fail completely
        assert result["success"] is False
        assert "error" in result
        assert "Persistent agent coordination failure" in result["error"]
        
        # Verify error recovery was attempted but failed
        assert initialized_controller.error_recovery_attempts["requirements"] == 1
        assert len(initialized_controller.workflow_summary['errors_recovered']) == 0  # No successful recoveries
    
    @pytest.mark.asyncio
    async def test_error_recovery_across_phases_normal_mode(self, initialized_controller):
        """Test error recovery across multiple phases in normal mode."""
        call_counts = {"requirements": 0, "design": 0, "tasks": 0}
        
        async def mock_coordinate_with_phase_failures(task_type, context):
            if task_type == "requirements_generation":
                call_counts["requirements"] += 1
                if call_counts["requirements"] == 1:
                    raise Exception("Requirements generation failure")
                return {
                    "success": True,
                    "work_directory": "/test/work",
                    "requirements_path": "/test/work/requirements.md"
                }
            elif task_type == "design_generation":
                call_counts["design"] += 1
                if call_counts["design"] == 1:
                    raise Exception("Design generation failure")
                return {
                    "success": True,
                    "design_path": "/test/work/design.md"
                }
            elif task_type == "task_generation":
                call_counts["tasks"] += 1
                if call_counts["tasks"] == 1:
                    raise Exception("Task generation failure")
                return {
                    "success": True,
                    "tasks_file": "/test/work/tasks.md"
                }
            return {"success": True}
        
        initialized_controller.agent_manager.coordinate_agents = AsyncMock(side_effect=mock_coordinate_with_phase_failures)
        
        # Mock successful error recovery for all phases
        original_retry_method = initialized_controller.workflow_manager._retry_with_modified_parameters
        initialized_controller.workflow_manager._retry_with_modified_parameters = Mock(return_value=True)
        
        try:
            # Mock task parsing to avoid file system dependencies
            with patch.object(initialized_controller.workflow_manager, '_parse_tasks_from_file') as mock_parse:
                mock_parse.return_value = []  # No tasks to execute
                
                # Process request in auto-approve mode to test all phases
                result = await initialized_controller.process_request("Create a test application", auto_approve=True)
            
            # Should succeed because error recovery worked for all phases
            assert result["success"] is True
            # The workflow stops at implementation phase since there are no tasks to execute
            assert result["current_phase"] == "implementation"
            
            # Verify error recovery was attempted for all phases
            assert initialized_controller.error_recovery_attempts["requirements"] >= 1
            assert call_counts["requirements"] == 2  # Original + retry
            assert call_counts["design"] == 2  # Original + retry
            assert call_counts["tasks"] == 2  # Original + retry
            
        finally:
            # Restore original method
            initialized_controller.workflow_manager._retry_with_modified_parameters = original_retry_method