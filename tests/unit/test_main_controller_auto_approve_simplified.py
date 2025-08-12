"""
Simplified unit tests for MainController auto-approve functionality.

This module contains unit tests for the simplified auto-approve features
after the error recovery system was simplified. These tests focus on:

- Auto-approve parameter handling in process_request method
- should_auto_approve method logic with critical checkpoint respect
- Simplified error recovery delegation to WorkflowManager
- Approval logging functionality for audit trail

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


class TestMainControllerAutoApproveSimplified:
    """Test cases for MainController simplified auto-approve functionality."""
    
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
            
            # Initialize the controller
            main_controller.initialize_framework(llm_config)
            
            return main_controller
    
    def test_auto_approve_initialization(self, main_controller):
        """Test that auto-approve data structures are initialized correctly."""
        assert hasattr(main_controller, 'user_approval_status')
        assert isinstance(main_controller.user_approval_status, dict)
        assert hasattr(main_controller, 'approval_log')
        assert isinstance(main_controller.approval_log, list)
    
    def test_should_auto_approve_with_manual_approval(self, main_controller):
        """Test should_auto_approve when manual approval is required."""
        # Test with auto_approve=False (manual approval required)
        result = main_controller.should_auto_approve("requirements", False)
        assert result is False
    
    def test_should_auto_approve_disabled(self, main_controller):
        """Test should_auto_approve when auto-approve is disabled."""
        result = main_controller.should_auto_approve("design", False)
        assert result is False
    
    def test_should_auto_approve_enabled(self, main_controller):
        """Test should_auto_approve when auto-approve is enabled."""
        with patch.object(main_controller, '_get_critical_checkpoints', return_value=[]):
            result = main_controller.should_auto_approve("tasks", True)
            assert result is True
    
    def test_should_auto_approve_critical_checkpoints(self, main_controller):
        """Test should_auto_approve respects critical checkpoints."""
        with patch.object(main_controller, '_get_critical_checkpoints', return_value=["requirements"]):
            # Critical checkpoint should require manual approval
            result = main_controller.should_auto_approve("requirements", True)
            assert result is False
            
            # Non-critical checkpoint should allow auto-approval
            result = main_controller.should_auto_approve("design", True)
            assert result is True
    
    def test_log_auto_approval(self, initialized_controller):
        """Test auto-approval logging functionality."""
        initialized_controller.log_auto_approval("requirements", True, "Auto-approved non-critical phase")
        
        # Check WorkflowManager's approval_log since MainController delegates
        assert len(initialized_controller.workflow_manager.approval_log) == 1
        log_entry = initialized_controller.workflow_manager.approval_log[0]
        assert log_entry["phase"] == "requirements"
        assert log_entry["decision"] is True  # The actual field name
        assert log_entry["reason"] == "Auto-approved non-critical phase"
        assert "timestamp" in log_entry
    
    def test_error_recovery_delegation_to_workflow_manager(self, initialized_controller):
        """Test that MainController delegates error recovery to WorkflowManager."""
        # Test the delegation pattern: MainController should delegate to WorkflowManager
        with patch.object(initialized_controller.workflow_manager, 'handle_error_recovery', return_value=False) as mock_wm_recovery:
            error = Exception("Test error")
            context = {"test": "context"}
            
            # Call MainController method
            result = initialized_controller.handle_error_recovery(error, "requirements", context)
            
            # Verify MainController delegates to WorkflowManager
            assert result is False  # Simplified error recovery always returns False
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
        main_controller.approval_log.append({
            "phase": "requirements",
            "decision": True,
            "reason": "Test approval",
            "timestamp": datetime.now().isoformat()
        })
        
        summary = main_controller.get_comprehensive_summary()
        
        assert "approval_log" in summary
        assert "workflow_summary" in summary
        assert "session_id" in summary
        assert len(summary["approval_log"]) == 1
    
    def test_simplified_error_recovery_behavior(self, initialized_controller):
        """Test that simplified error recovery always returns False."""
        # Test that error recovery is simplified and always returns False
        result = initialized_controller.handle_error_recovery(
            Exception("Test error"), "requirements", {}
        )
        assert result is False
        
        # Test that WorkflowManager provides user guidance
        guidance = initialized_controller.workflow_manager._get_phase_error_guidance("requirements")
        assert "--revise" in guidance
        assert "requirements" in guidance
    
    def test_get_critical_checkpoints_from_env(self, main_controller):
        """Test getting critical checkpoints from environment variable."""
        with patch.dict(os.environ, {'AUTO_APPROVE_CRITICAL_CHECKPOINTS': 'requirements,design'}):
            checkpoints = main_controller._get_critical_checkpoints()
            assert "requirements" in checkpoints
            assert "design" in checkpoints
            assert len(checkpoints) == 2
    
    def test_get_critical_checkpoints_default(self, main_controller):
        """Test getting critical checkpoints with default value."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing environment variable
            if 'AUTO_APPROVE_CRITICAL_CHECKPOINTS' in os.environ:
                del os.environ['AUTO_APPROVE_CRITICAL_CHECKPOINTS']
            
            checkpoints = main_controller._get_critical_checkpoints()
            assert checkpoints == []