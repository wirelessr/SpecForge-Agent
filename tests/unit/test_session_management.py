"""
Session Management Test for AutoGen Multi-Agent Framework
This test verifies that session management works correctly,
including session persistence across command invocations.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, Mock

from autogen_framework.main_controller import MainController
from autogen_framework.models import LLMConfig

class TestSessionManagement:
    """Test session management functionality."""
    # Using shared fixtures from conftest.py
    @pytest.fixture
    def mock_test_llm_config(self):
        """Create a mock LLM configuration for testing."""
        return LLMConfig(
            base_url="http://test.local:8888/openai/v1",
            model="test-model",
            api_key="test-key"
        )
    
    def test_session_creation_and_persistence(self, temp_workspace, mock_test_llm_config):
        """Test that sessions are created and persisted correctly."""
        # Create first controller instance
        controller1 = MainController(temp_workspace)
        
        # Mock the necessary components
        with patch.object(controller1, 'agent_manager') as mock_agent_manager:
            mock_agent_manager.setup_agents.return_value = True
            mock_agent_manager.get_agent_status.return_value = {"status": "ready"}
            
            with patch.object(controller1, 'memory_manager') as mock_memory_manager:
                mock_memory_manager.load_memory.return_value = {"context": "test"}
                mock_memory_manager.get_memory_stats.return_value = {"loaded": True}
                
                with patch.object(controller1, 'shell_executor') as mock_shell_executor:
                    mock_shell_executor.get_execution_stats.return_value = {"ready": True}
                    
                    # Initialize framework
                    assert controller1.initialize_framework(mock_test_llm_config) == True
                    
                    # Get session ID
                    session_id1 = controller1.get_session_id()
                    assert session_id1 is not None
                    assert len(session_id1) > 0
                    
                    # Check that session file was created
                    session_file = Path(temp_workspace) / "memory" / "session_state.json"
                    assert session_file.exists()
                    
                    # Verify session file content
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    assert session_data['session_id'] == session_id1
                    assert 'last_updated' in session_data
        
        # Create second controller instance (simulating new command invocation)
        controller2 = MainController(temp_workspace)
        with patch.object(controller2, 'agent_manager') as mock_agent_manager:
            mock_agent_manager.setup_agents.return_value = True
            mock_agent_manager.get_agent_status.return_value = {"status": "ready"}
            
            with patch.object(controller2, 'memory_manager') as mock_memory_manager:
                mock_memory_manager.load_memory.return_value = {"context": "test"}
                mock_memory_manager.get_memory_stats.return_value = {"loaded": True}
                
                with patch.object(controller2, 'shell_executor') as mock_shell_executor:
                    mock_shell_executor.get_execution_stats.return_value = {"ready": True}
                    
                    # Initialize framework
                    assert controller2.initialize_framework(mock_test_llm_config) == True
                    
                    # Get session ID - should be the same as first instance
                    session_id2 = controller2.get_session_id()
                    assert session_id2 == session_id1
                    
                    print(f"✅ Session persistence verified: {session_id1}")
    
    def test_session_reset(self, temp_workspace, mock_test_llm_config):
        """Test that session reset works correctly."""
        controller = MainController(temp_workspace)
        
        with patch.object(controller, 'agent_manager') as mock_agent_manager:
            mock_agent_manager.setup_agents.return_value = True
            mock_agent_manager.get_agent_status.return_value = {"status": "ready"}
            
            with patch.object(controller, 'memory_manager') as mock_memory_manager:
                mock_memory_manager.load_memory.return_value = {"context": "test"}
                mock_memory_manager.get_memory_stats.return_value = {"loaded": True}
                
                with patch.object(controller, 'shell_executor') as mock_shell_executor:
                    mock_shell_executor.get_execution_stats.return_value = {"ready": True}
                    
                    # Initialize framework
                    assert controller.initialize_framework(mock_test_llm_config) == True
                    
                    # Get initial session ID
                    session_id1 = controller.get_session_id()
                    
                    # Add some state
                    controller.phase_results["test"] = {"data": "test"}
                    from autogen_framework.main_controller import UserApprovalStatus
                    controller.user_approval_status["test"] = UserApprovalStatus.APPROVED
                    
                    # Reset session
                    controller.reset_session()
                    
                    # Get new session ID
                    session_id2 = controller.get_session_id()
                    
                    # Verify session was reset
                    assert session_id2 != session_id1
                    assert controller.phase_results == {}
                    assert controller.user_approval_status == {}
                    assert controller.current_workflow is None
                    
                    print(f"✅ Session reset verified: {session_id1} -> {session_id2}")
    
    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, temp_workspace, mock_test_llm_config):
        """Test that workflow state persists across controller instances."""
        # Mock coordinate_agents function
        async def mock_coordinate_agents(task_type: str, context: dict) -> dict:
            if task_type == "requirements_generation":
                work_dir = Path(context["workspace_path"]) / "test-project"
                work_dir.mkdir(exist_ok=True)
                requirements_path = work_dir / "requirements.md"
                requirements_path.write_text("# Test Requirements")
                return {
                    "success": True,
                    "requirements_path": str(requirements_path),
                    "work_directory": str(work_dir)
                }
            return {"success": False}
        
        # First controller instance - start workflow
        controller1 = MainController(temp_workspace)
        with patch.object(controller1, 'agent_manager') as mock_agent_manager:
            mock_agent_manager.coordinate_agents = mock_coordinate_agents
            mock_agent_manager.setup_agents.return_value = True
            mock_agent_manager.get_agent_status.return_value = {"status": "ready"}
            mock_agent_manager.update_agent_memory = Mock()
            
            with patch.object(controller1, 'memory_manager') as mock_memory_manager:
                mock_memory_manager.load_memory.return_value = {"context": "test"}
                mock_memory_manager.get_memory_stats.return_value = {"loaded": True}
                
                with patch.object(controller1, 'shell_executor') as mock_shell_executor:
                    mock_shell_executor.get_execution_stats.return_value = {"ready": True}
                    
                    # Initialize and start workflow
                    assert controller1.initialize_framework(mock_test_llm_config) == True
                    session_id = controller1.get_session_id()
                    
                    # Process a request
                    result = await controller1.process_user_request("Create a test API")
                    assert result.get("requires_user_approval") == True
                    assert result.get("approval_needed_for") == "requirements"
                    
                    # Verify workflow state
                    assert controller1.current_workflow is not None
                    assert len(controller1.phase_results) > 0
        
        # Second controller instance - should load existing workflow
        controller2 = MainController(temp_workspace)
        with patch.object(controller2, 'agent_manager') as mock_agent_manager:
            mock_agent_manager.coordinate_agents = mock_coordinate_agents
            mock_agent_manager.setup_agents.return_value = True
            mock_agent_manager.get_agent_status.return_value = {"status": "ready"}
            mock_agent_manager.update_agent_memory = Mock()
            
            with patch.object(controller2, 'memory_manager') as mock_memory_manager:
                mock_memory_manager.load_memory.return_value = {"context": "test"}
                mock_memory_manager.get_memory_stats.return_value = {"loaded": True}
                
                with patch.object(controller2, 'shell_executor') as mock_shell_executor:
                    mock_shell_executor.get_execution_stats.return_value = {"ready": True}
                    
                    # Initialize framework
                    assert controller2.initialize_framework(mock_test_llm_config) == True
                    
                    # Verify session ID is the same
                    assert controller2.get_session_id() == session_id
                    
                    # Verify workflow state was loaded
                    assert controller2.current_workflow is not None
                    assert len(controller2.phase_results) > 0
                    assert "requirements" in controller2.phase_results
                    
                    # Verify we can get pending approval
                    pending = controller2.get_pending_approval()
                    assert pending is not None
                    assert pending["phase"] == "requirements"
                    
                    print(f"✅ Workflow state persistence verified for session: {session_id}")
    
    def test_session_status_display(self, temp_workspace, mock_test_llm_config):
        """Test that session ID is included in status display."""
        controller = MainController(temp_workspace)
        
        with patch.object(controller, 'agent_manager') as mock_agent_manager:
            mock_agent_manager.setup_agents.return_value = True
            mock_agent_manager.get_agent_status.return_value = {"status": "ready"}
            
            with patch.object(controller, 'memory_manager') as mock_memory_manager:
                mock_memory_manager.load_memory.return_value = {"context": "test"}
                mock_memory_manager.get_memory_stats.return_value = {"loaded": True}
                
                with patch.object(controller, 'shell_executor') as mock_shell_executor:
                    mock_shell_executor.get_execution_stats.return_value = {"ready": True}
                    
                    # Initialize framework
                    assert controller.initialize_framework(mock_test_llm_config) == True
                    
                    # Get status
                    status = controller.get_framework_status()
                    
                    # Verify session ID is included
                    assert "session_id" in status
                    assert status["session_id"] == controller.get_session_id()
                    assert len(status["session_id"]) > 0
                    
                    print(f"✅ Session ID in status: {status['session_id']}")

if __name__ == "__main__":
    # Run the test with verbose output
    pytest.main([__file__, "-v", "-s"])