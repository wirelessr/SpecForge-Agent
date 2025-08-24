"""
Test the dependency container fixtures to ensure they work correctly.

This test file validates that the new dependency container fixtures provide
the expected functionality for both unit and integration testing scenarios.
"""

import pytest
from unittest.mock import Mock

from autogen_framework.dependency_container import DependencyContainer
from autogen_framework.agents.plan_agent import PlanAgent
from autogen_framework.agents.design_agent import DesignAgent
from autogen_framework.agents.tasks_agent import TasksAgent
from autogen_framework.agents.implement_agent import ImplementAgent


class TestDependencyContainerFixtures:
    """Test that dependency container fixtures work correctly."""
    
    def test_mock_dependency_container_fixture(self, mock_dependency_container, test_llm_config, temp_workspace):
        """Test that mock_dependency_container fixture provides a working container."""
        # Verify container is properly configured
        assert isinstance(mock_dependency_container, DependencyContainer)
        assert mock_dependency_container.work_dir == temp_workspace
        assert mock_dependency_container.llm_config == test_llm_config
        assert mock_dependency_container._is_test_mode is True
        
        # Verify managers can be created and are mocked
        token_manager = mock_dependency_container.get_token_manager()
        assert token_manager is not None
        assert hasattr(token_manager, 'get_model_limit')  # Mock method
        
        context_manager = mock_dependency_container.get_context_manager()
        assert context_manager is not None
        assert hasattr(context_manager, 'initialize')  # Mock method
    
    def test_simple_plan_agent_fixture(self, simple_plan_agent, test_llm_config):
        """Test that simple_plan_agent fixture creates a working PlanAgent."""
        assert isinstance(simple_plan_agent, PlanAgent)
        assert simple_plan_agent.name == "PlanAgent"
        assert simple_plan_agent.llm_config == test_llm_config
        
        # Verify agent has access to mocked managers
        assert simple_plan_agent.token_manager is not None
        assert simple_plan_agent.context_manager is not None
        assert simple_plan_agent.memory_manager is not None
    
    def test_simple_design_agent_fixture(self, simple_design_agent, test_llm_config):
        """Test that simple_design_agent fixture creates a working DesignAgent."""
        assert isinstance(simple_design_agent, DesignAgent)
        assert simple_design_agent.name == "DesignAgent"
        assert simple_design_agent.llm_config == test_llm_config
        
        # Verify agent has access to mocked managers
        assert simple_design_agent.token_manager is not None
        assert simple_design_agent.context_manager is not None
    
    def test_simple_tasks_agent_fixture(self, simple_tasks_agent, test_llm_config):
        """Test that simple_tasks_agent fixture creates a working TasksAgent."""
        assert isinstance(simple_tasks_agent, TasksAgent)
        assert simple_tasks_agent.name == "TasksAgent"
        assert simple_tasks_agent.llm_config == test_llm_config
        
        # Verify agent has access to mocked managers
        assert simple_tasks_agent.token_manager is not None
        assert simple_tasks_agent.context_manager is not None
    
    def test_simple_implement_agent_fixture(self, simple_implement_agent, test_llm_config):
        """Test that simple_implement_agent fixture creates a working ImplementAgent."""
        assert isinstance(simple_implement_agent, ImplementAgent)
        assert simple_implement_agent.name == "ImplementAgent"
        assert simple_implement_agent.llm_config == test_llm_config
        
        # Verify agent has access to mocked managers
        assert simple_implement_agent.token_manager is not None
        assert simple_implement_agent.context_manager is not None
        assert simple_implement_agent.shell_executor is not None
    
    def test_container_lazy_loading(self, mock_dependency_container):
        """Test that managers are created lazily."""
        # Initially no managers should be created
        assert mock_dependency_container.get_manager_count() == 0
        
        # Create one manager
        token_manager = mock_dependency_container.get_token_manager()
        assert mock_dependency_container.get_manager_count() == 1
        
        # Getting the same manager again shouldn't create a new one
        token_manager2 = mock_dependency_container.get_token_manager()
        assert token_manager is token_manager2
        assert mock_dependency_container.get_manager_count() == 1
    
    def test_fixture_isolation(self, test_llm_config, temp_workspace):
        """Test that each fixture provides isolated container instances."""
        # Create two separate containers
        container1 = DependencyContainer.create_test(temp_workspace, test_llm_config)
        container2 = DependencyContainer.create_test(temp_workspace, test_llm_config)
        
        # They should be different instances
        assert container1 is not container2
        
        # But should have the same configuration
        assert container1.work_dir == container2.work_dir
        assert container1.llm_config == container2.llm_config


