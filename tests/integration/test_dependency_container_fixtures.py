"""
Test the dependency container fixtures for integration tests.

This test file validates that the real dependency container fixtures provide
the expected functionality for integration testing scenarios.
"""

import pytest
from autogen_framework.dependency_container import DependencyContainer


class TestRealDependencyContainerFixtures:
    """Test that real dependency container fixtures work correctly for integration tests."""
    
    @pytest.mark.integration
    def test_real_dependency_container_fixture(self, real_dependency_container, temp_workspace, real_llm_config):
        """Test that real_dependency_container fixture provides a working container."""
        # Verify container is properly configured
        assert isinstance(real_dependency_container, DependencyContainer)
        assert real_dependency_container.work_dir == temp_workspace
        assert real_dependency_container._is_test_mode is False
        assert real_dependency_container.llm_config == real_llm_config
        
        # Verify container has the right configuration without creating managers yet
        assert real_dependency_container.llm_config is not None
        assert real_dependency_container.llm_config.base_url is not None
        assert real_dependency_container.llm_config.model is not None
        
        # Test that we can create lightweight managers without hanging
        config_manager = real_dependency_container.get_config_manager()
        assert config_manager is not None
        assert hasattr(config_manager, 'get_model_info')
        
        memory_manager = real_dependency_container.get_memory_manager()
        assert memory_manager is not None
        assert hasattr(memory_manager, 'load_memory')
        
        # Verify the container tracks created managers
        assert real_dependency_container.get_manager_count() == 2
        created_managers = real_dependency_container.get_created_managers()
        assert 'config_manager' in created_managers
        assert 'memory_manager' in created_managers
    
    @pytest.mark.integration
    def test_real_agent_fixtures_container_ready(self, real_dependency_container):
        """Test that real_dependency_container is ready for agent creation."""
        # Test that the container is properly configured for creating agents
        assert real_dependency_container is not None
        assert real_dependency_container._is_test_mode is False
        assert real_dependency_container.llm_config is not None
        
        # Verify all required manager factory methods are available
        manager_methods = [
            'get_token_manager',
            'get_context_manager', 
            'get_memory_manager',
            'get_shell_executor',
            'get_error_recovery',
            'get_task_decomposer',
            'get_config_manager',
            'get_context_compressor'
        ]
        
        for method_name in manager_methods:
            assert hasattr(real_dependency_container, method_name), f"Missing method: {method_name}"
    
    @pytest.mark.integration
    def test_container_isolation_integration(self, real_llm_config, temp_workspace):
        """Test that each real container instance is isolated."""
        # Create two separate containers
        container1 = DependencyContainer.create_production(temp_workspace, real_llm_config)
        container2 = DependencyContainer.create_production(temp_workspace, real_llm_config)
        
        # They should be different instances
        assert container1 is not container2
        
        # But should have the same configuration
        assert container1.work_dir == container2.work_dir
        assert container1.llm_config == real_llm_config
        assert container2.llm_config == real_llm_config
        
        # Managers should be separate instances
        config_manager1 = container1.get_config_manager()
        config_manager2 = container2.get_config_manager()
        assert config_manager1 is not config_manager2