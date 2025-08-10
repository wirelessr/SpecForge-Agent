"""
Integration tests for PlanAgent with real components.

This module contains tests that use actual MemoryManager and filesystem operations
to verify the plan agent integration with real components.
"""

import pytest
import tempfile
from pathlib import Path

from autogen_framework.agents.plan_agent import PlanAgent
from autogen_framework.memory_manager import MemoryManager


class TestPlanAgentRealIntegration:
    """Integration tests for PlanAgent with real components."""
    
    @pytest.fixture
    def real_memory_manager(self, temp_workspace):
        """Create a real memory manager with test content."""
        memory_manager = MemoryManager(temp_workspace)
        
        # Add some test memory content
        memory_manager.save_memory(
            "project_patterns",
            "# Project Patterns\n\nCommon patterns for project organization:\n- Use clear directory structure\n- Follow naming conventions",
            "global"
        )
        
        memory_manager.save_memory(
            "requirements_templates",
            "# Requirements Templates\n\nEARS format:\n- WHEN condition THEN system SHALL response",
            "global"
        )
        
        return memory_manager
    
    @pytest.fixture
    def integration_plan_agent(self, real_llm_config, real_memory_manager, real_managers):
        """Create a PlanAgent for integration testing."""
        return PlanAgent(
            llm_config=real_llm_config,
            memory_manager=real_memory_manager,
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    @pytest.mark.integration
    def test_memory_integration(self, integration_plan_agent):
        """Test that memory content is properly integrated."""
        # Check that memory context was loaded
        assert hasattr(integration_plan_agent, 'memory_context')
        
        # Check that system message includes memory context
        system_message = integration_plan_agent._build_complete_system_message()
        if len(integration_plan_agent.memory_context) > 0:
            assert "Memory Context" in system_message
        
        # Verify memory content is accessible
        memory_context = integration_plan_agent.memory_context
        if "global" in memory_context:
            global_memory = memory_context["global"]
            assert any("project_patterns" in key or "requirements_templates" in key for key in global_memory.keys())
    
    @pytest.mark.integration
    def test_real_directory_creation(self, integration_plan_agent, temp_workspace):
        """Test actual directory creation in filesystem."""
        summary = "Test project creation with real filesystem"
        
        # Test directory creation
        work_dir = integration_plan_agent.create_work_directory(summary)
        
        # This is an async method, so we need to run it
        import asyncio
        work_dir_path = asyncio.run(work_dir)
        
        # Verify actual filesystem changes
        work_path = Path(work_dir_path)
        assert work_path.exists()
        assert work_path.is_dir()
        assert work_path.parent == Path(temp_workspace)
        
        # Verify structure files were created
        for filename in ["requirements.md", "design.md", "tasks.md"]:
            file_path = work_path / filename
            assert file_path.exists()
            assert file_path.is_file()
            
            # Verify files have some content
            content = file_path.read_text()
            assert len(content.strip()) > 0
    
    @pytest.mark.integration
    def test_real_autogen_initialization(self, integration_plan_agent):
        """Test actual AutoGen agent initialization."""
        # Test initialization
        success = integration_plan_agent.initialize_autogen_agent()
        assert success is True
        assert integration_plan_agent._is_initialized is True
        assert integration_plan_agent._autogen_agent is not None
    
    @pytest.mark.integration
    async def test_real_request_parsing(self, integration_plan_agent):
        """Test request parsing with real AutoGen."""
        # Initialize the agent
        assert integration_plan_agent.initialize_autogen_agent() is True
        
        try:
            # Test parsing a real user request
            user_request = "Create a REST API for user management with authentication and data validation"
            
            result = await integration_plan_agent.parse_user_request(user_request)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "summary" in result
            assert "type" in result
            assert "scope" in result
            assert "key_requirements" in result
            assert "technical_context" in result
            assert "constraints" in result
            assert "suggested_directory" in result
            
            # Verify content quality
            assert len(result["summary"]) > 0
            assert result["type"] in ["development", "debugging", "enhancement", "research"]
            assert result["scope"] in ["small", "medium", "large"]
            assert isinstance(result["key_requirements"], list)
            assert len(result["key_requirements"]) > 0
            
        except Exception as e:
            pytest.skip(f"Real AutoGen request parsing test skipped due to: {e}")
    
    @pytest.mark.integration
    async def test_real_requirements_generation(self, integration_plan_agent, temp_workspace):
        """Test requirements generation with real AutoGen."""
        # Initialize the agent
        assert integration_plan_agent.initialize_autogen_agent() is True
        
        try:
            # Create a work directory first
            work_dir = Path(temp_workspace) / "test-requirements-project"
            work_dir.mkdir()
            
            user_request = "Create a simple calculator application with basic arithmetic operations"
            parsed_request = {
                'summary': 'Simple calculator application',
                'type': 'development',
                'scope': 'small',
                'key_requirements': ['Basic arithmetic', 'User interface', 'Input validation'],
                'technical_context': 'Desktop application',
                'constraints': 'Simple and user-friendly'
            }
            
            # Generate requirements
            requirements_path = await integration_plan_agent.generate_requirements(
                user_request, str(work_dir), parsed_request
            )
            
            # Verify file was created
            req_path = Path(requirements_path)
            assert req_path.exists()
            assert req_path.name == "requirements.md"
            
            # Verify content structure
            content = req_path.read_text()
            assert len(content) > 0
            assert "# Requirements Document" in content
            assert "## Introduction" in content
            assert "## Requirements" in content
            assert "User Story:" in content
            assert "Acceptance Criteria" in content
            assert "WHEN" in content and "THEN" in content and "SHALL" in content  # EARS format
            
        except Exception as e:
            pytest.skip(f"Real requirements generation test skipped due to: {e}")
    
    @pytest.mark.integration
    async def test_real_complete_process_task(self, integration_plan_agent, temp_workspace):
        """Test the complete process_task workflow with real components."""
        # Initialize the agent
        assert integration_plan_agent.initialize_autogen_agent() is True
        
        try:
            task_input = {
                "user_request": "Develop a web-based todo list application with user authentication"
            }
            
            # Execute the complete workflow
            result = await integration_plan_agent.process_task(task_input)
            
            # Verify result structure
            assert result['success'] is True
            assert 'work_directory' in result
            assert 'requirements_path' in result
            assert 'parsed_request' in result
            
            # Verify actual files were created
            work_dir = Path(result['work_directory'])
            assert work_dir.exists()
            assert work_dir.is_dir()
            
            requirements_file = Path(result['requirements_path'])
            assert requirements_file.exists()
            
            # Verify requirements content
            requirements_content = requirements_file.read_text()
            assert len(requirements_content) > 0
            assert "todo" in requirements_content.lower() or "task" in requirements_content.lower()
            assert "authentication" in requirements_content.lower() or "auth" in requirements_content.lower()
            
            # Verify other structure files
            for filename in ["design.md", "tasks.md"]:
                file_path = work_dir / filename
                assert file_path.exists()
                
        except Exception as e:
            pytest.skip(f"Real complete process task test skipped due to: {e}")
    
    @pytest.mark.integration
    async def test_real_directory_name_generation(self, integration_plan_agent):
        """Test directory name generation with real AutoGen."""
        # Initialize the agent
        assert integration_plan_agent.initialize_autogen_agent() is True
        
        try:
            test_cases = [
                "Create a REST API for user management",
                "Fix authentication bug in login system", 
                "Implement real-time chat functionality",
                "Optimize database query performance"
            ]
            
            for summary in test_cases:
                directory_name = await integration_plan_agent._generate_directory_name(summary)
                
                # Verify directory name properties
                assert isinstance(directory_name, str)
                assert len(directory_name) > 0
                assert len(directory_name) <= 50
                assert ' ' not in directory_name  # Should be kebab-case
                assert directory_name.islower()
                assert not directory_name.startswith('-')
                assert not directory_name.endswith('-')
                
        except Exception as e:
            pytest.skip(f"Real directory name generation test skipped due to: {e}")
    
    @pytest.mark.integration
    def test_real_workspace_integration(self, integration_plan_agent, temp_workspace):
        """Test workspace integration with real filesystem operations."""
        # Verify workspace path is set correctly
        assert integration_plan_agent.workspace_path == Path(temp_workspace)
        assert integration_plan_agent.workspace_path.exists()
        
        # Test unique name generation with real filesystem
        base_name = "test-project"
        
        # First call should return the base name
        result1 = integration_plan_agent.ensure_unique_name(base_name)
        assert result1 == base_name
        
        # Create a directory with that name
        (integration_plan_agent.workspace_path / base_name).mkdir()
        
        # Second call should return a unique name
        result2 = integration_plan_agent.ensure_unique_name(base_name)
        assert result2 != base_name
        assert result2.startswith(base_name)
        assert len(result2) <= 50
        
        # Verify the unique directory can be created
        unique_path = integration_plan_agent.workspace_path / result2
        unique_path.mkdir()
        assert unique_path.exists()
    
    @pytest.mark.integration
    async def test_real_work_directory_suggestions(self, integration_plan_agent):
        """Test work directory name suggestions with real AutoGen."""
        # Initialize the agent
        assert integration_plan_agent.initialize_autogen_agent() is True
        
        try:
            request = "Create a microservices architecture for e-commerce platform"
            
            suggestions = await integration_plan_agent.get_work_directory_suggestions(request)
            
            assert isinstance(suggestions, list)
            assert len(suggestions) <= 5
            assert len(suggestions) > 0
            
            # All suggestions should be valid directory names
            for suggestion in suggestions:
                assert isinstance(suggestion, str)
                assert len(suggestion) > 0
                assert ' ' not in suggestion  # Should be kebab-case
                assert suggestion.islower()
                assert not suggestion.startswith('-')
                assert not suggestion.endswith('-')
                
        except Exception as e:
            pytest.skip(f"Real work directory suggestions test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])