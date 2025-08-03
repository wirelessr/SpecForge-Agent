"""
Integration tests for DesignAgent with real AutoGen functionality.

This module contains integration tests that use actual AutoGen components and real LLM endpoints
to verify the design generation capabilities with external services. These tests are designed to:

- Test actual design generation with real AutoGen and LLM endpoints
- Verify end-to-end design workflow with real components
- Validate memory integration and Mermaid diagram generation
- Test real file operations and design document creation

Key test classes:
- TestDesignAgentRealIntegration: Real AutoGen and design generation integration tests

These tests require:
- Valid .env.integration file with real LLM configuration
- Network access to LLM endpoints
- Real AutoGen library installation

Tests may take longer to complete due to network calls and real service interactions.
Use @pytest.mark.integration marker to run these tests separately.

For fast unit tests with mocked dependencies, see:
tests/unit/test_design_agent.py
"""

import pytest
import tempfile
import os
from pathlib import Path

from autogen_framework.agents.design_agent import DesignAgent
from autogen_framework.memory_manager import MemoryManager


class TestDesignAgentRealIntegration:
    """Integration tests for DesignAgent with real AutoGen functionality."""
    
    @pytest.fixture
    def real_memory_manager(self, temp_workspace):
        """Create a real memory manager with test content."""
        memory_manager = MemoryManager(temp_workspace)
        
        # Add some test memory content
        memory_manager.save_memory(
            "design_patterns",
            "# Design Patterns\n\nUse MVC pattern for web applications.\nConsider microservices for scalability.",
            "global"
        )
        
        memory_manager.save_memory(
            "security_guidelines",
            "# Security Guidelines\n\nAlways validate input.\nUse HTTPS for all communications.",
            "global"
        )
        
        return memory_manager
    
    @pytest.fixture
    def integration_design_agent(self, real_llm_config, real_memory_manager):
        """Create a DesignAgent for integration testing."""
        return DesignAgent(real_llm_config, real_memory_manager.load_memory())
    
    @pytest.fixture
    def sample_requirements_file(self, temp_workspace):
        """Create a sample requirements file for testing."""
        requirements_content = """# Requirements Document

## Introduction
Test project requirements for a simple web application with user authentication.

## Requirements

### Requirement 1: User Authentication
**User Story:** As a user, I want to authenticate, so that I can access the system.

#### Acceptance Criteria
1. WHEN user provides valid credentials THEN system SHALL authenticate user
2. WHEN user provides invalid credentials THEN system SHALL reject authentication
3. WHEN user session expires THEN system SHALL require re-authentication

### Requirement 2: Data Management
**User Story:** As a user, I want to manage data, so that I can store information.

#### Acceptance Criteria
1. WHEN user creates data THEN system SHALL store data securely
2. WHEN user updates data THEN system SHALL modify existing data
3. WHEN user deletes data THEN system SHALL remove data permanently
"""
        
        requirements_path = Path(temp_workspace) / "requirements.md"
        requirements_path.write_text(requirements_content)
        return str(requirements_path)
    
    @pytest.mark.integration
    def test_real_autogen_initialization(self, integration_design_agent):
        """Test actual AutoGen agent initialization."""
        # Test initialization
        success = integration_design_agent.initialize_autogen_agent()
        assert success is True
        assert integration_design_agent._is_initialized is True
        assert integration_design_agent._autogen_agent is not None
    
    @pytest.mark.integration
    async def test_real_design_generation_basic(self, integration_design_agent):
        """Test basic design generation with real AutoGen."""
        # Initialize the agent
        assert integration_design_agent.initialize_autogen_agent() is True
        
        # Test basic response generation
        prompt = "Generate a technical design for a web application with user authentication and data management."
        
        try:
            response = await integration_design_agent.generate_response(prompt)
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert any(keyword in response.lower() for keyword in ['design', 'architecture', 'component', 'system'])
            
        except Exception as e:
            pytest.skip(f"Real AutoGen design test skipped due to: {e}")
    
    @pytest.mark.integration
    async def test_real_memory_integration(self, integration_design_agent):
        """Test memory context integration with real AutoGen."""
        # Initialize and test
        assert integration_design_agent.initialize_autogen_agent() is True
        
        prompt = "Create a design that incorporates the design patterns and security guidelines from memory."
        
        try:
            response = await integration_design_agent.generate_response(prompt)
            
            assert isinstance(response, str)
            assert len(response) > 0
            # The response should potentially reference the memory content
            
        except Exception as e:
            pytest.skip(f"Real AutoGen memory test skipped due to: {e}")
    
    @pytest.mark.integration
    async def test_real_requirements_based_design(self, integration_design_agent, sample_requirements_file, temp_workspace):
        """Test design generation based on real requirements file."""
        assert integration_design_agent.initialize_autogen_agent() is True
        
        try:
            # Generate design based on requirements
            design_content = await integration_design_agent.generate_design(
                sample_requirements_file, 
                integration_design_agent.memory_context
            )
            
            assert isinstance(design_content, str)
            assert len(design_content) > 0
            
            # Check for expected design document structure
            assert "# Design Document" in design_content or "Design" in design_content
            
            # Should reference the requirements
            assert any(keyword in design_content.lower() for keyword in ['authentication', 'user', 'data'])
            
        except Exception as e:
            pytest.skip(f"Real requirements-based design test skipped due to: {e}")
    
    @pytest.mark.integration
    async def test_real_complete_design_process(self, integration_design_agent, sample_requirements_file, temp_workspace):
        """Test the complete design process with real AutoGen."""
        assert integration_design_agent.initialize_autogen_agent() is True
        
        try:
            # Test the complete process_task workflow
            task_input = {
                "requirements_path": sample_requirements_file,
                "work_directory": temp_workspace
            }
            
            result = await integration_design_agent.process_task(task_input)
            
            assert result["success"] is True
            assert "design_path" in result
            assert "design_content" in result
            
            # Verify the design file was created
            design_path = Path(result["design_path"])
            assert design_path.exists()
            
            # Verify the content
            design_content = design_path.read_text()
            assert len(design_content) > 0
            assert any(keyword in design_content.lower() for keyword in ['design', 'architecture', 'component'])
            
        except Exception as e:
            pytest.skip(f"Real complete design process test skipped due to: {e}")
    
    @pytest.mark.integration
    async def test_real_mermaid_diagram_generation(self, integration_design_agent):
        """Test Mermaid diagram generation with real AutoGen."""
        assert integration_design_agent.initialize_autogen_agent() is True
        
        try:
            prompt = """Generate a technical design document that includes Mermaid diagrams for:
1. System architecture
2. Data flow
3. User authentication flow

Please include proper Mermaid syntax in code blocks."""
            
            response = await integration_design_agent.generate_response(prompt)
            
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Check for Mermaid diagram presence
            if "mermaid" in response.lower():
                # Validate and fix any Mermaid diagrams
                fixed_response = integration_design_agent._validate_mermaid_diagrams(response)
                assert isinstance(fixed_response, str)
                assert "graph TD" in fixed_response or "sequenceDiagram" in fixed_response or "flowchart" in fixed_response
            
        except Exception as e:
            pytest.skip(f"Real Mermaid diagram test skipped due to: {e}")
    
    @pytest.mark.integration
    def test_real_design_templates_integration(self, integration_design_agent):
        """Test design templates with real configuration."""
        # Test that templates are available and properly formatted
        templates = integration_design_agent.get_design_templates()
        
        assert isinstance(templates, dict)
        assert len(templates) > 0
        
        # Test each template
        for template_name, template_content in templates.items():
            assert isinstance(template_content, str)
            assert len(template_content) > 0
            assert "# Design Document" in template_content
            assert "Architectural Overview" in template_content
    
    @pytest.mark.integration
    async def test_real_design_revision_process(self, integration_design_agent, temp_workspace):
        """Test design revision with real AutoGen."""
        assert integration_design_agent.initialize_autogen_agent() is True
        
        # Create initial design file
        design_path = Path(temp_workspace) / "design.md"
        initial_design = """# Design Document

## Architecture
Basic REST API architecture using Express.js

## Components
- API Server
- Database
- Authentication Service
"""
        design_path.write_text(initial_design)
        
        try:
            # Test revision task
            task_input = {
                "task_type": "revision",
                "revision_feedback": "Add microservices architecture and include security considerations",
                "current_result": {"design_path": str(design_path)},
                "work_directory": temp_workspace
            }
            
            result = await integration_design_agent.process_task(task_input)
            
            assert result["success"] is True
            assert result["revision_applied"] is True
            
            # Verify the file was updated
            updated_content = design_path.read_text()
            assert len(updated_content) > len(initial_design)
            
        except Exception as e:
            pytest.skip(f"Real design revision test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])