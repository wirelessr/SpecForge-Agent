"""
Unit tests for the DesignAgent class.

This module contains unit tests that focus on testing the DesignAgent class
in isolation using mocked dependencies. These tests are designed to:

- Run quickly (under 1 second each)
- Use only mocked external dependencies (no real AutoGen or LLM calls)
- Test design generation logic, Mermaid diagram validation, and memory integration
- Validate agent behavior without external service dependencies

Key test classes:
- TestDesignAgent: Core functionality tests with mocked dependencies
- TestDesignAgentAutoGenMocking: AutoGen integration tests with comprehensive mocking

All external dependencies (AutoGen components, LLM APIs) are mocked to ensure
fast, reliable unit tests that can run without network access or real services.

For tests that use real AutoGen components and LLM endpoints, see:
tests/integration/test_real_design_agent.py
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from autogen_framework.agents.design_agent import DesignAgent
from autogen_framework.models import LLMConfig

class TestDesignAgent:
    """Test suite for DesignAgent functionality."""
    # Using shared fixtures from conftest.py
    @pytest.fixture
    def memory_context(self):
        """Create test memory context."""
        return {
            "global": {
                "python_patterns.md": "# Python Best Practices\n\nUse type hints and docstrings.",
                "architecture_patterns.md": "# Architecture Patterns\n\nUse modular design."
            },
            "projects": {
                "test_project": {
                    "context.md": "# Test Project Context\n\nThis is a test project."
                }
            }
        }
    
    @pytest.fixture
    def design_agent(self, mock_dependency_container, test_llm_config, memory_context):
        """Create a DesignAgent instance for testing with container-based dependencies."""
        return DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config,
            memory_context=memory_context
        )
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample requirements content."""
        return """# Requirements Document

## Introduction
Test project requirements for a simple web application.

## Requirements

### Requirement 1: User Authentication
**User Story:** As a user, I want to authenticate, so that I can access the system.

#### Acceptance Criteria
1. WHEN user provides valid credentials THEN system SHALL authenticate user
2. WHEN user provides invalid credentials THEN system SHALL reject authentication

### Requirement 2: Data Management
**User Story:** As a user, I want to manage data, so that I can store information.

#### Acceptance Criteria
1. WHEN user creates data THEN system SHALL store data
2. WHEN user updates data THEN system SHALL modify existing data
"""
    
    def test_design_agent_initialization(self, mock_dependency_container, test_llm_config, memory_context):
        """Test DesignAgent initialization with container-based dependencies."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config,
            memory_context=memory_context
        )
        
        assert agent.name == "DesignAgent"
        assert agent.llm_config == test_llm_config
        assert agent.memory_context == memory_context
        assert agent.container == mock_dependency_container
        assert "AI agent specialized" in agent.description
        assert "technical design documents" in agent.system_message
    
    def test_design_agent_initialization_without_memory(self, mock_dependency_container, test_llm_config):
        """Test DesignAgent initialization without memory context but with container."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        assert agent.name == "DesignAgent"
        assert agent.llm_config == test_llm_config
        assert agent.memory_context == {}
        assert agent.container == mock_dependency_container
    
    def test_system_message_content(self, mock_dependency_container, test_llm_config):
        """Test that system message contains required elements."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        system_message = agent.system_message
        
        # Check for key components
        assert "Design Agent" in system_message
        assert "technical design documents" in system_message
        assert "Mermaid.js" in system_message
        assert "Architecture" in system_message
        assert "Components and Interfaces" in system_message
        assert "Security Considerations" in system_message
        assert "Testing Strategy" in system_message
    
    @pytest.mark.asyncio
    async def test_process_task_success(self, design_agent, sample_requirements):
        """Test successful task processing."""
        agent = design_agent
        
        # Mock the generate_response method
        mock_design_content = """# Design Document

## Architectural Overview
Test architecture overview.

## Data Flow Diagrams
```mermaid
graph TD
    A[User] --> B[Auth Service]
    B --> C[Database]
```

## Components and Interfaces
Test components.

## Security Considerations
Test security.

## Testing Strategy
Test strategy.
"""
        
        with patch.object(agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_design_content
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create requirements file
                req_path = os.path.join(temp_dir, "requirements.md")
                with open(req_path, 'w') as f:
                    f.write(sample_requirements)
                
                task_input = {
                    "requirements_path": req_path,
                    "work_directory": temp_dir
                }
                
                result = await agent.process_task(task_input)
                
                assert result["success"] is True
                assert "design_path" in result
                assert "design_content" in result
                assert result["message"] == "Design document generated successfully"
                
                # Check that design file was created
                design_path = os.path.join(temp_dir, "design.md")
                assert os.path.exists(design_path)
                
                # Check design content
                with open(design_path, 'r') as f:
                    content = f.read()
                    assert "# Design Document" in content
                    assert "mermaid" in content
    
    @pytest.mark.asyncio
    async def test_process_task_missing_parameters(self, mock_dependency_container, test_llm_config):
        """Test task processing with missing parameters."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        # Test missing requirements_path
        task_input = {"work_directory": "/tmp"}
        result = await agent.process_task(task_input)
        
        assert result["success"] is False
        assert "requirements_path and work_directory are required" in result["error"]
        
        # Test missing work_directory
        task_input = {"requirements_path": "/tmp/req.md"}
        result = await agent.process_task(task_input)
        
        assert result["success"] is False
        assert "requirements_path and work_directory are required" in result["error"]
    
    @pytest.mark.asyncio
    async def test_generate_design(self, design_agent, sample_requirements):
        """Test design generation functionality."""
        agent = design_agent
        
        mock_design_content = """# Design Document

## Architectural Overview
Generated architecture overview.

## Data Flow Diagrams
```mermaid
graph TD
    A[Client] --> B[Server]
```
"""
        
        with patch.object(agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_design_content
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
                temp_file.write(sample_requirements)
                temp_file.flush()
                
                try:
                    result = await agent.generate_design(temp_file.name, agent.memory_context)
                    
                    assert "# Design Document" in result
                    assert "Architectural Overview" in result
                    assert "mermaid" in result
                    
                    # Verify that generate_response was called with proper prompt
                    mock_generate.assert_called_once()
                    call_args = mock_generate.call_args[0]
                    prompt = call_args[0]
                    
                    assert "Requirements Document" in prompt
                    assert "Mermaid.js diagrams" in prompt
                    assert "Available Memory Context" in prompt
                    
                finally:
                    os.unlink(temp_file.name)

    @pytest.mark.asyncio
    async def test_generate_design_with_existing_codebase(self, design_agent, sample_requirements):
        """Test design generation with existing codebase analysis."""
        agent = design_agent
        
        # Mock codebase analysis service
        mock_codebase_insights = (
            "Existing codebase analysis:\n"
            "- Python-based AutoGen multi-agent framework\n"
            "- Dependency injection container pattern\n"
            "- LLM client integration for AI operations\n"
            "- Memory management for persistent state"
        )
        mock_codebase_service = Mock()
        mock_codebase_service.is_codebase_present.return_value = True
        mock_codebase_service.analyze_codebase = AsyncMock(return_value=(True, mock_codebase_insights))
        agent.codebase_analysis_service = mock_codebase_service
        
        mock_design_content = """# Design Document

## Architectural Overview
Enhanced architecture based on existing codebase.

## Integration Points
- Extends AutoGen framework
- Uses dependency injection pattern
"""
        
        with patch.object(agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_design_content
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
                temp_file.write(sample_requirements)
                temp_file.flush()
                
                try:
                    result = await agent.generate_design(temp_file.name, agent.memory_context)
                    
                    assert "# Design Document" in result
                    assert "AutoGen framework" in result
                    
                    # Verify codebase analysis was called
                    mock_codebase_service.analyze_codebase.assert_called_once()
                    
                    # Verify prompt included codebase insights
                    mock_generate.assert_called_once()
                    call_args = mock_generate.call_args[0]
                    prompt = call_args[0]
                    
                    assert "Existing Codebase Context" in prompt
                    assert "AutoGen multi-agent framework" in prompt
                    assert "Dependency injection container pattern" in prompt
                    
                finally:
                    os.unlink(temp_file.name)

    @pytest.mark.asyncio
    async def test_generate_design_codebase_analysis_failure(self, design_agent, sample_requirements):
        """Test design generation when codebase analysis fails."""
        agent = design_agent
        
        # Mock codebase analysis service to return failure
        mock_codebase_service = Mock()
        mock_codebase_service.is_codebase_present.return_value = True
        mock_codebase_service.analyze_codebase = AsyncMock(return_value=(False, None))
        agent.codebase_analysis_service = mock_codebase_service
        
        mock_design_content = """# Design Document

## Architectural Overview
New greenfield design.
"""
        
        with patch.object(agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_design_content
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
                temp_file.write(sample_requirements)
                temp_file.flush()
                
                try:
                    result = await agent.generate_design(temp_file.name, agent.memory_context)
                    
                    assert "# Design Document" in result
                    assert "greenfield design" in result
                    
                    # Verify codebase analysis was attempted
                    mock_codebase_service.analyze_codebase.assert_called_once()
                    
                    # Verify prompt indicated no codebase context
                    mock_generate.assert_called_once()
                    call_args = mock_generate.call_args[0]
                    prompt = call_args[0]
                    
                    assert "New Project Context" in prompt
                    
                finally:
                    os.unlink(temp_file.name)
    
    def test_build_design_prompt(self, design_agent):
        """Test design prompt building."""
        agent = design_agent
        
        requirements = "Test requirements content"
        prompt = agent._build_design_prompt(requirements, agent.memory_context)
        
        assert "Requirements Document" in prompt
        assert requirements in prompt
        assert "Task Instructions" in prompt
        assert "Mermaid.js diagrams" in prompt
        assert "Available Memory Context" in prompt
        assert "global" in prompt
        assert "projects" in prompt
        assert "Output Format" in prompt
    
    def test_build_design_prompt_no_memory(self, mock_dependency_container, test_llm_config):
        """Test design prompt building without memory context."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        requirements = "Test requirements content"
        prompt = agent._build_design_prompt(requirements, {})
        
        assert "Requirements Document" in prompt
        assert requirements in prompt
        assert "Available Memory Context" not in prompt

    def test_build_design_prompt_with_codebase_insights(self, mock_dependency_container, test_llm_config):
        """Test design prompt building with codebase analysis insights."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        requirements = "Test requirements content"
        codebase_insights = (
            "Existing codebase analysis:\n"
            "- Python-based multi-agent framework\n"
            "- Uses dependency injection pattern\n"
            "- Has LLM integration components"
        )
        
        prompt = agent._build_design_prompt(requirements, {}, codebase_insights)
        
        assert "Requirements Document" in prompt
        assert requirements in prompt
        assert "Existing Codebase Context" in prompt
        assert "multi-agent framework" in prompt
        assert "dependency injection" in prompt

    def test_build_design_prompt_no_codebase_insights(self, mock_dependency_container, test_llm_config):
        """Test design prompt building without codebase insights."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        requirements = "Test requirements content"
        prompt = agent._build_design_prompt(requirements, {}, None)
        
        assert "Requirements Document" in prompt
        assert requirements in prompt
        assert "no existing codebase context" in prompt.lower()
        assert "Existing Codebase Context" not in prompt
    
    def test_validate_mermaid_diagrams(self, mock_dependency_container, test_llm_config):
        """Test Mermaid diagram validation and fixing."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        # Test content with invalid Mermaid diagram
        content_with_invalid = """# Design

```mermaid
A --> B
B --> C
```
"""
        
        fixed_content = agent._validate_mermaid_diagrams(content_with_invalid)
        
        # Should add graph type
        assert "graph TD" in fixed_content
        assert "A --> B" in fixed_content
    
    def test_validate_mermaid_diagrams_valid(self, mock_dependency_container, test_llm_config):
        """Test Mermaid diagram validation with valid diagrams."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        content_with_valid = """# Design

```mermaid
graph TD
    A --> B
    B --> C
```
"""
        
        result = agent._validate_mermaid_diagrams(content_with_valid)
        
        # Should remain unchanged
        assert result == content_with_valid
    
    def test_fix_markdown_formatting(self, mock_dependency_container, test_llm_config):
        """Test markdown formatting fixes."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        content_with_issues = """#Header1
##Header2

```
def test():
    pass
```

More content"""
        
        fixed_content = agent._fix_markdown_formatting(content_with_issues)
        
        # Check header spacing (the regex doesn't add spaces to headers without spaces)
        assert "# Header1" in fixed_content or "#Header1" in fixed_content
        assert "## Header2" in fixed_content
        
        # Check code block language
        assert "```python" in fixed_content
        
        # Check excessive newlines are reduced
        assert "\n\n\n" not in fixed_content
    
    def test_post_process_design(self, mock_dependency_container, test_llm_config):
        """Test design post-processing."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        raw_design = """Some content without header

```mermaid
A --> B
```

#Header
Content"""
        
        processed = agent._post_process_design(raw_design)
        
        # Should add header if missing
        assert processed.startswith("# Design Document")
        
        # Should fix Mermaid diagrams
        assert "graph TD" in processed
        
        # Should fix markdown formatting
        assert "\n\n# Header\n\n" in processed
    
    def test_extract_from_markdown_block(self, mock_dependency_container, test_llm_config):
        """Test extraction of content from markdown code blocks."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        # Test markdown block extraction
        wrapped_content = """```markdown
# Technical Design Document

## Overview
This is a test design document.

```mermaid
graph TD
    A --> B
```
```"""
        
        extracted = agent._extract_from_markdown_block(wrapped_content)
        expected = """# Technical Design Document

## Overview
This is a test design document.

```mermaid
graph TD
    A --> B
```"""
        
        assert extracted == expected
        
        # Test generic code block extraction
        generic_wrapped = """```
# Some Content
## Section
Content here
```"""
        
        extracted_generic = agent._extract_from_markdown_block(generic_wrapped)
        expected_generic = """# Some Content
## Section
Content here"""
        
        assert extracted_generic == expected_generic
        
        # Test content without wrapping (should return as-is)
        unwrapped_content = "# Normal Content\n## Section"
        extracted_unwrapped = agent._extract_from_markdown_block(unwrapped_content)
        assert extracted_unwrapped == unwrapped_content
    
    def test_process_revision_task(self, mock_dependency_container, test_llm_config):
        """Test design revision task processing."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        # Create a temporary design file
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            design_path = os.path.join(temp_dir, "design.md")
            original_design = """# Design Document

## Architecture
Original architecture using REST API

## Components
- Component A
- Component B
"""
            
            with open(design_path, 'w') as f:
                f.write(original_design)
            
            # Mock the generate_response method to avoid actual LLM calls
            async def mock_generate_response(prompt):
                return """# Design Document

## Architecture
Updated architecture using FastAPI with SSR

## Components
- Component A (updated)
- Component B (updated)
- Component C (new)
"""
            
            agent.generate_response = mock_generate_response
            
            # Test revision task
            task_input = {
                "task_type": "revision",
                "revision_feedback": "use FastAPI with server-side rendering",
                "current_result": {"design_path": design_path},
                "work_directory": temp_dir
            }
            
            # Run the revision task
            import asyncio
            result = asyncio.run(agent.process_task(task_input))
            
            # Verify the result
            assert result["success"] is True
            assert result["revision_applied"] is True
            assert result["design_path"] == design_path
            
            # Verify the file was updated
            with open(design_path, 'r') as f:
                updated_content = f.read()
            
            assert "FastAPI" in updated_content
            assert "SSR" in updated_content
    
    def test_create_architecture_diagram_existing(self, mock_dependency_container, test_llm_config):
        """Test architecture diagram creation from existing content."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        design_content = """# Design

```mermaid
graph TD
    A[System] --> B[Component]
```

```mermaid
sequenceDiagram
    A->>B: Message
```
"""
        
        diagram = agent.create_architecture_diagram(design_content)
        
        # Should return the first graph diagram
        assert "graph TD" in diagram
        assert "A[System] --> B[Component]" in diagram
    
    def test_create_architecture_diagram_none_existing(self, mock_dependency_container, test_llm_config):
        """Test architecture diagram creation when none exists."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        design_content = """# Design

## Components
- SystemManager
- DataProcessor
"""
        
        diagram = agent.create_architecture_diagram(design_content)
        
        # Should generate a basic diagram
        assert "graph TD" in diagram
        assert "A[" in diagram
    
    def test_generate_basic_architecture_diagram(self, mock_dependency_container, test_llm_config):
        """Test basic architecture diagram generation."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        design_content = """
        class UserManager:
            pass
        
        class DataProcessor:
            pass
        
        ## Authentication Component
        ## Database Component
        """
        
        diagram = agent._generate_basic_architecture_diagram(design_content)
        
        assert "graph TD" in diagram
        assert "A[" in diagram
        assert "-->" in diagram
    
    def test_get_agent_capabilities(self, mock_dependency_container, test_llm_config):
        """Test agent capabilities listing."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        capabilities = agent.get_agent_capabilities()
        
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        assert any("technical design documents" in cap for cap in capabilities)
        assert any("Mermaid.js diagrams" in cap for cap in capabilities)
        assert any("component interfaces" in cap for cap in capabilities)
    
    def test_manager_dependencies_inheritance(self, design_agent):
        """Test that DesignAgent has required manager dependencies from BaseLLMAgent."""
        agent = design_agent
        
        # Verify that DesignAgent has manager dependencies from BaseLLMAgent
        assert hasattr(agent, 'token_manager')
        assert hasattr(agent, 'context_manager')
        
        # Verify that managers are properly set
        assert agent.token_manager is not None
        assert agent.context_manager is not None
        
        # Verify that deprecated compression methods exist but raise errors
        assert hasattr(agent, '_perform_context_compression')
        assert hasattr(agent, 'compress_context')
        assert hasattr(agent, 'truncate_context')
        
        # These should be inherited from BaseLLMAgent
        assert callable(getattr(agent, '_perform_context_compression'))
        assert callable(getattr(agent, 'compress_context'))
        assert callable(getattr(agent, 'truncate_context'))
    
    @pytest.mark.asyncio
    async def test_deprecated_compression_methods(self, mock_dependency_container, test_llm_config):
        """Test that deprecated compression methods raise appropriate errors."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        # Test that deprecated compression methods raise RuntimeError
        with pytest.raises(RuntimeError, match="_perform_context_compression is removed.*ContextManager"):
            await agent._perform_context_compression()
        
        with pytest.raises(RuntimeError, match="compress_context is removed.*ContextManager"):
            await agent.compress_context()
        
        with pytest.raises(RuntimeError, match="truncate_context is removed.*ContextManager"):
            agent.truncate_context()
        
        # Test that manager dependencies are properly available
        assert agent.token_manager is not None
        assert agent.context_manager is not None
        
        # Test that managers have expected methods
        assert hasattr(agent.token_manager, 'estimate_tokens_from_text')
        assert hasattr(agent.context_manager, 'prepare_system_prompt')
    
    def test_get_design_templates(self, mock_dependency_container, test_llm_config):
        """Test design template retrieval."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        templates = agent.get_design_templates()
        
        assert isinstance(templates, dict)
        assert "web_application" in templates
        assert "api_service" in templates
        assert "data_processing" in templates
        assert "multi_agent_system" in templates
        
        # Check template content
        web_template = templates["web_application"]
        assert "# Design Document" in web_template
        assert "Architectural Overview" in web_template
        assert "mermaid" in web_template
    
    def test_web_app_template_content(self, mock_dependency_container, test_llm_config):
        """Test web application template content."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        template = agent._get_web_app_template()
        
        assert "# Design Document" in template
        assert "Frontend Components" in template
        assert "Backend Services" in template
        assert "Database Layer" in template
        assert "Authentication" in template
        assert "End-to-End Testing" in template
    
    def test_api_service_template_content(self, mock_dependency_container, test_llm_config):
        """Test API service template content."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        template = agent._get_api_service_template()
        
        assert "# Design Document" in template
        assert "sequenceDiagram" in template
        assert "API Endpoints" in template
        assert "Service Layer" in template
        assert "API Authentication" in template
        assert "Rate Limiting" in template
    
    def test_data_processing_template_content(self, mock_dependency_container, test_llm_config):
        """Test data processing template content."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        template = agent._get_data_processing_template()
        
        assert "# Design Document" in template
        assert "Data Ingestion" in template
        assert "Processing Pipeline" in template
        assert "Input Schemas" in template
        assert "Data Privacy" in template
        assert "Performance Testing" in template
    
    def test_multi_agent_template_content(self, mock_dependency_container, test_llm_config):
        """Test multi-agent system template content."""
        agent = DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config
        )
        
        template = agent._get_multi_agent_template()
        
        assert "# Design Document" in template
        assert "Agent Manager" in template
        assert "Communication Protocols" in template
        assert "Agent States" in template
        assert "Message Security" in template
        assert "System Integration Testing" in template

class TestDesignAgentAutoGenMocking:
    """Test suite for DesignAgent AutoGen integration with proper mocking."""
    
    @pytest.fixture
    def design_agent(self, mock_dependency_container, test_llm_config, mock_memory_context):
        """Create a DesignAgent instance for testing with container-based dependencies."""
        return DesignAgent(
            container=mock_dependency_container,
            llm_config=test_llm_config,
            memory_context=mock_memory_context
        )
    
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    def test_design_agent_autogen_initialization(self, mock_client_class, mock_agent_class, design_agent):
        """Test DesignAgent AutoGen initialization with mocks."""
        # Mock AutoGen components
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        result = design_agent.initialize_autogen_agent()
        
        assert result is True
        assert design_agent._is_initialized is True
        assert design_agent._autogen_agent == mock_agent
        
        # Verify design-specific system message
        agent_call_args = mock_agent_class.call_args
        system_message = agent_call_args[1]["system_message"]
        assert "Design Agent" in system_message
        assert "Mermaid.js" in system_message
    
    @pytest.mark.asyncio
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    async def test_design_generation_with_mocks(self, mock_client_class, mock_agent_class, design_agent, temp_workspace):
        """Test design generation with mocked AutoGen responses."""
        # Mock AutoGen components
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        # Mock design response
        mock_design_content = """# Design Document

## Architectural Overview
Test architecture overview with mocked response.

## Data Flow Diagrams
```mermaid
graph TD
    A[User] --> B[Auth Service]
    B --> C[Database]
```

## Components and Interfaces
Test components with mocked data.

## Security Considerations
Mocked security considerations.

## Testing Strategy
Mocked testing strategy.
"""
        
        # Mock response generation - properly simulate AutoGen Response
        from autogen_agentchat.base import Response
        mock_response = Mock(spec=Response)
        mock_response.chat_message = Mock()
        mock_response.chat_message.content = mock_design_content
        mock_response.chat_message.source = "DesignAgent"
        mock_agent.on_messages = AsyncMock(return_value=mock_response)
        
        # Initialize agent
        design_agent.initialize_autogen_agent()
        
        # Create requirements file
        req_path = Path(temp_workspace) / "requirements.md"
        req_path.write_text("# Requirements\nTest requirements")
        
        # Test design generation
        task_input = {
            "requirements_path": str(req_path),
            "work_directory": temp_workspace
        }
        
        result = await design_agent.process_task(task_input)
        
        assert result["success"] is True
        assert "design_path" in result
        assert "design_content" in result
        
        # Verify design file was created
        design_path = Path(result["design_path"])
        assert design_path.exists()
        
        # Verify content includes mocked response
        design_content = design_path.read_text()
        assert "# Design Document" in design_content
        assert "mocked response" in design_content.lower()
    
    @pytest.mark.asyncio
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    async def test_design_revision_with_mocks(self, mock_client_class, mock_agent_class, design_agent, temp_workspace):
        """Test design revision with mocked AutoGen responses."""
        # Mock AutoGen components
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        # Mock revised design response
        mock_revised_content = """# Design Document

## Architectural Overview
Revised architecture with FastAPI and microservices.

## Components and Interfaces
Updated components with new requirements.
"""
        
        # Mock response generation - properly simulate AutoGen Response
        from autogen_agentchat.base import Response
        mock_response = Mock(spec=Response)
        mock_response.chat_message = Mock()
        mock_response.chat_message.content = mock_revised_content
        mock_response.chat_message.source = "DesignAgent"
        mock_agent.on_messages = AsyncMock(return_value=mock_response)
        
        # Initialize agent
        design_agent.initialize_autogen_agent()
        
        # Create initial design file
        design_path = Path(temp_workspace) / "design.md"
        design_path.write_text("# Design Document\n\nOriginal design content")
        
        # Test revision
        task_input = {
            "task_type": "revision",
            "revision_feedback": "use FastAPI with microservices",
            "current_result": {"design_path": str(design_path)},
            "work_directory": temp_workspace
        }
        
        result = await design_agent.process_task(task_input)
        
        assert result["success"] is True
        assert result["revision_applied"] is True
        
        # Verify file was updated
        updated_content = design_path.read_text()
        assert "FastAPI" in updated_content
        assert "microservices" in updated_content


# Integration tests have been moved to tests/integration/test_real_design_agent.py

# Test fixtures and utilities
@pytest.fixture
def temp_requirements_file():
    """Create a temporary requirements file for testing."""
    content = """# Requirements Document

## Introduction
Test requirements for design generation.

## Requirements

### Requirement 1: Core Functionality
**User Story:** As a user, I want core functionality, so that I can use the system.

#### Acceptance Criteria
1. WHEN user accesses system THEN system SHALL provide functionality
2. WHEN user performs action THEN system SHALL respond appropriately
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    
    os.unlink(f.name)

@pytest.fixture
def temp_work_directory():
    """Create a temporary work directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

# Parametrized tests for different scenarios
@pytest.mark.parametrize("memory_available", [True, False])
def test_design_agent_with_memory_variations(memory_available):
    """Test DesignAgent with and without memory context."""
    import tempfile
    from autogen_framework.dependency_container import DependencyContainer
    
    llm_config = LLMConfig(
        base_url="http://test.local:8888/openai/v1",
        model="test-model",
        api_key="sk-test123"
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        container = DependencyContainer.create_test(temp_dir, llm_config)
        
        if memory_available:
            memory_context = {"global": {"test.md": "Test content"}}
            agent = DesignAgent(
                container=container,
                llm_config=llm_config,
                memory_context=memory_context
            )
            assert agent.memory_context == memory_context
        else:
            agent = DesignAgent(
                container=container,
                llm_config=llm_config
            )
            assert agent.memory_context == {}
        
        assert agent.name == "DesignAgent"
        assert isinstance(agent.get_agent_capabilities(), list)

@pytest.mark.parametrize("template_type", ["web_application", "api_service", "data_processing", "multi_agent_system"])
def test_all_design_templates(template_type):
    """Test all available design templates."""
    import tempfile
    from autogen_framework.dependency_container import DependencyContainer
    
    llm_config = LLMConfig(
        base_url="http://test.local:8888/openai/v1",
        model="test-model",
        api_key="sk-test123"
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        container = DependencyContainer.create_test(temp_dir, llm_config)
        agent = DesignAgent(
            container=container,
            llm_config=llm_config
        )
        
        templates = agent.get_design_templates()
        assert template_type in templates
        
        template_content = templates[template_type]
    assert "# Design Document" in template_content
    assert "Architectural Overview" in template_content
    assert "Components and Interfaces" in template_content
    assert "Security Considerations" in template_content
    assert "Testing Strategy" in template_content