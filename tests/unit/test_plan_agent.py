"""
Unit tests for the PlanAgent class.

This module contains unit tests that focus on testing the PlanAgent class
in isolation using mocked dependencies. These tests are designed to:

- Run quickly (under 1 second each)
- Use only mocked external dependencies (no real AutoGen or LLM calls)
- Test request parsing, directory creation logic, and requirements generation
- Validate agent behavior without external service dependencies

Key test classes:
- TestPlanAgent: Core functionality tests with mocked dependencies
- TestPlanAgentMocking: Comprehensive mocking tests for AutoGen integration

All external dependencies (AutoGen components, LLM APIs, file system operations) are mocked to ensure
fast, reliable unit tests that can run without network access or real services.

For tests that use real AutoGen components and file system operations, see:
tests/integration/test_real_plan_agent.py
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from autogen_framework.agents.plan_agent import PlanAgent
from autogen_framework.models import LLMConfig
from autogen_framework.memory_manager import MemoryManager

class TestPlanAgent:
    """Test suite for PlanAgent functionality."""
    # Using shared fixtures from conftest.py
    # Using shared fixtures from conftest.py
    @pytest.fixture
    def memory_manager(self, temp_workspace):
        """Create a test memory manager."""
        return MemoryManager(temp_workspace)
    
    @pytest.fixture
    def plan_agent(self, llm_config, memory_manager, mock_token_manager, mock_context_manager):
        """Create a PlanAgent instance for testing with required manager dependencies."""
        return PlanAgent(llm_config, memory_manager, mock_token_manager, mock_context_manager)
    
    def test_plan_agent_initialization(self, plan_agent, llm_config):
        """Test that PlanAgent initializes correctly."""
        assert plan_agent.name == "PlanAgent"
        assert plan_agent.llm_config == llm_config
        assert "Plan Agent" in plan_agent.description
        assert plan_agent.memory_manager is not None
        assert plan_agent.workspace_path.exists()
    
    def test_system_message_content(self, plan_agent):
        """Test that the system message contains required content."""
        system_message = plan_agent.system_message
        
        # Check for key responsibilities
        assert "Parse User Requests" in system_message
        assert "Create Work Directories" in system_message
        assert "Generate Requirements Documents" in system_message
        
        # Check for format requirements
        assert "EARS" in system_message
        assert "kebab-case" in system_message
        assert "user stories" in system_message
    
    def test_memory_context_loading(self, plan_agent):
        """Test that memory context is loaded on initialization."""
        # The agent should have attempted to load memory context
        assert hasattr(plan_agent, 'memory_context')
        # Memory context might be empty in test environment, but should be a dict
        assert isinstance(plan_agent.memory_context, dict)
    
    @pytest.mark.asyncio
    async def test_parse_user_request_basic(self, plan_agent):
        """Test basic user request parsing functionality."""
        # Mock the generate_response method to return a structured response
        mock_response = """## Summary
Fix authentication bug in user login system

## Request Type
Type: debugging

## Scope Estimate
Scope: medium

## Key Requirements
- Fix login authentication issue
- Ensure secure password handling
- Add proper error messages

## Technical Context
Authentication system using JWT tokens

## Constraints
Must maintain backward compatibility

## Suggested Directory Name
[fix-authentication-bug]"""
        
        with patch.object(plan_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response
            
            result = await plan_agent.parse_user_request("Fix the authentication bug in our login system")
            
            assert result['summary'] == "Fix authentication bug in user login system"
            assert result['type'] == "debugging"
            assert result['scope'] == "medium"
            assert "Fix login authentication issue" in result['key_requirements']
            assert result['technical_context'] == "Authentication system using JWT tokens"
            assert result['constraints'] == "Must maintain backward compatibility"
            assert result['suggested_directory'] == "fix-authentication-bug"
    
    @pytest.mark.asyncio
    async def test_parse_user_request_fallback(self, plan_agent):
        """Test fallback behavior when LLM response parsing fails."""
        # Mock generate_response to raise an exception
        with patch.object(plan_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("API Error")
            
            request = "Implement user dashboard"
            result = await plan_agent.parse_user_request(request)
            
            # Should return fallback analysis
            assert result['summary'] == request
            assert result['type'] == "development"
            assert result['scope'] == "medium"
            assert result['key_requirements'] == [request]
            assert result['suggested_directory'] == "implement-user-dashboard"
    
    def test_generate_fallback_directory_name(self, plan_agent):
        """Test fallback directory name generation."""
        test_cases = [
            ("Fix authentication bug", "fix-authentication-bug"),
            ("Implement user dashboard with charts", "implement-user-dashboard-charts"),
            ("The quick brown fox jumps", "quick-brown-fox-jumps"),
            ("", "task-"),  # Should include timestamp
            ("a", "task-"),  # Too short, should fallback
        ]
        
        for request, expected_prefix in test_cases:
            result = plan_agent._generate_fallback_directory_name(request)
            if expected_prefix == "task-":
                assert result.startswith("task-")
            else:
                assert result == expected_prefix
    
    def test_clean_directory_name(self, plan_agent):
        """Test directory name cleaning and sanitization."""
        test_cases = [
            ("fix-authentication-bug", "fix-authentication-bug"),
            ('"implement-user-dashboard"', "implement-user-dashboard"),
            ("[optimize-api-performance]", "optimize-api-performance"),
            ("Fix: API endpoint returns 500 error", "fix-api-endpoint-returns-500-error"),
            ("", "task-"),  # Should include timestamp for empty input
            ("a", "task-"),  # Too short, should fallback
        ]
        
        for input_name, expected in test_cases:
            result = plan_agent.clean_directory_name(input_name)
            if expected.startswith("task-"):
                assert result.startswith("task-")
            else:
                assert result == expected
    
    def test_ensure_unique_name(self, plan_agent, temp_workspace):
        """Test unique name generation with conflict resolution."""
        base_name = "test-project"
        
        # First call should return the base name
        result1 = plan_agent.ensure_unique_name(base_name)
        assert result1 == base_name
        
        # Create a directory with that name
        (plan_agent.workspace_path / base_name).mkdir()
        
        # Second call should return a unique name
        result2 = plan_agent.ensure_unique_name(base_name)
        assert result2 != base_name
        assert result2.startswith(base_name)
        assert len(result2) <= 50
    
    @pytest.mark.asyncio
    async def test_generate_directory_name_with_llm(self, plan_agent):
        """Test LLM-based directory name generation."""
        summary = "Fix authentication bug in user login system"
        
        # Mock the LLM response
        with patch.object(plan_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "fix-authentication-bug"
            
            result = await plan_agent._generate_directory_name(summary)
            
            # Should call LLM with appropriate prompt
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0][0]
            assert "Generate a concise, descriptive directory name" in call_args
            assert summary in call_args
            assert "kebab-case" in call_args
            
            # Should return cleaned name
            assert result == "fix-authentication-bug"
    
    @pytest.mark.asyncio
    async def test_generate_directory_name_fallback(self, plan_agent):
        """Test fallback behavior when LLM fails."""
        summary = "Fix authentication bug"
        
        # Mock LLM to raise an exception
        with patch.object(plan_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("API Error")
            
            result = await plan_agent._generate_directory_name(summary)
            
            # Should fallback to the old method
            assert isinstance(result, str)
            assert len(result) > 0
            assert "fix" in result or "authentication" in result or "bug" in result
    
    @pytest.mark.asyncio
    async def test_create_work_directory(self, plan_agent, temp_workspace):
        """Test work directory creation."""
        summary = "Fix authentication bug"
        
        # Mock the LLM response for directory name generation
        with patch.object(plan_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "fix-authentication-bug"
            
            result = await plan_agent.create_work_directory(summary)
            
            # Check that directory was created
            work_dir = Path(result)
            assert work_dir.exists()
            assert work_dir.is_dir()
            assert work_dir.name == "fix-authentication-bug"
            
            # Check that basic files were created
            assert (work_dir / "requirements.md").exists()
            assert (work_dir / "design.md").exists()
            assert (work_dir / "tasks.md").exists()
    
    @pytest.mark.asyncio
    async def test_create_work_directory_unique_names(self, plan_agent, temp_workspace):
        """Test that work directories get unique names when conflicts occur."""
        summary = "Fix authentication bug"
        
        # Mock the LLM response for directory name generation
        with patch.object(plan_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "fix-authentication-bug"
            
            # Create first directory
            first_dir = await plan_agent.create_work_directory(summary)
            
            # Create second directory with same summary
            second_dir = await plan_agent.create_work_directory(summary)
            
            # Should be different directories
            assert first_dir != second_dir
            assert Path(first_dir).exists()
            assert Path(second_dir).exists()
            
            # Second directory should have timestamp suffix
            assert "fix-authentication-bug" in Path(second_dir).name
    
    @pytest.mark.asyncio
    async def test_generate_requirements(self, plan_agent, temp_workspace):
        """Test requirements document generation."""
        # Create a work directory first
        work_dir = Path(temp_workspace) / "test-project"
        work_dir.mkdir()
        
        user_request = "Fix authentication bug in login system"
        parsed_request = {
            'summary': 'Fix authentication bug',
            'type': 'debugging',
            'scope': 'medium',
            'key_requirements': ['Fix login issue', 'Add error handling'],
            'technical_context': 'JWT authentication',
            'constraints': 'Backward compatibility'
        }
        
        # Mock the generate_response method
        mock_requirements = """# Requirements Document

## Introduction

This project aims to fix authentication bugs in the login system.

## Requirements

### Requirement 1: Fix Authentication Issue

**User Story:** As a user, I want to login successfully, so that I can access the application.

#### Acceptance Criteria

1. WHEN user enters valid credentials THEN system SHALL authenticate successfully
2. WHEN user enters invalid credentials THEN system SHALL show appropriate error message
"""
        
        with patch.object(plan_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_requirements
            
            result = await plan_agent.generate_requirements(
                user_request, str(work_dir), parsed_request
            )
            
            # Check that file was created
            requirements_path = Path(result)
            assert requirements_path.exists()
            assert requirements_path.name == "requirements.md"
            
            # Check content
            content = requirements_path.read_text()
            assert "Requirements Document" in content
            assert "User Story:" in content
            assert "Acceptance Criteria" in content
    
    @pytest.mark.asyncio
    async def test_process_task_complete_workflow(self, plan_agent):
        """Test the complete process_task workflow."""
        task_input = {
            "user_request": "Implement user dashboard with charts and analytics"
        }
        
        # Mock all the async methods
        mock_parsed = {
            'summary': 'Implement user dashboard',
            'type': 'development',
            'scope': 'large',
            'key_requirements': ['Dashboard UI', 'Charts', 'Analytics'],
            'technical_context': 'React frontend',
            'constraints': 'Mobile responsive'
        }
        
        with patch.object(plan_agent, 'parse_user_request', new_callable=AsyncMock) as mock_parse, \
             patch.object(plan_agent, 'create_work_directory', new_callable=AsyncMock) as mock_create, \
             patch.object(plan_agent, 'generate_requirements', new_callable=AsyncMock) as mock_generate:
            
            mock_parse.return_value = mock_parsed
            mock_create.return_value = "/test/work/directory"
            mock_generate.return_value = "/test/work/directory/requirements.md"
            
            result = await plan_agent.process_task(task_input)
            
            # Check that all methods were called
            mock_parse.assert_called_once_with("Implement user dashboard with charts and analytics")
            mock_create.assert_called_once_with("Implement user dashboard")
            mock_generate.assert_called_once()
            
            # Check result structure
            assert result['success'] is True
            assert result['work_directory'] == "/test/work/directory"
            assert result['requirements_path'] == "/test/work/directory/requirements.md"
            assert result['parsed_request'] == mock_parsed
    
    @pytest.mark.asyncio
    async def test_process_task_error_handling(self, plan_agent):
        """Test error handling in process_task."""
        task_input = {}  # Missing user_request
        
        result = await plan_agent.process_task(task_input)
        
        assert result['success'] is False
        assert 'error' in result
        assert "user_request is required" in result['error']
    
    def test_get_agent_capabilities(self, plan_agent):
        """Test that agent capabilities are properly defined."""
        capabilities = plan_agent.get_agent_capabilities()
        
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        
        # Check for key capabilities
        capability_text = ' '.join(capabilities)
        assert "Parse" in capability_text
        assert "requirements" in capability_text
        assert "directories" in capability_text
        assert "EARS" in capability_text
    
    def test_manager_dependencies_inheritance(self, plan_agent):
        """Test that PlanAgent has required manager dependencies from BaseLLMAgent."""
        # Verify that PlanAgent has manager dependencies from BaseLLMAgent
        assert hasattr(plan_agent, 'token_manager')
        assert hasattr(plan_agent, 'context_manager')
        
        # Verify that managers are properly set
        assert plan_agent.token_manager is not None
        assert plan_agent.context_manager is not None
        
        # Verify that deprecated compression methods exist but raise errors
        assert hasattr(plan_agent, '_perform_context_compression')
        assert hasattr(plan_agent, 'compress_context')
        assert hasattr(plan_agent, 'truncate_context')
        
        # These should be inherited from BaseLLMAgent
        assert callable(getattr(plan_agent, '_perform_context_compression'))
        assert callable(getattr(plan_agent, 'compress_context'))
        assert callable(getattr(plan_agent, 'truncate_context'))
    
    @pytest.mark.asyncio
    async def test_deprecated_compression_methods(self, plan_agent):
        """Test that deprecated compression methods raise appropriate errors."""
        # Test that deprecated compression methods raise RuntimeError
        with pytest.raises(RuntimeError, match="_perform_context_compression is removed.*ContextManager"):
            await plan_agent._perform_context_compression()
        
        with pytest.raises(RuntimeError, match="compress_context is removed.*ContextManager"):
            await plan_agent.compress_context()
        
        with pytest.raises(RuntimeError, match="truncate_context is removed.*ContextManager"):
            plan_agent.truncate_context()
        
        # Test that manager dependencies are properly available
        assert plan_agent.token_manager is not None
        assert plan_agent.context_manager is not None
        
        # Test that managers have expected methods
        assert hasattr(plan_agent.token_manager, 'estimate_tokens_from_text')
        assert hasattr(plan_agent.context_manager, 'prepare_system_prompt')
    
    @pytest.mark.asyncio
    async def test_get_work_directory_suggestions(self, plan_agent):
        """Test work directory name suggestions."""
        request = "Fix authentication bug in user login system"
        
        # Mock the LLM response for the primary suggestion
        with patch.object(plan_agent, 'generate_response', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "fix-authentication-bug"
            
            suggestions = await plan_agent.get_work_directory_suggestions(request)
            
            assert isinstance(suggestions, list)
            assert len(suggestions) <= 5
            assert len(suggestions) > 0
            
            # All suggestions should be valid directory names
            for suggestion in suggestions:
                assert isinstance(suggestion, str)
                assert len(suggestion) > 0
                assert ' ' not in suggestion  # Should be kebab-case
    
    def test_parse_analysis_response(self, plan_agent):
        """Test parsing of structured LLM analysis response."""
        response = """## Summary
Fix authentication bug in login system

## Request Type
Type: debugging

## Scope Estimate
Scope: medium

## Key Requirements
- Fix login authentication issue
- Ensure secure password handling
- Add proper error messages
- Test with different browsers

## Technical Context
Authentication system using JWT tokens and OAuth

## Constraints
Must maintain backward compatibility with existing API

## Suggested Directory Name
[fix-authentication-bug]"""
        
        result = plan_agent._parse_analysis_response(response)
        
        assert result['summary'] == "Fix authentication bug in login system"
        assert result['type'] == "debugging"
        assert result['scope'] == "medium"
        assert len(result['key_requirements']) == 4
        assert "Fix login authentication issue" in result['key_requirements']
        assert result['technical_context'] == "Authentication system using JWT tokens and OAuth"
        assert result['constraints'] == "Must maintain backward compatibility with existing API"
        assert result['suggested_directory'] == "fix-authentication-bug"
    
    def test_parse_analysis_response_malformed(self, plan_agent):
        """Test parsing of malformed or incomplete LLM response."""
        response = """Some random text without proper structure
        
## Summary
Partial response

## Request Type
Missing type info"""
        
        result = plan_agent._parse_analysis_response(response)
        
        # Should handle missing sections gracefully
        assert result['summary'] == "Partial response"
        assert result['type'] == "missing"  # Extracted from the actual content
        assert result['scope'] == "medium"  # Default value (no scope section)
        assert isinstance(result['key_requirements'], list)
        assert result['technical_context'] == ""
        assert result['constraints'] == ""
        assert result['suggested_directory'] == ""

class TestPlanAgentMocking:
    """Test suite for PlanAgent with comprehensive mocking."""
    
    @pytest.fixture
    def memory_manager(self, temp_workspace):
        """Create a test memory manager."""
        return MemoryManager(temp_workspace)
    
    @pytest.fixture
    def plan_agent(self, llm_config, memory_manager, mock_token_manager, mock_context_manager):
        """Create a PlanAgent instance for testing with required manager dependencies."""
        return PlanAgent(llm_config, memory_manager, mock_token_manager, mock_context_manager)
    
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    def test_plan_agent_autogen_initialization(self, mock_client_class, mock_agent_class, plan_agent):
        """Test PlanAgent AutoGen initialization with mocks."""
        # Mock AutoGen components
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        result = plan_agent.initialize_autogen_agent()
        
        assert result is True
        assert plan_agent._is_initialized is True
        assert plan_agent._autogen_agent == mock_agent
        
        # Verify plan-specific system message
        agent_call_args = mock_agent_class.call_args
        system_message = agent_call_args[1]["system_message"]
        assert "Plan Agent" in system_message
        assert "EARS" in system_message
        assert "kebab-case" in system_message
    
    @pytest.mark.asyncio
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    async def test_parse_user_request_with_mocks(self, mock_client_class, mock_agent_class, plan_agent):
        """Test user request parsing with mocked AutoGen responses."""
        # Mock AutoGen components
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        # Mock parsing response
        mock_response_content = """## Summary
Create REST API for user management

## Request Type
Type: development

## Scope Estimate
Scope: medium

## Key Requirements
- User authentication system
- CRUD operations for users
- Input validation and error handling

## Technical Context
REST API using modern web framework

## Constraints
Must be scalable and secure

## Suggested Directory Name
[user-management-api]"""
        
        # Mock response generation - properly simulate AutoGen Response
        from autogen_agentchat.base import Response
        mock_response = Mock(spec=Response)
        mock_response.chat_message = Mock()
        mock_response.chat_message.content = mock_response_content
        mock_response.chat_message.source = "PlanAgent"
        mock_agent.on_messages = AsyncMock(return_value=mock_response)
        
        # Initialize agent
        plan_agent.initialize_autogen_agent()
        
        # Test request parsing
        result = await plan_agent.parse_user_request("Create REST API for user management")
        
        assert result['summary'] == "Create REST API for user management"
        assert result['type'] == "development"
        assert result['scope'] == "medium"
        assert "User authentication system" in result['key_requirements']
        assert result['technical_context'] == "REST API using modern web framework"
        assert result['constraints'] == "Must be scalable and secure"
        assert result['suggested_directory'] == "user-management-api"
    
    @pytest.mark.asyncio
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    async def test_generate_requirements_with_mocks(self, mock_client_class, mock_agent_class, plan_agent, temp_workspace):
        """Test requirements generation with mocked AutoGen responses."""
        # Mock AutoGen components
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        # Mock requirements response
        mock_requirements_content = """# Requirements Document

## Introduction

This document outlines the requirements for a user management REST API.

## Requirements

### Requirement 1: User Authentication

**User Story:** As a system administrator, I want to authenticate users, so that I can control access.

#### Acceptance Criteria

1. WHEN user provides valid credentials THEN system SHALL authenticate successfully
2. WHEN user provides invalid credentials THEN system SHALL reject with appropriate error
3. WHEN authentication fails 3 times THEN system SHALL lock account temporarily

### Requirement 2: User Management

**User Story:** As an administrator, I want to manage user accounts, so that I can maintain the system.

#### Acceptance Criteria

1. WHEN creating new user THEN system SHALL validate all required fields
2. WHEN updating user THEN system SHALL preserve data integrity
3. WHEN deleting user THEN system SHALL archive data securely
"""
        
        # Mock response generation - properly simulate AutoGen Response
        from autogen_agentchat.base import Response
        mock_response = Mock(spec=Response)
        mock_response.chat_message = Mock()
        mock_response.chat_message.content = mock_requirements_content
        mock_response.chat_message.source = "PlanAgent"
        mock_agent.on_messages = AsyncMock(return_value=mock_response)
        
        # Initialize agent
        plan_agent.initialize_autogen_agent()
        
        # Create work directory
        work_dir = Path(temp_workspace) / "test-project"
        work_dir.mkdir()
        
        # Test requirements generation
        user_request = "Create REST API for user management"
        parsed_request = {
            'summary': 'User management API',
            'type': 'development',
            'scope': 'medium',
            'key_requirements': ['Authentication', 'User CRUD'],
            'technical_context': 'REST API',
            'constraints': 'Secure and scalable'
        }
        
        result = await plan_agent.generate_requirements(
            user_request, str(work_dir), parsed_request
        )
        
        # Verify file was created
        requirements_path = Path(result)
        assert requirements_path.exists()
        assert requirements_path.name == "requirements.md"
        
        # Verify content
        content = requirements_path.read_text()
        assert "# Requirements Document" in content
        assert "User Story:" in content
        assert "Acceptance Criteria" in content
        assert "WHEN" in content and "THEN" in content and "SHALL" in content
    
    @pytest.mark.asyncio
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    async def test_complete_process_task_with_mocks(self, mock_client_class, mock_agent_class, plan_agent, temp_workspace):
        """Test complete process_task workflow with mocked responses."""
        # Mock AutoGen components
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        # Mock responses for different calls
        responses = [
            # First call: parse_user_request
            """## Summary
Test application development

## Request Type
Type: development

## Scope Estimate
Scope: small

## Key Requirements
- Basic functionality
- User interface
- Testing

## Technical Context
Web application

## Constraints
Simple and maintainable

## Suggested Directory Name
[test-application]""",
            # Second call: generate_directory_name
            "test-application",
            # Third call: generate_requirements
            """# Requirements Document

## Introduction
Test application requirements.

## Requirements

### Requirement 1: Basic Functionality
**User Story:** As a user, I want basic functionality, so that I can use the application.

#### Acceptance Criteria
1. WHEN user accesses application THEN system SHALL provide interface
2. WHEN user performs action THEN system SHALL respond appropriately
"""
        ]
        
        # Mock response generation - properly simulate AutoGen Response
        from autogen_agentchat.base import Response
        mock_responses = []
        for resp in responses:
            mock_response = Mock(spec=Response)
            mock_response.chat_message = Mock()
            mock_response.chat_message.content = resp
            mock_response.chat_message.source = "PlanAgent"
            mock_responses.append(mock_response)
        mock_agent.on_messages = AsyncMock(side_effect=mock_responses)
        
        # Initialize agent
        plan_agent.initialize_autogen_agent()
        
        # Test complete workflow
        task_input = {
            "user_request": "Create a test application with basic functionality"
        }
        
        result = await plan_agent.process_task(task_input)
        
        assert result['success'] is True
        assert 'work_directory' in result
        assert 'requirements_path' in result
        assert 'parsed_request' in result
        
        # Verify work directory was created
        work_dir = Path(result['work_directory'])
        assert work_dir.exists()
        assert work_dir.is_dir()
        
        # Verify requirements file was created
        requirements_file = Path(result['requirements_path'])
        assert requirements_file.exists()
        
        # Verify structure files were created
        for filename in ["design.md", "tasks.md"]:
            file_path = work_dir / filename
            assert file_path.exists()


# Integration tests have been moved to tests/integration/test_real_plan_agent.py

if __name__ == "__main__":
    pytest.main([__file__, "-v"])