"""
Interactive LLM feature integration tests.

This module tests interactive LLM features including:
- Shell command enhancement with conditional logic addition
- Error handling integration in enhanced commands
- Feedback processing and meaningful response generation
- Quality assessment and improvement tracking
- Interactive feature responsiveness

These tests validate Requirements 6.2, 6.3, and 6.5:
- 6.2: ImplementAgent command enhancement with conditional logic and error handling
- 6.3: Context compression with coherent summaries and essential information preservation
- 6.5: Revision capabilities with meaningful improvements based on feedback

Key Test Categories:
1. Directory Naming Tests (PlanAgent kebab-case generation)
2. Command Enhancement Tests (ImplementAgent conditional logic)
3. Context Compression Tests (ContextCompressor coherent summaries)
4. System Message Building Tests (memory and context integration)
5. Revision Capability Tests (meaningful improvement validation)

Usage:
    pytest tests/integration/test_llm_interactive_features.py -v
    pytest tests/integration/test_llm_interactive_features.py::TestDirectoryNaming -v
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

from autogen_framework.agents.plan_agent import PlanAgent
from autogen_framework.agents.implement_agent import ImplementAgent
from autogen_framework.agents.task_decomposer import TaskDecomposer
from autogen_framework.agents.error_recovery import ErrorRecovery
from autogen_framework.context_compressor import ContextCompressor
from autogen_framework.shell_executor import ShellExecutor
from autogen_framework.models import TaskDefinition, LLMConfig

from .test_llm_base import (
    LLMIntegrationTestBase,
    sequential_test_execution,
    QUALITY_THRESHOLDS_STRICT
)


@dataclass
class DirectoryNamingTestCase:
    """Test case for directory naming validation."""
    user_request: str
    expected_pattern: str  # Regex pattern for validation
    description: str


@dataclass
class CommandEnhancementTestCase:
    """Test case for command enhancement validation."""
    original_command: str
    task_context: str
    expected_features: List[str]  # Features that should be added
    description: str


@dataclass
class RevisionTestCase:
    """Test case for revision capability validation."""
    original_content: str
    feedback: str
    content_type: str  # 'requirements', 'design', 'tasks'
    expected_improvements: List[str]
    description: str


class TestDirectoryNaming(LLMIntegrationTestBase):
    """
    Test PlanAgent directory naming capabilities.
    
    Validates Requirement 6.1: WHEN testing PlanAgent directory naming THEN the system 
    SHALL validate that LLM generates appropriate kebab-case directory names from user requests.
    """
    
    @pytest.fixture(autouse=True)
    def setup_plan_agent(self, real_llm_config, initialized_real_managers, temp_workspace, real_memory_manager):
        """Setup PlanAgent for directory naming tests."""
        self.llm_config = real_llm_config
        self.managers = initialized_real_managers
        self.workspace = temp_workspace
        self.memory_manager = real_memory_manager
        
        # Create PlanAgent with real dependencies using container
        from autogen_framework.dependency_container import DependencyContainer
        self.container = DependencyContainer.create_production(temp_workspace, self.llm_config)
        
        self.plan_agent = PlanAgent(
            container=self.container,
            name="PlanAgent",
            llm_config=self.llm_config,
            system_message="Generate project requirements"
        )
    
    def get_directory_naming_test_cases(self) -> List[DirectoryNamingTestCase]:
        """Get test cases for directory naming validation."""
        return [
            DirectoryNamingTestCase(
                user_request="I need to fix the authentication bug in the login system",
                expected_pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",  # kebab-case pattern
                description="Authentication bug fix should generate kebab-case directory"
            ),
            DirectoryNamingTestCase(
                user_request="Implement a user dashboard with charts and analytics",
                expected_pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
                description="Dashboard implementation should generate kebab-case directory"
            ),
            DirectoryNamingTestCase(
                user_request="Optimize database queries for better performance",
                expected_pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
                description="Database optimization should generate kebab-case directory"
            ),
            DirectoryNamingTestCase(
                user_request="Create API endpoints for user management",
                expected_pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
                description="API creation should generate kebab-case directory"
            ),
            DirectoryNamingTestCase(
                user_request="Add unit tests for the payment processing module",
                expected_pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$",
                description="Test addition should generate kebab-case directory"
            )
        ]
    
    @sequential_test_execution()
    @pytest.mark.integration
    async def test_directory_name_generation_kebab_case(self):
        """
        Test that PlanAgent generates proper kebab-case directory names.
        
        Validates that LLM generates directory names that:
        - Use kebab-case format (lowercase with hyphens)
        - Are descriptive but concise (max 50 characters)
        - Include the main action or problem being addressed
        """
        test_cases = self.get_directory_naming_test_cases()
        
        for test_case in test_cases:
            # Generate directory name using create_work_directory method
                directory_path = await self.execute_with_rate_limit_handling(
                    lambda: self.plan_agent.create_work_directory(test_case.user_request)
                )
                
                # Extract directory name from path
                directory_name = Path(directory_path).name
                
                # Validate kebab-case format
                import re
                assert re.match(test_case.expected_pattern, directory_name), (
                    f"Directory name '{directory_name}' does not match kebab-case pattern. "
                    f"Expected pattern: {test_case.expected_pattern}"
                )
                
                # Validate length constraint
                assert len(directory_name) <= 50, (
                    f"Directory name '{directory_name}' exceeds 50 character limit. "
                    f"Length: {len(directory_name)}"
                )
                
                # Validate descriptiveness (should contain relevant keywords)
                request_keywords = self._extract_keywords(test_case.user_request)
                name_keywords = self._extract_keywords(directory_name.replace('-', ' '))
                
                # At least one keyword should match
                keyword_overlap = set(request_keywords) & set(name_keywords)
                assert len(keyword_overlap) > 0, (
                    f"Directory name '{directory_name}' does not contain relevant keywords. "
                    f"Request keywords: {request_keywords}, Name keywords: {name_keywords}"
                )
                
                self.logger.info(
                    f"✓ Generated valid directory name: '{directory_name}' "
                    f"for request: '{test_case.user_request[:50]}...'"
                )
    
    @sequential_test_execution()
    @pytest.mark.integration
    async def test_directory_name_uniqueness_and_relevance(self):
        """
        Test that directory names are unique and relevant to the request.
        
        Validates that different requests generate different directory names
        and that names are contextually appropriate.
        """
        requests = [
            "Fix authentication bug",
            "Implement user dashboard", 
            "Optimize database queries",
            "Create REST API",
            "Add unit tests"
        ]
        
        generated_names = []
        
        for request in requests:
            directory_path = await self.execute_with_rate_limit_handling(
                lambda r=request: self.plan_agent.create_work_directory(r)
            )
            directory_name = Path(directory_path).name
            
            # Check uniqueness
            assert directory_name not in generated_names, (
                f"Directory name '{directory_name}' was generated multiple times. "
                f"Names should be unique for different requests."
            )
            
            generated_names.append(directory_name)
            
            # Check relevance by ensuring name relates to request
            request_lower = request.lower()
            name_lower = directory_name.lower()
            
            # Simple relevance check - at least one word should be related
            request_words = set(request_lower.split())
            name_words = set(name_lower.replace('-', ' ').split())
            
            # Allow for synonyms and related terms
            relevance_score = len(request_words & name_words) / len(request_words)
            assert relevance_score > 0.2, (
                f"Directory name '{directory_name}' seems unrelated to request '{request}'. "
                f"Relevance score: {relevance_score:.2f}"
            )
        
        self.logger.info(f"✓ Generated {len(generated_names)} unique, relevant directory names")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        import re
        
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]


class TestCommandEnhancement(LLMIntegrationTestBase):
    """
    Test ImplementAgent command enhancement capabilities.
    
    Validates Requirement 6.2: WHEN testing ImplementAgent command enhancement THEN the system 
    SHALL validate that LLM adds proper conditional logic and error handling to shell commands.
    """
    
    @pytest.fixture(autouse=True)
    def setup_implement_agent(self, real_llm_config, initialized_real_managers, temp_workspace, real_memory_manager):
        """Setup ImplementAgent for command enhancement tests."""
        self.llm_config = real_llm_config
        self.managers = initialized_real_managers
        self.workspace = temp_workspace
        self.memory_manager = real_memory_manager
        
        # Create shell executor
        self.shell_executor = ShellExecutor()
        
        # Create dependency container for all agents
        from autogen_framework.dependency_container import DependencyContainer
        self.container = DependencyContainer.create_production(temp_workspace, self.llm_config)
        
        # Create task decomposer and error recovery using container
        self.task_decomposer = TaskDecomposer(
            name="TaskDecomposer",
            llm_config=self.llm_config,
            system_message="Decompose tasks into executable commands",
            container=self.container
        )
        
        self.error_recovery = ErrorRecovery(
            name="ErrorRecovery",
            llm_config=self.llm_config,
            system_message="Analyze errors and generate recovery strategies",
            container=self.container
        )
        
        # Create ImplementAgent with real dependencies using container
        self.implement_agent = ImplementAgent(
            container=self.container,
            name="TestImplementAgent",
            llm_config=self.llm_config,
            system_message="Test implementation agent for command enhancement"
        )
    
    def get_command_enhancement_test_cases(self) -> List[CommandEnhancementTestCase]:
        """Get test cases for command enhancement validation."""
        return [
            CommandEnhancementTestCase(
                original_command="pip install requests",
                task_context="Install Python package for API calls",
                expected_features=["error_handling", "conditional_logic", "validation"],
                description="Package installation should add error handling and validation"
            ),
            CommandEnhancementTestCase(
                original_command="mkdir project_dir",
                task_context="Create project directory structure",
                expected_features=["existence_check", "error_handling", "permissions"],
                description="Directory creation should check existence and handle permissions"
            ),
            CommandEnhancementTestCase(
                original_command="git clone https://github.com/user/repo.git",
                task_context="Clone repository for development",
                expected_features=["error_handling", "validation", "cleanup"],
                description="Git clone should handle network errors and validate success"
            ),
            CommandEnhancementTestCase(
                original_command="python script.py",
                task_context="Run Python script for data processing",
                expected_features=["error_handling", "output_validation", "logging"],
                description="Script execution should handle errors and validate output"
            ),
            CommandEnhancementTestCase(
                original_command="cp file.txt backup/",
                task_context="Backup important configuration file",
                expected_features=["existence_check", "error_handling", "verification"],
                description="File copy should verify source exists and copy succeeded"
            )
        ]
    
    @pytest.mark.skip(reason="TaskDecomposer command enhancement functionality needs to be fully implemented")
    @sequential_test_execution()
    @pytest.mark.integration
    async def test_command_enhancement_with_conditional_logic(self):
        """
        Test that ImplementAgent enhances commands with proper conditional logic.
        
        Validates that enhanced commands include:
        - Conditional checks before execution
        - Error handling mechanisms
        - Success validation
        - Appropriate fallback strategies
        """
        test_cases = self.get_command_enhancement_test_cases()
        
        for test_case in test_cases:
            # Create a simple task for command enhancement
                task = TaskDefinition(
                    id="test_task",
                    title=f"Test command enhancement: {test_case.original_command}",
                    description=test_case.task_context,
                    steps=[test_case.original_command],
                    requirements_ref=["6.2"]
                )
                
                # Use TaskDecomposer to generate enhanced command sequence
                execution_plan = await self.execute_with_rate_limit_handling(
                    lambda: self.task_decomposer.decompose_task(task)
                )
                
                # Extract enhanced commands from execution plan
                enhanced_commands = [cmd.command for cmd in execution_plan.commands]
                
                # Validate that enhancement was applied
                assert isinstance(enhanced_commands, (list, str)), (
                    f"Enhanced commands should be a list or string, got: {type(enhanced_commands)}"
                )
                
                if isinstance(enhanced_commands, str):
                    enhanced_commands = [enhanced_commands]
                
                # Check that original command is included or referenced
                command_text = " ".join(enhanced_commands).lower()
                original_parts = test_case.original_command.lower().split()
                
                # At least some parts of the original command should be present
                found_parts = sum(1 for part in original_parts if part in command_text)
                assert found_parts >= len(original_parts) * 0.5, (
                    f"Enhanced commands don't seem to include the original command. "
                    f"Original: {test_case.original_command}, Enhanced: {enhanced_commands}"
                )
                
                # Validate expected features are present
                for feature in test_case.expected_features:
                    feature_present = self._check_feature_in_commands(enhanced_commands, feature)
                    assert feature_present, (
                        f"Expected feature '{feature}' not found in enhanced commands. "
                        f"Commands: {enhanced_commands}"
                    )
                
                # Validate that commands are executable (basic syntax check)
                for cmd in enhanced_commands:
                    assert self._is_valid_shell_command(cmd), (
                        f"Enhanced command appears to have invalid syntax: {cmd}"
                    )
                
                self.logger.info(
                    f"✓ Successfully enhanced command '{test_case.original_command}' "
                    f"with features: {test_case.expected_features}"
                )
    
    @pytest.mark.skip(reason="TaskDecomposer error handling integration functionality needs to be fully implemented")
    @sequential_test_execution()
    @pytest.mark.integration
    async def test_error_handling_integration(self):
        """
        Test that enhanced commands include proper error handling integration.
        
        Validates that error handling includes:
        - Exit code checking
        - Error message capture
        - Appropriate error responses
        - Recovery suggestions
        """
        # Test with a command that might fail
        failing_command = "ls /nonexistent/directory"
        task_context = "List contents of directory that might not exist"
        
        # Create task for decomposition
        failing_task = TaskDefinition(
            id="failing_test",
            title="Test error handling",
            description=f"{task_context}: {failing_command}",
            steps=[failing_command],
            requirements_ref=["6.2"]
        )
        
        execution_plan = await self.execute_with_rate_limit_handling(
            lambda: self.task_decomposer.decompose_task(failing_task)
        )
        
        enhanced_commands = [cmd.command for cmd in execution_plan.commands]
        
        if isinstance(enhanced_commands, str):
            enhanced_commands = [enhanced_commands]
        
        command_text = " ".join(enhanced_commands).lower()
        
        # Check for error handling patterns
        error_patterns = [
            r'\$\?',  # Exit code checking
            r'if.*then',  # Conditional logic
            r'else',  # Alternative handling
            r'error',  # Error handling
            r'fail',  # Failure handling
            r'exist',  # Existence checking
        ]
        
        import re
        found_patterns = []
        for pattern in error_patterns:
            if re.search(pattern, command_text):
                found_patterns.append(pattern)
        
        assert len(found_patterns) >= 2, (
            f"Enhanced commands should include error handling patterns. "
            f"Found: {found_patterns}, Commands: {enhanced_commands}"
        )
        
        self.logger.info(f"✓ Enhanced commands include error handling patterns: {found_patterns}")
    
    def _check_feature_in_commands(self, commands: List[str], feature: str) -> bool:
        """Check if a specific feature is present in the enhanced commands."""
        command_text = " ".join(commands).lower()
        
        feature_indicators = {
            "error_handling": ["error", "fail", "$?", "exit", "return"],
            "conditional_logic": ["if", "then", "else", "case", "&&", "||"],
            "validation": ["test", "check", "verify", "validate", "exist"],
            "existence_check": ["exist", "test -", "[ -", "-f", "-d"],
            "permissions": ["chmod", "permission", "access", "-w", "-r"],
            "cleanup": ["cleanup", "remove", "rm", "clean"],
            "output_validation": ["output", "result", "success", "complete"],
            "logging": ["log", "echo", "print", "verbose"]
        }
        
        indicators = feature_indicators.get(feature, [feature])
        return any(indicator in command_text for indicator in indicators)
    
    def _is_valid_shell_command(self, command: str) -> bool:
        """Basic validation that command has valid shell syntax."""
        # Very basic syntax checking
        if not command.strip():
            return False
        
        # Check for balanced quotes (simple check)
        single_quotes = command.count("'")
        double_quotes = command.count('"')
        
        if single_quotes % 2 != 0 or double_quotes % 2 != 0:
            return False
        
        # Check for basic shell command structure
        # Should not start with invalid characters
        invalid_starts = ['|', '&', '>', '<', ';']
        if any(command.strip().startswith(char) for char in invalid_starts):
            return False
        
        return True


class TestContextCompression(LLMIntegrationTestBase):
    """
    Test ContextCompressor capabilities for coherent summarization.
    
    Validates Requirement 6.3: WHEN testing context compression THEN the system 
    SHALL validate that ContextCompressor produces coherent summaries while preserving essential information.
    """
    
    @pytest.fixture(autouse=True)
    def setup_context_compressor(self, real_llm_config, initialized_real_managers):
        """Setup ContextCompressor for compression tests."""
        self.llm_config = real_llm_config
        self.managers = initialized_real_managers
        
        # Create ContextCompressor with real dependencies
        self.context_compressor = ContextCompressor(
            llm_config=self.llm_config,
            token_manager=self.managers.token_manager
        )
    
    def get_compression_test_content(self) -> Dict[str, str]:
        """Get test content for compression validation."""
        return {
            "requirements_document": """
# Requirements Document

## Introduction
This project aims to create a comprehensive user management system for a web application.
The system should handle user registration, authentication, profile management, and role-based access control.

## Requirements

### Requirement 1: User Registration
**User Story:** As a new user, I want to register for an account, so that I can access the application.

#### Acceptance Criteria
1. WHEN a user provides valid registration information THEN the system SHALL create a new user account
2. WHEN a user provides an email that already exists THEN the system SHALL display an appropriate error message
3. WHEN a user provides invalid email format THEN the system SHALL validate and reject the input
4. WHEN registration is successful THEN the system SHALL send a confirmation email

### Requirement 2: User Authentication
**User Story:** As a registered user, I want to log into my account, so that I can access personalized features.

#### Acceptance Criteria
1. WHEN a user provides correct credentials THEN the system SHALL authenticate and grant access
2. WHEN a user provides incorrect credentials THEN the system SHALL deny access and show error message
3. WHEN a user fails authentication 3 times THEN the system SHALL temporarily lock the account
4. WHEN a user requests password reset THEN the system SHALL send a secure reset link
            """,
            
            "design_document": """
# Design Document

## Overview
The user management system will be built using a microservices architecture with separate services for authentication, user profiles, and authorization. The system will use JWT tokens for session management and implement OAuth2 for third-party authentication.

## Architecture
The system consists of the following components:
- Authentication Service: Handles login, logout, and token management
- User Profile Service: Manages user information and preferences
- Authorization Service: Handles role-based access control
- API Gateway: Routes requests and handles cross-cutting concerns
- Database: PostgreSQL for user data storage
- Cache: Redis for session and temporary data storage

## Components and Interfaces
### Authentication Service
- POST /auth/login: Authenticate user credentials
- POST /auth/logout: Invalidate user session
- POST /auth/refresh: Refresh authentication token
- POST /auth/register: Create new user account

### User Profile Service
- GET /users/{id}: Retrieve user profile
- PUT /users/{id}: Update user profile
- DELETE /users/{id}: Delete user account
- GET /users/{id}/preferences: Get user preferences

## Data Models
### User Model
- id: UUID (primary key)
- email: String (unique, required)
- password_hash: String (required)
- first_name: String (required)
- last_name: String (required)
- created_at: Timestamp
- updated_at: Timestamp
- is_active: Boolean
- email_verified: Boolean

### Role Model
- id: UUID (primary key)
- name: String (unique, required)
- description: String
- permissions: Array of permission strings

## Error Handling
The system implements comprehensive error handling with standardized error responses, logging, and monitoring. All errors are categorized and include appropriate HTTP status codes and user-friendly messages.

## Testing Strategy
The testing strategy includes unit tests for individual components, integration tests for service interactions, and end-to-end tests for complete user workflows.
            """,
            
            "execution_log": """
# Project Execution Log

## Task 1: Setup Project Structure - 2024-01-15 10:00:00

### What Was Done
Created the basic project structure with separate directories for each microservice. Set up the development environment with Docker containers for PostgreSQL and Redis.

### What Was Completed
- Created project directory structure
- Set up Docker Compose configuration
- Initialized Git repository
- Created basic README and documentation

### How It Was Done
Used Docker Compose to set up the development environment. Created separate directories for auth-service, user-service, and api-gateway. Initialized each service with basic Node.js/Express setup.

### Difficulties Encountered
- Docker networking configuration took longer than expected
- Had to resolve port conflicts with existing services
- PostgreSQL initialization scripts needed debugging

## Task 2: Implement Authentication Service - 2024-01-15 14:30:00

### What Was Done
Implemented the core authentication functionality including user registration, login, and JWT token management.

### What Was Completed
- User registration endpoint with validation
- Login endpoint with credential verification
- JWT token generation and validation
- Password hashing using bcrypt
- Basic error handling and logging

### How It Was Done
Used Express.js framework with bcrypt for password hashing and jsonwebtoken for JWT management. Implemented middleware for request validation and error handling.

### Difficulties Encountered
- JWT token expiration handling required multiple iterations
- Database connection pooling configuration was complex
- Input validation edge cases needed additional testing
            """
        }
    
    @sequential_test_execution()
    @pytest.mark.integration
    async def test_coherent_summarization(self):
        """
        Test that ContextCompressor produces coherent summaries.
        
        Validates that compressed content:
        - Maintains logical flow and structure
        - Preserves key information
        - Reduces token count significantly
        - Remains readable and useful
        """
        test_content = self.get_compression_test_content()
        
        for content_type, original_content in test_content.items():
            # Prepare context for compression
                context_dict = {
                    "content": original_content,
                    "type": content_type,
                    "metadata": {"source": "test_content"}
                }
                
                # Compress the content
                compressed_result = await self.execute_with_rate_limit_handling(
                    lambda: self.context_compressor.compress_context(
                        context_dict,
                        target_reduction=0.5  # Reduce to 50% of original size
                    )
                )
                
                # Validate compression result structure
                assert hasattr(compressed_result, 'compressed_content'), (
                    f"Compression result should have compressed_content attribute, got: {type(compressed_result)}"
                )
                
                compressed_content = compressed_result.compressed_content
                compression_ratio = compressed_result.compression_ratio
                
                # Validate that content was actually compressed
                assert len(compressed_content) < len(original_content), (
                    f"Compressed content should be shorter than original. "
                    f"Original: {len(original_content)}, Compressed: {len(compressed_content)}"
                )
                
                # Validate compression ratio is reasonable
                assert 0.3 <= compression_ratio <= 0.8, (
                    f"Compression ratio should be between 0.3 and 0.8, got: {compression_ratio}"
                )
                
                # Validate coherence by checking structure preservation
                coherence_score = self._assess_coherence(original_content, compressed_content)
                assert coherence_score >= 0.6, (
                    f"Compressed content coherence score too low: {coherence_score:.2f}. "
                    f"Content may have lost important structure or meaning."
                )
                
                # Validate essential information preservation
                preservation_score = self._assess_information_preservation(
                    original_content, compressed_content, content_type
                )
                assert preservation_score >= 0.8, (
                    f"Essential information preservation score too low: {preservation_score:.2f}. "
                    f"Important information may have been lost during compression."
                )
                
                self.logger.info(
                    f"✓ Successfully compressed {content_type}: "
                    f"ratio={compression_ratio:.2f}, coherence={coherence_score:.2f}, "
                    f"preservation={preservation_score:.2f}"
                )
    
    @sequential_test_execution()
    @pytest.mark.integration
    async def test_essential_information_preservation(self):
        """
        Test that ContextCompressor preserves essential information during compression.
        
        Validates that compressed content retains:
        - Key concepts and entities
        - Important relationships
        - Critical details
        - Structural elements
        """
        # Test with requirements document (has specific structure)
        requirements_content = self.get_compression_test_content()["requirements_document"]
        
        # Prepare context for compression
        context_dict = {
            "content": requirements_content,
            "type": "requirements_document",
            "metadata": {"preserve_structure": True}
        }
        
        compressed_result = await self.execute_with_rate_limit_handling(
            lambda: self.context_compressor.compress_context(
                context_dict,
                target_reduction=0.6
            )
        )
        
        compressed_content = compressed_result.compressed_content
        
        # Check that essential elements are preserved
        essential_elements = [
            "Requirements Document",  # Title
            "User Registration",      # Key requirement
            "User Authentication",    # Key requirement
            "User Story",            # Structure element
            "Acceptance Criteria",   # Structure element
            "WHEN",                  # EARS format
            "THEN",                  # EARS format
            "SHALL"                  # EARS format
        ]
        
        preserved_elements = []
        for element in essential_elements:
            if element.lower() in compressed_content.lower():
                preserved_elements.append(element)
        
        preservation_ratio = len(preserved_elements) / len(essential_elements)
        assert preservation_ratio >= 0.7, (
            f"Too many essential elements lost during compression. "
            f"Preserved: {preserved_elements}, Missing: {set(essential_elements) - set(preserved_elements)}"
        )
        
        # Check that key relationships are maintained
        # Requirements should still reference user stories and acceptance criteria
        has_user_stories = "user story" in compressed_content.lower()
        has_acceptance_criteria = "acceptance criteria" in compressed_content.lower()
        
        assert has_user_stories and has_acceptance_criteria, (
            f"Key structural relationships lost. "
            f"Has user stories: {has_user_stories}, Has acceptance criteria: {has_acceptance_criteria}"
        )
        
        self.logger.info(
            f"✓ Essential information preserved: {preservation_ratio:.2f} "
            f"({len(preserved_elements)}/{len(essential_elements)} elements)"
        )
    
    def _assess_coherence(self, original: str, compressed: str) -> float:
        """Assess coherence of compressed content compared to original."""
        # Simple coherence assessment based on structure preservation
        
        # Check if major sections are preserved
        original_sections = self._extract_sections(original)
        compressed_sections = self._extract_sections(compressed)
        
        if not original_sections:
            return 1.0  # No sections to preserve
        
        section_preservation = len(compressed_sections) / len(original_sections)
        
        # Check if logical flow is maintained (simple heuristic)
        original_sentences = [s.strip() for s in original.split('.') if s.strip()]
        compressed_sentences = [s.strip() for s in compressed.split('.') if s.strip()]
        
        # Check if sentence order is roughly maintained
        order_score = self._assess_order_preservation(original_sentences, compressed_sentences)
        
        # Combine scores
        coherence_score = (section_preservation * 0.6) + (order_score * 0.4)
        return min(1.0, coherence_score)
    
    def _assess_information_preservation(self, original: str, compressed: str, content_type: str) -> float:
        """Assess how well essential information is preserved."""
        # Extract key information based on content type
        if content_type == "requirements_document":
            return self._assess_requirements_preservation(original, compressed)
        elif content_type == "design_document":
            return self._assess_design_preservation(original, compressed)
        elif content_type == "execution_log":
            return self._assess_log_preservation(original, compressed)
        else:
            # Generic assessment
            return self._assess_generic_preservation(original, compressed)
    
    def _assess_requirements_preservation(self, original: str, compressed: str) -> float:
        """Assess preservation of requirements-specific information."""
        # Key elements that should be preserved in requirements
        key_patterns = [
            r'requirement \d+',
            r'user story',
            r'acceptance criteria',
            r'when.*then.*shall',
            r'as a.*i want.*so that'
        ]
        
        import re
        original_matches = sum(len(re.findall(pattern, original, re.IGNORECASE)) for pattern in key_patterns)
        compressed_matches = sum(len(re.findall(pattern, compressed, re.IGNORECASE)) for pattern in key_patterns)
        
        if original_matches == 0:
            return 1.0
        
        return min(1.0, compressed_matches / original_matches)
    
    def _assess_design_preservation(self, original: str, compressed: str) -> float:
        """Assess preservation of design-specific information."""
        # Key elements for design documents
        key_elements = [
            "architecture", "components", "interfaces", "data models",
            "error handling", "testing", "api", "service", "database"
        ]
        
        original_count = sum(1 for element in key_elements if element in original.lower())
        compressed_count = sum(1 for element in key_elements if element in compressed.lower())
        
        if original_count == 0:
            return 1.0
        
        return min(1.0, compressed_count / original_count)
    
    def _assess_log_preservation(self, original: str, compressed: str) -> float:
        """Assess preservation of execution log information."""
        # Key elements for execution logs
        key_patterns = [
            r'what was done',
            r'what was completed',
            r'how it was done',
            r'difficulties encountered',
            r'task \d+',
            r'\d{4}-\d{2}-\d{2}'  # Dates
        ]
        
        import re
        original_matches = sum(len(re.findall(pattern, original, re.IGNORECASE)) for pattern in key_patterns)
        compressed_matches = sum(len(re.findall(pattern, compressed, re.IGNORECASE)) for pattern in key_patterns)
        
        if original_matches == 0:
            return 1.0
        
        return min(1.0, compressed_matches / original_matches)
    
    def _assess_generic_preservation(self, original: str, compressed: str) -> float:
        """Generic assessment of information preservation."""
        # Simple word overlap assessment
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())
        
        if not original_words:
            return 1.0
        
        overlap = len(original_words & compressed_words)
        return overlap / len(original_words)
    
    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headers from content."""
        import re
        # Look for markdown headers
        sections = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        return sections
    
    def _assess_order_preservation(self, original_sentences: List[str], compressed_sentences: List[str]) -> float:
        """Assess if logical order is preserved in compressed content."""
        if not original_sentences or not compressed_sentences:
            return 1.0
        
        # Simple heuristic: check if first and last sentences are preserved
        first_preserved = any(
            self._sentence_similarity(original_sentences[0], comp_sent) > 0.5
            for comp_sent in compressed_sentences[:3]  # Check first 3 sentences
        )
        
        last_preserved = any(
            self._sentence_similarity(original_sentences[-1], comp_sent) > 0.5
            for comp_sent in compressed_sentences[-3:]  # Check last 3 sentences
        )
        
        return (int(first_preserved) + int(last_preserved)) / 2
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate simple similarity between two sentences."""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class TestRevisionCapabilities(LLMIntegrationTestBase):
    """
    Test revision capabilities across all agents.
    
    Validates Requirement 6.5: WHEN testing revision capabilities THEN the system 
    SHALL validate that all agents can meaningfully improve their outputs based on user feedback using LLM calls.
    """
    
    @pytest.fixture(autouse=True)
    def setup_agents(self, real_llm_config, initialized_real_managers, temp_workspace, real_memory_manager):
        """Setup agents for revision testing."""
        self.llm_config = real_llm_config
        self.managers = initialized_real_managers
        self.workspace = temp_workspace
        self.memory_manager = real_memory_manager
        
        # Create agents for revision testing using container
        from autogen_framework.dependency_container import DependencyContainer
        self.container = DependencyContainer.create_production(temp_workspace, self.llm_config)
        
        self.plan_agent = PlanAgent(
            container=self.container,
            name="PlanAgent",
            llm_config=self.llm_config,
            system_message="Generate project requirements"
        )
    
    def get_revision_test_cases(self) -> List[RevisionTestCase]:
        """Get test cases for revision capability validation."""
        return [
            RevisionTestCase(
                original_content="""# Requirements Document

## Requirement 1
**User Story:** As a user, I want to login.

#### Acceptance Criteria
1. User can enter credentials
2. System validates credentials
""",
                feedback="The requirements are too vague. Please add more specific acceptance criteria using EARS format with WHEN/THEN/SHALL structure. Also include error handling scenarios.",
                content_type="requirements",
                expected_improvements=["EARS format", "error handling", "specific criteria"],
                description="Requirements revision should add EARS format and error handling"
            ),
            RevisionTestCase(
                original_content="""# Design Document

## Overview
Simple web application.

## Architecture
Uses database and web server.
""",
                feedback="The design lacks detail. Please add specific technology choices, component diagrams, data models, and error handling strategies. Include API specifications.",
                content_type="design",
                expected_improvements=["technology choices", "component diagrams", "data models", "API specifications"],
                description="Design revision should add technical details and specifications"
            ),
            RevisionTestCase(
                original_content="""# Implementation Plan

- [ ] 1. Create files
- [ ] 2. Write code
- [ ] 3. Test
""",
                feedback="Tasks are too generic. Please break down into specific, actionable steps with clear deliverables, file names, and requirement references.",
                content_type="tasks",
                expected_improvements=["specific steps", "file names", "requirement references", "actionable tasks"],
                description="Tasks revision should add specificity and actionable details"
            )
        ]
    
    @pytest.mark.skip(reason="Revision capabilities testing needs agent method implementations to be completed")
    @sequential_test_execution()
    @pytest.mark.integration
    async def test_meaningful_revision_improvement(self):
        """
        Test that agents can meaningfully improve their outputs based on feedback.
        
        Validates that revised content:
        - Addresses the specific feedback provided
        - Shows measurable improvement in quality
        - Maintains original intent while adding requested elements
        - Demonstrates understanding of the feedback
        """
        test_cases = self.get_revision_test_cases()
        
        for test_case in test_cases:
            # Generate revised content using appropriate agent
                revised_content = await self.execute_with_rate_limit_handling(
                    lambda: self._generate_revision(
                        test_case.original_content,
                        test_case.feedback,
                        test_case.content_type
                    )
                )
                
                # Validate that content was actually revised
                assert revised_content != test_case.original_content, (
                    f"Revised content should be different from original. "
                    f"Agent may not have processed the feedback properly."
                )
                
                # Validate that revision is longer (more detailed)
                assert len(revised_content) > len(test_case.original_content), (
                    f"Revised content should be more detailed than original. "
                    f"Original: {len(test_case.original_content)} chars, "
                    f"Revised: {len(revised_content)} chars"
                )
                
                # Validate that expected improvements are present
                improvement_scores = []
                for improvement in test_case.expected_improvements:
                    score = self._check_improvement_present(revised_content, improvement)
                    improvement_scores.append(score)
                    
                    assert score > 0.3, (
                        f"Expected improvement '{improvement}' not sufficiently addressed. "
                        f"Score: {score:.2f}, Content may not have incorporated feedback properly."
                    )
                
                average_improvement = sum(improvement_scores) / len(improvement_scores)
                assert average_improvement > 0.6, (
                    f"Overall improvement score too low: {average_improvement:.2f}. "
                    f"Revision may not have adequately addressed the feedback."
                )
                
                # Validate revision quality using quality framework
                revision_quality = self.quality_validator.validate_revision_improvement(
                    test_case.original_content,
                    revised_content,
                    test_case.feedback
                )
                
                assert revision_quality['shows_improvement'], (
                    f"Quality framework indicates no improvement detected. "
                    f"Assessment: {revision_quality['improvement_assessment']}"
                )
                
                assert revision_quality['improvement_score'] > 0.5, (
                    f"Improvement score too low: {revision_quality['improvement_score']:.2f}. "
                    f"Revision may not meet quality standards."
                )
                
                self.logger.info(
                    f"✓ Successful revision for {test_case.content_type}: "
                    f"improvement_score={revision_quality['improvement_score']:.2f}, "
                    f"average_improvement={average_improvement:.2f}"
                )
    
    @pytest.mark.skip(reason="Feedback understanding testing needs agent method implementations to be completed")
    @sequential_test_execution()
    @pytest.mark.integration
    async def test_feedback_understanding_and_incorporation(self):
        """
        Test that agents understand and incorporate specific feedback elements.
        
        Validates that agents can:
        - Parse feedback to identify specific requests
        - Incorporate requested changes appropriately
        - Maintain document coherence while making changes
        - Address multiple feedback points simultaneously
        """
        # Test with complex, multi-part feedback
        original_content = """# Requirements Document

## Requirement 1
**User Story:** As a user, I want to access the system.

#### Acceptance Criteria
1. User provides credentials
2. System responds
"""
        
        complex_feedback = """Please improve this requirements document by:
1. Using proper EARS format (WHEN/THEN/SHALL) for acceptance criteria
2. Adding error handling scenarios for invalid credentials
3. Including security requirements for password complexity
4. Adding user story for password reset functionality
5. Specifying response time requirements for authentication"""
        
        revised_content = await self.execute_with_rate_limit_handling(
            lambda: self._generate_revision(original_content, complex_feedback, "requirements")
        )
        
        # Check that each feedback point was addressed
        feedback_points = [
            ("EARS format", ["WHEN", "THEN", "SHALL"]),
            ("error handling", ["error", "invalid", "fail"]),
            ("security requirements", ["security", "password", "complexity"]),
            ("password reset", ["reset", "forgot", "recovery"]),
            ("response time", ["time", "performance", "response"])
        ]
        
        addressed_points = []
        for point_name, keywords in feedback_points:
            content_lower = revised_content.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
            
            if keyword_matches >= 2:  # At least 2 keywords present
                addressed_points.append(point_name)
        
        # Should address at least 80% of feedback points
        address_ratio = len(addressed_points) / len(feedback_points)
        assert address_ratio >= 0.8, (
            f"Not enough feedback points addressed. "
            f"Addressed: {addressed_points}, Total: {len(feedback_points)}, "
            f"Ratio: {address_ratio:.2f}"
        )
        
        # Validate document structure is maintained
        assert "# Requirements Document" in revised_content, (
            "Document title should be preserved during revision"
        )
        
        assert "User Story:" in revised_content, (
            "User story structure should be maintained during revision"
        )
        
        self.logger.info(
            f"✓ Successfully addressed {len(addressed_points)}/{len(feedback_points)} "
            f"feedback points: {addressed_points}"
        )
    
    async def _generate_revision(self, original_content: str, feedback: str, content_type: str) -> str:
        """Generate revised content using appropriate agent."""
        # For this test, we'll use PlanAgent for all revisions
        # In a full implementation, different agents would handle different content types
        
        revision_prompt = f"""Please revise the following {content_type} document based on the provided feedback.

Original Content:
{original_content}

Feedback:
{feedback}

Please provide an improved version that addresses all the feedback points while maintaining the document's purpose and structure."""
        
        # Use the agent's generate_response method with proper context
        context = {
            "user_request": revision_prompt,
            "work_directory": self.workspace,
            "additional_context": {"content_type": content_type, "revision_request": True}
        }
        
        revised_content = await self.plan_agent.generate_response(revision_prompt, context)
        
        return revised_content
    
    def _check_improvement_present(self, content: str, improvement: str) -> float:
        """Check if a specific improvement is present in the content."""
        content_lower = content.lower()
        improvement_lower = improvement.lower()
        
        # Define improvement indicators
        improvement_indicators = {
            "ears format": ["when", "then", "shall", "if"],
            "error handling": ["error", "fail", "exception", "invalid", "timeout"],
            "specific criteria": ["specific", "detailed", "precise", "exactly"],
            "technology choices": ["technology", "framework", "database", "language", "tool"],
            "component diagrams": ["diagram", "component", "architecture", "flow"],
            "data models": ["model", "schema", "entity", "table", "field"],
            "api specifications": ["api", "endpoint", "rest", "http", "json"],
            "specific steps": ["step", "action", "create", "implement", "configure"],
            "file names": [".py", ".js", ".md", ".json", ".txt", "file"],
            "requirement references": ["requirement", "ref", "reference", "addresses"],
            "actionable tasks": ["actionable", "implement", "create", "configure", "test"]
        }
        
        indicators = improvement_indicators.get(improvement_lower, [improvement_lower])
        matches = sum(1 for indicator in indicators if indicator in content_lower)
        
        # Score based on number of indicators found
        return min(1.0, matches / len(indicators))


# Test execution configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])