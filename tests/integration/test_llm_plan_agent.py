"""
PlanAgent LLM Integration Tests.

This module tests the PlanAgent's LLM interactions and output quality validation,
focusing on real LLM calls and document generation quality assessment.

Test Categories:
1. Requirements.md generation with EARS format validation
2. Directory name generation with kebab-case validation
3. User request parsing and analysis capabilities
4. Requirements revision with meaningful improvement validation
5. Memory context integration in requirements generation

All tests use real LLM configurations and validate output quality using the
enhanced QualityMetricsFramework with LLM-specific validation methods.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import replace

from autogen_framework.agents.plan_agent import PlanAgent
from autogen_framework.models import LLMConfig
from autogen_framework.memory_manager import MemoryManager
from tests.integration.test_llm_base import (
    LLMIntegrationTestBase,
    sequential_test_execution,
    QUALITY_THRESHOLDS_STRICT
)


class TestPlanAgentLLMIntegration:
    """
    Integration tests for PlanAgent LLM interactions.
    
    These tests validate the PlanAgent's ability to:
    - Generate high-quality requirements documents using real LLM calls
    - Parse user requests and extract meaningful information
    - Create appropriate directory names in kebab-case format
    - Revise requirements based on feedback with measurable improvements
    - Integrate memory context for better understanding
    """
    
    @pytest.fixture(autouse=True)
    def setup_plan_agent(self, real_llm_config, initialized_real_managers, temp_workspace):
        """Setup PlanAgent with real LLM configuration and managers."""
        # Initialize test base functionality
        self.test_base = LLMIntegrationTestBase()
        self.test_base.setup_method(self.setup_plan_agent)
        
        self.llm_config = real_llm_config
        self.managers = initialized_real_managers
        self.workspace_path = Path(temp_workspace)
        
        # Create memory manager for the workspace
        self.memory_manager = MemoryManager(workspace_path=temp_workspace)
        
        # Initialize PlanAgent with real dependencies using container
        from autogen_framework.dependency_container import DependencyContainer
        self.container = DependencyContainer.create_production(temp_workspace, self.llm_config)
        
        self.plan_agent = PlanAgent(
            container=self.container,
            name="PlanAgent",
            llm_config=self.llm_config,
            system_message="Generate project requirements"
        )
        
        # Use strict quality thresholds for PlanAgent tests
        self.test_base.quality_validator.thresholds = QUALITY_THRESHOLDS_STRICT
    
    async def execute_with_rate_limit_handling(self, llm_operation):
        """Delegate to test base."""
        return await self.test_base.execute_with_rate_limit_handling(llm_operation)
    
    async def execute_with_retry(self, llm_operation, max_attempts=3):
        """Delegate to test base."""
        return await self.test_base.execute_with_retry(llm_operation, max_attempts)
    
    def assert_quality_threshold(self, validation_result, custom_message=None):
        """Delegate to test base."""
        return self.test_base.assert_quality_threshold(validation_result, custom_message)
    
    def log_quality_assessment(self, validation_result):
        """Delegate to test base."""
        return self.test_base.log_quality_assessment(validation_result)
    
    @property
    def quality_validator(self):
        """Access quality validator from test base."""
        return self.test_base.quality_validator
    
    @property
    def logger(self):
        """Access logger from test base."""
        return self.test_base.logger
    
    @sequential_test_execution()
    async def test_requirements_generation_with_ears_validation(self):
        """
        Test requirements.md generation with EARS format validation.
        
        Validates:
        - Requirements document is generated with proper structure
        - EARS format compliance (WHEN/THEN/SHALL statements)
        - User stories follow "As a [role], I want [feature], so that [benefit]" format
        - Acceptance criteria are numbered and testable
        - Document meets quality thresholds for structure and completeness
        """
        user_request = (
            "Create a user authentication system that allows users to register, "
            "login, and manage their profiles. The system should support password "
            "reset functionality and email verification."
        )
        
        task_input = {
            "user_request": user_request,
            "task_type": "planning"
        }
        
        # Execute requirements generation with rate limit handling
        result = await self.execute_with_rate_limit_handling(
            lambda: self.plan_agent._process_task_impl(task_input)
        )
        
        # Verify task execution succeeded
        assert result["success"], f"Requirements generation failed: {result.get('error', 'Unknown error')}"
        assert "requirements_path" in result
        
        # Read generated requirements document
        requirements_path = Path(result["requirements_path"])
        assert requirements_path.exists(), "Requirements file was not created"
        
        requirements_content = requirements_path.read_text(encoding='utf-8')
        
        # Validate requirements quality with EARS format checking
        validation_result = self.quality_validator.validate_requirements_quality(requirements_content)
        
        # Log quality assessment for debugging
        self.log_quality_assessment(validation_result)
        
        # Assert quality thresholds are met
        self.assert_quality_threshold(
            validation_result,
            "Requirements document quality below threshold"
        )
        
        # Specific EARS format validation
        assert validation_result['ears_compliant'], (
            f"Requirements document does not comply with EARS format. "
            f"EARS validation: {validation_result['ears_validation']}"
        )
        
        # Verify document structure
        assert "# Requirements Document" in requirements_content
        assert "## Introduction" in requirements_content
        assert "## Requirements" in requirements_content
        
        # Verify user stories format
        assert "As a" in requirements_content and "I want" in requirements_content
        assert "so that" in requirements_content
        
        # Verify EARS format acceptance criteria
        ears_patterns = ["WHEN", "THEN", "SHALL"]
        for pattern in ears_patterns:
            assert pattern in requirements_content, f"Missing EARS pattern: {pattern}"
        
        # Verify content relevance to user request
        key_terms = ["authentication", "register", "login", "profile", "password", "email"]
        content_lower = requirements_content.lower()
        found_terms = [term for term in key_terms if term in content_lower]
        assert len(found_terms) >= 4, f"Requirements missing key terms. Found: {found_terms}"
        
        self.logger.info(f"Requirements generation test passed with quality score: {validation_result['assessment']['overall_score']:.2f}")
    
    @sequential_test_execution()
    async def test_directory_name_generation_kebab_case(self):
        """
        Test directory name generation with kebab-case validation.
        
        Validates:
        - Directory names are generated in proper kebab-case format
        - Names are descriptive and relevant to the request
        - Names are within length constraints (max 50 characters)
        - Names are unique and don't conflict with existing directories
        - LLM generates appropriate names for various request types
        """
        test_requests = [
            {
                "request": "Build a REST API for managing user accounts",
                "expected_patterns": ["api", "user", "account", "rest"]
            },
            {
                "request": "Fix the database connection timeout issue in production",
                "expected_patterns": ["fix", "database", "connection", "timeout"]
            },
            {
                "request": "Implement real-time chat functionality with WebSocket support",
                "expected_patterns": ["chat", "real-time", "websocket", "implement"]
            },
            {
                "request": "Optimize the search algorithm for better performance",
                "expected_patterns": ["optimize", "search", "algorithm", "performance"]
            }
        ]
        
        generated_names = []
        
        for test_case in test_requests:
            user_request = test_case["request"]
            expected_patterns = test_case["expected_patterns"]
            
            # Parse user request to get summary
            parsed_request = await self.execute_with_rate_limit_handling(
                lambda: self.plan_agent.parse_user_request(user_request)
            )
            
            # Generate directory name
            directory_name = await self.execute_with_rate_limit_handling(
                lambda: self.plan_agent._generate_directory_name(parsed_request["summary"])
            )
            
            # Validate kebab-case format
            kebab_case_pattern = r'^[a-z]+(?:-[a-z]+)*$'
            assert re.match(kebab_case_pattern, directory_name), (
                f"Directory name '{directory_name}' is not in kebab-case format"
            )
            
            # Validate length constraint
            assert len(directory_name) <= 50, (
                f"Directory name '{directory_name}' exceeds 50 character limit"
            )
            
            # Validate descriptiveness - should contain at least one expected pattern
            name_words = directory_name.split('-')
            found_patterns = []
            for pattern in expected_patterns:
                if any(pattern.lower() in word.lower() for word in name_words):
                    found_patterns.append(pattern)
            
            assert len(found_patterns) >= 1, (
                f"Directory name '{directory_name}' doesn't contain expected patterns. "
                f"Expected: {expected_patterns}, Found: {found_patterns}"
            )
            
            # Ensure uniqueness
            assert directory_name not in generated_names, (
                f"Directory name '{directory_name}' is not unique"
            )
            generated_names.append(directory_name)
            
            self.logger.info(f"Generated directory name: '{directory_name}' for request: '{user_request[:50]}...'")
        
        self.logger.info(f"Directory name generation test passed for {len(test_requests)} requests")
    
    @sequential_test_execution()
    async def test_user_request_parsing_and_analysis(self):
        """
        Test user request parsing and analysis capabilities.
        
        Validates:
        - LLM correctly extracts key information from user requests
        - Request type classification is accurate
        - Scope estimation is reasonable
        - Key requirements are identified
        - Technical context is extracted when present
        - Constraints are identified when mentioned
        """
        test_cases = [
            {
                "request": "Create a microservices architecture for an e-commerce platform with Docker containers, Redis caching, and PostgreSQL database. Must handle 10,000 concurrent users.",
                "expected_type": "development",
                "expected_scope": "large",
                "expected_requirements": ["microservices", "e-commerce", "docker", "redis", "postgresql"],
                "expected_constraints": ["10,000 concurrent users"]
            },
            {
                "request": "Fix the memory leak in the image processing module that's causing the application to crash after processing 100 images.",
                "expected_type": "debugging",
                "expected_scope": "medium",
                "expected_requirements": ["memory leak", "image processing", "crash"],
                "expected_constraints": ["100 images"]
            },
            {
                "request": "Add a simple contact form to the website with name, email, and message fields.",
                "expected_type": "development",
                "expected_scope": "small",
                "expected_requirements": ["contact form", "name", "email", "message"],
                "expected_constraints": []
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            user_request = test_case["request"]
            
            # Parse user request with LLM
            parsed_result = await self.execute_with_rate_limit_handling(
                lambda: self.plan_agent.parse_user_request(user_request)
            )
            
            # Validate parsing results
            assert "summary" in parsed_result, "Missing summary in parsed result"
            assert "type" in parsed_result, "Missing type in parsed result"
            assert "scope" in parsed_result, "Missing scope in parsed result"
            assert "key_requirements" in parsed_result, "Missing key_requirements in parsed result"
            
            # Validate request type classification
            assert parsed_result["type"] == test_case["expected_type"], (
                f"Incorrect type classification. Expected: {test_case['expected_type']}, "
                f"Got: {parsed_result['type']}"
            )
            
            # Validate scope estimation
            assert parsed_result["scope"] == test_case["expected_scope"], (
                f"Incorrect scope estimation. Expected: {test_case['expected_scope']}, "
                f"Got: {parsed_result['scope']}"
            )
            
            # Validate key requirements extraction
            extracted_requirements = [req.lower() for req in parsed_result["key_requirements"]]
            expected_requirements = [req.lower() for req in test_case["expected_requirements"]]
            
            found_requirements = []
            for expected_req in expected_requirements:
                if any(expected_req in extracted_req for extracted_req in extracted_requirements):
                    found_requirements.append(expected_req)
            
            assert len(found_requirements) >= len(expected_requirements) // 2, (
                f"Insufficient key requirements extracted. "
                f"Expected: {expected_requirements}, Found: {found_requirements}"
            )
            
            # Validate constraints identification (if any expected)
            if test_case["expected_constraints"]:
                constraints_text = parsed_result.get("constraints", "").lower()
                for constraint in test_case["expected_constraints"]:
                    # Check for key words from the constraint instead of exact match
                    constraint_words = constraint.lower().split()
                    key_words = [word for word in constraint_words if len(word) > 3]  # Skip short words
                    found_words = [word for word in key_words if word in constraints_text]
                    
                    assert len(found_words) >= len(key_words) // 2, (
                        f"Expected constraint keywords from '{constraint}' not found in: {constraints_text}. "
                        f"Key words: {key_words}, Found: {found_words}"
                    )
            
            # Validate summary quality
            summary = parsed_result["summary"]
            assert len(summary) > 10, "Summary is too short"
            assert len(summary) <= 200, "Summary is too long"
            
            self.logger.info(f"Request parsing test {i+1} passed: {parsed_result['type']}/{parsed_result['scope']}")
        
        self.logger.info(f"User request parsing test passed for {len(test_cases)} requests")
    
    @sequential_test_execution()
    async def test_requirements_revision_with_improvement_validation(self):
        """
        Test requirements revision with meaningful improvement validation.
        
        Validates:
        - LLM can revise requirements based on feedback
        - Revisions show measurable quality improvements
        - Feedback is properly incorporated into the document
        - Document structure and format are maintained
        - Changes address the specific feedback provided
        """
        # First, generate initial requirements
        user_request = (
            "Create a simple blog application where users can create, edit, "
            "and delete blog posts. Include basic user authentication."
        )
        
        task_input = {
            "user_request": user_request,
            "task_type": "planning"
        }
        
        # Generate initial requirements
        initial_result = await self.execute_with_rate_limit_handling(
            lambda: self.plan_agent._process_task_impl(task_input)
        )
        
        assert initial_result["success"], "Initial requirements generation failed"
        
        requirements_path = Path(initial_result["requirements_path"])
        original_content = requirements_path.read_text(encoding='utf-8')
        
        # Validate initial quality
        initial_validation = self.quality_validator.validate_requirements_quality(original_content)
        self.log_quality_assessment(initial_validation)
        
        # Apply revision with specific feedback
        revision_feedback = (
            "Please add more detailed security requirements for user authentication, "
            "include requirements for comment functionality on blog posts, "
            "and add performance requirements for handling multiple concurrent users."
        )
        
        revision_task_input = {
            "task_type": "revision",
            "revision_feedback": revision_feedback,
            "current_result": initial_result,
            "work_directory": initial_result["work_directory"]
        }
        
        # Apply revision
        revision_result = await self.execute_with_rate_limit_handling(
            lambda: self.plan_agent._process_task_impl(revision_task_input)
        )
        
        assert revision_result["success"], f"Requirements revision failed: {revision_result.get('error')}"
        assert revision_result["revision_applied"], "Revision was not applied"
        
        # Read revised content
        revised_content = requirements_path.read_text(encoding='utf-8')
        
        # Validate revision improvement
        improvement_validation = self.quality_validator.validate_revision_improvement(
            original_content, revised_content, revision_feedback
        )
        
        # Assert meaningful improvement
        assert improvement_validation['shows_improvement'], (
            f"Revision does not show meaningful improvement. "
            f"Assessment: {improvement_validation['improvement_assessment']}"
        )
        
        # Validate that feedback was incorporated
        revised_lower = revised_content.lower()
        feedback_terms = ["security", "comment", "performance", "concurrent"]
        
        incorporated_terms = [term for term in feedback_terms if term in revised_lower]
        assert len(incorporated_terms) >= 3, (
            f"Revision did not incorporate enough feedback terms. "
            f"Expected: {feedback_terms}, Found: {incorporated_terms}"
        )
        
        # Validate document structure is maintained
        assert "# Requirements Document" in revised_content
        assert "## Introduction" in revised_content
        assert "## Requirements" in revised_content
        
        # Validate EARS format is still present
        ears_patterns = ["WHEN", "THEN", "SHALL"]
        for pattern in ears_patterns:
            assert pattern in revised_content, f"EARS pattern '{pattern}' lost during revision"
        
        # Validate revised quality meets thresholds
        revised_validation = self.quality_validator.validate_requirements_quality(revised_content)
        self.log_quality_assessment(revised_validation)
        
        self.assert_quality_threshold(
            revised_validation,
            "Revised requirements quality below threshold"
        )
        
        # Ensure improvement in quality score
        initial_score = initial_validation['assessment']['overall_score']
        revised_score = revised_validation['assessment']['overall_score']
        
        self.logger.info(f"Quality improvement: {initial_score:.2f} -> {revised_score:.2f}")
        
        # Allow for small variations in LLM output, but expect general improvement or maintenance
        assert revised_score >= initial_score - 0.1, (
            f"Revised quality score significantly decreased: {initial_score:.2f} -> {revised_score:.2f}"
        )
        
        self.logger.info("Requirements revision test passed with meaningful improvements")
    
    @sequential_test_execution()
    async def test_memory_context_integration(self):
        """
        Test memory context integration in requirements generation.
        
        Validates:
        - PlanAgent loads and uses memory context
        - Memory patterns influence requirements generation
        - Historical learnings are applied to new requests
        - Context integration improves output quality
        - Memory updates are properly handled
        """
        # Setup memory context with historical patterns
        memory_content = {
            "successful_patterns": [
                {
                    "pattern_type": "authentication_requirements",
                    "description": "Comprehensive authentication requirements pattern",
                    "requirements": [
                        "Multi-factor authentication support",
                        "Password complexity requirements",
                        "Session timeout management",
                        "Account lockout after failed attempts",
                        "Secure password reset workflow"
                    ]
                },
                {
                    "pattern_type": "api_design_requirements",
                    "description": "RESTful API design requirements pattern",
                    "requirements": [
                        "RESTful endpoint design",
                        "API versioning strategy",
                        "Rate limiting implementation",
                        "Comprehensive error handling",
                        "API documentation generation"
                    ]
                }
            ],
            "project_history": [
                {
                    "project_name": "user-management-system",
                    "success_factors": ["Clear security requirements", "Detailed API specifications"],
                    "lessons_learned": ["Always include rate limiting", "Document all error codes"]
                }
            ]
        }
        
        # Save memory content
        import json
        self.memory_manager.save_memory(
            key="test_patterns",
            content=json.dumps(memory_content, indent=2),
            category="global"
        )
        
        # Create new PlanAgent instance to load memory
        from autogen_framework.dependency_container import DependencyContainer
        container = DependencyContainer.create_production(str(self.workspace_path), self.llm_config)
        
        plan_agent_with_memory = PlanAgent(
            container=container,
            name="PlanAgent",
            llm_config=self.llm_config,
            system_message="Generate project requirements"
        )
        
        # Test request that should benefit from memory patterns
        user_request = (
            "Build a secure API for user authentication and profile management "
            "with proper error handling and documentation."
        )
        
        task_input = {
            "user_request": user_request,
            "task_type": "planning"
        }
        
        # Generate requirements with memory context
        result = await self.execute_with_rate_limit_handling(
            lambda: plan_agent_with_memory._process_task_impl(task_input)
        )
        
        assert result["success"], f"Requirements generation with memory failed: {result.get('error')}"
        
        # Read generated requirements
        requirements_path = Path(result["requirements_path"])
        requirements_content = requirements_path.read_text(encoding='utf-8')
        
        # Validate memory pattern integration
        content_lower = requirements_content.lower()
        
        # Check for authentication pattern elements
        auth_patterns = ["multi-factor", "password complexity", "session timeout", "account lockout"]
        found_auth_patterns = [pattern for pattern in auth_patterns if pattern in content_lower]
        
        # Check for API pattern elements
        api_patterns = ["rate limiting", "api versioning", "error handling", "documentation"]
        found_api_patterns = [pattern for pattern in api_patterns if pattern in content_lower]
        
        # Should find at least some patterns from memory
        total_patterns_found = len(found_auth_patterns) + len(found_api_patterns)
        assert total_patterns_found >= 3, (
            f"Memory patterns not sufficiently integrated. "
            f"Auth patterns found: {found_auth_patterns}, "
            f"API patterns found: {found_api_patterns}"
        )
        
        # Validate quality with memory integration
        validation_result = self.quality_validator.validate_requirements_quality(requirements_content)
        self.log_quality_assessment(validation_result)
        
        # Memory integration should result in high-quality output
        self.assert_quality_threshold(
            validation_result,
            "Requirements with memory integration below quality threshold"
        )
        
        # Verify comprehensive coverage due to memory patterns
        assert validation_result['assessment']['completeness_score'] >= 0.8, (
            f"Memory integration should improve completeness. "
            f"Score: {validation_result['assessment']['completeness_score']}"
        )
        
        self.logger.info(f"Memory context integration test passed. Patterns found: {total_patterns_found}")
    
    @sequential_test_execution()
    async def test_work_directory_creation_and_structure(self):
        """
        Test work directory creation with proper structure.
        
        Validates:
        - Work directories are created successfully
        - Directory names are unique and descriptive
        - Basic file structure is created (requirements.md, design.md, tasks.md)
        - Directory paths are returned correctly
        - Multiple directories can be created without conflicts
        """
        test_summaries = [
            "Build user authentication system",
            "Fix database performance issues",
            "Implement real-time notifications",
            "Create API documentation generator"
        ]
        
        created_directories = []
        
        for summary in test_summaries:
            # Create work directory
            work_directory = await self.execute_with_rate_limit_handling(
                lambda: self.plan_agent.create_work_directory(summary)
            )
            
            work_path = Path(work_directory)
            
            # Validate directory was created
            assert work_path.exists(), f"Work directory was not created: {work_directory}"
            assert work_path.is_dir(), f"Work directory is not a directory: {work_directory}"
            
            # Validate basic file structure
            expected_files = ["requirements.md", "design.md", "tasks.md"]
            for expected_file in expected_files:
                file_path = work_path / expected_file
                assert file_path.exists(), f"Expected file not created: {expected_file}"
                
                # Validate files have placeholder content
                content = file_path.read_text(encoding='utf-8')
                assert len(content) > 10, f"File {expected_file} has insufficient placeholder content"
            
            # Validate directory name format
            dir_name = work_path.name
            kebab_case_pattern = r'^[a-z]+(?:-[a-z]+)*$'
            assert re.match(kebab_case_pattern, dir_name), (
                f"Directory name '{dir_name}' is not in kebab-case format"
            )
            
            # Validate uniqueness
            assert work_directory not in created_directories, (
                f"Directory name is not unique: {work_directory}"
            )
            created_directories.append(work_directory)
            
            self.logger.info(f"Created work directory: {dir_name}")
        
        # Validate all directories are under workspace
        for directory in created_directories:
            assert str(self.workspace_path) in directory, (
                f"Directory not created under workspace: {directory}"
            )
        
        self.logger.info(f"Work directory creation test passed for {len(test_summaries)} directories")
    
    @sequential_test_execution()
    async def test_error_handling_and_recovery(self):
        """
        Test error handling and recovery in PlanAgent operations.
        
        Validates:
        - Graceful handling of invalid inputs
        - Proper error messages for missing parameters
        - Recovery from LLM API errors
        - Fallback mechanisms for directory naming
        - Consistent behavior under error conditions
        """
        # Test invalid task input
        invalid_task_input = {
            "task_type": "planning"
            # Missing required user_request
        }
        
        result = await self.execute_with_rate_limit_handling(
            lambda: self.plan_agent._process_task_impl(invalid_task_input)
        )
        
        # Should handle error gracefully
        assert not result["success"], "Should fail with missing user_request"
        assert "error" in result, "Should provide error message"
        assert "user_request" in result["error"], "Error should mention missing user_request"
        
        # Test revision without required parameters
        invalid_revision_input = {
            "task_type": "revision"
            # Missing revision_feedback and work_directory
        }
        
        revision_result = await self.execute_with_rate_limit_handling(
            lambda: self.plan_agent._process_task_impl(invalid_revision_input)
        )
        
        assert not revision_result["success"], "Should fail with missing revision parameters"
        assert "error" in revision_result, "Should provide error message"
        
        # Test fallback directory naming
        problematic_summary = "!@#$%^&*()_+{}|:<>?[]\\;'\",./"
        
        try:
            directory_name = await self.execute_with_rate_limit_handling(
                lambda: self.plan_agent._generate_directory_name(problematic_summary)
            )
            
            # Should generate a valid fallback name
            assert directory_name, "Should generate fallback directory name"
            assert len(directory_name) > 0, "Directory name should not be empty"
            
            # Should be valid kebab-case or timestamp-based fallback
            is_kebab_case = re.match(r'^[a-z]+(?:-[a-z]+)*$', directory_name)
            is_timestamp_fallback = re.match(r'^task-\d{8}-\d{4}$', directory_name)
            
            assert is_kebab_case or is_timestamp_fallback, (
                f"Fallback directory name should be valid format: {directory_name}"
            )
            
        except Exception as e:
            # If LLM fails, should still handle gracefully
            self.logger.info(f"LLM directory generation failed as expected: {e}")
        
        self.logger.info("Error handling and recovery test passed")


# Additional test utilities for PlanAgent-specific scenarios

import re


def validate_ears_format_compliance(content: str) -> Dict[str, Any]:
    """
    Validate EARS format compliance in requirements document.
    
    Args:
        content: Requirements document content
        
    Returns:
        Dictionary with compliance details
    """
    ears_validation = {
        'has_when_then_shall': False,
        'has_user_stories': False,
        'has_acceptance_criteria': False,
        'when_count': 0,
        'then_count': 0,
        'shall_count': 0,
        'user_story_count': 0,
        'compliance_score': 0.0
    }
    
    content_upper = content.upper()
    
    # Count EARS keywords
    ears_validation['when_count'] = content_upper.count('WHEN')
    ears_validation['then_count'] = content_upper.count('THEN')
    ears_validation['shall_count'] = content_upper.count('SHALL')
    
    # Check for user stories
    user_story_pattern = r'As\s+a\s+.*?,\s*I\s+want\s+.*?,\s*so\s+that'
    user_stories = re.findall(user_story_pattern, content, re.IGNORECASE | re.DOTALL)
    ears_validation['user_story_count'] = len(user_stories)
    
    # Validate presence of key elements
    ears_validation['has_when_then_shall'] = (
        ears_validation['when_count'] > 0 and 
        ears_validation['then_count'] > 0 and 
        ears_validation['shall_count'] > 0
    )
    
    ears_validation['has_user_stories'] = ears_validation['user_story_count'] > 0
    
    # Check for acceptance criteria sections
    acceptance_criteria_pattern = r'acceptance\s+criteria'
    ears_validation['has_acceptance_criteria'] = bool(
        re.search(acceptance_criteria_pattern, content, re.IGNORECASE)
    )
    
    # Calculate compliance score
    score_components = [
        ears_validation['has_when_then_shall'],
        ears_validation['has_user_stories'],
        ears_validation['has_acceptance_criteria'],
        ears_validation['when_count'] >= 3,  # At least 3 WHEN statements
        ears_validation['user_story_count'] >= 2  # At least 2 user stories
    ]
    
    ears_validation['compliance_score'] = sum(score_components) / len(score_components)
    
    return ears_validation


def validate_kebab_case_format(name: str) -> bool:
    """
    Validate that a string follows kebab-case format.
    
    Args:
        name: String to validate
        
    Returns:
        True if string is in kebab-case format
    """
    kebab_case_pattern = r'^[a-z]+(?:-[a-z]+)*$'
    return bool(re.match(kebab_case_pattern, name))


def extract_key_terms_from_request(request: str) -> List[str]:
    """
    Extract key terms from a user request for validation.
    
    Args:
        request: User request string
        
    Returns:
        List of key terms found in the request
    """
    # Common technical terms and action words
    technical_terms = [
        'api', 'database', 'authentication', 'user', 'system', 'application',
        'service', 'interface', 'security', 'performance', 'scalability',
        'microservice', 'rest', 'graphql', 'websocket', 'cache', 'redis',
        'postgresql', 'mysql', 'mongodb', 'docker', 'kubernetes'
    ]
    
    action_words = [
        'create', 'build', 'implement', 'develop', 'fix', 'debug', 'optimize',
        'improve', 'enhance', 'add', 'remove', 'update', 'modify', 'refactor'
    ]
    
    all_terms = technical_terms + action_words
    request_lower = request.lower()
    
    found_terms = [term for term in all_terms if term in request_lower]
    
    # Also extract custom terms (words longer than 3 characters)
    words = re.findall(r'\b\w{4,}\b', request_lower)
    custom_terms = [word for word in words if word not in found_terms]
    
    return found_terms + custom_terms[:5]  # Limit custom terms