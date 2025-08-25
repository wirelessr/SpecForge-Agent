"""
LLM Memory Integration Tests.

This module tests system message building and memory integration functionality
with real LLM interactions. It validates that agents can properly incorporate
memory context into their system messages and use historical patterns in their
decision-making processes.

Test Categories:
1. System Message Construction - Tests building system messages with memory context
2. Historical Pattern Incorporation - Tests using memory patterns in agent decisions  
3. Context Formatting - Tests proper formatting of memory context for agent consumption
4. Memory Context Updates - Tests persistence and updates of memory context
5. Cross-Agent Memory Sharing - Tests memory consistency across different agents

Requirements Covered:
- 6.4: System message building with memory context integration
- 6.5: Cross-agent memory sharing and consistency
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

from tests.integration.test_llm_base import (
    LLMIntegrationTestBase, 
    sequential_test_execution,
    QUALITY_THRESHOLDS_LENIENT
)
from autogen_framework.agents.plan_agent import PlanAgent
from autogen_framework.agents.design_agent import DesignAgent
from autogen_framework.agents.tasks_agent import TasksAgent
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.context_manager import ContextManager
from autogen_framework.models import LLMConfig


@dataclass
class MemoryTestScenario:
    """Test scenario for memory integration testing."""
    name: str
    description: str
    memory_content: Dict[str, Any]
    user_request: str
    expected_memory_usage: List[str]  # Expected memory patterns to be used
    validation_criteria: Dict[str, Any]


class TestLLMMemoryIntegration(LLMIntegrationTestBase):
    """
    Test suite for LLM memory integration and system message building.
    
    This class tests the integration between agents and the memory system,
    ensuring that memory context is properly incorporated into system messages
    and influences agent decision-making.
    """
    
    def setup_method(self, method):
        """Setup for each test method."""
        super().setup_method(method)
        self.test_scenarios = self._create_test_scenarios()
    
    def _create_test_scenarios(self) -> List[MemoryTestScenario]:
        """Create test scenarios for memory integration testing."""
        return [
            MemoryTestScenario(
                name="python_project_with_patterns",
                description="Python project with existing patterns and best practices",
                memory_content={
                    "global": {
                        "python_best_practices.md": """# Python Best Practices

## Code Organization
- Use clear module structure with __init__.py files
- Follow PEP 8 naming conventions
- Implement proper error handling with try/except blocks

## Testing Patterns
- Use pytest for testing framework
- Implement unit tests with proper mocking
- Create integration tests for external dependencies

## Documentation
- Use docstrings for all public functions and classes
- Include type hints for better code clarity
- Maintain README.md with setup instructions
""",
                        "project_patterns.md": """# Project Patterns

## Successful Project Structures
- src/ directory for source code
- tests/ directory with unit and integration tests
- config/ directory for configuration files
- docs/ directory for documentation

## Common Implementation Approaches
- Use dependency injection for better testability
- Implement logging with proper levels
- Create configuration management systems
- Use virtual environments for dependency isolation
"""
                    },
                    "projects": {
                        "similar_python_project": {
                            "context.md": """# Similar Python Project Context

This project successfully implemented a multi-module Python application with:
- Clean architecture separation
- Comprehensive test suite
- Configuration management
- Proper error handling

## Key Learnings
- Start with clear requirements and design
- Implement tests early in development
- Use consistent coding standards
- Document all public interfaces
""",
                            "learnings.md": """# Project Learnings

## What Worked Well
- Test-driven development approach
- Modular design with clear interfaces
- Comprehensive documentation
- Regular code reviews

## Challenges Overcome
- Complex dependency management resolved with dependency injection
- Performance issues resolved with caching strategies
- Testing challenges resolved with proper mocking
"""
                        }
                    }
                },
                user_request="Create a Python web application with user authentication and data management",
                expected_memory_usage=[
                    "python_best_practices",
                    "project_patterns", 
                    "similar_python_project"
                ],
                validation_criteria={
                    "mentions_testing": True,
                    "mentions_structure": True,
                    "mentions_documentation": True,
                    "uses_memory_patterns": True
                }
            ),
            MemoryTestScenario(
                name="api_development_patterns",
                description="API development with security and performance patterns",
                memory_content={
                    "global": {
                        "api_security.md": """# API Security Best Practices

## Authentication
- Use JWT tokens for stateless authentication
- Implement proper token expiration and refresh
- Use HTTPS for all API communications

## Authorization
- Implement role-based access control (RBAC)
- Use principle of least privilege
- Validate all input parameters

## Rate Limiting
- Implement rate limiting per user/IP
- Use sliding window algorithms
- Provide clear error messages for rate limit exceeded
""",
                        "performance_patterns.md": """# Performance Optimization Patterns

## Caching Strategies
- Use Redis for session and data caching
- Implement cache invalidation strategies
- Use CDN for static content delivery

## Database Optimization
- Use connection pooling
- Implement proper indexing strategies
- Use query optimization techniques

## Monitoring
- Implement comprehensive logging
- Use metrics collection and alerting
- Monitor response times and error rates
"""
                    }
                },
                user_request="Design a REST API for a e-commerce platform with high performance requirements",
                expected_memory_usage=[
                    "api_security",
                    "performance_patterns"
                ],
                validation_criteria={
                    "mentions_security": True,
                    "mentions_performance": True,
                    "mentions_caching": True,
                    "uses_memory_patterns": True
                }
            )
        ]
    
    @pytest.mark.skip(reason="Memory integration functionality needs to be updated to match current implementation")
    @sequential_test_execution()
    async def test_system_message_construction_with_memory_context(self, initialized_real_managers, real_llm_config, temp_workspace):
        """
        Test system message construction with memory context integration.
        
        Validates that agents properly build system messages that include
        relevant memory context and that this context influences their responses.
        """
        managers = initialized_real_managers
        scenario = self.test_scenarios[0]  # Python project scenario
        
        # Create container and get memory manager
        from autogen_framework.dependency_container import DependencyContainer
        container = DependencyContainer.create_production(temp_workspace, real_llm_config)
        memory_manager = container.get_memory_manager()
        
        # Setup memory content
        await self._setup_memory_content(memory_manager, scenario.memory_content)
        
        plan_agent = PlanAgent(
            container=container,
            name="PlanAgent",
            llm_config=real_llm_config,
            system_message="Generate project requirements"
        )
        
        # Test system message construction
        original_system_message = plan_agent.system_message
        
        # Load memory context into agent
        plan_agent._load_memory_context()
        
        # Verify memory context was loaded
        assert plan_agent.memory_context is not None
        assert len(plan_agent.memory_context) > 0
        
        # Build complete system message
        complete_system_message = plan_agent._build_complete_system_message()
        
        # Validate system message includes memory context
        assert len(complete_system_message) > len(original_system_message)
        assert "Memory Context" in complete_system_message
        assert "python_best_practices" in complete_system_message
        assert "project_patterns" in complete_system_message
        
        # Test that memory context influences response
        response = await self.execute_with_rate_limit_handling(
            lambda: plan_agent.generate_response(scenario.user_request)
        )
        
        # Validate response quality and memory usage
        validation_result = self.quality_validator.validate_llm_output(response, 'requirements')
        
        # Check that memory patterns are referenced in response
        response_lower = response.lower()
        memory_usage_count = 0
        for pattern in scenario.expected_memory_usage:
            if any(keyword in response_lower for keyword in pattern.split('_')):
                memory_usage_count += 1
        
        # Validate memory integration
        assert memory_usage_count >= 2, f"Expected memory patterns not found in response. Found {memory_usage_count} patterns."
        
        # Validate response incorporates memory guidance
        for criterion, expected in scenario.validation_criteria.items():
            if criterion == "mentions_testing":
                assert "test" in response_lower or "testing" in response_lower
            elif criterion == "mentions_structure":
                assert "structure" in response_lower or "organization" in response_lower
            elif criterion == "mentions_documentation":
                assert "document" in response_lower or "readme" in response_lower
        
        self.log_quality_assessment(validation_result)
        self.logger.info(f"System message construction test passed with {memory_usage_count} memory patterns used")
    
    @pytest.mark.skip(reason="Memory integration functionality needs to be updated to match current implementation")
    @sequential_test_execution()
    async def test_historical_pattern_incorporation_in_decisions(self, initialized_real_managers, real_llm_config, temp_workspace):
        """
        Test historical pattern incorporation into agent decisions.
        
        Validates that agents use historical patterns from memory to make
        better decisions and provide more informed recommendations.
        """
        managers = initialized_real_managers
        scenario = self.test_scenarios[1]  # API development scenario
        
        # Create container and get memory manager
        from autogen_framework.dependency_container import DependencyContainer
        container = DependencyContainer.create_production(temp_workspace, real_llm_config)
        memory_manager = container.get_memory_manager()
        
        # Setup memory content with API patterns
        await self._setup_memory_content(memory_manager, scenario.memory_content)
        
        # Create DesignAgent with memory context using container
        design_agent = DesignAgent(
            container=container,
            name="DesignAgent",
            llm_config=real_llm_config,
            system_message="Generate technical design documents"
        )
        
        # Create a requirements document for the design agent
        requirements_content = """# Requirements Document

## Introduction
This document outlines requirements for a REST API for an e-commerce platform.

## Requirements

### Requirement 1: User Authentication
**User Story:** As a user, I want to authenticate securely, so that my account is protected.

#### Acceptance Criteria
1. WHEN a user logs in THEN the system SHALL authenticate using secure methods
2. WHEN authentication succeeds THEN the system SHALL provide access tokens
3. WHEN tokens expire THEN the system SHALL require re-authentication

### Requirement 2: High Performance
**User Story:** As a user, I want fast API responses, so that the application is responsive.

#### Acceptance Criteria
1. WHEN API requests are made THEN the system SHALL respond within 200ms for 95% of requests
2. WHEN high load occurs THEN the system SHALL maintain performance through caching
3. WHEN scaling is needed THEN the system SHALL support horizontal scaling
"""
        
        # Create temporary requirements file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(requirements_content)
            requirements_path = f.name
        
        try:
            # Generate design using memory context
            design_response = await self.execute_with_rate_limit_handling(
                lambda: design_agent.generate_design(requirements_path, design_agent.memory_context)
            )
            
            # Validate design quality
            validation_result = self.quality_validator.validate_design_quality(design_response)
            
            # Check that historical patterns influenced the design
            design_lower = design_response.lower()
            
            # Validate security patterns are incorporated
            security_indicators = ["jwt", "authentication", "authorization", "https", "security"]
            security_mentions = sum(1 for indicator in security_indicators if indicator in design_lower)
            assert security_mentions >= 3, f"Expected security patterns from memory not found. Found {security_mentions} indicators."
            
            # Validate performance patterns are incorporated
            performance_indicators = ["cache", "caching", "redis", "performance", "optimization", "rate limit"]
            performance_mentions = sum(1 for indicator in performance_indicators if indicator in design_lower)
            assert performance_mentions >= 3, f"Expected performance patterns from memory not found. Found {performance_mentions} indicators."
            
            # Validate overall design quality
            self.assert_quality_threshold(validation_result, "Design should meet quality thresholds with memory integration")
            
            self.log_quality_assessment(validation_result)
            self.logger.info(f"Historical pattern incorporation test passed with {security_mentions} security and {performance_mentions} performance patterns")
            
        finally:
            # Cleanup temporary file
            Path(requirements_path).unlink(missing_ok=True)
    
    @pytest.mark.skip(reason="Memory integration functionality needs to be updated to match current implementation")
    @sequential_test_execution()
    async def test_context_formatting_and_agent_consumption(self, initialized_real_managers, real_llm_config, temp_workspace):
        """
        Test context formatting and agent consumption of memory context.
        
        Validates that memory context is properly formatted for different agent types
        and that agents can effectively consume and utilize the formatted context.
        """
        managers = initialized_real_managers
        scenario = self.test_scenarios[0]  # Python project scenario
        
        # Create container and get memory manager
        from autogen_framework.dependency_container import DependencyContainer
        container = DependencyContainer.create_production(temp_workspace, real_llm_config)
        memory_manager = container.get_memory_manager()
        
        # Setup memory content
        await self._setup_memory_content(memory_manager, scenario.memory_content)
        
        # Test context formatting for different agent types
        agents_to_test = [
            ("PlanAgent", PlanAgent),
            ("DesignAgent", DesignAgent),
            ("TasksAgent", TasksAgent)
        ]
        
        context_consumption_results = {}
        
        for agent_name, agent_class in agents_to_test:
            # Create agent with memory context using container
            if agent_name == "PlanAgent":
                agent = agent_class(
                    container=container,
                    name=agent_name,
                    llm_config=real_llm_config,
                    system_message="Generate project requirements"
                )
            elif agent_name == "TasksAgent":
                agent = agent_class(
                    container=container,
                    name=agent_name,
                    llm_config=real_llm_config,
                    system_message="Generate implementation task lists"
                )
            else:  # DesignAgent
                agent = agent_class(
                    container=container,
                    name=agent_name,
                    llm_config=real_llm_config,
                    system_message="Generate technical design documents"
                )
            
            # Test context formatting
            if hasattr(agent, '_load_memory_context'):
                agent._load_memory_context()
            
            complete_system_message = agent._build_complete_system_message()
            
            # Validate context formatting
            assert "Memory Context" in complete_system_message or len(agent.memory_context) > 0
            
            # Test agent consumption of context
            test_prompt = f"Based on available context, what are the key considerations for {scenario.user_request}?"
            
            response = await self.execute_with_rate_limit_handling(
                lambda: agent.generate_response(test_prompt)
            )
            
            # Analyze context consumption
            response_lower = response.lower()
            context_usage_score = 0
            
            # Check for memory pattern usage
            for pattern in scenario.expected_memory_usage:
                pattern_keywords = pattern.replace('_', ' ').split()
                if any(keyword in response_lower for keyword in pattern_keywords):
                    context_usage_score += 1
            
            # Check for specific memory content references
            memory_keywords = ["best practices", "patterns", "testing", "structure", "documentation"]
            memory_references = sum(1 for keyword in memory_keywords if keyword in response_lower)
            
            context_consumption_results[agent_name] = {
                "context_usage_score": context_usage_score,
                "memory_references": memory_references,
                "response_length": len(response),
                "system_message_length": len(complete_system_message)
            }
            
            # Validate minimum context consumption
            assert context_usage_score >= 1, f"{agent_name} did not effectively use memory context"
            assert memory_references >= 2, f"{agent_name} did not reference enough memory content"
            
            self.logger.info(f"{agent_name} context consumption: {context_usage_score} patterns, {memory_references} references")
        
        # Validate that all agents consumed context effectively
        total_usage = sum(result["context_usage_score"] for result in context_consumption_results.values())
        assert total_usage >= len(agents_to_test), "Insufficient context consumption across agents"
        
        self.logger.info(f"Context formatting test passed with total usage score: {total_usage}")
    
    @pytest.mark.skip(reason="Memory integration functionality needs to be updated to match current implementation")
    @sequential_test_execution()
    async def test_memory_context_updates_and_persistence(self, initialized_real_managers, real_llm_config, temp_workspace):
        """
        Test memory context updates and persistence.
        
        Validates that memory context can be updated during execution and that
        these updates persist across agent interactions and sessions.
        """
        managers = initialized_real_managers
        
        # Create container and get memory manager
        from autogen_framework.dependency_container import DependencyContainer
        container = DependencyContainer.create_production(temp_workspace, real_llm_config)
        memory_manager = container.get_memory_manager()
        
        # Initial memory setup
        initial_memory = {
            "global": {
                "initial_pattern.md": "# Initial Pattern\n\nThis is an initial pattern for testing."
            }
        }
        await self._setup_memory_content(memory_manager, initial_memory)
        
        # Create agent and verify initial memory
        plan_agent = PlanAgent(
            container=container,
            name="PlanAgent",
            llm_config=real_llm_config,
            system_message="Generate project requirements"
        )
        
        initial_memory_content = memory_manager.load_memory()
        # Memory manager has categories as top-level keys
        assert len(initial_memory_content) > 0
        assert "global" in initial_memory_content
        # The global category contains the actual memory files
        global_memory = initial_memory_content["global"]
        assert any("initial_pattern.md" in key for key in global_memory.keys())
        
        # Add new memory content
        new_pattern_content = """# New Learning Pattern

## Key Insights
- Memory updates should be persistent
- Context should be immediately available
- Patterns should influence future decisions

## Implementation Notes
- Use proper file organization
- Maintain memory consistency
- Validate memory integration
"""
        
        success = memory_manager.save_memory(
            key="new_learning_pattern",
            content=new_pattern_content,
            category="global"
        )
        assert success, "Failed to save new memory content"
        
        # Verify memory update
        updated_memory_content = memory_manager.load_memory()
        global_memory = updated_memory_content["global"]
        assert any("new_learning_pattern.md" in key for key in global_memory.keys())
        
        # Test persistence across agent reload
        plan_agent._load_memory_context()
        agent_memory = plan_agent.memory_context
        
        # Validate updated memory is available to agent
        # Agent memory context has flattened keys like "global/new_learning_pattern.md"
        assert len(agent_memory) > 0
        assert any("new_learning_pattern.md" in key for key in agent_memory.keys())
        assert "Key Insights" in agent_memory["global"]["new_learning_pattern.md"]
        
        # Test that updated memory influences responses
        test_prompt = "What patterns should I consider for memory management in applications?"
        
        response = await self.execute_with_rate_limit_handling(
            lambda: plan_agent.generate_response(test_prompt)
        )
        
        # Validate that new memory content influences response
        response_lower = response.lower()
        new_memory_indicators = ["persistent", "consistency", "integration", "memory"]
        memory_influence_count = sum(1 for indicator in new_memory_indicators if indicator in response_lower)
        
        assert memory_influence_count >= 2, f"New memory content did not influence response. Found {memory_influence_count} indicators."
        
        # Test project-specific memory updates
        project_memory_content = """# Project Context

This project involves testing memory integration patterns.

## Current Status
- Memory persistence: Implemented
- Context updates: Working
- Agent integration: Validated
"""
        
        project_success = memory_manager.save_memory(
            key="project_context",
            content=project_memory_content,
            category="project",
            project_name="memory_integration_test"
        )
        assert project_success, "Failed to save project-specific memory"
        
        # Verify project memory persistence
        final_memory_content = memory_manager.load_memory()
        assert "projects" in final_memory_content
        assert "memory_integration_test" in final_memory_content["projects"]
        assert "project_context.md" in final_memory_content["projects"]["memory_integration_test"]
        
        self.logger.info("Memory context updates and persistence test passed")
    
    @pytest.mark.skip(reason="Memory integration functionality needs to be updated to match current implementation")
    @sequential_test_execution()
    async def test_cross_agent_memory_sharing_and_consistency(self, initialized_real_managers, real_llm_config, temp_workspace):
        """
        Test cross-agent memory sharing and consistency.
        
        Validates that memory context is consistently shared across different
        agent types and that all agents have access to the same memory patterns.
        """
        managers = initialized_real_managers
        scenario = self.test_scenarios[0]  # Python project scenario
        
        # Create container and get memory manager
        from autogen_framework.dependency_container import DependencyContainer
        container = DependencyContainer.create_production(temp_workspace, real_llm_config)
        memory_manager = container.get_memory_manager()
        
        # Setup comprehensive memory content
        await self._setup_memory_content(memory_manager, scenario.memory_content)
        
        # Add shared learning that should be available to all agents
        shared_learning = """# Shared Development Patterns

## Cross-Agent Consistency Requirements
- All agents should reference the same patterns
- Memory context should be synchronized
- Decisions should be consistent across agents

## Best Practices for Multi-Agent Systems
- Use centralized memory management
- Ensure consistent context formatting
- Validate cross-agent communication patterns
- Maintain shared knowledge base
"""
        
        memory_manager.save_memory(
            key="shared_development_patterns",
            content=shared_learning,
            category="global"
        )
        
        # Create multiple agents with shared memory
        agents = {
            "plan": PlanAgent(
                container=container,
                name="PlanAgent",
                llm_config=real_llm_config,
                system_message="Generate project requirements"
            ),
            "design": DesignAgent(
                container=container,
                name="DesignAgent",
                llm_config=real_llm_config,
                system_message="Generate technical design documents"
            ),
            "tasks": TasksAgent(
                container=container,
                name="TasksAgent",
                llm_config=real_llm_config,
                system_message="Generate implementation task lists"
            )
        }
        
        # Load memory context for all agents
        for agent_name, agent in agents.items():
            if hasattr(agent, '_load_memory_context'):
                agent._load_memory_context()
        
        # Test memory consistency across agents
        memory_consistency_results = {}
        
        for agent_name, agent in agents.items():
            # Verify each agent has access to shared memory
            agent_memory = agent.memory_context
            assert "global" in agent_memory, f"{agent_name} missing global memory"
            assert "shared_development_patterns.md" in agent_memory["global"], f"{agent_name} missing shared patterns"
            
            # Test agent response to shared context
            consistency_prompt = "What development patterns should be followed for consistency across the project?"
            
            response = await self.execute_with_rate_limit_handling(
                lambda: agent.generate_response(consistency_prompt)
            )
            
            # Analyze consistency indicators
            response_lower = response.lower()
            consistency_keywords = ["consistent", "pattern", "shared", "standard", "practice"]
            consistency_score = sum(1 for keyword in consistency_keywords if keyword in response_lower)
            
            memory_consistency_results[agent_name] = {
                "consistency_score": consistency_score,
                "response_length": len(response),
                "mentions_shared_patterns": "shared" in response_lower and "pattern" in response_lower
            }
            
            # Validate minimum consistency
            assert consistency_score >= 2, f"{agent_name} did not demonstrate sufficient consistency awareness"
            
            self.logger.info(f"{agent_name} consistency score: {consistency_score}")
        
        # Validate cross-agent consistency
        all_agents_mention_shared = all(
            result["mentions_shared_patterns"] 
            for result in memory_consistency_results.values()
        )
        assert all_agents_mention_shared, "Not all agents referenced shared patterns"
        
        # Test memory synchronization
        # Add new memory and verify all agents can access it
        sync_test_content = """# Synchronization Test Pattern

This pattern was added during cross-agent testing to verify memory synchronization.

## Validation Points
- All agents should see this content
- Responses should reference synchronization
- Memory consistency should be maintained
"""
        
        memory_manager.save_memory(
            key="sync_test_pattern",
            content=sync_test_content,
            category="global"
        )
        
        # Reload memory for all agents
        for agent_name, agent in agents.items():
            if hasattr(agent, '_load_memory_context'):
                agent._load_memory_context()
            elif hasattr(agent, 'update_memory_context'):
                agent.update_memory_context(memory_manager.load_memory())
        
        # Verify synchronization
        sync_test_prompt = "What synchronization patterns are important for this project?"
        
        sync_results = {}
        for agent_name, agent in agents.items():
            response = await self.execute_with_rate_limit_handling(
                lambda: agent.generate_response(sync_test_prompt)
            )
            
            sync_results[agent_name] = "synchronization" in response.lower()
        
        # Validate that all agents can access the new memory
        agents_with_sync = sum(1 for has_sync in sync_results.values() if has_sync)
        assert agents_with_sync >= 2, f"Only {agents_with_sync} agents referenced synchronization patterns"
        
        self.logger.info(f"Cross-agent memory sharing test passed with {agents_with_sync}/{len(agents)} agents synchronized")
    
    async def _setup_memory_content(self, memory_manager: MemoryManager, memory_content: Dict[str, Any]) -> None:
        """
        Setup memory content for testing.
        
        Args:
            memory_manager: MemoryManager instance
            memory_content: Memory content to setup
        """
        for category, content in memory_content.items():
            if category == "global":
                for file_key, file_content in content.items():
                    key = file_key.replace('.md', '')
                    success = memory_manager.save_memory(
                        key=key,
                        content=file_content,
                        category="global"
                    )
                    assert success, f"Failed to save global memory: {key}"
            
            elif category == "projects":
                for project_name, project_content in content.items():
                    for file_key, file_content in project_content.items():
                        key = file_key.replace('.md', '')
                        success = memory_manager.save_memory(
                            key=key,
                            content=file_content,
                            category="project",
                            project_name=project_name
                        )
                        assert success, f"Failed to save project memory: {project_name}/{key}"
        
        self.logger.info(f"Setup memory content with {len(memory_content)} categories")
    
    def get_agent_capabilities(self) -> List[str]:
        """Get capabilities tested by this test suite."""
        return [
            "System message construction with memory context",
            "Historical pattern incorporation in agent decisions",
            "Context formatting for different agent types",
            "Memory context updates and persistence",
            "Cross-agent memory sharing and consistency",
            "Memory-influenced response generation",
            "Context synchronization across agents",
            "Memory pattern validation and usage"
        ]


# Additional test utilities for memory integration testing

class MemoryIntegrationValidator:
    """Utility class for validating memory integration in tests."""
    
    @staticmethod
    def validate_memory_usage_in_response(response: str, expected_patterns: List[str]) -> Dict[str, Any]:
        """
        Validate that a response uses expected memory patterns.
        
        Args:
            response: Agent response to validate
            expected_patterns: List of expected memory patterns
            
        Returns:
            Validation results dictionary
        """
        response_lower = response.lower()
        
        pattern_usage = {}
        total_matches = 0
        
        for pattern in expected_patterns:
            pattern_keywords = pattern.replace('_', ' ').split()
            matches = sum(1 for keyword in pattern_keywords if keyword in response_lower)
            pattern_usage[pattern] = matches
            if matches > 0:
                total_matches += 1
        
        return {
            "total_patterns_used": total_matches,
            "pattern_usage": pattern_usage,
            "usage_percentage": total_matches / len(expected_patterns) if expected_patterns else 0,
            "response_length": len(response)
        }
    
    @staticmethod
    def validate_cross_agent_consistency(agent_responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate consistency across multiple agent responses.
        
        Args:
            agent_responses: Dictionary mapping agent names to their responses
            
        Returns:
            Consistency validation results
        """
        if len(agent_responses) < 2:
            return {"consistent": True, "reason": "Only one agent response"}
        
        # Extract common themes and keywords
        all_responses = list(agent_responses.values())
        common_keywords = set()
        
        for response in all_responses:
            words = response.lower().split()
            # Filter for meaningful words (length > 3)
            meaningful_words = [word for word in words if len(word) > 3]
            common_keywords.update(meaningful_words)
        
        # Check for shared concepts across responses
        shared_concepts = {}
        for keyword in common_keywords:
            appearances = sum(1 for response in all_responses if keyword in response.lower())
            if appearances > 1:
                shared_concepts[keyword] = appearances
        
        consistency_score = len(shared_concepts) / len(common_keywords) if common_keywords else 0
        
        return {
            "consistent": consistency_score > 0.1,  # At least 10% shared concepts
            "consistency_score": consistency_score,
            "shared_concepts": len(shared_concepts),
            "total_concepts": len(common_keywords),
            "agent_count": len(agent_responses)
        }


# Test configuration for memory integration
MEMORY_INTEGRATION_CONFIG = {
    "test_timeout": 300,  # 5 minutes per test
    "memory_patterns_threshold": 2,  # Minimum memory patterns to reference
    "consistency_threshold": 0.15,  # Minimum consistency score across agents
    "quality_thresholds": QUALITY_THRESHOLDS_LENIENT  # Use lenient thresholds for memory tests
}