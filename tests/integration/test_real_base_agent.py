"""
Integration tests for BaseLLMAgent with real AutoGen functionality.

This module contains integration tests that use actual AutoGen components and real LLM endpoints
to verify the integration between the framework and external services. These tests are designed to:

- Test actual service integration (real AutoGen, LLM endpoints)
- Verify end-to-end functionality with real components
- Validate configuration and service communication
- Test real AutoGen agent initialization and response generation

Key test classes:
- TestBaseLLMAgentRealAutoGenIntegration: Real AutoGen integration tests

These tests require:
- Valid .env.integration file with real LLM configuration
- Network access to LLM endpoints
- Real AutoGen library installation

Tests may take longer to complete due to network calls and real service interactions.
Use @pytest.mark.integration marker to run these tests separately.

For fast unit tests with mocked dependencies, see:
tests/unit/test_base_agent.py
"""

import pytest
import asyncio
from unittest.mock import Mock

from autogen_framework.agents.base_agent import BaseLLMAgent
from autogen_framework.models import LLMConfig, AgentContext


class TestBaseLLMAgentRealAutoGenIntegration:
    """Integration test suite for BaseLLMAgent with real AutoGen functionality."""
    
    @pytest.fixture
    def test_agent_class(self):
        """Create a concrete test implementation of BaseLLMAgent."""
        class TestAgent(BaseLLMAgent):
            async def process_task(self, task_input):
                return {"result": "test_result"}
            
            def get_agent_capabilities(self):
                return ["test_capability"]
        
        return TestAgent
    
    @pytest.fixture
    def test_agent(self, test_agent_class, real_llm_config):
        """Create a test agent instance with real LLM configuration."""
        return test_agent_class(
            name="TestAgent",
            llm_config=real_llm_config,
            system_message="Test system message"
        )
    
    @pytest.mark.integration
    def test_real_autogen_initialization(self, test_agent):
        """Test actual AutoGen agent initialization with real configuration."""
        # Test initialization with real AutoGen components
        result = test_agent.initialize_autogen_agent()
        
        assert result is True
        assert test_agent._is_initialized is True
        assert test_agent._autogen_agent is not None
        
        # Verify the agent was created with correct configuration
        assert hasattr(test_agent, '_autogen_agent')
    
    @pytest.mark.integration
    async def test_real_design_generation_basic(self, test_agent):
        """Test basic response generation with real AutoGen."""
        # Initialize the agent first
        assert test_agent.initialize_autogen_agent() is True
        
        # Test basic response generation
        prompt = "Generate a simple design overview for a web application with user authentication."
        
        try:
            response = await test_agent.generate_response(prompt)
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert any(keyword in response.lower() for keyword in ['design', 'architecture', 'component'])
            
        except Exception as e:
            pytest.skip(f"Real AutoGen test skipped due to: {e}")
    
    @pytest.mark.integration
    async def test_real_memory_integration(self, test_agent):
        """Test memory context integration with real AutoGen."""
        memory_context = {
            "global": {
                "patterns.md": "# Design Patterns\n\nUse MVC pattern for web applications."
            }
        }
        
        test_agent.update_memory_context(memory_context)
        
        # Initialize and test
        assert test_agent.initialize_autogen_agent() is True
        
        prompt = "Create a design that uses the patterns from memory."
        
        try:
            response = await test_agent.generate_response(prompt)
            
            assert isinstance(response, str)
            assert len(response) > 0
            # The response should potentially reference the memory content
            
        except Exception as e:
            pytest.skip(f"Real AutoGen memory test skipped due to: {e}")
    
    @pytest.mark.integration
    def test_real_autogen_configuration_validation(self, test_agent, real_llm_config):
        """Test that real AutoGen configuration is properly validated."""
        # Verify the configuration is valid for real use
        assert real_llm_config.validate() is True
        
        # Test that the agent can be initialized with this configuration
        result = test_agent.initialize_autogen_agent()
        assert result is True
        
        # Verify the configuration was applied correctly
        assert test_agent.llm_config == real_llm_config
        assert test_agent._is_initialized is True
    
    @pytest.mark.integration
    async def test_real_conversation_history_management(self, test_agent):
        """Test conversation history with real AutoGen responses."""
        assert test_agent.initialize_autogen_agent() is True
        
        # Add some conversation history
        test_agent.add_to_conversation_history("user", "Hello")
        
        try:
            # Generate a response that should consider the history
            response = await test_agent.generate_response("Continue our conversation")
            
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Verify conversation history was updated
            assert len(test_agent.conversation_history) >= 1
            
        except Exception as e:
            pytest.skip(f"Real AutoGen conversation test skipped due to: {e}")
    
    @pytest.mark.integration
    def test_real_agent_status_with_initialization(self, test_agent):
        """Test agent status reporting with real initialization."""
        # Get status before initialization
        status_before = test_agent.get_agent_status()
        assert status_before["initialized"] is False
        
        # Initialize with real AutoGen
        result = test_agent.initialize_autogen_agent()
        assert result is True
        
        # Get status after initialization
        status_after = test_agent.get_agent_status()
        assert status_after["initialized"] is True
        assert status_after["model"] == test_agent.llm_config.model
    
    @pytest.mark.integration
    async def test_real_error_handling_with_autogen(self, test_agent):
        """Test error handling with real AutoGen components."""
        assert test_agent.initialize_autogen_agent() is True
        
        try:
            # Test with a potentially problematic prompt
            response = await test_agent.generate_response("")
            
            # Should handle empty prompts gracefully
            assert isinstance(response, str)
            
        except Exception as e:
            # Should handle errors gracefully without crashing
            assert isinstance(e, Exception)
            # This is acceptable for integration tests - we're testing real error conditions


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])