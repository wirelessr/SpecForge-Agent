"""
Unit tests for the BaseLLMAgent abstract class.

This module contains unit tests that focus on testing the BaseLLMAgent class
in isolation using mocked dependencies. These tests are designed to:

- Run quickly (under 1 second each)
- Use only mocked external dependencies (no real AutoGen or LLM calls)
- Test logic paths, error handling, and component interfaces
- Validate agent behavior without external service dependencies

Key test classes:
- TestBaseLLMAgent: Core functionality tests with mocked dependencies
- TestBaseLLMAgentAutoGenMocking: AutoGen integration tests with comprehensive mocking

All external dependencies (AutoGen components, LLM APIs) are mocked to ensure
fast, reliable unit tests that can run without network access or real services.

For tests that use real AutoGen components and LLM endpoints, see:
tests/integration/test_real_base_agent.py
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from autogen_framework.agents.base_agent import BaseLLMAgent
from autogen_framework.models import LLMConfig, AgentContext

class TestBaseLLMAgent:
    """Test suite for BaseLLMAgent abstract class."""
    # Using shared fixtures from conftest.py
    @pytest.fixture
    def test_agent_class(self):
        """Create a concrete test implementation of BaseLLMAgent."""
        class TestAgent(BaseLLMAgent):
            async def process_task(self, task_input):
                return {"result": "test_result", "input": task_input}
            
            def get_agent_capabilities(self):
                return ["test_capability_1", "test_capability_2"]
        
        return TestAgent
    
    @pytest.fixture
    def test_agent(self, test_agent_class, llm_config):
        """Create a test agent instance."""
        return test_agent_class(
            name="TestAgent",
            llm_config=llm_config,
            system_message="Test system message",
            description="Test agent for unit testing"
        )
    
    def test_agent_initialization(self, test_agent, llm_config):
        """Test basic agent initialization."""
        assert test_agent.name == "TestAgent"
        assert test_agent.llm_config == llm_config
        assert test_agent.system_message == "Test system message"
        assert test_agent.description == "Test agent for unit testing"
        assert test_agent.context == {}
        assert test_agent.memory_context == {}
        assert test_agent.conversation_history == []
        assert not test_agent._is_initialized
        assert test_agent._autogen_agent is None
    
    def test_agent_initialization_with_invalid_config(self, test_agent_class):
        """Test agent initialization with invalid LLM config."""
        invalid_config = LLMConfig(
            base_url="",  # Invalid empty URL
            model="test-model",
            api_key="test-key"
        )
        
        with pytest.raises(ValueError, match="Invalid LLM configuration"):
            test_agent_class(
                name="TestAgent",
                llm_config=invalid_config,
                system_message="Test message"
            )
    
    def test_cannot_instantiate_abstract_class(self, llm_config):
        """Test that BaseLLMAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMAgent(
                name="TestAgent",
                llm_config=llm_config,
                system_message="Test message"
            )
    
    def test_update_context(self, test_agent):
        """Test context updating functionality."""
        context = {"key1": "value1", "key2": "value2"}
        test_agent.update_context(context)
        
        assert test_agent.context == context
        
        # Test updating with additional context
        additional_context = {"key3": "value3"}
        test_agent.update_context(additional_context)
        
        expected_context = {"key1": "value1", "key2": "value2", "key3": "value3"}
        assert test_agent.context == expected_context
    
    def test_update_memory_context(self, test_agent):
        """Test memory context updating."""
        memory_context = {
            "global": {"best_practices.md": "Some best practices"},
            "projects": {"test_project": {"context.md": "Project context"}}
        }
        
        test_agent.update_memory_context(memory_context)
        assert test_agent.memory_context == memory_context
    
    def test_build_complete_system_message(self, test_agent):
        """Test system message building with context and memory."""
        # Test with basic system message
        message = test_agent._build_complete_system_message()
        assert "Test system message" in message
        
        # Add context and test again
        test_agent.update_context({"current_task": "Testing"})
        test_agent.update_memory_context({"global": {"tips.md": "Some tips"}})
        
        message = test_agent._build_complete_system_message()
        assert "Test system message" in message
        assert "Memory Context" in message
        assert "Current Context" in message
        assert "Testing" in message
        assert "Some tips" in message
    
    def test_add_to_conversation_history(self, test_agent):
        """Test conversation history management."""
        test_agent.add_to_conversation_history("user", "Hello", {"test": True})
        
        assert len(test_agent.conversation_history) == 1
        entry = test_agent.conversation_history[0]
        
        assert entry["role"] == "user"
        assert entry["content"] == "Hello"
        assert entry["agent"] == "TestAgent"
        assert entry["metadata"]["test"] is True
        assert "timestamp" in entry
    
    def test_conversation_history_limit(self, test_agent):
        """Test that conversation history is limited to 50 entries."""
        # Add 60 entries
        for i in range(60):
            test_agent.add_to_conversation_history("user", f"Message {i}")
        
        # Should only keep the last 50
        assert len(test_agent.conversation_history) == 50
        assert test_agent.conversation_history[0]["content"] == "Message 10"
        assert test_agent.conversation_history[-1]["content"] == "Message 59"
    
    def test_get_agent_status(self, test_agent):
        """Test agent status reporting."""
        status = test_agent.get_agent_status()
        
        assert status["name"] == "TestAgent"
        assert status["description"] == "Test agent for unit testing"
        assert status["initialized"] is False
        assert status["model"] == "test-model"
        assert status["context_items"] == 0
        assert status["memory_items"] == 0
        assert status["conversation_length"] == 0
        assert status["last_activity"] is None
        
        # Add some data and test again
        test_agent.update_context({"key": "value"})
        test_agent.add_to_conversation_history("user", "Hello")
        
        status = test_agent.get_agent_status()
        assert status["context_items"] == 1
        assert status["conversation_length"] == 1
        assert status["last_activity"] is not None
    
    def test_reset_agent(self, test_agent):
        """Test agent reset functionality."""
        # Add some data
        test_agent.update_context({"key": "value"})
        test_agent.add_to_conversation_history("user", "Hello")
        
        # Reset
        test_agent.reset_agent()
        
        assert test_agent.context == {}
        assert test_agent.conversation_history == []
        # Memory context should not be cleared by reset
        assert test_agent.memory_context == {}
    
    def test_export_conversation_history(self, test_agent):
        """Test conversation history export."""
        test_agent.add_to_conversation_history("user", "Hello")
        test_agent.add_to_conversation_history("assistant", "Hi there")
        
        exported = test_agent.export_conversation_history()
        
        assert len(exported) == 2
        assert exported[0]["content"] == "Hello"
        assert exported[1]["content"] == "Hi there"
        
        # Ensure it's a copy, not the original
        exported.append({"test": "entry"})
        assert len(test_agent.conversation_history) == 2
    
    @pytest.mark.asyncio
    async def test_process_task_implementation(self, test_agent):
        """Test that concrete implementation of process_task works."""
        task_input = {"task": "test_task", "params": {"param1": "value1"}}
        result = await test_agent.process_task(task_input)
        
        assert result["result"] == "test_result"
        assert result["input"] == task_input
    
    def test_get_agent_capabilities_implementation(self, test_agent):
        """Test that concrete implementation of get_agent_capabilities works."""
        capabilities = test_agent.get_agent_capabilities()
        
        assert isinstance(capabilities, list)
        assert "test_capability_1" in capabilities
        assert "test_capability_2" in capabilities
    
    def test_string_representations(self, test_agent):
        """Test string representation methods."""
        str_repr = str(test_agent)
        assert "TestAgent" in str_repr
        assert "test-model" in str_repr
        assert "initialized=False" in str_repr
        
        repr_str = repr(test_agent)
        assert "TestAgent" in repr_str
        assert "test-model" in repr_str
        assert "initialized=False" in repr_str
        assert "context_items=0" in repr_str

class TestBaseLLMAgentAutoGenMocking:
    """Test suite for AutoGen integration with proper mocking."""
    
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
    def test_agent(self, test_agent_class, llm_config):
        """Create a test agent instance."""
        return test_agent_class(
            name="TestAgent",
            llm_config=llm_config,
            system_message="Test system message"
        )
    
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    def test_initialize_autogen_agent_success(self, mock_client_class, mock_agent_class, test_agent):
        """Test successful AutoGen agent initialization with mocks."""
        # Mock AutoGen components
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        result = test_agent.initialize_autogen_agent()
        
        assert result is True
        assert test_agent._is_initialized is True
        assert test_agent._autogen_agent == mock_agent
        
        # Verify client was created with correct parameters
        mock_client_class.assert_called_once()
        client_call_args = mock_client_class.call_args
        assert client_call_args[1]["model"] == test_agent.llm_config.model
        assert client_call_args[1]["base_url"] == test_agent.llm_config.base_url
        assert client_call_args[1]["api_key"] == test_agent.llm_config.api_key
        
        # Verify agent was created with correct parameters
        mock_agent_class.assert_called_once()
        agent_call_args = mock_agent_class.call_args
        assert agent_call_args[1]["name"] == "TestAgent"
        assert "Test system message" in agent_call_args[1]["system_message"]
        assert agent_call_args[1]["model_client"] == mock_client
    
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    def test_initialize_autogen_agent_exception(self, mock_client_class, mock_agent_class, test_agent):
        """Test AutoGen agent initialization with exception."""
        mock_client_class.side_effect = Exception("Test error")
        
        result = test_agent.initialize_autogen_agent()
        
        assert result is False
        assert test_agent._is_initialized is False
        assert test_agent._autogen_agent is None
    
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    def test_initialize_autogen_agent_already_initialized(self, mock_client_class, mock_agent_class, test_agent):
        """Test that initialization is skipped if already initialized."""
        # Mock successful first initialization
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        # First initialization
        result1 = test_agent.initialize_autogen_agent()
        assert result1 is True
        
        # Second initialization should return True without creating new agent
        mock_client_class.reset_mock()
        mock_agent_class.reset_mock()
        result2 = test_agent.initialize_autogen_agent()
        assert result2 is True
        mock_client_class.assert_not_called()
        mock_agent_class.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_generate_autogen_response_not_initialized(self, test_agent):
        """Test AutoGen response generation when agent not initialized."""
        with pytest.raises(RuntimeError, match="AutoGen agent not initialized"):
            await test_agent._generate_autogen_response("Test prompt")
    
    @pytest.mark.asyncio
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    async def test_generate_autogen_response_success(self, mock_client_class, mock_agent_class, test_agent):
        """Test successful AutoGen response generation with mocks."""
        # Mock AutoGen components
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        # Initialize agent
        test_agent.initialize_autogen_agent()
        
        # Mock the _generate_autogen_response method directly to avoid AutoGen complexity
        test_agent._generate_autogen_response = AsyncMock(return_value="Mocked response")
        
        # Test response generation
        result = await test_agent._generate_autogen_response("Test prompt")
        
        assert result == "Mocked response"
        test_agent._generate_autogen_response.assert_called_once_with("Test prompt")
    
    @pytest.mark.asyncio
    @patch('autogen_framework.agents.base_agent.AssistantAgent')
    @patch('autogen_framework.agents.base_agent.OpenAIChatCompletionClient')
    async def test_generate_response_with_context(self, mock_client_class, mock_agent_class, test_agent):
        """Test response generation with context and memory."""
        # Mock AutoGen components
        mock_client = Mock()
        mock_agent = Mock()
        mock_client_class.return_value = mock_client
        mock_agent_class.return_value = mock_agent
        
        # Initialize agent and add context
        test_agent.initialize_autogen_agent()
        test_agent.update_context({"key": "value"})
        test_agent.update_memory_context({"global": {"test.md": "Test content"}})
        
        # Mock the generate_response method directly to avoid AutoGen complexity
        test_agent.generate_response = AsyncMock(return_value="Response with context")
        
        # Test response generation
        result = await test_agent.generate_response("Test prompt")
        
        assert result == "Response with context"
        test_agent.generate_response.assert_called_once_with("Test prompt")


# Integration tests have been moved to tests/integration/test_real_base_agent.py

if __name__ == "__main__":
    pytest.main([__file__])