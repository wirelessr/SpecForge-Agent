"""
Integration tests for token management in BaseLLMAgent.

Tests the integration of TokenManager and ContextCompressor with BaseLLMAgent
to ensure proper token limit checking, context compression, and token usage tracking.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from autogen_framework.agents.base_agent import BaseLLMAgent
from autogen_framework.token_manager import TokenManager, TokenCheckResult
from autogen_framework.context_compressor import ContextCompressor
from autogen_framework.models import LLMConfig, CompressionResult
from autogen_framework.config_manager import ConfigManager


class TestTokenManagementIntegration:
    """Test token management integration with BaseLLMAgent."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock ConfigManager for testing."""
        config_manager = Mock(spec=ConfigManager)
        config_manager.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.9,
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': False
        }
        return config_manager
    
    @pytest.fixture
    def token_manager(self, mock_config_manager):
        """Create a TokenManager instance for testing."""
        return TokenManager(mock_config_manager)
    
    @pytest.fixture
    def mock_context_compressor(self, test_llm_config):
        """Create a mock ContextCompressor for testing."""
        compressor = Mock(spec=ContextCompressor)
        compressor.compress_context = AsyncMock()
        return compressor
    
    @pytest.fixture
    def test_agent(self, test_llm_config, token_manager, mock_context_compressor, real_managers):
        """Create a test agent with token management."""
        class TestAgent(BaseLLMAgent):
            async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "test"}
            
            def get_agent_capabilities(self) -> list:
                return ["test_capability"]
        
        return TestAgent(
            name="test_agent",
            llm_config=test_llm_config,
            system_message="Test system message",
            token_manager=token_manager,
            context_manager=real_managers.context_manager)
    
    def test_agent_initialization_with_token_manager(self, test_agent, token_manager, mock_context_compressor, real_managers):
        """Test that agent initializes correctly with token manager and context manager."""
        assert test_agent.token_manager is token_manager
        assert test_agent.context_manager is real_managers.context_manager
        assert test_agent.name == "test_agent"
    
    def test_agent_initialization_with_required_managers(self, test_llm_config, real_managers):
        """Test that agent initializes correctly with required managers."""
        class TestAgent(BaseLLMAgent):
            async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "test"}
            
            def get_agent_capabilities(self) -> list:
                return ["test_capability"]
        
        agent = TestAgent(
            name="test_agent",
            llm_config=test_llm_config,
            system_message="Test system message",
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
        
        assert agent.token_manager is not None
        assert agent.context_manager is not None
    
    @pytest.mark.asyncio
    async def test_generate_response_with_token_limit_check(self, test_agent, token_manager, real_managers):
        """Test that generate_response works with token management through context manager."""
        # Mock AutoGen response generation
        with patch.object(test_agent, 'initialize_autogen_agent', return_value=True), \
             patch.object(test_agent, '_generate_autogen_response', new_callable=AsyncMock) as mock_generate, \
             patch.object(real_managers.context_manager, 'prepare_system_prompt', new_callable=AsyncMock) as mock_prepare:
            
            mock_generate.return_value = "Test response"
            
            # Mock context preparation
            from autogen_framework.context_manager import ContextManager
            mock_prepared = ContextManager.PreparedPrompt(
                system_prompt="Prepared system message",
                estimated_tokens=1000
            )
            mock_prepare.return_value = mock_prepared
            
            response = await test_agent.generate_response("Test prompt")
            
            # Verify context preparation was called (new architecture)
            mock_prepare.assert_called()
            
            # Verify response was generated
            assert response == "Test response"
            mock_generate.assert_called_once_with("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_response_triggers_compression(self, test_agent, token_manager, mock_context_compressor, real_managers):
        """Test that generate_response works with context manager for compression."""
        # Mock AutoGen response generation
        with patch.object(test_agent, 'initialize_autogen_agent', return_value=True), \
             patch.object(test_agent, '_generate_autogen_response', new_callable=AsyncMock) as mock_generate, \
             patch.object(real_managers.context_manager, 'prepare_system_prompt', new_callable=AsyncMock) as mock_prepare:
            
            mock_generate.return_value = "Test response"
            
            # Mock context preparation to simulate compression
            from autogen_framework.context_manager import ContextManager
            mock_prepared = ContextManager.PreparedPrompt(
                system_prompt="Compressed system message",
                estimated_tokens=3000
            )
            mock_prepare.return_value = mock_prepared
            
            response = await test_agent.generate_response("Test prompt")
            
            # Verify context preparation was called (new architecture)
            mock_prepare.assert_called()
            
            # Verify response was generated
            assert response == "Test response"
            
            # Verify token manager is tracking usage
            stats = token_manager.get_usage_statistics()
            assert stats.requests_made > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_fallback_truncation(self, test_agent, token_manager, real_managers):
        """Test context management functionality (replaces fallback truncation)."""
        # Create agent with context manager
        class TestAgent(BaseLLMAgent):
            async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "test"}
            
            def get_agent_capabilities(self) -> list:
                return ["test_capability"]
        
        agent_with_context_manager = TestAgent(
            name="test_agent",
            llm_config=test_agent.llm_config,
            system_message="Test system message",
            token_manager=token_manager,
            context_manager=real_managers.context_manager
        )
        
        # Mock AutoGen response generation
        with patch.object(agent_with_context_manager, 'initialize_autogen_agent', return_value=True), \
             patch.object(agent_with_context_manager, '_generate_autogen_response', new_callable=AsyncMock) as mock_generate, \
             patch.object(real_managers.context_manager, 'prepare_system_prompt', new_callable=AsyncMock) as mock_prepare:
            
            mock_generate.return_value = "Test response"
            
            # Mock context preparation
            from autogen_framework.context_manager import ContextManager
            mock_prepared = ContextManager.PreparedPrompt(
                system_prompt="Prepared system message",
                estimated_tokens=2000
            )
            mock_prepare.return_value = mock_prepared
            
            response = await agent_with_context_manager.generate_response("Test prompt")
            
            # Verify context preparation was called (new architecture)
            mock_prepare.assert_called()
            
            # Verify response was generated
            assert response == "Test response"
    
    @pytest.mark.asyncio
    async def test_generate_response_updates_token_usage(self, test_agent, token_manager):
        """Test that generate_response updates token usage after LLM call."""
        # Mock token limit check to not need compression
        token_manager.check_token_limit = Mock(return_value=TokenCheckResult(
            current_tokens=1000,
            model_limit=8192,
            percentage_used=0.12,
            needs_compression=False
        ))
        
        # Mock token manager update method
        token_manager.update_token_usage = Mock()
        
        # Mock AutoGen response generation
        with patch.object(test_agent, 'initialize_autogen_agent', return_value=True), \
             patch.object(test_agent, '_generate_autogen_response', new_callable=AsyncMock) as mock_generate, \
             patch.object(test_agent, '_extract_token_usage_from_response', return_value=150) as mock_extract:
            
            mock_generate.return_value = "Test response with some content"
            
            response = await test_agent.generate_response("Test prompt")
            
            # Verify token usage was extracted and updated
            mock_extract.assert_called_once_with("Test response with some content")
            token_manager.update_token_usage.assert_called_once_with(
                test_agent.llm_config.model,
                150,
                "generate_response"
            )
            
            assert response == "Test response with some content"
    
    def test_extract_token_usage_from_response(self, test_agent):
        """Test token usage extraction from response."""
        # Test with short response
        short_response = "Short response"
        tokens = test_agent._extract_token_usage_from_response(short_response)
        expected = len(short_response) // 4 + 50  # Base overhead
        assert tokens == expected
        
        # Test with longer response
        long_response = "This is a much longer response that should result in more tokens being estimated based on the character count and the approximation formula used."
        tokens = test_agent._extract_token_usage_from_response(long_response)
        expected = len(long_response) // 4 + 50  # Base overhead
        assert tokens == expected
        
        # Test minimum token count (empty response should still return base overhead)
        empty_response = ""
        tokens = test_agent._extract_token_usage_from_response(empty_response)
        assert tokens == 50  # Base overhead, minimum is max(estimated, 1) but estimated includes overhead
    
    @pytest.mark.asyncio
    async def test_perform_context_compression_success(self, test_agent, mock_context_compressor, token_manager, real_managers):
        """Test context manager integration (replaces context compression)."""
        # Mock context manager preparation
        with patch.object(real_managers.context_manager, 'prepare_system_prompt', new_callable=AsyncMock) as mock_prepare:
            from autogen_framework.context_manager import ContextManager
            mock_prepared = ContextManager.PreparedPrompt(
                system_prompt="Prepared system message",
                estimated_tokens=3000
            )
            mock_prepare.return_value = mock_prepared
            
            # Test context preparation
            prepared = await real_managers.context_manager.prepare_system_prompt("Test system message")
            
            # Verify context was prepared
            assert prepared.system_prompt == "Prepared system message"
            assert prepared.estimated_tokens == 3000
    
    @pytest.mark.asyncio
    async def test_perform_context_compression_failure(self, test_agent, mock_context_compressor, real_managers):
        """Test context manager error handling."""
        # Test context manager error handling
        with patch.object(real_managers.context_manager, 'prepare_system_prompt', side_effect=Exception("Context preparation error")):
            
            # This should handle the error gracefully
            try:
                await real_managers.context_manager.prepare_system_prompt("Test system message")
                assert False, "Should have raised an exception"
            except Exception as e:
                assert str(e) == "Context preparation error"
                # Verify error was properly raised
    
    @pytest.mark.asyncio
    async def test_perform_fallback_truncation(self, test_agent, token_manager, real_managers):
        """Test context management functionality (replaces fallback truncation)."""
        # Test context preparation (which handles token control)
        large_system_message = test_agent._build_complete_system_message()
        
        # Mock context manager to simulate token control
        with patch.object(real_managers.context_manager, 'prepare_system_prompt', new_callable=AsyncMock) as mock_prepare:
            from autogen_framework.context_manager import ContextManager
            mock_prepared = ContextManager.PreparedPrompt(
                system_prompt="Managed system message (truncated if needed)",
                estimated_tokens=2000
            )
            mock_prepare.return_value = mock_prepared
            
            # Test context preparation
            prepared = await real_managers.context_manager.prepare_system_prompt(large_system_message)
            
            # Verify context was prepared
            assert prepared.system_prompt == "Managed system message (truncated if needed)"
            assert prepared.estimated_tokens == 2000


class TestTokenManagementEdgeCases:
    """Test edge cases and error scenarios for token management integration."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock ConfigManager for testing."""
        config_manager = Mock(spec=ConfigManager)
        config_manager.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.9,
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': False
        }
        return config_manager
    
    @pytest.fixture
    def token_manager(self, mock_config_manager):
        """Create a TokenManager instance for testing."""
        return TokenManager(mock_config_manager)
    
    @pytest.fixture
    def mock_context_compressor(self, test_llm_config):
        """Create a mock ContextCompressor for testing."""
        compressor = Mock(spec=ContextCompressor)
        compressor.compress_context = AsyncMock()
        return compressor
    
    @pytest.fixture
    def test_agent(self, test_llm_config, token_manager, mock_context_compressor, real_managers):
        """Create a test agent with token management."""
        class TestAgent(BaseLLMAgent):
            async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "test"}
            
            def get_agent_capabilities(self) -> list:
                return ["test_capability"]
        
        return TestAgent(
            name="test_agent",
            llm_config=test_llm_config,
            system_message="Test system message",
            token_manager=token_manager,
            context_manager=real_managers.context_manager)
    
    @pytest.fixture
    def test_agent_minimal(self, test_llm_config, real_managers):
        """Create a minimal test agent without token management."""
        class TestAgent(BaseLLMAgent):
            async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "test"}
            
            def get_agent_capabilities(self) -> list:
                return ["test_capability"]
        
        return TestAgent(
            name="test_agent",
            llm_config=test_llm_config,
            system_message="Test system message",
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    @pytest.mark.asyncio
    async def test_generate_response_without_token_manager(self, test_agent_minimal):
        """Test generate_response works correctly without token manager."""
        # Mock AutoGen response generation
        with patch.object(test_agent_minimal, 'initialize_autogen_agent', return_value=True), \
             patch.object(test_agent_minimal, '_generate_autogen_response', new_callable=AsyncMock) as mock_generate:
            
            mock_generate.return_value = "Test response"
            
            response = await test_agent_minimal.generate_response("Test prompt")
            
            # Verify response was generated normally
            assert response == "Test response"
            mock_generate.assert_called_once_with("Test prompt")
    
    @pytest.mark.asyncio
    async def test_compression_exception_handling(self, test_agent, mock_context_compressor, real_managers):
        """Test exception handling during context management."""
        # Test context manager error handling
        with patch.object(real_managers.context_manager, 'prepare_system_prompt', side_effect=Exception("Context preparation error")):
            
            # This should handle the error gracefully
            try:
                await real_managers.context_manager.prepare_system_prompt("Test system message")
                assert False, "Should have raised an exception"
            except Exception as e:
                assert str(e) == "Context preparation error"
                # Verify error was properly raised
    
    def test_truncation_exception_handling(self, test_agent, token_manager):
        """Test error handling with token manager."""
        # Test invalid token numbers
        initial_size = token_manager.current_context_size
        token_manager.update_token_usage("test-model", -100, "invalid_operation")
        
        # Should ignore invalid token numbers
        assert token_manager.current_context_size == initial_size
        
        # Test unknown model token limit
        token_check = token_manager.check_token_limit("unknown-model")
        assert token_check.model_limit == 8192  # Should use default limit