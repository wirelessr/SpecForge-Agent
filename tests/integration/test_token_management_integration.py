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
    def test_agent(self, test_llm_config, token_manager, mock_context_compressor):
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
            context_compressor=mock_context_compressor
        )
    
    def test_agent_initialization_with_token_manager(self, test_agent, token_manager, mock_context_compressor):
        """Test that agent initializes correctly with token manager and context compressor."""
        assert test_agent.token_manager is token_manager
        assert test_agent.context_compressor is mock_context_compressor
        assert test_agent.name == "test_agent"
    
    def test_agent_initialization_without_token_manager(self, test_llm_config):
        """Test that agent initializes correctly without token manager."""
        class TestAgent(BaseLLMAgent):
            async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "test"}
            
            def get_agent_capabilities(self) -> list:
                return ["test_capability"]
        
        agent = TestAgent(
            name="test_agent",
            llm_config=test_llm_config,
            system_message="Test system message"
        )
        
        assert agent.token_manager is None
        assert agent.context_compressor is None
    
    @pytest.mark.asyncio
    async def test_generate_response_with_token_limit_check(self, test_agent, token_manager):
        """Test that generate_response checks token limits before LLM requests."""
        # Mock token limit check to not need compression
        token_manager.check_token_limit = Mock(return_value=TokenCheckResult(
            current_tokens=1000,
            model_limit=8192,
            percentage_used=0.12,
            needs_compression=False
        ))
        
        # Mock AutoGen response generation
        with patch.object(test_agent, 'initialize_autogen_agent', return_value=True), \
             patch.object(test_agent, '_generate_autogen_response', new_callable=AsyncMock) as mock_generate:
            
            mock_generate.return_value = "Test response"
            
            response = await test_agent.generate_response("Test prompt")
            
            # Verify token limit was checked (may include estimated_static_tokens parameter)
            token_manager.check_token_limit.assert_called_once()
            # Check that the first argument is the model
            call_args = token_manager.check_token_limit.call_args
            assert call_args[0][0] == test_agent.llm_config.model
            
            # Verify response was generated
            assert response == "Test response"
            mock_generate.assert_called_once_with("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_response_triggers_compression(self, test_agent, token_manager, mock_context_compressor):
        """Test that generate_response triggers compression when token limit is exceeded."""
        # Mock token limit check to need compression
        token_manager.check_token_limit = Mock(return_value=TokenCheckResult(
            current_tokens=7500,
            model_limit=8192,
            percentage_used=0.92,
            needs_compression=True
        ))
        
        # Mock token manager methods
        token_manager.reset_context_size = Mock()
        token_manager.increment_compression_count = Mock()
        
        # Mock successful compression
        compression_result = CompressionResult(
            original_size=10000,
            compressed_size=5000,
            compression_ratio=0.5,
            compressed_content="Compressed content",
            method_used="llm_compression",
            success=True
        )
        mock_context_compressor.compress_context.return_value = compression_result
        
        # Mock AutoGen response generation
        with patch.object(test_agent, 'initialize_autogen_agent', return_value=True), \
             patch.object(test_agent, '_generate_autogen_response', new_callable=AsyncMock) as mock_generate:
            
            mock_generate.return_value = "Test response"
            
            response = await test_agent.generate_response("Test prompt")
            
            # Verify token limit was checked (may include estimated_static_tokens parameter)
            token_manager.check_token_limit.assert_called_once()
            # Check that the first argument is the model
            call_args = token_manager.check_token_limit.call_args
            assert call_args[0][0] == test_agent.llm_config.model
            
            # Verify compression was triggered
            mock_context_compressor.compress_context.assert_called_once()
            
            # Verify token manager methods were called
            token_manager.reset_context_size.assert_called_once()
            token_manager.increment_compression_count.assert_called_once()
            
            # Verify response was generated
            assert response == "Test response"
    
    @pytest.mark.asyncio
    async def test_generate_response_fallback_truncation(self, test_agent, token_manager):
        """Test fallback truncation when no context compressor is available."""
        # Create agent without context compressor
        class TestAgent(BaseLLMAgent):
            async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "test"}
            
            def get_agent_capabilities(self) -> list:
                return ["test_capability"]
        
        agent_without_compressor = TestAgent(
            name="test_agent",
            llm_config=test_agent.llm_config,
            system_message="Test system message",
            token_manager=token_manager
        )
        
        # Mock token limit check to need compression
        token_manager.check_token_limit = Mock(return_value=TokenCheckResult(
            current_tokens=7500,
            model_limit=8192,
            percentage_used=0.92,
            needs_compression=True
        ))
        
        # Mock AutoGen response generation
        with patch.object(agent_without_compressor, 'initialize_autogen_agent', return_value=True), \
             patch.object(agent_without_compressor, '_generate_autogen_response', new_callable=AsyncMock) as mock_generate, \
             patch.object(agent_without_compressor, '_perform_fallback_truncation') as mock_truncate:
            
            mock_generate.return_value = "Test response"
            
            response = await agent_without_compressor.generate_response("Test prompt")
            
            # Verify fallback truncation was called
            mock_truncate.assert_called_once()
            
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
    async def test_perform_context_compression_success(self, test_agent, mock_context_compressor, token_manager):
        """Test successful context compression."""
        # Mock token manager methods
        token_manager.reset_context_size = Mock()
        token_manager.increment_compression_count = Mock()
        
        # Mock successful compression
        compression_result = CompressionResult(
            original_size=10000,
            compressed_size=5000,
            compression_ratio=0.5,
            compressed_content="Compressed system message",
            method_used="llm_compression",
            success=True
        )
        mock_context_compressor.compress_context.return_value = compression_result
        
        # Mock AutoGen initialization
        with patch.object(test_agent, 'initialize_autogen_agent', return_value=True):
            await test_agent._perform_context_compression()
            
            # Verify compression was called
            mock_context_compressor.compress_context.assert_called_once()
            
            # Verify system message was updated
            assert test_agent.system_message == "Compressed system message"
            
            # Verify token manager methods were called
            token_manager.reset_context_size.assert_called_once()
            token_manager.increment_compression_count.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_perform_context_compression_failure(self, test_agent, mock_context_compressor):
        """Test context compression failure with fallback."""
        # Mock failed compression
        compression_result = CompressionResult(
            original_size=10000,
            compressed_size=10000,
            compression_ratio=0.0,
            compressed_content="Original content",
            method_used="compression_failed",
            success=False,
            error="Compression failed"
        )
        mock_context_compressor.compress_context.return_value = compression_result
        
        # Mock fallback truncation
        with patch.object(test_agent, '_perform_fallback_truncation') as mock_truncate:
            await test_agent._perform_context_compression()
            
            # Verify compression was attempted
            mock_context_compressor.compress_context.assert_called_once()
            
            # Verify fallback truncation was called
            mock_truncate.assert_called_once()
    
    def test_perform_fallback_truncation(self, test_agent, token_manager):
        """Test fallback truncation functionality."""
        # Mock token manager methods
        token_manager.get_model_limit = Mock(return_value=8192)
        token_manager.reset_context_size = Mock()
        
        # Mock truncate_context method
        with patch.object(test_agent, 'truncate_context') as mock_truncate, \
             patch.object(test_agent, 'initialize_autogen_agent', return_value=True):
            
            mock_truncate.return_value = {'truncated_content': 'Truncated system message'}
            
            test_agent._perform_fallback_truncation()
            
            # Verify truncation was called with correct parameters
            mock_truncate.assert_called_once_with(max_tokens=4096)  # 50% of model limit
            
            # Verify system message was updated
            assert test_agent.system_message == 'Truncated system message'
            
            # Verify token manager reset was called
            token_manager.reset_context_size.assert_called_once()


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
    def test_agent(self, test_llm_config, token_manager, mock_context_compressor):
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
            context_compressor=mock_context_compressor
        )
    
    @pytest.fixture
    def test_agent_minimal(self, test_llm_config):
        """Create a minimal test agent without token management."""
        class TestAgent(BaseLLMAgent):
            async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "test"}
            
            def get_agent_capabilities(self) -> list:
                return ["test_capability"]
        
        return TestAgent(
            name="test_agent",
            llm_config=test_llm_config,
            system_message="Test system message"
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
    async def test_compression_exception_handling(self, test_agent, mock_context_compressor):
        """Test exception handling during compression."""
        # Mock compression to raise exception
        mock_context_compressor.compress_context.side_effect = Exception("Compression error")
        
        # Mock fallback truncation
        with patch.object(test_agent, '_perform_fallback_truncation') as mock_truncate:
            await test_agent._perform_context_compression()
            
            # Verify fallback truncation was called due to exception
            mock_truncate.assert_called_once()
    
    def test_truncation_exception_handling(self, test_agent, token_manager):
        """Test exception handling during fallback truncation."""
        # Mock token manager to raise exception
        with patch.object(token_manager, 'get_model_limit', side_effect=Exception("Token manager error")):
            # Should not raise exception, just log error
            test_agent._perform_fallback_truncation()
            
            # Test passes if no exception is raised
            assert True