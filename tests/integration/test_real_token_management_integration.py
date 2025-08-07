"""
Real integration tests for token management in BaseLLMAgent.

Tests the actual integration of real TokenManager and ContextCompressor with BaseLLMAgent
using real instances instead of mocks to verify the complete integration works.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from typing import Dict, Any

from autogen_framework.agents.base_agent import BaseLLMAgent
from autogen_framework.token_manager import TokenManager
from autogen_framework.context_compressor import ContextCompressor
from autogen_framework.models import LLMConfig
from autogen_framework.config_manager import ConfigManager


class RealTestAgent(BaseLLMAgent):
    """Real test agent for integration testing."""
    
    async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": "Real integration test completed"}
    
    def get_agent_capabilities(self) -> list:
        return ["real_integration_test", "token_management", "context_compression"]


@pytest.mark.integration
class TestRealTokenManagementIntegration:
    """Test real token management integration with actual components."""
    
    @pytest.fixture
    def real_config_manager(self):
        """Create a real ConfigManager for testing."""
        config_manager = ConfigManager()
        # Override with test configuration
        config_manager._config = {
            'llm': {
                'base_url': 'http://test.local:8888/openai/v1',
                'model': 'test-model',
                'api_key': 'test-key',
                'temperature': 0.7,
                'max_output_tokens': 8192,
                'timeout': 60
            },
            'token_management': {
                'default_token_limit': 8192,
                'compression_threshold': 0.9,
                'compression_enabled': True,
                'compression_target_ratio': 0.5,
                'verbose_logging': False
            }
        }
        return config_manager
    
    @pytest.fixture
    def real_token_manager(self, real_config_manager):
        """Create a real TokenManager instance."""
        return TokenManager(real_config_manager)
    
    @pytest.fixture
    def real_context_compressor(self, real_llm_config):
        """Create a real ContextCompressor instance."""
        return ContextCompressor(real_llm_config)
    
    @pytest.fixture
    def real_agent(self, real_llm_config, real_token_manager, real_context_compressor):
        """Create a real agent with real token management components."""
        return RealTestAgent(
            name="real_test_agent",
            llm_config=real_llm_config,
            system_message="This is a real integration test agent with comprehensive context for testing token management and compression capabilities.",
            token_manager=real_token_manager,
            context_compressor=real_context_compressor
        )
    
    def test_real_agent_initialization(self, real_agent, real_token_manager, real_context_compressor):
        """Test that real agent initializes correctly with real components."""
        assert real_agent.token_manager is real_token_manager
        assert real_agent.context_compressor is real_context_compressor
        assert isinstance(real_agent.token_manager, TokenManager)
        assert isinstance(real_agent.context_compressor, ContextCompressor)
        assert real_agent.name == "real_test_agent"
    
    def test_real_token_manager_functionality(self, real_token_manager):
        """Test real TokenManager functionality."""
        # Test token usage tracking
        real_token_manager.update_token_usage("test-model", 1000, "test_operation")
        assert real_token_manager.current_context_size == 1000
        
        # Test token limit checking
        token_check = real_token_manager.check_token_limit("test-model")
        assert token_check.current_tokens == 1000
        assert token_check.model_limit == 8192
        assert not token_check.needs_compression  # Should be under threshold
        
        # Test usage statistics
        stats = real_token_manager.get_usage_statistics()
        assert stats.total_tokens_used == 1000
        assert stats.requests_made == 1
        assert stats.compressions_performed == 0
    
    def test_real_token_limit_threshold(self, real_token_manager):
        """Test real token limit threshold detection."""
        # Simulate high token usage (over 90% threshold)
        real_token_manager.current_context_size = 7500  # 91.6% of 8192
        
        token_check = real_token_manager.check_token_limit("test-model")
        assert token_check.current_tokens == 7500
        assert token_check.percentage_used > 0.9
        assert token_check.needs_compression  # Should trigger compression
    
    @pytest.mark.asyncio
    async def test_real_context_compression(self, real_context_compressor):
        """Test real ContextCompressor functionality."""
        # Create test context to compress
        test_context = {
            'system_message': 'This is a comprehensive system message with lots of detailed information about the agent capabilities and instructions.',
            'conversation_history': [
                {'role': 'user', 'content': 'First user message with detailed context'},
                {'role': 'assistant', 'content': 'Detailed assistant response with comprehensive information'},
                {'role': 'user', 'content': 'Follow-up user message with additional context'},
                {'role': 'assistant', 'content': 'Another detailed assistant response with more information'}
            ],
            'memory_context': {
                'project_info': 'Detailed project information with specifications and requirements',
                'user_preferences': 'User preferences and configuration settings',
                'workflow_state': 'Current workflow state and progress information'
            }
        }
        
        # Mock the LLM call since we don't have a real LLM endpoint
        with patch.object(real_context_compressor, '_generate_autogen_response', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Compressed context: System message with agent capabilities. Conversation includes user queries and assistant responses. Memory contains project info, user preferences, and workflow state."
            
            # Test compression
            result = await real_context_compressor.compress_context(test_context, target_reduction=0.5)
            
            # Verify compression result
            assert result.success
            assert result.original_size > 0
            assert result.compressed_size > 0
            assert result.compression_ratio > 0
            assert result.method_used == "llm_compression"
            assert "Compressed context" in result.compressed_content
    
    @pytest.mark.asyncio
    async def test_real_agent_token_integration(self, real_agent, real_token_manager):
        """Test real agent integration with token management."""
        # Mock AutoGen to avoid real LLM calls
        with patch.object(real_agent, 'initialize_autogen_agent', return_value=True), \
             patch.object(real_agent, '_generate_autogen_response', new_callable=AsyncMock) as mock_generate:
            
            mock_generate.return_value = "This is a test response from the agent with token management integration."
            
            # Test normal response generation
            response = await real_agent.generate_response("Test prompt for token management")
            
            # Verify response
            assert response == "This is a test response from the agent with token management integration."
            
            # Verify token usage was tracked
            assert real_token_manager.current_context_size > 0
            stats = real_token_manager.get_usage_statistics()
            assert stats.requests_made > 0
            assert stats.total_tokens_used > 0
    
    @pytest.mark.asyncio
    async def test_real_agent_compression_trigger(self, real_agent, real_token_manager, real_context_compressor):
        """Test real agent compression trigger with real components."""
        # Set high token usage to trigger compression
        real_token_manager.current_context_size = 7500  # Over threshold
        initial_compressions = real_token_manager.usage_stats['compressions_performed']
        
        # Mock token limit check to return needs_compression=True
        from autogen_framework.token_manager import TokenCheckResult
        mock_token_check = TokenCheckResult(
            current_tokens=7500,
            model_limit=8192,
            percentage_used=0.916,
            needs_compression=True
        )
        
        # Mock AutoGen and compression calls
        with patch.object(real_agent, 'initialize_autogen_agent', return_value=True), \
             patch.object(real_agent, '_generate_autogen_response', new_callable=AsyncMock) as mock_generate, \
             patch.object(real_agent, '_perform_context_compression', new_callable=AsyncMock) as mock_compress, \
             patch.object(real_token_manager, 'check_token_limit', return_value=mock_token_check):
            
            mock_generate.return_value = "Response after compression trigger"
            
            # Mock the compression to simulate successful compression
            async def mock_compression():
                # Simulate what real compression does
                real_token_manager.reset_context_size()
                real_token_manager.increment_compression_count()
                real_agent.system_message = "Compressed system message"
            
            mock_compress.side_effect = mock_compression
            
            # This should trigger compression due to high token usage
            response = await real_agent.generate_response("Test prompt that should trigger compression")
            
            # Verify response was generated
            assert response == "Response after compression trigger"
            
            # Verify compression was called
            mock_compress.assert_called_once()
            
            # Verify compression was triggered by checking compression count increased
            stats = real_token_manager.get_usage_statistics()
            assert stats.compressions_performed > initial_compressions
            
            # The context size should be reset and then have new token usage added
            # It should be much less than the original 7500
            assert real_token_manager.current_context_size < 1000  # Should be just the new response tokens
    
    def test_real_token_extraction(self, real_agent):
        """Test real token extraction from responses."""
        # Test with various response lengths
        test_cases = [
            ("Short", 50 + len("Short") // 4),  # Base overhead + estimated tokens
            ("Medium length response with more content", 50 + len("Medium length response with more content") // 4),
            ("Very long response with extensive content that should result in a higher token count estimation based on the character count approximation method used by the system", 50 + len("Very long response with extensive content that should result in a higher token count estimation based on the character count approximation method used by the system") // 4)
        ]
        
        for response_text, expected_tokens in test_cases:
            actual_tokens = real_agent._extract_token_usage_from_response(response_text)
            assert actual_tokens == expected_tokens
    
    def test_real_fallback_truncation(self, real_agent, real_token_manager):
        """Test real fallback truncation functionality."""
        # Add some context to the agent
        real_agent.context = {
            'large_context': 'This is a large context that should be truncated when fallback truncation is applied. ' * 100,
            'important_info': 'Critical information that should be preserved',
            'workflow_state': 'Current workflow state information'
        }
        
        # Mock AutoGen initialization
        with patch.object(real_agent, 'initialize_autogen_agent', return_value=True):
            # Test fallback truncation
            real_agent._perform_fallback_truncation()
            
            # Verify system message was updated (truncated)
            assert len(real_agent.system_message) > 0
            
            # Verify token manager context was reset
            assert real_token_manager.current_context_size == 0
    
    @pytest.mark.asyncio
    async def test_real_error_handling(self, real_agent, real_context_compressor):
        """Test real error handling in integration scenarios."""
        # Test compression failure handling
        with patch.object(real_context_compressor, 'compress_context', side_effect=Exception("Real compression error")), \
             patch.object(real_agent, '_perform_fallback_truncation') as mock_fallback:
            
            # This should handle the compression error gracefully
            await real_agent._perform_context_compression()
            
            # Verify fallback was called
            mock_fallback.assert_called_once()
    
    def test_real_component_compatibility(self, real_agent, real_token_manager, real_context_compressor):
        """Test that real components work together correctly."""
        # Verify all components are properly connected
        assert real_agent.token_manager is real_token_manager
        assert real_agent.context_compressor is real_context_compressor
        
        # Test token manager configuration
        assert real_token_manager.token_config['compression_threshold'] == 0.9
        assert real_token_manager.token_config['compression_enabled'] is True
        
        # Test context compressor configuration
        assert real_context_compressor.llm_config.model == real_agent.llm_config.model
        assert real_context_compressor.name == "context_compressor"
    
    def test_real_usage_statistics_accuracy(self, real_token_manager):
        """Test accuracy of real usage statistics."""
        # Simulate multiple operations
        operations = [
            ("operation1", 500),
            ("operation2", 750),
            ("operation3", 1200),
            ("compression", 0)  # Compression resets context
        ]
        
        for operation, tokens in operations:
            if operation == "compression":
                real_token_manager.increment_compression_count()
                real_token_manager.reset_context_size()
            else:
                real_token_manager.update_token_usage("test-model", tokens, operation)
        
        # Verify statistics
        stats = real_token_manager.get_usage_statistics()
        assert stats.total_tokens_used == 2450  # Sum of all tokens
        assert stats.requests_made == 3  # Three non-compression operations
        assert stats.compressions_performed == 1
        assert stats.average_tokens_per_request == 2450 / 3
        
        # Verify detailed report
        report = real_token_manager.get_detailed_usage_report()
        assert report['current_context_size'] == 0  # Reset after compression
        assert len(report['usage_history']) > 0
        assert report['configuration']['compression_threshold'] == 0.9