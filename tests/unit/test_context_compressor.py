"""
Unit tests for the ContextCompressor class and BaseLLMAgent compression functionality.

Tests the context compression functionality including LLM-based compression,
fallback truncation strategies, statistics tracking, and error handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json

from autogen_framework.context_compressor import ContextCompressor
from autogen_framework.agents.base_agent import BaseLLMAgent
from autogen_framework.models import LLMConfig, CompressionResult
from autogen_framework.config_manager import ConfigManager


class TestContextCompressor:
    """Test cases for ContextCompressor class."""
    
    @pytest.fixture
    def test_llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            base_url="http://test.local/v1",
            model="test-model",
            api_key="test-key",
            temperature=0.7,
            max_output_tokens=4096,
            timeout=30
        )
    
    @pytest.fixture
    def mock_token_manager(self):
        """Create mock TokenManager."""
        mock_manager = Mock()
        mock_manager.increment_compression_count = Mock()
        return mock_manager
    
    @pytest.fixture
    def context_compressor(self, test_llm_config, mock_token_manager):
        """Create ContextCompressor instance for testing."""
        return ContextCompressor(test_llm_config, mock_token_manager)
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context for testing."""
        return {
            "system_message": "You are a helpful AI assistant for software development.",
            "workflow_state": {
                "phase": "implementation",
                "current_task": "Create unit tests",
                "progress": "50%"
            },
            "conversation_history": [
                {"role": "user", "content": "Please create unit tests for the new feature"},
                {"role": "assistant", "content": "I'll create comprehensive unit tests..."}
            ],
            "memory_context": {
                "project_info": "Multi-agent framework for AutoGen",
                "recent_decisions": "Using pytest for testing framework"
            }
        }
    
    def test_initialization(self, test_llm_config, mock_token_manager):
        """Test ContextCompressor initialization."""
        compressor = ContextCompressor(test_llm_config, mock_token_manager)
        
        assert compressor.name == "context_compressor"
        assert compressor.llm_config == test_llm_config
        assert compressor.token_manager == mock_token_manager
        assert compressor.description == "Specialized agent for intelligent context compression"
        assert len(compressor.compression_history) == 0
        assert compressor.compression_stats['total_compressions'] == 0
    
    def test_initialization_without_token_manager(self, test_llm_config):
        """Test ContextCompressor initialization without token manager."""
        compressor = ContextCompressor(test_llm_config)
        
        assert compressor.token_manager is None
        assert compressor.name == "context_compressor"
    
    def test_build_compression_system_message(self, context_compressor):
        """Test compression system message building."""
        system_message = context_compressor._build_compression_system_message()
        
        assert isinstance(system_message, str)
        assert len(system_message) > 0
        assert "context compression specialist" in system_message.lower()
        assert "critical information" in system_message.lower()
        assert "workflow state" in system_message.lower()
        assert "user requirements" in system_message.lower()
    
    def test_get_agent_capabilities(self, context_compressor):
        """Test getting agent capabilities."""
        capabilities = context_compressor.get_agent_capabilities()
        
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        assert "Intelligent context compression using LLM" in capabilities
        assert "Batch compression operations" in capabilities
    
    @pytest.mark.asyncio
    async def test_process_task(self, context_compressor, sample_context):
        """Test processing a compression task."""
        task_input = {
            'context': sample_context,
            'target_reduction': 0.4
        }
        
        # Mock the compress_context method
        mock_result = CompressionResult(
            original_size=1000,
            compressed_size=600,
            compression_ratio=0.4,
            compressed_content="Compressed content",
            method_used="llm_compression",
            success=True
        )
        
        with patch.object(context_compressor, 'compress_context', return_value=mock_result):
            result = await context_compressor.process_task(task_input)
            
            assert result['success'] is True
            assert result['original_size'] == 1000
            assert result['compressed_size'] == 600
            assert result['compression_ratio'] == 0.4
            assert result['method_used'] == "llm_compression"
            assert result['error'] is None
    
    @pytest.mark.asyncio
    async def test_compress_multiple_contexts(self, context_compressor, sample_context):
        """Test compressing multiple contexts in batch."""
        contexts = [sample_context, sample_context.copy()]
        
        # Mock the compress_context method
        mock_result = CompressionResult(
            original_size=1000,
            compressed_size=600,
            compression_ratio=0.4,
            compressed_content="Compressed content",
            method_used="llm_compression",
            success=True
        )
        
        with patch.object(context_compressor, 'compress_context', return_value=mock_result):
            results = await context_compressor.compress_multiple_contexts(contexts)
            
            assert len(results) == 2
            assert all(result.success for result in results)
            assert all(result.compression_ratio == 0.4 for result in results)
    
    @pytest.mark.asyncio
    async def test_compress_agent_contexts(self, context_compressor, test_llm_config):
        """Test compressing contexts from multiple agents."""
        # Create mock agents
        class MockAgent(BaseLLMAgent):
            async def process_task(self, task_input):
                return {"result": "test"}
            
            def get_agent_capabilities(self):
                return ["test capability"]
        
        agent1 = MockAgent("agent1", test_llm_config, "System message 1")
        agent2 = MockAgent("agent2", test_llm_config, "System message 2")
        
        # Mock the compress_context method
        mock_result = CompressionResult(
            original_size=1000,
            compressed_size=600,
            compression_ratio=0.4,
            compressed_content="Compressed content",
            method_used="llm_compression",
            success=True
        )
        
        with patch.object(context_compressor, 'compress_context', return_value=mock_result):
            results = await context_compressor.compress_agent_contexts([agent1, agent2])
            
            assert len(results) == 2
            assert "agent1" in results
            assert "agent2" in results
            assert all(result.success for result in results.values())


class TestBaseLLMAgentCompression:
    """Test cases for BaseLLMAgent compression functionality."""
    
    @pytest.fixture
    def test_llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            base_url="http://test.local/v1",
            model="test-model",
            api_key="test-key",
            temperature=0.7,
            max_output_tokens=4096,
            timeout=30
        )
    
    @pytest.fixture
    def mock_agent(self, test_llm_config):
        """Create mock BaseLLMAgent for testing."""
        class TestAgent(BaseLLMAgent):
            async def process_task(self, task_input):
                return {"result": "test"}
            
            def get_agent_capabilities(self):
                return ["test capability"]
        
        return TestAgent("test_agent", test_llm_config, "Test system message")
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context for testing."""
        return {
            "system_message": "You are a helpful AI assistant for software development.",
            "workflow_state": {
                "phase": "implementation",
                "current_task": "Create unit tests",
                "progress": "50%"
            },
            "conversation_history": [
                {"role": "user", "content": "Please create unit tests for the new feature"},
                {"role": "assistant", "content": "I'll create comprehensive unit tests..."}
            ],
            "memory_context": {
                "project_info": "Multi-agent framework for AutoGen",
                "recent_decisions": "Using pytest for testing framework"
            }
        }
    
    def test_compression_stats_initialization(self, mock_agent):
        """Test that compression stats are initialized in BaseLLMAgent."""
        assert hasattr(mock_agent, 'compression_history')
        assert hasattr(mock_agent, 'compression_stats')
        assert len(mock_agent.compression_history) == 0
        assert mock_agent.compression_stats['total_compressions'] == 0
        assert mock_agent.compression_stats['successful_compressions'] == 0
        assert mock_agent.compression_stats['failed_compressions'] == 0
    
    def test_get_full_context(self, mock_agent):
        """Test getting full context from agent."""
        # Add some context to the agent
        mock_agent.context = {"test": "value"}
        mock_agent.memory_context = {"memory": "data"}
        mock_agent.conversation_history = [{"role": "user", "content": "test"}]
        
        full_context = mock_agent._get_full_context()
        
        assert isinstance(full_context, dict)
        assert full_context['agent_name'] == "test_agent"
        assert full_context['current_context'] == {"test": "value"}
        assert full_context['memory_context'] == {"memory": "data"}
        assert len(full_context['conversation_history']) == 1
    
    def test_context_to_string(self, mock_agent, sample_context):
        """Test converting context dictionary to string."""
        result = mock_agent._context_to_string(sample_context)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "system_message" in result
        assert "workflow_state" in result
        assert "conversation_history" in result
        assert "memory_context" in result
    
    def test_context_to_string_simple_string(self, mock_agent):
        """Test converting simple string context."""
        context = "Simple string context"
        result = mock_agent._context_to_string(context)
        
        assert result == context
    
    def test_build_compression_prompt(self, mock_agent):
        """Test compression prompt building."""
        content = "Sample content to compress"
        target_reduction = 0.6
        
        prompt = mock_agent._build_compression_prompt(content, target_reduction)
        
        assert isinstance(prompt, str)
        assert content in prompt
        assert "60%" in prompt
        assert "compress" in prompt.lower()
        assert "preserve" in prompt.lower()
    
    def test_apply_additional_truncation(self, mock_agent):
        """Test additional truncation when compression is insufficient."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8"
        original_size = 100
        target_reduction = 0.5
        
        result = mock_agent._apply_additional_truncation(content, original_size, target_reduction)
        
        assert isinstance(result, str)
        assert len(result) <= original_size * (1 - target_reduction) + 100  # Allow some margin
    
    def test_apply_additional_truncation_with_important_content(self, mock_agent):
        """Test additional truncation preserving important content."""
        content = """Normal line 1
ERROR: Critical error occurred
Normal line 2
WARNING: Important warning
Normal line 3
REQUIREMENT: Must preserve this
Normal line 4"""
        
        original_size = 500  # Larger size to accommodate important content
        target_reduction = 0.5  # Less aggressive reduction
        
        result = mock_agent._apply_additional_truncation(content, original_size, target_reduction)
        
        # Important lines should be preserved
        assert "ERROR: Critical error occurred" in result
        assert "WARNING: Important warning" in result
        assert "REQUIREMENT: Must preserve this" in result
    
    def test_truncate_context_small_content(self, mock_agent, sample_context):
        """Test context truncation when content is already small enough."""
        max_tokens = 10000  # Large limit
        
        result = mock_agent.truncate_context(sample_context, max_tokens)
        
        assert result == sample_context
    
    def test_truncate_context_large_content(self, mock_agent):
        """Test context truncation for large content."""
        # Create large context
        large_context = {
            "large_content": "This is a very long content. " * 1000
        }
        max_tokens = 100
        
        result = mock_agent.truncate_context(large_context, max_tokens)
        
        assert isinstance(result, dict)
        assert "truncated_content" in result
        assert len(result["truncated_content"]) < len(mock_agent._context_to_string(large_context))
        
        # Check that fallback truncation count was incremented
        assert mock_agent.compression_stats['fallback_truncations'] == 1
    
    def test_update_compression_stats_success(self, mock_agent):
        """Test updating compression statistics with successful result."""
        result = CompressionResult(
            original_size=1000,
            compressed_size=500,
            compression_ratio=0.5,
            compressed_content="compressed",
            method_used="llm_compression",
            success=True
        )
        
        mock_agent._update_compression_stats(result)
        
        stats = mock_agent.compression_stats
        assert stats['total_compressions'] == 1
        assert stats['successful_compressions'] == 1
        assert stats['failed_compressions'] == 0
        assert stats['total_original_size'] == 1000
        assert stats['total_compressed_size'] == 500
        assert stats['average_compression_ratio'] == 0.5
    
    def test_get_compression_stats(self, mock_agent):
        """Test getting compression statistics."""
        # Add some test data
        result = CompressionResult(
            original_size=1000,
            compressed_size=500,
            compression_ratio=0.5,
            compressed_content="compressed",
            method_used="llm_compression",
            success=True
        )
        mock_agent.compression_history.append(result)
        mock_agent._update_compression_stats(result)
        
        stats = mock_agent.get_compression_stats()
        
        assert isinstance(stats, dict)
        assert 'agent_name' in stats
        assert 'statistics' in stats
        assert 'recent_compressions' in stats
        assert 'total_history_entries' in stats
        
        assert stats['statistics']['total_compressions'] == 1
        assert stats['total_history_entries'] == 1
        assert len(stats['recent_compressions']) == 1
    
    def test_reset_compression_stats(self, mock_agent):
        """Test resetting compression statistics."""
        # Add some test data
        result = CompressionResult(
            original_size=1000,
            compressed_size=500,
            compression_ratio=0.5,
            compressed_content="compressed",
            method_used="llm_compression",
            success=True
        )
        mock_agent.compression_history.append(result)
        mock_agent._update_compression_stats(result)
        
        # Verify data exists
        assert len(mock_agent.compression_history) == 1
        assert mock_agent.compression_stats['total_compressions'] == 1
        
        # Reset
        mock_agent.reset_compression_stats()
        
        # Verify reset
        assert len(mock_agent.compression_history) == 0
        assert mock_agent.compression_stats['total_compressions'] == 0
        assert mock_agent.compression_stats['successful_compressions'] == 0
        assert mock_agent.compression_stats['failed_compressions'] == 0


class TestCompressionResult:
    """Test cases for CompressionResult data class."""
    
    def test_compression_result_creation(self):
        """Test CompressionResult creation."""
        result = CompressionResult(
            original_size=1000,
            compressed_size=500,
            compression_ratio=0.5,
            compressed_content="compressed content",
            method_used="llm_compression",
            success=True
        )
        
        assert result.original_size == 1000
        assert result.compressed_size == 500
        assert result.compression_ratio == 0.5
        assert result.compressed_content == "compressed content"
        assert result.method_used == "llm_compression"
        assert result.success is True
        assert result.error is None
        assert result.timestamp != ""  # Should be set by __post_init__
    
    def test_compression_result_with_error(self):
        """Test CompressionResult creation with error."""
        result = CompressionResult(
            original_size=1000,
            compressed_size=1000,
            compression_ratio=0.0,
            compressed_content="original content",
            method_used="compression_failed",
            success=False,
            error="Test error message"
        )
        
        assert result.success is False
        assert result.error == "Test error message"
    
    def test_compression_result_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        result = CompressionResult(
            original_size=100,
            compressed_size=50,
            compression_ratio=0.5,
            compressed_content="test",
            method_used="test",
            success=True
        )
        
        # Timestamp should be set automatically
        assert result.timestamp != ""
        assert isinstance(result.timestamp, str)
        
        # Should be a valid ISO format timestamp
        datetime.fromisoformat(result.timestamp)