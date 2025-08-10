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
        
        assert compressor.llm_config == test_llm_config
        assert compressor.token_manager == mock_token_manager
        # ContextCompressor doesn't inherit from BaseLLMAgent, so it doesn't have name or description attributes
    
    def test_initialization_without_token_manager(self, test_llm_config):
        """Test ContextCompressor initialization without token manager."""
        compressor = ContextCompressor(test_llm_config)
        
        assert compressor.token_manager is None
        # ContextCompressor doesn't inherit from BaseLLMAgent, so it doesn't have name attribute
    
    def test_build_compression_system_message(self, context_compressor):
        """Test compression system message building."""
        system_message = context_compressor._build_compression_system_message()
        
        assert isinstance(system_message, str)
        assert len(system_message) > 0
        assert "context compression specialist" in system_message.lower()
        assert "critical information" in system_message.lower()
        assert "workflow state" in system_message.lower()
        assert "user requirements" in system_message.lower()
    
    def test_get_capabilities(self, context_compressor):
        """Test getting compressor capabilities."""
        capabilities = context_compressor.get_capabilities()
        
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
    
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
    
    # ContextCompressor doesn't have compress_agent_contexts method


# BaseLLMAgent compression functionality has been moved to ContextManager
# These tests are no longer relevant as the functionality has been refactored


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