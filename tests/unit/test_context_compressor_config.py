"""
Integration tests for ContextCompressor with dynamic model configuration.

Tests the ContextCompressor with real ConfigManager instances to verify
dynamic model family detection, token limits, and compression settings work correctly.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import json
from pathlib import Path

from autogen_framework.context_compressor import ContextCompressor
from autogen_framework.config_manager import ConfigManager
from autogen_framework.models import LLMConfig


class TestContextCompressorDynamicConfig:
    """Integration tests for ContextCompressor with dynamic configuration."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory with test configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create models.json
            models_config = {
                "models": {
                    "test-model": {
                        "family": "GPT_4",
                        "token_limit": 16384,
                        "capabilities": {
                            "vision": False,
                            "function_calling": True,
                            "streaming": True
                        }
                    }
                },
                "patterns": [
                    {
                        "pattern": "^test-.*",
                        "family": "GPT_4",
                        "token_limit": 16384,
                        "capabilities": {
                            "vision": False,
                            "function_calling": True,
                            "streaming": True
                        }
                    }
                ],
                "defaults": {
                    "family": "GPT_4",
                    "token_limit": 8192,
                    "capabilities": {
                        "vision": False,
                        "function_calling": False,
                        "streaming": False
                    }
                }
            }
            
            models_file = config_dir / "models.json"
            with open(models_file, 'w') as f:
                json.dump(models_config, f, indent=2)
            
            # Create framework.json
            framework_config = {
                "context": {
                    "compression_threshold": 0.85,
                    "context_size_ratio": 0.75
                },
                "llm": {
                    "temperature": 0.5,
                    "max_output_tokens": 2048
                }
            }
            
            framework_file = config_dir / "framework.json"
            with open(framework_file, 'w') as f:
                json.dump(framework_config, f, indent=2)
            
            yield str(config_dir)
    
    @pytest.fixture
    def real_config_manager(self, temp_config_dir):
        """Create ConfigManager with test configuration files."""
        return ConfigManager(config_dir=temp_config_dir)
    
    @pytest.fixture
    def test_llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            base_url="http://test.local/v1",
            model="test-model",
            api_key="sk-test123",
            temperature=0.7,
            max_output_tokens=4096,
            timeout=30
        )
    
    def test_context_compressor_with_real_config_manager(self, test_llm_config, real_config_manager):
        """Test ContextCompressor initialization with real ConfigManager."""
        compressor = ContextCompressor(test_llm_config, config_manager=real_config_manager)
        
        # Verify model info was loaded correctly
        assert compressor.model_info["family"] == "GPT_4"
        assert compressor.model_info["token_limit"] == 16384
        assert compressor.model_info["capabilities"]["function_calling"] is True
        
        # Verify max context size calculation
        max_context_size = compressor.get_max_context_size()
        expected_size = int(16384 * 0.75)  # token_limit * context_size_ratio from config
        assert max_context_size == expected_size
    
    def test_model_pattern_matching(self, real_config_manager):
        """Test that model pattern matching works correctly."""
        # Test with a model that matches the pattern
        test_config = LLMConfig(
            base_url="http://test.local/v1",
            model="test-custom-model",  # Should match "^test-.*" pattern
            api_key="sk-test123"
        )
        
        compressor = ContextCompressor(test_config, config_manager=real_config_manager)
        
        # Should use pattern-matched configuration
        assert compressor.model_info["family"] == "GPT_4"
        assert compressor.model_info["token_limit"] == 16384
    
    def test_fallback_to_defaults(self, real_config_manager):
        """Test fallback to default configuration for unknown models."""
        # Test with a model that doesn't match any pattern
        test_config = LLMConfig(
            base_url="http://test.local/v1",
            model="unknown-model",
            api_key="sk-test123"
        )
        
        compressor = ContextCompressor(test_config, config_manager=real_config_manager)
        
        # Should use default configuration
        assert compressor.model_info["family"] == "GPT_4"
        assert compressor.model_info["token_limit"] == 8192
    
    def test_framework_config_integration(self, test_llm_config, real_config_manager):
        """Test integration with framework configuration settings."""
        compressor = ContextCompressor(test_llm_config, config_manager=real_config_manager)
        
        # Get framework config to verify settings
        framework_config = real_config_manager.get_framework_config()
        
        # Verify compression settings are loaded
        assert framework_config.get("compression_threshold") == 0.85
        assert framework_config.get("context_size_ratio") == 0.75
        
        # Verify max context size uses the configured ratio
        max_context_size = compressor.get_max_context_size()
        expected_size = int(16384 * 0.75)
        assert max_context_size == expected_size
    
    @pytest.mark.asyncio
    async def test_compression_with_dynamic_settings(self, test_llm_config, real_config_manager):
        """Test context compression using dynamic configuration settings."""
        compressor = ContextCompressor(test_llm_config, config_manager=real_config_manager)
        
        # Mock the AutoGen agent response
        with patch.object(compressor, '_generate_autogen_response', return_value="Compressed content"):
            test_context = {
                "test_data": "This is test content for compression",
                "workflow_state": {"phase": "test", "progress": "50%"}
            }
            
            result = await compressor.compress_context(test_context)
            
            # Verify compression used dynamic settings
            assert result.success is True
            assert result.method_used == "llm_compression_dynamic"
            assert result.compressed_content == "Compressed content"
    
    def test_capabilities_include_dynamic_features(self, test_llm_config, real_config_manager):
        """Test that capabilities reflect dynamic configuration features."""
        compressor = ContextCompressor(test_llm_config, config_manager=real_config_manager)
        
        capabilities = compressor.get_capabilities()
        
        # Check for dynamic configuration capabilities
        capability_text = " ".join(capabilities).lower()
        assert "dynamic model" in capability_text
        assert "model-specific" in capability_text
        assert "configurable" in capability_text