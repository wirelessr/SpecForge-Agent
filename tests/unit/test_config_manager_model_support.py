"""
Unit tests for ConfigManager model configuration support.

Tests the enhanced ConfigManager functionality for model family detection,
token limit resolution, and configuration loading.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from autogen_framework.config_manager import ConfigManager, ModelConfigurationError


class TestConfigManagerModelSupport:
    """Test ConfigManager model configuration functionality."""
    
    def test_builtin_model_detection_gemini_2_0(self):
        """Test built-in detection for Gemini 2.0 models."""
        config_manager = ConfigManager(load_env=False)
        
        # Test exact model name
        model_info = config_manager.get_model_info("models/gemini-2.0-flash")
        assert model_info["family"] == "GEMINI_2_0_FLASH"
        assert model_info["token_limit"] == 1048576
        assert model_info["capabilities"]["function_calling"] is True
        assert model_info["capabilities"]["vision"] is False
        
        # Test pattern matching
        model_info = config_manager.get_model_info("custom/gemini-2.0-pro")
        assert model_info["family"] == "GEMINI_2_0_FLASH"
        assert model_info["token_limit"] == 1048576
    
    def test_builtin_model_detection_gemini_1_5_pro(self):
        """Test built-in detection for Gemini 1.5 Pro models."""
        config_manager = ConfigManager(load_env=False)
        
        model_info = config_manager.get_model_info("models/gemini-1.5-pro")
        assert model_info["family"] == "GEMINI_1_5_PRO"
        assert model_info["token_limit"] == 2097152
        assert model_info["capabilities"]["vision"] is True
    
    def test_builtin_model_detection_gemini_1_5_flash(self):
        """Test built-in detection for Gemini 1.5 Flash models."""
        config_manager = ConfigManager(load_env=False)
        
        model_info = config_manager.get_model_info("models/gemini-1.5-flash")
        assert model_info["family"] == "GEMINI_1_5_FLASH"
        assert model_info["token_limit"] == 1048576
        assert model_info["capabilities"]["vision"] is True
    
    def test_builtin_model_detection_gpt_4_turbo(self):
        """Test built-in detection for GPT-4 Turbo models."""
        config_manager = ConfigManager(load_env=False)
        
        model_info = config_manager.get_model_info("gpt-4-turbo")
        assert model_info["family"] == "GPT_4"
        assert model_info["token_limit"] == 128000
        assert model_info["capabilities"]["vision"] is True
    
    def test_builtin_model_detection_gpt_4(self):
        """Test built-in detection for GPT-4 models."""
        config_manager = ConfigManager(load_env=False)
        
        model_info = config_manager.get_model_info("gpt-4")
        assert model_info["family"] == "GPT_4"
        assert model_info["token_limit"] == 8192
        assert model_info["capabilities"]["vision"] is False
    
    def test_builtin_model_detection_claude_3_opus(self):
        """Test built-in detection for Claude 3 Opus models."""
        config_manager = ConfigManager(load_env=False)
        
        model_info = config_manager.get_model_info("claude-3-opus")
        assert model_info["family"] == "CLAUDE_3_OPUS"
        assert model_info["token_limit"] == 200000
        assert model_info["capabilities"]["vision"] is True
    
    def test_builtin_model_detection_gpt_oss(self):
        """Test built-in detection for gpt-oss models."""
        config_manager = ConfigManager(load_env=False)
        
        # Test exact match
        model_info = config_manager.get_model_info("gpt-oss:20b")
        assert model_info["family"] == "GPT_4"
        assert model_info["token_limit"] == 128000
        
        # Test pattern match
        model_info = config_manager.get_model_info("gpt-oss:7b")
        assert model_info["family"] == "GPT_4"
        assert model_info["token_limit"] == 128000
    
    def test_unknown_model_fallback(self):
        """Test fallback to defaults for unknown models."""
        config_manager = ConfigManager(load_env=False)
        
        model_info = config_manager.get_model_info("unknown-model")
        assert model_info["family"] == "GPT_4"
        assert model_info["token_limit"] == 8192
        assert model_info["capabilities"]["vision"] is False
        assert model_info["capabilities"]["function_calling"] is False
    
    def test_get_model_token_limit(self):
        """Test get_model_token_limit method."""
        config_manager = ConfigManager(load_env=False)
        
        assert config_manager.get_model_token_limit("models/gemini-2.0-flash") == 1048576
        assert config_manager.get_model_token_limit("gpt-4-turbo") == 128000
        assert config_manager.get_model_token_limit("unknown-model") == 8192
    
    def test_get_model_family(self):
        """Test get_model_family method."""
        config_manager = ConfigManager(load_env=False)
        
        assert config_manager.get_model_family("models/gemini-2.0-flash") == "GEMINI_2_0_FLASH"
        assert config_manager.get_model_family("gpt-4-turbo") == "GPT_4"
        assert config_manager.get_model_family("claude-3-opus") == "CLAUDE_3_OPUS"
        assert config_manager.get_model_family("unknown-model") == "GPT_4"
    
    def test_get_model_capabilities(self):
        """Test get_model_capabilities method."""
        config_manager = ConfigManager(load_env=False)
        
        capabilities = config_manager.get_model_capabilities("models/gemini-1.5-pro")
        assert capabilities["vision"] is True
        assert capabilities["function_calling"] is True
        assert capabilities["streaming"] is True
        
        capabilities = config_manager.get_model_capabilities("unknown-model")
        assert capabilities["vision"] is False
        assert capabilities["function_calling"] is False
        assert capabilities["streaming"] is False
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_custom_config_loading(self, mock_file, mock_exists):
        """Test loading custom model configurations from file."""
        # Mock config file exists
        mock_exists.return_value = True
        
        # Mock config file content
        custom_config = {
            "models": {
                "custom-model": {
                    "family": "GEMINI_2_0_FLASH",
                    "token_limit": 500000,
                    "capabilities": {
                        "vision": True,
                        "function_calling": True,
                        "streaming": False
                    }
                }
            },
            "patterns": [
                {
                    "pattern": "^custom-.*",
                    "family": "CLAUDE_3_OPUS",
                    "token_limit": 300000,
                    "capabilities": {
                        "vision": True,
                        "function_calling": True,
                        "streaming": True
                    }
                }
            ],
            "defaults": {
                "family": "GEMINI_1_5_FLASH",
                "token_limit": 16384,
                "capabilities": {
                    "vision": True,
                    "function_calling": True,
                    "streaming": True
                }
            }
        }
        
        mock_file.return_value.read.return_value = json.dumps(custom_config)
        
        config_manager = ConfigManager(load_env=False)
        
        # Test exact model match (custom config takes precedence)
        model_info = config_manager.get_model_info("custom-model")
        assert model_info["family"] == "GEMINI_2_0_FLASH"
        assert model_info["token_limit"] == 500000
        assert model_info["capabilities"]["streaming"] is False
        
        # Test pattern match (custom pattern takes precedence)
        model_info = config_manager.get_model_info("custom-other")
        assert model_info["family"] == "CLAUDE_3_OPUS"
        assert model_info["token_limit"] == 300000
        
        # Test defaults (custom defaults used)
        model_info = config_manager.get_model_info("totally-unknown")
        assert model_info["family"] == "GEMINI_1_5_FLASH"
        assert model_info["token_limit"] == 16384
    
    @patch('pathlib.Path.exists')
    def test_no_custom_config_file(self, mock_exists):
        """Test behavior when no custom config file exists."""
        mock_exists.return_value = False
        
        config_manager = ConfigManager(load_env=False)
        
        # Should still work with built-in configurations
        model_info = config_manager.get_model_info("models/gemini-2.0-flash")
        assert model_info["family"] == "GEMINI_2_0_FLASH"
        assert model_info["token_limit"] == 1048576
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_invalid_config_file_format(self, mock_file, mock_exists):
        """Test handling of invalid JSON in config file."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = "invalid json content"
        
        # Should not raise exception, should fall back to built-in configs
        config_manager = ConfigManager(load_env=False)
        
        # Should still work with built-in configurations
        model_info = config_manager.get_model_info("models/gemini-2.0-flash")
        assert model_info["family"] == "GEMINI_2_0_FLASH"
        assert model_info["token_limit"] == 1048576
    
    @patch.dict(os.environ, {'CONFIG_DIR': '/custom/config/path'})
    def test_custom_config_directory_from_env(self):
        """Test using custom config directory from environment variable."""
        config_manager = ConfigManager(load_env=False)
        config_dir = config_manager._get_config_directory()
        assert str(config_dir) == "/custom/config/path"
    
    def test_config_directory_default(self):
        """Test default config directory resolution."""
        with patch.dict(os.environ, {}, clear=True):
            config_manager = ConfigManager(load_env=False)
            config_dir = config_manager._get_config_directory()
            expected_path = Path.cwd() / "config"
            assert config_dir == expected_path
    
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_load_config_file_not_found(self, mock_file):
        """Test error handling when config file is not found."""
        config_manager = ConfigManager(load_env=False)
        
        with pytest.raises(ModelConfigurationError):
            config_manager._load_config_file("/nonexistent/path.json")
    
    @patch('builtins.open', new_callable=mock_open)
    def test_load_config_file_invalid_json(self, mock_file):
        """Test error handling for invalid JSON in config file."""
        mock_file.return_value.read.return_value = "invalid json"
        
        config_manager = ConfigManager(load_env=False)
        
        with pytest.raises(ModelConfigurationError):
            config_manager._load_config_file("/path/to/config.json")
    
    def test_pattern_precedence(self):
        """Test that pattern matching follows correct precedence order."""
        config_manager = ConfigManager(load_env=False)
        
        # Test that more specific patterns match before general ones
        # gemini-1.5-pro should match the pro pattern, not the general 1.5 pattern
        model_info = config_manager.get_model_info("models/gemini-1.5-pro-latest")
        assert model_info["family"] == "GEMINI_1_5_PRO"
        assert model_info["token_limit"] == 2097152
        
        # gemini-1.5-flash should match the general 1.5 pattern
        model_info = config_manager.get_model_info("models/gemini-1.5-flash")
        assert model_info["family"] == "GEMINI_1_5_FLASH"
        assert model_info["token_limit"] == 1048576
    
    def test_case_insensitive_matching(self):
        """Test that pattern matching is case insensitive."""
        config_manager = ConfigManager(load_env=False)
        
        # Test uppercase model name
        model_info = config_manager.get_model_info("MODELS/GEMINI-2.0-FLASH")
        assert model_info["family"] == "GEMINI_2_0_FLASH"
        
        # Test mixed case
        model_info = config_manager.get_model_info("Models/Gemini-2.0-Flash")
        assert model_info["family"] == "GEMINI_2_0_FLASH"