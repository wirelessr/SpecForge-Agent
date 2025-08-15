"""
Tests for enhanced logging and error handling in ConfigManager.

This module tests the comprehensive logging and error handling features
added to the ConfigManager for the dynamic model family detection system.
"""

import pytest
import tempfile
import json
import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from autogen_framework.config_manager import ConfigManager, ModelConfigurationError, ConfigurationError


class TestConfigManagerLogging:
    """Test enhanced logging functionality in ConfigManager."""
    
    def test_model_info_detection_logging(self, caplog):
        """Test that model info detection logs detailed information."""
        with caplog.at_level(logging.INFO):
            config_manager = ConfigManager(load_env=False)
            
            # Test exact match logging
            model_info = config_manager.get_model_info('models/gemini-2.0-flash')
            
            # Check that detailed logging occurred
            assert "Model configuration detected for 'models/gemini-2.0-flash'" in caplog.text
            assert "family=GEMINI_2_0_FLASH" in caplog.text
            assert "token_limit=1048576" in caplog.text
            assert "source: exact match" in caplog.text
    
    def test_unknown_model_logging_with_suggestions(self, caplog):
        """Test that unknown models log warnings and provide suggestions."""
        with caplog.at_level(logging.INFO):
            config_manager = ConfigManager(load_env=False)
            
            # Test unknown model
            model_info = config_manager.get_model_info('unknown-test-model')
            
            # Check warning and suggestions
            assert "not recognized by any pattern" in caplog.text
            assert "Attempted patterns:" in caplog.text
            assert "MODEL CONFIGURATION SUGGESTIONS" in caplog.text
            assert "Add a specific configuration" in caplog.text
            assert "Add a pattern to match similar models" in caplog.text
    
    def test_token_limit_source_logging(self, caplog):
        """Test that token limit logging includes source information."""
        with caplog.at_level(logging.INFO):
            config_manager = ConfigManager(load_env=False)
            
            # Test token limit for exact match
            token_limit = config_manager.get_model_token_limit('gpt-oss:20b')
            
            assert "Token limit applied for 'gpt-oss:20b': 128000" in caplog.text
            assert "source: custom configuration" in caplog.text
    
    def test_model_family_source_logging(self, caplog):
        """Test that model family logging includes source information."""
        with caplog.at_level(logging.INFO):
            config_manager = ConfigManager(load_env=False)
            
            # Test model family for pattern match
            family = config_manager.get_model_family('models/gemini-2.0-flash-experimental')
            
            assert "Model family applied for" in caplog.text
            assert "GEMINI_2_0_FLASH" in caplog.text
    
    def test_configuration_loading_logging(self, caplog):
        """Test that configuration loading logs detailed information."""
        with caplog.at_level(logging.INFO):
            config_manager = ConfigManager(load_env=False)
            
            # Check that built-in and custom config loading is logged
            assert "Loaded built-in model configurations:" in caplog.text
            assert "specific models" in caplog.text
            assert "detection patterns" in caplog.text
            assert "Successfully loaded custom model configurations" in caplog.text
            assert "Configuration precedence:" in caplog.text


class TestConfigManagerErrorHandling:
    """Test enhanced error handling functionality in ConfigManager."""
    
    def test_invalid_json_error_handling(self, caplog):
        """Test error handling for invalid JSON configuration files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"models": {"test": invalid json}')
            invalid_json_path = f.name
        
        try:
            with caplog.at_level(logging.INFO):  # Changed to INFO to capture suggestions
                config_manager = ConfigManager(load_env=False, models_config_file=invalid_json_path)
                
                # Check error logging and suggestions
                assert "Failed to load model config" in caplog.text
                assert "Invalid JSON" in caplog.text
                assert "CONFIGURATION FILE FIX SUGGESTIONS" in caplog.text
                assert "JSON Syntax Issues:" in caplog.text
                assert "Check for missing commas" in caplog.text
        finally:
            os.unlink(invalid_json_path)
    
    def test_missing_required_fields_error_handling(self, caplog):
        """Test error handling for missing required model configuration fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'models': {
                    'test-model': {
                        'family': 'GPT_4'
                        # Missing token_limit
                    }
                }
            }, f)
            invalid_structure_path = f.name
        
        try:
            with caplog.at_level(logging.INFO):  # Changed to INFO to capture suggestions
                config_manager = ConfigManager(load_env=False, models_config_file=invalid_structure_path)
                
                # Check error logging
                assert "missing required fields: token_limit" in caplog.text
                assert "CONFIGURATION FILE FIX SUGGESTIONS" in caplog.text
        finally:
            os.unlink(invalid_structure_path)
    
    def test_invalid_pattern_regex_error_handling(self, caplog):
        """Test error handling for invalid regex patterns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'patterns': [
                    {
                        'pattern': '[invalid regex',  # Invalid regex
                        'family': 'GPT_4',
                        'token_limit': 8192
                    }
                ]
            }, f)
            invalid_pattern_path = f.name
        
        try:
            with caplog.at_level(logging.ERROR):
                config_manager = ConfigManager(load_env=False, models_config_file=invalid_pattern_path)
                
                # Check error logging
                assert "invalid regex" in caplog.text
        finally:
            os.unlink(invalid_pattern_path)
    
    def test_llm_config_validation_error_messages(self):
        """Test that LLM config validation provides helpful error messages."""
        config_manager = ConfigManager(load_env=False)
        
        # Test invalid base URL
        with pytest.raises(ConfigurationError) as exc_info:
            config_manager._validate_llm_config({
                'base_url': 'invalid-url',
                'temperature': 0.7,
                'max_output_tokens': 4096,
                'timeout': 60
            })
        
        assert "Must start with http:// or https://" in str(exc_info.value)
        assert "Examples:" in str(exc_info.value)
        
        # Test invalid temperature
        with pytest.raises(ConfigurationError) as exc_info:
            config_manager._validate_llm_config({
                'base_url': 'http://localhost:8888/openai/v1',
                'temperature': 3.0,  # Invalid
                'max_output_tokens': 4096,
                'timeout': 60
            })
        
        assert "Must be between 0.0 and 2.0" in str(exc_info.value)
        assert "Lower values" in str(exc_info.value)
        assert "higher values" in str(exc_info.value)
    
    def test_framework_config_validation_error_messages(self):
        """Test that framework config validation provides helpful error messages."""
        config_manager = ConfigManager(load_env=False)
        
        # Test invalid log level
        with pytest.raises(ConfigurationError) as exc_info:
            config_manager._validate_framework_config({
                'log_level': 'INVALID',
                'shell_timeout': 30,
                'shell_max_retries': 2
            })
        
        assert "Must be one of:" in str(exc_info.value)
        assert "DEBUG: Most verbose" in str(exc_info.value)
        
        # Test invalid shell timeout
        with pytest.raises(ConfigurationError) as exc_info:
            config_manager._validate_framework_config({
                'log_level': 'INFO',
                'shell_timeout': -1,  # Invalid
                'shell_max_retries': 2
            })
        
        assert "Must be greater than" in str(exc_info.value)
        assert "Shell command timeout" in str(exc_info.value)


class TestConfigManagerMigrationWarnings:
    """Test migration warning functionality in ConfigManager."""
    
    def test_migration_warnings_for_deprecated_vars(self, caplog):
        """Test that deprecated environment variables trigger migration warnings."""
        with patch.dict(os.environ, {
            'LLM_TEMPERATURE': '0.8',
            'LLM_MAX_OUTPUT_TOKENS': '4096',
            'SHELL_TIMEOUT_SECONDS': '45',
            'LLM_BASE_URL': 'http://localhost:8888/openai/v1',
            'LLM_MODEL': 'test-model',
            'LLM_API_KEY': 'test-key'
        }):
            with caplog.at_level(logging.WARNING):
                config_manager = ConfigManager(load_env=False)
                config_manager.get_framework_config()
                
                # Check migration warnings
                assert "CONFIGURATION MIGRATION NOTICE" in caplog.text
                assert "LLM_TEMPERATURE = 0.8" in caplog.text
                assert "Move to: config/framework.json" in caplog.text
                assert "MIGRATION STEPS:" in caplog.text
                assert "Example config file structure" in caplog.text
    
    def test_migration_guidance_messages(self):
        """Test that migration guidance provides helpful information."""
        with patch.dict(os.environ, {
            'LLM_TEMPERATURE': '0.8',
            'SHELL_MAX_RETRIES': '3'
        }):
            config_manager = ConfigManager(load_env=False)
            guidance = config_manager.provide_migration_guidance()
            
            # Check guidance content
            assert len(guidance) > 0
            assert any("DEPRECATED" in msg for msg in guidance)
            assert any("config/framework.json" in msg for msg in guidance)
            assert any("migration" in msg.lower() for msg in guidance)


class TestConfigManagerValidation:
    """Test configuration validation functionality."""
    
    def test_model_config_structure_validation(self):
        """Test validation of model configuration structure."""
        config_manager = ConfigManager(load_env=False)
        
        # Test invalid top-level structure
        with pytest.raises(ModelConfigurationError) as exc_info:
            config_manager._validate_model_config_structure({
                'invalid_section': {}
            })
        
        assert "Invalid configuration sections" in str(exc_info.value)
        assert "Valid sections are:" in str(exc_info.value)
        
        # Test empty configuration
        with pytest.raises(ModelConfigurationError) as exc_info:
            config_manager._validate_model_config_structure({})
        
        assert "must contain at least one of" in str(exc_info.value)
    
    def test_individual_model_config_validation(self):
        """Test validation of individual model configurations."""
        config_manager = ConfigManager(load_env=False)
        
        # Test missing required fields
        with pytest.raises(ModelConfigurationError) as exc_info:
            config_manager._validate_individual_model_config('test-model', {
                'family': 'GPT_4'
                # Missing token_limit
            })
        
        assert "missing required fields: token_limit" in str(exc_info.value)
        
        # Test invalid token limit
        with pytest.raises(ModelConfigurationError) as exc_info:
            config_manager._validate_individual_model_config('test-model', {
                'family': 'GPT_4',
                'token_limit': -1  # Invalid
            })
        
        assert "must be a positive integer" in str(exc_info.value)
        
        # Test invalid capabilities
        with pytest.raises(ModelConfigurationError) as exc_info:
            config_manager._validate_individual_model_config('test-model', {
                'family': 'GPT_4',
                'token_limit': 8192,
                'capabilities': {
                    'vision': 'yes'  # Should be boolean
                }
            })
        
        assert "must be true or false" in str(exc_info.value)
    
    def test_pattern_config_validation(self):
        """Test validation of pattern configurations."""
        config_manager = ConfigManager(load_env=False)
        
        # Test invalid regex pattern
        with pytest.raises(ModelConfigurationError) as exc_info:
            config_manager._validate_pattern_config({
                'pattern': '[invalid regex',
                'family': 'GPT_4',
                'token_limit': 8192
            }, 0)
        
        assert "invalid regex" in str(exc_info.value)
        
        # Test missing required fields
        with pytest.raises(ModelConfigurationError) as exc_info:
            config_manager._validate_pattern_config({
                'pattern': '^test.*'
                # Missing family and token_limit
            }, 0)
        
        assert "missing required fields" in str(exc_info.value)