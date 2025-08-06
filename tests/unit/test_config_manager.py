"""
Tests for the ConfigManager class.

This module tests the configuration management functionality including
environment variable loading, validation, and error handling.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from autogen_framework.config_manager import ConfigManager, ConfigurationError


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Clear any existing config manager instance
        from autogen_framework.config_manager import reset_config_manager
        reset_config_manager()
    
    def test_init_without_env_file(self):
        """Test ConfigManager initialization without .env file."""
        config_manager = ConfigManager(load_env=False)
        assert config_manager is not None
    
    def test_load_env_file_success(self):
        """Test successful loading of .env file."""
        # Save original environment values
        original_test_var = os.getenv('TEST_VAR')
        original_llm_base_url = os.getenv('LLM_BASE_URL')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("TEST_VAR=test_value\n")
            f.write("LLM_BASE_URL=http://test.local:8888/openai/v1\n")
            env_file = f.name
        
        try:
            config_manager = ConfigManager(env_file=env_file)
            assert os.getenv('TEST_VAR') == 'test_value'
            assert os.getenv('LLM_BASE_URL') == 'http://test.local:8888/openai/v1'
        finally:
            os.unlink(env_file)
            # Restore original environment values
            if original_test_var is None:
                if 'TEST_VAR' in os.environ:
                    del os.environ['TEST_VAR']
            else:
                os.environ['TEST_VAR'] = original_test_var
                
            if original_llm_base_url is None:
                if 'LLM_BASE_URL' in os.environ:
                    del os.environ['LLM_BASE_URL']
            else:
                os.environ['LLM_BASE_URL'] = original_llm_base_url
    
    def test_get_llm_config_success(self):
        """Test successful LLM configuration retrieval."""
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
            'LLM_MODEL': 'test-model',
            'LLM_API_KEY': 'test-key',
            'LLM_TEMPERATURE': '0.8',
            'LLM_MAX_OUTPUT_TOKENS': '4096',
            'LLM_TIMEOUT_SECONDS': '30'
        }):
            config_manager = ConfigManager(load_env=False)
            config = config_manager.get_llm_config()
            
            assert config['base_url'] == 'http://test.local:8888/openai/v1'
            assert config['model'] == 'test-model'
            assert config['api_key'] == 'test-key'
            assert config['temperature'] == 0.8
            assert config['max_output_tokens'] == 4096
            assert config['timeout'] == 30
    
    def test_get_llm_config_missing_required(self):
        """Test LLM configuration with missing required variables."""
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
            # Missing LLM_MODEL and LLM_API_KEY
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager.get_llm_config()
            
            assert "Missing required LLM configuration" in str(exc_info.value)
            assert "LLM_MODEL" in str(exc_info.value)
            assert "LLM_API_KEY" in str(exc_info.value)
    
    def test_get_llm_config_with_defaults(self):
        """Test LLM configuration uses defaults for optional values."""
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
            'LLM_MODEL': 'test-model',
            'LLM_API_KEY': 'test-key'
            # No optional parameters set
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            config = config_manager.get_llm_config()
            
            assert config['temperature'] == 0.7  # default
            assert config['max_output_tokens'] == 8192  # default
            assert config['timeout'] == 60  # default
    
    def test_validate_llm_config_invalid_url(self):
        """Test LLM configuration validation with invalid URL."""
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'invalid-url',
            'LLM_MODEL': 'test-model',
            'LLM_API_KEY': 'test-key'
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager.get_llm_config()
            
            assert "Invalid LLM_BASE_URL" in str(exc_info.value)
            assert "Must start with http://" in str(exc_info.value)
    
    def test_validate_llm_config_invalid_temperature(self):
        """Test LLM configuration validation with invalid temperature."""
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
            'LLM_MODEL': 'test-model',
            'LLM_API_KEY': 'test-key',
            'LLM_TEMPERATURE': '3.0'  # Invalid: > 2.0
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager.get_llm_config()
            
            assert "Invalid LLM_TEMPERATURE" in str(exc_info.value)
            assert "Must be between 0.0 and 2.0" in str(exc_info.value)
    
    def test_get_framework_config_success(self):
        """Test successful framework configuration retrieval."""
        with patch.dict(os.environ, {
            'WORKSPACE_PATH': '/test/workspace',
            'LOG_LEVEL': 'DEBUG',
            'LOG_FILE': 'test.log',
            'SHELL_TIMEOUT_SECONDS': '45',
            'SHELL_MAX_RETRIES': '3'
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            config = config_manager.get_framework_config()
            
            assert config['workspace_path'] == '/test/workspace'
            assert config['log_level'] == 'DEBUG'
            assert config['log_file'] == 'test.log'
            assert config['shell_timeout'] == 45
            assert config['shell_max_retries'] == 3
    
    def test_get_framework_config_with_defaults(self):
        """Test framework configuration uses defaults when not set."""
        with patch.dict(os.environ, {}, clear=True):
            config_manager = ConfigManager(load_env=False)
            config = config_manager.get_framework_config()
            
            assert config['workspace_path'] == '.'
            assert config['log_level'] == 'INFO'
            assert config['log_file'] == 'logs/framework.log'
            assert config['shell_timeout'] == 30
            assert config['shell_max_retries'] == 2
    
    def test_validate_framework_config_invalid_log_level(self):
        """Test framework configuration validation with invalid log level."""
        with patch.dict(os.environ, {
            'LOG_LEVEL': 'INVALID'
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager.get_framework_config()
            
            assert "Invalid LOG_LEVEL" in str(exc_info.value)
            assert "Must be one of:" in str(exc_info.value)
    
    def test_get_test_config(self):
        """Test test configuration retrieval."""
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
            'LLM_MODEL': 'test-model',
            'LLM_API_KEY': 'test-key',
            'WORKSPACE_PATH': 'test_workspace',
            'LOG_LEVEL': 'DEBUG'
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            config = config_manager.get_test_config()
            
            assert config['base_url'] == 'http://test.local:8888/openai/v1'
            assert config['model'] == 'test-model'
            assert config['api_key'] == 'test-key'
            assert config['workspace_path'] == 'test_workspace'
            assert config['log_level'] == 'DEBUG'
    
    def test_get_test_config_fallback_to_main(self):
        """Test test configuration falls back to main config when not set."""
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://main.local:8888/openai/v1',
            'LLM_MODEL': 'main-model',
            'LLM_API_KEY': 'main-key'
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            config = config_manager.get_test_config()
            
            # Should fall back to main config values
            assert config['base_url'] == 'http://main.local:8888/openai/v1'
            assert config['model'] == 'main-model'
            assert config['api_key'] == 'main-key'
    
    def test_validate_required_config_success(self):
        """Test validation passes with all required config present."""
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
            'LLM_MODEL': 'test-model',
            'LLM_API_KEY': 'test-key'
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            errors = config_manager.validate_required_config()
            
            assert errors == []
    
    def test_validate_required_config_with_errors(self):
        """Test validation returns errors when config is missing."""
        with patch.dict(os.environ, {}, clear=True):
            config_manager = ConfigManager(load_env=False)
            errors = config_manager.validate_required_config()
            
            assert len(errors) > 0
            assert any("LLM Configuration" in error for error in errors)
    
    def test_get_all_config(self):
        """Test getting complete configuration."""
        with patch.dict(os.environ, {
            'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
            'LLM_MODEL': 'test-model',
            'LLM_API_KEY': 'test-key',
            'WORKSPACE_PATH': '/test/workspace',
            'LOG_LEVEL': 'DEBUG'
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            config = config_manager.get_all_config()
            
            # Should contain both LLM and framework config
            assert 'base_url' in config
            assert 'model' in config
            assert 'api_key' in config
            assert 'workspace_path' in config
            assert 'log_level' in config
    
    def test_get_token_config_success(self):
        """Test successful token configuration retrieval."""
        with patch.dict(os.environ, {
            'DEFAULT_TOKEN_LIMIT': '16384',
            'TOKEN_COMPRESSION_THRESHOLD': '0.8',
            'CONTEXT_COMPRESSION_ENABLED': 'true',
            'COMPRESSION_TARGET_RATIO': '0.4',
            'VERBOSE_TOKEN_LOGGING': 'true'
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            config = config_manager.get_token_config()
            
            assert config['default_token_limit'] == 16384
            assert config['compression_threshold'] == 0.8
            assert config['compression_enabled'] is True
            assert config['compression_target_ratio'] == 0.4
            assert config['verbose_logging'] is True
    
    def test_get_token_config_with_defaults(self):
        """Test token configuration uses defaults when not set."""
        with patch.dict(os.environ, {}, clear=True):
            config_manager = ConfigManager(load_env=False)
            config = config_manager.get_token_config()
            
            assert config['default_token_limit'] == 8192
            assert config['compression_threshold'] == 0.9
            assert config['compression_enabled'] is True
            assert config['compression_target_ratio'] == 0.5
            assert config['verbose_logging'] is False
    
    def test_get_token_config_boolean_parsing(self):
        """Test token configuration boolean parsing."""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('invalid', False)  # Should default to False for invalid values
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {
                'CONTEXT_COMPRESSION_ENABLED': env_value,
                'VERBOSE_TOKEN_LOGGING': env_value
            }, clear=True):
                config_manager = ConfigManager(load_env=False)
                config = config_manager.get_token_config()
                
                assert config['compression_enabled'] is expected
                assert config['verbose_logging'] is expected
    
    def test_validate_token_config_invalid_threshold(self):
        """Test token configuration validation with invalid compression threshold."""
        config_manager = ConfigManager(load_env=False)
        
        # Test threshold <= 0
        invalid_config = {
            'compression_threshold': 0.0,
            'default_token_limit': 8192,
            'compression_target_ratio': 0.5
        }
        
        with patch.object(config_manager.logger, 'warning') as mock_warning:
            validated = config_manager.validate_token_config(invalid_config)
            mock_warning.assert_called_once()
            assert validated['compression_threshold'] == 0.9  # Should use default
        
        # Test threshold > 1
        invalid_config['compression_threshold'] = 1.5
        
        with patch.object(config_manager.logger, 'warning') as mock_warning:
            validated = config_manager.validate_token_config(invalid_config)
            mock_warning.assert_called_once()
            assert validated['compression_threshold'] == 0.9  # Should use default
    
    def test_validate_token_config_invalid_token_limit(self):
        """Test token configuration validation with invalid token limit."""
        config_manager = ConfigManager(load_env=False)
        
        invalid_config = {
            'compression_threshold': 0.9,
            'default_token_limit': -100,  # Invalid: negative
            'compression_target_ratio': 0.5
        }
        
        with patch.object(config_manager.logger, 'warning') as mock_warning:
            validated = config_manager.validate_token_config(invalid_config)
            mock_warning.assert_called_once()
            assert validated['default_token_limit'] == 8192  # Should use default
    
    def test_validate_token_config_invalid_target_ratio(self):
        """Test token configuration validation with invalid target ratio."""
        config_manager = ConfigManager(load_env=False)
        
        # Test ratio <= 0
        invalid_config = {
            'compression_threshold': 0.9,
            'default_token_limit': 8192,
            'compression_target_ratio': 0.0
        }
        
        with patch.object(config_manager.logger, 'warning') as mock_warning:
            validated = config_manager.validate_token_config(invalid_config)
            mock_warning.assert_called_once()
            assert validated['compression_target_ratio'] == 0.5  # Should use default
        
        # Test ratio >= 1
        invalid_config['compression_target_ratio'] = 1.0
        
        with patch.object(config_manager.logger, 'warning') as mock_warning:
            validated = config_manager.validate_token_config(invalid_config)
            mock_warning.assert_called_once()
            assert validated['compression_target_ratio'] == 0.5  # Should use default
    
    def test_validate_token_config_valid_values(self):
        """Test token configuration validation with valid values."""
        config_manager = ConfigManager(load_env=False)
        
        valid_config = {
            'compression_threshold': 0.85,
            'default_token_limit': 16384,
            'compression_target_ratio': 0.3
        }
        
        with patch.object(config_manager.logger, 'warning') as mock_warning:
            validated = config_manager.validate_token_config(valid_config)
            mock_warning.assert_not_called()  # No warnings for valid config
            
            # Values should remain unchanged
            assert validated['compression_threshold'] == 0.85
            assert validated['default_token_limit'] == 16384
            assert validated['compression_target_ratio'] == 0.3
    
    def test_get_token_config_invalid_values_fallback(self):
        """Test token configuration falls back to defaults for invalid environment values."""
        with patch.dict(os.environ, {
            'DEFAULT_TOKEN_LIMIT': 'invalid_number',
            'TOKEN_COMPRESSION_THRESHOLD': 'not_a_float',
            'COMPRESSION_TARGET_RATIO': 'invalid'
        }, clear=True):
            config_manager = ConfigManager(load_env=False)
            
            with patch.object(config_manager.logger, 'warning') as mock_warning:
                config = config_manager.get_token_config()
                
                # Should fall back to defaults and log warnings
                assert mock_warning.call_count >= 3  # At least 3 warnings for invalid values
                assert config['default_token_limit'] == 8192
                assert config['compression_threshold'] == 0.9
                assert config['compression_target_ratio'] == 0.5
    
    def test_global_config_manager(self):
        """Test global configuration manager functions."""
        from autogen_framework.config_manager import get_config_manager, reset_config_manager
        
        # Get first instance
        manager1 = get_config_manager()
        assert manager1 is not None
        
        # Get second instance (should be same)
        manager2 = get_config_manager()
        assert manager1 is manager2
        
        # Reset and get new instance
        reset_config_manager()
        manager3 = get_config_manager()
        assert manager3 is not manager1