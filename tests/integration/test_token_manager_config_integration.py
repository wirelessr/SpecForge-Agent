"""
Integration tests for TokenManager with enhanced ConfigManager.

Tests the integration of TokenManager with the enhanced ConfigManager
to verify dynamic model configuration detection and context size calculation.
"""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path
import tempfile
import json

from autogen_framework.token_manager import TokenManager
from autogen_framework.config_manager import ConfigManager


class TestTokenManagerConfigIntegration:
    """Integration tests for TokenManager with enhanced ConfigManager."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory with test configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create models.json with test configurations
            models_config = {
                "models": {
                    "test-model-1": {
                        "family": "GPT_4",
                        "token_limit": 16000,
                        "capabilities": {
                            "vision": False,
                            "function_calling": True,
                            "streaming": True
                        }
                    },
                    "test-model-2": {
                        "family": "GEMINI_2_0_FLASH",
                        "token_limit": 32000,
                        "capabilities": {
                            "vision": True,
                            "function_calling": True,
                            "streaming": True
                        }
                    }
                },
                "patterns": [
                    {
                        "pattern": "^test-custom-.*",
                        "family": "GPT_4",
                        "token_limit": 8000,
                        "capabilities": {
                            "vision": False,
                            "function_calling": True,
                            "streaming": True
                        }
                    }
                ],
                "defaults": {
                    "family": "GPT_4",
                    "token_limit": 4000,
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
            
            # Create framework.json with test configurations
            framework_config = {
                "llm": {
                    "temperature": 0.7,
                    "max_output_tokens": 4096,
                    "timeout_seconds": 60
                },
                "shell": {
                    "timeout_seconds": 30,
                    "max_retries": 2
                },
                "context": {
                    "compression_threshold": 0.9,
                    "context_size_ratio": 0.75  # Custom 75% ratio
                }
            }
            
            framework_file = config_dir / "framework.json"
            with open(framework_file, 'w') as f:
                json.dump(framework_config, f, indent=2)
            
            yield config_dir
    
    @pytest.fixture
    def real_config_manager(self, temp_config_dir):
        """Create a real ConfigManager with test configurations."""
        return ConfigManager(
            load_env=False,
            config_dir=str(temp_config_dir)
        )
    
    @pytest.fixture
    def token_manager(self, real_config_manager):
        """Create a TokenManager with real ConfigManager."""
        return TokenManager(real_config_manager)
    
    def test_dynamic_model_limit_detection(self, token_manager):
        """Test that TokenManager uses ConfigManager for dynamic model limit detection."""
        # Test known models from config
        assert token_manager.get_model_limit('test-model-1') == 16000
        assert token_manager.get_model_limit('test-model-2') == 32000
        
        # Test pattern matching
        assert token_manager.get_model_limit('test-custom-abc') == 8000
        assert token_manager.get_model_limit('test-custom-xyz') == 8000
        
        # Test unknown model uses default
        assert token_manager.get_model_limit('unknown-model') == 4000
    
    def test_context_size_calculation(self, token_manager):
        """Test context size calculation using custom ratio from config."""
        # Test with custom ratio (0.75 from framework.json)
        context_size_1 = token_manager.get_model_context_size('test-model-1')
        expected_size_1 = int(16000 * 0.75)  # 12000
        assert context_size_1 == expected_size_1
        
        context_size_2 = token_manager.get_model_context_size('test-model-2')
        expected_size_2 = int(32000 * 0.75)  # 24000
        assert context_size_2 == expected_size_2
        
        # Test with pattern-matched model
        context_size_custom = token_manager.get_model_context_size('test-custom-abc')
        expected_size_custom = int(8000 * 0.75)  # 6000
        assert context_size_custom == expected_size_custom
    
    def test_token_limit_checking_with_context_size(self, token_manager):
        """Test token limit checking using calculated context sizes."""
        # Set context size that triggers compression with context size but not token limit
        token_manager.current_context_size = 10000
        
        # Test with full token limit (16000)
        result_token = token_manager.check_token_limit('test-model-1', use_context_size=False)
        assert result_token.model_limit == 16000
        assert result_token.current_tokens == 10000
        assert result_token.percentage_used == 10000 / 16000  # 62.5%
        assert result_token.needs_compression is False  # Below 90% threshold
        
        # Test with calculated context size (12000)
        result_context = token_manager.check_token_limit('test-model-1', use_context_size=True)
        assert result_context.model_limit == 12000
        assert result_context.current_tokens == 10000
        assert result_context.percentage_used == 10000 / 12000  # 83.3%
        assert result_context.needs_compression is False  # Still below 90% threshold
        
        # Test with higher usage that triggers compression
        token_manager.current_context_size = 11000
        result_context_high = token_manager.check_token_limit('test-model-1', use_context_size=True)
        assert result_context_high.percentage_used == 11000 / 12000  # 91.7%
        assert result_context_high.needs_compression is True  # Above 90% threshold
    
    def test_backward_compatibility(self, token_manager):
        """Test that existing TokenManager interface remains compatible."""
        # Test that all existing methods work
        token_manager.update_token_usage('test-model-1', 1000, 'test_operation')
        
        # Check token limit (backward compatible)
        result = token_manager.check_token_limit('test-model-1')
        assert result.current_tokens == 1000
        assert result.model_limit == 16000  # Uses ConfigManager
        
        # Get usage statistics
        stats = token_manager.get_usage_statistics()
        assert stats.total_tokens_used == 1000
        assert stats.requests_made == 1
        
        # Get detailed report
        report = token_manager.get_detailed_usage_report()
        assert report['current_context_size'] == 1000
        assert 'statistics' in report
        assert 'configuration' in report
    
    def test_config_manager_error_handling(self, real_config_manager):
        """Test TokenManager fallback when ConfigManager fails."""
        token_manager = TokenManager(real_config_manager)
        
        # Mock ConfigManager to raise exception
        with patch.object(real_config_manager, 'get_model_token_limit', side_effect=Exception("Config error")):
            # Should fallback to default limit
            limit = token_manager.get_model_limit('test-model-1')
            assert limit == 8192  # Default fallback limit
    
    def test_verbose_logging_with_config_manager(self, real_config_manager):
        """Test verbose logging shows ConfigManager source."""
        # Enable verbose logging
        with patch.object(real_config_manager, 'get_token_config') as mock_get_config:
            mock_get_config.return_value = {
                'default_token_limit': 8192,
                'compression_threshold': 0.9,
                'compression_enabled': True,
                'compression_target_ratio': 0.5,
                'verbose_logging': True
            }
            
            token_manager = TokenManager(real_config_manager)
            
            with patch.object(token_manager.logger, 'debug') as mock_debug:
                limit = token_manager.get_model_limit('test-model-1')
                
                # Should log with ConfigManager source
                mock_debug.assert_called_once()
                call_args = mock_debug.call_args[0][0]
                assert 'from ConfigManager' in call_args
                assert limit == 16000
    
    def test_model_info_integration(self, token_manager, real_config_manager):
        """Test that TokenManager can access full model info from ConfigManager."""
        # Test that ConfigManager methods are accessible
        model_info = real_config_manager.get_model_info('test-model-1')
        
        assert model_info['family'] == 'GPT_4'
        assert model_info['token_limit'] == 16000
        assert model_info['capabilities']['function_calling'] is True
        assert model_info['capabilities']['vision'] is False
        
        # Test with model that has vision
        model_info_2 = real_config_manager.get_model_info('test-model-2')
        assert model_info_2['family'] == 'GEMINI_2_0_FLASH'
        assert model_info_2['token_limit'] == 32000
        assert model_info_2['capabilities']['vision'] is True
    
    def test_framework_config_integration(self, token_manager, real_config_manager):
        """Test that TokenManager can access framework configuration."""
        framework_config = real_config_manager.get_framework_config()
        
        # Verify custom context size ratio is loaded
        assert framework_config['context_size_ratio'] == 0.75
        
        # Verify other framework settings
        assert framework_config.get('shell_timeout') == 30
        assert framework_config.get('shell_max_retries') == 2
    
    def test_configuration_source_tracking(self, token_manager, real_config_manager):
        """Test that configuration sources are properly tracked."""
        # Get configuration sources info
        sources_info = real_config_manager.get_configuration_sources_info()
        
        assert 'models_config' in sources_info
        assert 'framework_config' in sources_info
        assert sources_info['models_config']['exists'] is True
        assert sources_info['framework_config']['exists'] is True
        
        # Verify paths are correct
        assert 'models.json' in str(sources_info['models_config']['resolved_path'])
        assert 'framework.json' in str(sources_info['framework_config']['resolved_path'])


class TestTokenManagerConfigEdgeCases:
    """Edge case tests for TokenManager with ConfigManager integration."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock ConfigManager for edge case testing."""
        mock_config = Mock(spec=ConfigManager)
        mock_config.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.9,
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': False
        }
        return mock_config
    
    def test_config_manager_unavailable(self, mock_config_manager):
        """Test TokenManager behavior when ConfigManager methods are unavailable."""
        # Mock ConfigManager to not have enhanced methods
        del mock_config_manager.get_model_token_limit
        del mock_config_manager.get_model_info
        
        token_manager = TokenManager(mock_config_manager)
        
        # Should fallback gracefully
        limit = token_manager.get_model_limit('any-model')
        assert limit == 8192  # Default limit
    
    def test_partial_config_manager_failure(self, mock_config_manager):
        """Test TokenManager when some ConfigManager methods fail."""
        # Mock get_model_token_limit to work but get_model_info to fail
        mock_config_manager.get_model_token_limit.return_value = 16000
        mock_config_manager.get_model_info.side_effect = Exception("Model info error")
        mock_config_manager.get_framework_config.return_value = {'context_size_ratio': 0.8}
        
        token_manager = TokenManager(mock_config_manager)
        
        # get_model_limit should work
        assert token_manager.get_model_limit('test-model') == 16000
        
        # get_model_context_size should fallback gracefully
        context_size = token_manager.get_model_context_size('test-model')
        expected_fallback = int(16000 * 0.8)  # Uses get_model_limit as fallback
        assert context_size == expected_fallback
    
    def test_invalid_context_size_ratio(self, mock_config_manager):
        """Test TokenManager with invalid context size ratio."""
        # Mock framework config with invalid ratio
        mock_config_manager.get_framework_config.return_value = {
            'context_size_ratio': 1.5  # Invalid ratio > 1
        }
        mock_config_manager.get_model_info.return_value = {
            'token_limit': 8192,
            'family': 'GPT_4',
            'capabilities': {}
        }
        
        token_manager = TokenManager(mock_config_manager)
        
        # Should handle invalid ratio gracefully
        context_size = token_manager.get_model_context_size('test-model')
        # Should use fallback calculation
        assert context_size > 0
        assert context_size <= 8192  # Should not exceed token limit