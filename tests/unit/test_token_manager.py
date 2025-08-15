"""
Unit tests for TokenManager.

Tests token usage tracking, limit checking, and configuration management
using mocked dependencies to ensure fast, isolated testing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from autogen_framework.token_manager import TokenManager, TokenCheckResult, TokenUsageStats
from autogen_framework.config_manager import ConfigManager


class TestTokenManager:
    """Test cases for TokenManager functionality."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock ConfigManager for testing."""
        mock_config = Mock(spec=ConfigManager)
        mock_config.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.9,
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': False
        }
        
        # Mock the enhanced model configuration methods
        def mock_get_model_token_limit(model_name):
            # Return known model limits or default
            known_limits = {
                'models/gemini-2.0-flash': 1048576,
                'models/gemini-1.5-pro': 2097152,
                'models/gemini-1.5-flash': 1048576,
                'gpt-4': 8192,
                'gpt-4-turbo': 128000,
                'gpt-3.5-turbo': 16385,
                'claude-3-opus': 200000,
                'claude-3-sonnet': 200000,
                'claude-3-haiku': 200000,
            }
            return known_limits.get(model_name, 8192)  # Default limit
        
        def mock_get_model_info(model_name):
            token_limit = mock_get_model_token_limit(model_name)
            return {
                'family': 'GPT_4',  # Default family
                'token_limit': token_limit,
                'capabilities': {
                    'vision': False,
                    'function_calling': True,
                    'streaming': True
                }
            }
        
        mock_config.get_model_token_limit.side_effect = mock_get_model_token_limit
        mock_config.get_model_info.side_effect = mock_get_model_info
        mock_config.get_framework_config.return_value = {
            'context_size_ratio': 0.8,
            'workspace_path': '.',
            'log_level': 'INFO'
        }
        
        return mock_config
    
    @pytest.fixture
    def token_manager(self, mock_config_manager):
        """Create a TokenManager instance for testing."""
        return TokenManager(mock_config_manager)
    
    def test_initialization(self, mock_config_manager):
        """Test TokenManager initialization."""
        token_manager = TokenManager(mock_config_manager)
        
        # Verify configuration was loaded
        mock_config_manager.get_token_config.assert_called_once()
        
        # Verify initial state
        assert token_manager.current_context_size == 0
        assert token_manager.usage_stats['total_tokens_used'] == 0
        assert token_manager.usage_stats['requests_made'] == 0
        assert token_manager.usage_stats['compressions_performed'] == 0
        assert token_manager.usage_stats['peak_token_usage'] == 0
        assert isinstance(token_manager.usage_stats['usage_history'], list)
        assert len(token_manager.usage_stats['usage_history']) == 0
    
    def test_load_token_config(self, token_manager, mock_config_manager):
        """Test token configuration loading."""
        expected_config = {
            'default_token_limit': 8192,
            'compression_threshold': 0.9,
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': False
        }
        
        assert token_manager.token_config == expected_config
        mock_config_manager.get_token_config.assert_called_once()
    
    def test_load_model_limits(self, token_manager):
        """Test model limits loading with dynamic ConfigManager detection."""
        # Test known models (now using ConfigManager)
        assert token_manager.get_model_limit('models/gemini-2.0-flash') == 1048576
        assert token_manager.get_model_limit('gpt-4') == 8192
        assert token_manager.get_model_limit('claude-3-opus') == 200000
        
        # Test unknown model uses default
        assert token_manager.get_model_limit('unknown-model') == 8192
    
    def test_update_token_usage_basic(self, token_manager):
        """Test basic token usage update."""
        # Update token usage
        token_manager.update_token_usage('gpt-4', 100, 'test_operation')
        
        # Verify context size updated
        assert token_manager.current_context_size == 100
        
        # Verify statistics updated
        assert token_manager.usage_stats['total_tokens_used'] == 100
        assert token_manager.usage_stats['requests_made'] == 1
        assert token_manager.usage_stats['peak_token_usage'] == 100
        assert token_manager.usage_stats['average_tokens_per_request'] == 100.0
        
        # Verify usage history recorded
        assert len(token_manager.usage_stats['usage_history']) == 1
        history_entry = token_manager.usage_stats['usage_history'][0]
        assert history_entry['model'] == 'gpt-4'
        assert history_entry['tokens_used'] == 100
        assert history_entry['operation'] == 'test_operation'
        assert history_entry['current_context_size'] == 100
        assert 'timestamp' in history_entry
    
    def test_update_token_usage_multiple(self, token_manager):
        """Test multiple token usage updates."""
        # First update
        token_manager.update_token_usage('gpt-4', 100, 'operation1')
        
        # Second update
        token_manager.update_token_usage('gpt-4', 150, 'operation2')
        
        # Verify cumulative context size
        assert token_manager.current_context_size == 250
        
        # Verify statistics
        assert token_manager.usage_stats['total_tokens_used'] == 250
        assert token_manager.usage_stats['requests_made'] == 2
        assert token_manager.usage_stats['peak_token_usage'] == 250
        assert token_manager.usage_stats['average_tokens_per_request'] == 125.0
        
        # Verify usage history
        assert len(token_manager.usage_stats['usage_history']) == 2
    
    def test_update_token_usage_invalid(self, token_manager):
        """Test token usage update with invalid values."""
        with patch.object(token_manager.logger, 'warning') as mock_warning:
            # Test zero tokens
            token_manager.update_token_usage('gpt-4', 0, 'invalid_operation')
            mock_warning.assert_called_once()
            
            # Test negative tokens
            token_manager.update_token_usage('gpt-4', -50, 'invalid_operation')
            assert mock_warning.call_count == 2
        
        # Verify no changes to context or stats
        assert token_manager.current_context_size == 0
        assert token_manager.usage_stats['total_tokens_used'] == 0
        assert token_manager.usage_stats['requests_made'] == 0
    
    def test_check_token_limit_below_threshold(self, token_manager):
        """Test token limit checking when below threshold."""
        # Set context size below threshold (90% of 8192 = 7372.8)
        token_manager.current_context_size = 5000
        
        result = token_manager.check_token_limit('gpt-4')
        
        assert isinstance(result, TokenCheckResult)
        assert result.current_tokens == 5000
        assert result.model_limit == 8192
        assert result.percentage_used == pytest.approx(0.610, rel=1e-3)
        assert result.needs_compression is False
    
    def test_check_token_limit_above_threshold(self, token_manager):
        """Test token limit checking when above threshold."""
        # Set context size above threshold (90% of 8192 = 7372.8)
        token_manager.current_context_size = 7500
        
        with patch.object(token_manager.logger, 'warning') as mock_warning:
            result = token_manager.check_token_limit('gpt-4')
            mock_warning.assert_called_once()
        
        assert result.current_tokens == 7500
        assert result.model_limit == 8192
        assert result.percentage_used == pytest.approx(0.915, rel=1e-3)
        assert result.needs_compression is True
    
    def test_check_token_limit_different_models(self, token_manager):
        """Test token limit checking with different models."""
        token_manager.current_context_size = 100000
        
        # Test with Gemini model (higher limit)
        result_gemini = token_manager.check_token_limit('models/gemini-2.0-flash')
        assert result_gemini.model_limit == 1048576
        assert result_gemini.needs_compression is False
        
        # Test with GPT-4 (lower limit)
        result_gpt4 = token_manager.check_token_limit('gpt-4')
        assert result_gpt4.model_limit == 8192
        assert result_gpt4.needs_compression is True
    
    def test_get_model_limit_known_models(self, token_manager):
        """Test getting limits for known models."""
        test_cases = [
            ('models/gemini-2.0-flash', 1048576),
            ('models/gemini-1.5-pro', 2097152),
            ('gpt-4', 8192),
            ('gpt-4-turbo', 128000),
            ('claude-3-opus', 200000),
        ]
        
        for model, expected_limit in test_cases:
            assert token_manager.get_model_limit(model) == expected_limit
    
    def test_get_model_limit_unknown_model(self, token_manager):
        """Test getting limit for unknown model uses default."""
        unknown_models = ['unknown-model', 'custom-model', 'test-model']
        
        for model in unknown_models:
            assert token_manager.get_model_limit(model) == 8192
    
    def test_get_model_limit_verbose_logging(self, mock_config_manager):
        """Test model limit retrieval with verbose logging."""
        # Enable verbose logging
        mock_config_manager.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.9,
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': True
        }
        
        token_manager = TokenManager(mock_config_manager)
        
        with patch.object(token_manager.logger, 'debug') as mock_debug:
            limit = token_manager.get_model_limit('gpt-4')
            mock_debug.assert_called_once()
            assert limit == 8192
    
    def test_get_model_context_size(self, token_manager):
        """Test calculated context size using ratio."""
        # Test with default ratio (0.8)
        context_size = token_manager.get_model_context_size('gpt-4')
        expected_size = int(8192 * 0.8)  # 6553
        assert context_size == expected_size
        
        # Test with larger model
        context_size = token_manager.get_model_context_size('models/gemini-2.0-flash')
        expected_size = int(1048576 * 0.8)  # 838860
        assert context_size == expected_size
    
    def test_get_model_context_size_custom_ratio(self, mock_config_manager):
        """Test calculated context size with custom ratio."""
        # Set custom context size ratio
        mock_config_manager.get_framework_config.return_value = {
            'context_size_ratio': 0.7,  # Custom 70% ratio
            'workspace_path': '.',
            'log_level': 'INFO'
        }
        
        token_manager = TokenManager(mock_config_manager)
        
        context_size = token_manager.get_model_context_size('gpt-4')
        expected_size = int(8192 * 0.7)  # 5734
        assert context_size == expected_size
    
    def test_check_token_limit_with_context_size(self, token_manager):
        """Test token limit checking using calculated context size."""
        # Set context size that will trigger compression with context size but not with token limit
        token_manager.current_context_size = 6000  # Below context size (6553) but above 90% of token limit
        
        # Test with full token limit (backward compatibility)
        result_token = token_manager.check_token_limit('gpt-4', use_context_size=False)
        assert result_token.model_limit == 8192
        assert result_token.needs_compression is False  # 6000/8192 = 73% < 90%
        
        # Test with calculated context size
        result_context = token_manager.check_token_limit('gpt-4', use_context_size=True)
        expected_context_size = int(8192 * 0.8)  # 6553
        assert result_context.model_limit == expected_context_size
        assert result_context.needs_compression is True  # 6000/6553 = 91% >= 90%
    
    def test_log_token_usage_normal_logging(self, token_manager):
        """Test token usage logging with normal verbosity."""
        token_manager.current_context_size = 1000
        
        with patch.object(token_manager.logger, 'debug') as mock_debug:
            token_manager.log_token_usage('gpt-4', 100, 'test_operation')
            mock_debug.assert_called_once()
    
    def test_log_token_usage_verbose_logging(self, mock_config_manager):
        """Test token usage logging with verbose mode."""
        # Enable verbose logging
        mock_config_manager.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.9,
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': True
        }
        
        token_manager = TokenManager(mock_config_manager)
        token_manager.current_context_size = 1000
        
        with patch.object(token_manager.logger, 'info') as mock_info:
            token_manager.log_token_usage('gpt-4', 100, 'test_operation')
            mock_info.assert_called_once()
    
    def test_reset_context_size(self, token_manager):
        """Test context size reset functionality."""
        # Set up initial state
        token_manager.current_context_size = 5000
        
        with patch.object(token_manager.logger, 'info') as mock_info:
            token_manager.reset_context_size()
            mock_info.assert_called_once()
        
        # Verify context size reset
        assert token_manager.current_context_size == 0
        
        # Verify reset recorded in history
        assert len(token_manager.usage_stats['usage_history']) == 1
        reset_entry = token_manager.usage_stats['usage_history'][0]
        assert reset_entry['operation'] == 'context_reset'
        assert reset_entry['old_context_size'] == 5000
        assert reset_entry['new_context_size'] == 0
        assert 'timestamp' in reset_entry
    
    def test_get_usage_statistics(self, token_manager):
        """Test usage statistics retrieval."""
        # Set up some usage data
        token_manager.usage_stats.update({
            'total_tokens_used': 1000,
            'requests_made': 5,
            'compressions_performed': 2,
            'average_tokens_per_request': 200.0,
            'peak_token_usage': 800
        })
        
        stats = token_manager.get_usage_statistics()
        
        assert isinstance(stats, TokenUsageStats)
        assert stats.total_tokens_used == 1000
        assert stats.requests_made == 5
        assert stats.compressions_performed == 2
        assert stats.average_tokens_per_request == 200.0
        assert stats.peak_token_usage == 800
    
    def test_increment_compression_count(self, token_manager):
        """Test compression count increment."""
        initial_count = token_manager.usage_stats['compressions_performed']
        
        with patch.object(token_manager.logger, 'info') as mock_info:
            token_manager.increment_compression_count()
            mock_info.assert_called_once()
        
        # Verify count incremented
        assert token_manager.usage_stats['compressions_performed'] == initial_count + 1
        
        # Verify compression recorded in history
        assert len(token_manager.usage_stats['usage_history']) == 1
        compression_entry = token_manager.usage_stats['usage_history'][0]
        assert compression_entry['operation'] == 'compression_performed'
        assert compression_entry['compressions_total'] == initial_count + 1
        assert 'timestamp' in compression_entry
    
    def test_get_detailed_usage_report(self, token_manager):
        """Test detailed usage report generation."""
        # Set up usage data
        token_manager.current_context_size = 2000
        token_manager.update_token_usage('gpt-4', 500, 'operation1')
        token_manager.update_token_usage('gpt-4', 300, 'operation2')
        token_manager.increment_compression_count()
        
        report = token_manager.get_detailed_usage_report()
        
        # Verify report structure
        assert 'current_context_size' in report
        assert 'statistics' in report
        assert 'configuration' in report
        assert 'model_limits' in report
        assert 'usage_history' in report
        
        # Verify report content
        assert report['current_context_size'] == 2800  # 2000 + 500 + 300
        assert report['statistics']['total_tokens_used'] == 800
        assert report['statistics']['requests_made'] == 2
        assert report['statistics']['compressions_performed'] == 1
        
        # Verify history is limited to last 10 entries
        assert len(report['usage_history']) <= 10


class TestTokenManagerIntegration:
    """Integration tests for TokenManager with real configuration."""
    
    @pytest.fixture
    def mock_config_manager_integration(self):
        """Create a mock ConfigManager for integration testing."""
        mock_config = Mock(spec=ConfigManager)
        mock_config.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.9,
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': False
        }
        
        # Mock the enhanced model configuration methods
        def mock_get_model_token_limit(model_name):
            known_limits = {
                'models/gemini-2.0-flash': 1048576,
                'models/gemini-1.5-pro': 2097152,
                'gpt-4': 8192,
                'test-model': 4096,  # For integration tests
            }
            return known_limits.get(model_name, 8192)
        
        def mock_get_model_info(model_name):
            token_limit = mock_get_model_token_limit(model_name)
            return {
                'family': 'GPT_4',
                'token_limit': token_limit,
                'capabilities': {
                    'vision': False,
                    'function_calling': True,
                    'streaming': True
                }
            }
        
        mock_config.get_model_token_limit.side_effect = mock_get_model_token_limit
        mock_config.get_model_info.side_effect = mock_get_model_info
        mock_config.get_framework_config.return_value = {
            'context_size_ratio': 0.8,
            'workspace_path': '.',
            'log_level': 'INFO'
        }
        
        return mock_config
    
    @pytest.fixture
    def token_manager_integration(self, mock_config_manager_integration):
        """Create a TokenManager instance for integration testing."""
        return TokenManager(mock_config_manager_integration)
    
    def test_with_real_config_manager(self):
        """Test TokenManager with real ConfigManager instance."""
        from autogen_framework.config_manager import ConfigManager
        
        # Create real config manager
        config_manager = ConfigManager(load_env=False)  # Don't load .env for test
        
        # Mock the methods to return test values
        with patch.object(config_manager, 'get_token_config') as mock_get_config, \
             patch.object(config_manager, 'get_model_token_limit') as mock_get_limit, \
             patch.object(config_manager, 'get_model_info') as mock_get_info, \
             patch.object(config_manager, 'get_framework_config') as mock_get_framework:
            
            mock_get_config.return_value = {
                'default_token_limit': 4096,
                'compression_threshold': 0.8,
                'compression_enabled': True,
                'compression_target_ratio': 0.4,
                'verbose_logging': True
            }
            
            mock_get_limit.return_value = 4096
            mock_get_info.return_value = {
                'family': 'GPT_4',
                'token_limit': 4096,
                'capabilities': {'vision': False, 'function_calling': True, 'streaming': True}
            }
            mock_get_framework.return_value = {
                'context_size_ratio': 0.8,
                'workspace_path': '.',
                'log_level': 'INFO'
            }
            
            token_manager = TokenManager(config_manager)
            
            # Verify configuration loaded correctly
            assert token_manager.token_config['default_token_limit'] == 4096
            assert token_manager.token_config['compression_threshold'] == 0.8
            assert token_manager.token_config['verbose_logging'] is True
            
            # Test functionality with real config
            token_manager.update_token_usage('test-model', 1000, 'integration_test')
            
            result = token_manager.check_token_limit('test-model')
            assert result.current_tokens == 1000
            assert result.model_limit == 4096  # Uses ConfigManager
            assert result.needs_compression is False  # Below 80% threshold
    
    def test_edge_case_zero_model_limit(self, token_manager_integration):
        """Test behavior with zero model limit."""
        # Mock ConfigManager to return zero limit for a specific model
        token_manager_integration.config_manager.get_model_token_limit.side_effect = lambda model: 0 if model == 'zero-limit-model' else 8192
        
        result = token_manager_integration.check_token_limit('zero-limit-model')
        
        assert result.model_limit == 0
        assert result.percentage_used == 0  # Should handle division by zero
        assert result.needs_compression is False
    
    def test_peak_usage_tracking(self, token_manager_integration):
        """Test peak usage tracking across multiple operations."""
        # Simulate varying token usage
        token_manager_integration.update_token_usage('gpt-4', 1000, 'op1')  # Peak: 1000
        token_manager_integration.update_token_usage('gpt-4', 500, 'op2')   # Peak: 1500
        token_manager_integration.reset_context_size()                      # Context: 0, Peak: 1500
        token_manager_integration.update_token_usage('gpt-4', 800, 'op3')   # Context: 800, Peak: 1500
        token_manager_integration.update_token_usage('gpt-4', 1200, 'op4')  # Context: 2000, Peak: 2000
        
        stats = token_manager_integration.get_usage_statistics()
        assert stats.peak_token_usage == 2000
        assert stats.total_tokens_used == 3500  # 1000 + 500 + 800 + 1200
        assert stats.requests_made == 4
    
    def test_usage_history_management(self, token_manager_integration):
        """Test usage history recording and management."""
        # Generate multiple operations
        operations = [
            ('gpt-4', 100, 'op1'),
            ('claude-3-opus', 200, 'op2'),
            ('models/gemini-2.0-flash', 150, 'op3')
        ]
        
        for model, tokens, operation in operations:
            token_manager_integration.update_token_usage(model, tokens, operation)
        
        # Add compression and reset events
        token_manager_integration.increment_compression_count()
        token_manager_integration.reset_context_size()
        
        # Verify history contains all events
        history = token_manager_integration.usage_stats['usage_history']
        assert len(history) == 5  # 3 updates + 1 compression + 1 reset
        
        # Verify detailed report limits history to 10 entries
        report = token_manager_integration.get_detailed_usage_report()
        assert len(report['usage_history']) == 5  # All entries since < 10


class TestTokenManagerEdgeCases:
    """Additional edge case tests for TokenManager functionality."""
    
    @pytest.fixture
    def mock_config_manager_edge(self):
        """Create a mock ConfigManager for edge case testing."""
        mock_config = Mock(spec=ConfigManager)
        mock_config.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.9,
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': False
        }
        
        # Mock the enhanced model configuration methods
        def mock_get_model_token_limit(model_name):
            known_limits = {
                'models/gemini-2.0-flash': 1048576,
                'gpt-4': 8192,
            }
            return known_limits.get(model_name, 8192)
        
        def mock_get_model_info(model_name):
            token_limit = mock_get_model_token_limit(model_name)
            return {
                'family': 'GPT_4',
                'token_limit': token_limit,
                'capabilities': {
                    'vision': False,
                    'function_calling': True,
                    'streaming': True
                }
            }
        
        mock_config.get_model_token_limit.side_effect = mock_get_model_token_limit
        mock_config.get_model_info.side_effect = mock_get_model_info
        mock_config.get_framework_config.return_value = {
            'context_size_ratio': 0.8,
            'workspace_path': '.',
            'log_level': 'INFO'
        }
        
        return mock_config
    
    @pytest.fixture
    def token_manager_edge(self, mock_config_manager_edge):
        """Create a TokenManager instance for edge case testing."""
        return TokenManager(mock_config_manager_edge)
    
    def test_check_token_limit_exactly_at_threshold(self, token_manager_edge):
        """Test token limit checking when exactly at threshold."""
        # Set context size exactly at threshold (90% of 8192 = 7372.8, round up to ensure >= 90%)
        token_manager_edge.current_context_size = int(8192 * 0.9) + 1  # 7373
        
        result = token_manager_edge.check_token_limit('gpt-4')
        
        assert result.current_tokens == 7373
        assert result.model_limit == 8192
        assert result.percentage_used >= 0.9  # Should be at or above 90%
        assert result.needs_compression is True  # Should trigger at exactly 90%
    
    def test_check_token_limit_just_below_threshold(self, token_manager_edge):
        """Test token limit checking when just below threshold."""
        # Set context size just below threshold
        token_manager_edge.current_context_size = int(8192 * 0.9) - 1  # 7371
        
        result = token_manager_edge.check_token_limit('gpt-4')
        
        assert result.current_tokens == 7371
        assert result.needs_compression is False  # Should not trigger below 90%
    
    def test_check_token_limit_custom_threshold(self, mock_config_manager_edge):
        """Test token limit checking with custom threshold."""
        # Set custom threshold
        mock_config_manager_edge.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.75,  # Custom 75% threshold
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': False
        }
        
        token_manager = TokenManager(mock_config_manager_edge)
        token_manager.current_context_size = int(8192 * 0.8)  # 6553 (80% usage)
        
        result = token_manager.check_token_limit('gpt-4')
        
        assert result.needs_compression is True  # Should trigger at 80% with 75% threshold
    
    def test_update_token_usage_large_numbers(self, token_manager_edge):
        """Test token usage update with very large token counts."""
        large_token_count = 1000000
        
        token_manager_edge.update_token_usage('models/gemini-2.0-flash', large_token_count, 'large_operation')
        
        assert token_manager_edge.current_context_size == large_token_count
        assert token_manager_edge.usage_stats['total_tokens_used'] == large_token_count
        assert token_manager_edge.usage_stats['peak_token_usage'] == large_token_count
        
        # Verify it doesn't cause overflow or other issues
        result = token_manager_edge.check_token_limit('models/gemini-2.0-flash')
        assert result.current_tokens == large_token_count
        assert result.model_limit == 1048576  # Gemini limit
        assert result.needs_compression is True  # Should need compression
    
    def test_multiple_resets_and_usage_tracking(self, token_manager_edge):
        """Test multiple context resets and usage tracking."""
        # First session
        token_manager_edge.update_token_usage('gpt-4', 1000, 'session1_op1')
        token_manager_edge.update_token_usage('gpt-4', 500, 'session1_op2')
        assert token_manager_edge.current_context_size == 1500
        
        # Reset context
        token_manager_edge.reset_context_size()
        assert token_manager_edge.current_context_size == 0
        
        # Second session
        token_manager_edge.update_token_usage('gpt-4', 800, 'session2_op1')
        assert token_manager_edge.current_context_size == 800
        
        # Verify total usage is cumulative but context is reset
        stats = token_manager_edge.get_usage_statistics()
        assert stats.total_tokens_used == 2300  # 1000 + 500 + 800
        assert stats.requests_made == 3
        assert stats.peak_token_usage == 1500  # Peak from first session
        
        # Verify history includes reset event
        history = token_manager_edge.usage_stats['usage_history']
        reset_entries = [entry for entry in history if entry.get('operation') == 'context_reset']
        assert len(reset_entries) == 1
    
    def test_compression_count_tracking(self, token_manager_edge):
        """Test compression count tracking and statistics."""
        initial_stats = token_manager_edge.get_usage_statistics()
        assert initial_stats.compressions_performed == 0
        
        # Perform multiple compressions
        for i in range(3):
            token_manager_edge.increment_compression_count()
        
        final_stats = token_manager_edge.get_usage_statistics()
        assert final_stats.compressions_performed == 3
        
        # Verify compression events in history
        history = token_manager_edge.usage_stats['usage_history']
        compression_entries = [entry for entry in history if entry.get('operation') == 'compression_performed']
        assert len(compression_entries) == 3
        
        # Verify compression totals are tracked correctly
        for i, entry in enumerate(compression_entries):
            assert entry['compressions_total'] == i + 1
    
    def test_detailed_report_with_empty_history(self, token_manager_edge):
        """Test detailed usage report with no usage history."""
        report = token_manager_edge.get_detailed_usage_report()
        
        assert report['current_context_size'] == 0
        assert report['statistics']['total_tokens_used'] == 0
        assert report['statistics']['requests_made'] == 0
        assert report['statistics']['compressions_performed'] == 0
        assert report['statistics']['average_tokens_per_request'] == 0.0
        assert report['statistics']['peak_token_usage'] == 0
        assert len(report['usage_history']) == 0
        assert 'configuration' in report
        assert 'model_limits' in report
    
    def test_average_tokens_calculation_edge_cases(self, token_manager_edge):
        """Test average tokens per request calculation edge cases."""
        # Initially should be 0
        stats = token_manager_edge.get_usage_statistics()
        assert stats.average_tokens_per_request == 0.0
        
        # Single request
        token_manager_edge.update_token_usage('gpt-4', 100, 'single_request')
        stats = token_manager_edge.get_usage_statistics()
        assert stats.average_tokens_per_request == 100.0
        
        # Multiple requests with different sizes
        token_manager_edge.update_token_usage('gpt-4', 200, 'request2')
        token_manager_edge.update_token_usage('gpt-4', 300, 'request3')
        
        stats = token_manager_edge.get_usage_statistics()
        expected_average = (100 + 200 + 300) / 3  # 200.0
        assert stats.average_tokens_per_request == expected_average
    
    def test_model_limits_case_sensitivity(self, token_manager_edge):
        """Test model limits handling with different case variations."""
        # Test exact match
        assert token_manager_edge.get_model_limit('gpt-4') == 8192
        
        # Test case variations (should use default for unknown)
        assert token_manager_edge.get_model_limit('GPT-4') == 8192  # Uses default
        assert token_manager_edge.get_model_limit('Gpt-4') == 8192  # Uses default
        
        # Test partial matches (should use default)
        assert token_manager_edge.get_model_limit('gpt-4-custom') == 8192  # Uses default
        assert token_manager_edge.get_model_limit('gpt') == 8192  # Uses default