"""
Configuration Manager for the AutoGen Multi-Agent Framework.

This module provides centralized configuration management using environment variables
with validation, default values, and clear error messages. It supports loading
configuration from .env files and environment variables.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


class ConfigManager:
    """
    Centralized configuration manager for the AutoGen framework.
    
    Handles loading configuration from environment variables and .env files,
    with validation and clear error messages for missing or invalid values.
    """
    
    def __init__(self, env_file: Optional[str] = None, load_env: bool = True):
        """
        Initialize the configuration manager.
        
        Args:
            env_file: Path to .env file to load. If None, looks for .env in current directory.
            load_env: Whether to automatically load environment variables from .env file.
        """
        self.logger = logging.getLogger(__name__)
        self._config_cache: Dict[str, Any] = {}
        
        if load_env:
            self._load_env_file(env_file)
    
    def _load_env_file(self, env_file: Optional[str] = None) -> None:
        """
        Load environment variables from .env file.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in current directory.
        """
        if env_file is None:
            # Look for .env file in current directory and parent directories
            current_dir = Path.cwd()
            for path in [current_dir] + list(current_dir.parents):
                env_path = path / ".env"
                if env_path.exists():
                    env_file = str(env_path)
                    break
        
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
            self.logger.info(f"Loaded environment configuration from {env_file}")
        else:
            self.logger.info("No .env file found, using system environment variables only")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration from environment variables.
        
        Returns:
            Dictionary containing LLM configuration parameters.
            
        Raises:
            ConfigurationError: If required LLM configuration is missing or invalid.
        """
        config = {}
        
        # Required LLM configuration
        required_vars = {
            'LLM_BASE_URL': 'base_url',
            'LLM_MODEL': 'model', 
            'LLM_API_KEY': 'api_key'
        }
        
        missing_vars = []
        for env_var, config_key in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                missing_vars.append(env_var)
            else:
                config[config_key] = value
        
        if missing_vars:
            raise ConfigurationError(
                f"Missing required LLM configuration environment variables: {', '.join(missing_vars)}. "
                f"Please set these variables in your .env file or environment. "
                f"See .env.example for the required format."
            )
        
        # Optional LLM configuration with defaults
        optional_vars = {
            'LLM_TEMPERATURE': ('temperature', 0.7, float),
            'LLM_MAX_OUTPUT_TOKENS': ('max_output_tokens', 8192, int),
            'LLM_TIMEOUT_SECONDS': ('timeout', 60, int)
        }
        
        for env_var, (config_key, default_value, value_type) in optional_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    config[config_key] = value_type(value)
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid value for {env_var}: '{value}'. Expected {value_type.__name__}."
                    )
            else:
                config[config_key] = default_value
        
        # Validate configuration values
        self._validate_llm_config(config)
        
        return config
    
    def _validate_llm_config(self, config: Dict[str, Any]) -> None:
        """
        Validate LLM configuration values.
        
        Args:
            config: LLM configuration dictionary to validate.
            
        Raises:
            ConfigurationError: If configuration values are invalid.
        """
        # Validate base_url format
        base_url = config.get('base_url', '')
        if not (base_url.startswith('http://') or base_url.startswith('https://')):
            raise ConfigurationError(
                f"Invalid LLM_BASE_URL: '{base_url}'. Must start with http:// or https://"
            )
        
        # Validate temperature range
        temperature = config.get('temperature', 0.7)
        if not (0.0 <= temperature <= 2.0):
            raise ConfigurationError(
                f"Invalid LLM_TEMPERATURE: {temperature}. Must be between 0.0 and 2.0"
            )
        
        # Validate positive integers
        for key, min_value in [('max_output_tokens', 1), ('timeout', 1)]:
            value = config.get(key, 1)
            if value < min_value:
                env_var = f"LLM_{key.upper()}"
                raise ConfigurationError(
                    f"Invalid {env_var}: {value}. Must be greater than {min_value - 1}"
                )
    
    def get_framework_config(self) -> Dict[str, Any]:
        """
        Get framework configuration from environment variables.
        
        Returns:
            Dictionary containing framework configuration parameters.
        """
        config = {}
        
        # Framework configuration with defaults
        framework_vars = {
            'WORKSPACE_PATH': ('workspace_path', '.', str),
            'LOG_LEVEL': ('log_level', 'INFO', str),
            'LOG_FILE': ('log_file', 'logs/framework.log', str),
            'SHELL_TIMEOUT_SECONDS': ('shell_timeout', 30, int),
            'SHELL_MAX_RETRIES': ('shell_max_retries', 2, int)
        }
        
        for env_var, (config_key, default_value, value_type) in framework_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    config[config_key] = value_type(value)
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid value for {env_var}: '{value}'. Expected {value_type.__name__}."
                    )
            else:
                config[config_key] = default_value
        
        # Validate framework configuration
        self._validate_framework_config(config)
        
        return config
    
    def _validate_framework_config(self, config: Dict[str, Any]) -> None:
        """
        Validate framework configuration values.
        
        Args:
            config: Framework configuration dictionary to validate.
            
        Raises:
            ConfigurationError: If configuration values are invalid.
        """
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        log_level = config.get('log_level', 'INFO').upper()
        if log_level not in valid_log_levels:
            raise ConfigurationError(
                f"Invalid LOG_LEVEL: '{log_level}'. Must be one of: {', '.join(valid_log_levels)}"
            )
        config['log_level'] = log_level
        
        # Validate positive integers
        for key, min_value in [('shell_timeout', 1), ('shell_max_retries', 0)]:
            value = config.get(key, 1)
            if value < min_value:
                env_var = f"{key.upper().replace('_', '_')}"
                raise ConfigurationError(
                    f"Invalid {env_var}: {value}. Must be greater than {min_value - 1}"
                )
    
    def get_token_config(self) -> Dict[str, Any]:
        """
        Get token management configuration from environment variables.
        
        Returns:
            Dictionary containing token management configuration parameters.
        """
        config = {}
        
        # Token management configuration with defaults
        token_vars = {
            'DEFAULT_TOKEN_LIMIT': ('default_token_limit', 8192, int),
            'TOKEN_COMPRESSION_THRESHOLD': ('compression_threshold', 0.9, float),
            'CONTEXT_COMPRESSION_ENABLED': ('compression_enabled', True, lambda x: x.lower() == 'true'),
            'COMPRESSION_TARGET_RATIO': ('compression_target_ratio', 0.5, float),
            'VERBOSE_TOKEN_LOGGING': ('verbose_logging', False, lambda x: x.lower() == 'true')
        }
        
        for env_var, (config_key, default_value, value_type) in token_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    if callable(value_type):
                        config[config_key] = value_type(value)
                    else:
                        config[config_key] = value_type(value)
                except (ValueError, AttributeError):
                    self.logger.warning(
                        f"Invalid value for {env_var}: '{value}'. Using default: {default_value}"
                    )
                    config[config_key] = default_value
            else:
                config[config_key] = default_value
        
        # Validate token configuration
        validated_config = self.validate_token_config(config)
        
        return validated_config
    
    def validate_token_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate token configuration and log warnings for invalid values.
        
        Args:
            config: Token configuration dictionary to validate.
            
        Returns:
            Validated configuration dictionary with corrected values.
        """
        validated_config = config.copy()
        
        # Validate compression threshold (must be between 0 and 1)
        threshold = config.get('compression_threshold', 0.9)
        if threshold <= 0 or threshold > 1:
            self.logger.warning(f"Invalid compression_threshold: {threshold}. Using default: 0.9")
            validated_config['compression_threshold'] = 0.9
        
        # Validate default token limit (must be positive)
        token_limit = config.get('default_token_limit', 8192)
        if token_limit <= 0:
            self.logger.warning(f"Invalid default_token_limit: {token_limit}. Using default: 8192")
            validated_config['default_token_limit'] = 8192
        
        # Validate compression target ratio (must be between 0 and 1)
        target_ratio = config.get('compression_target_ratio', 0.5)
        if target_ratio <= 0 or target_ratio >= 1:
            self.logger.warning(f"Invalid compression_target_ratio: {target_ratio}. Using default: 0.5")
            validated_config['compression_target_ratio'] = 0.5
        
        return validated_config

    def get_test_config(self) -> Dict[str, Any]:
        """
        Get test-specific configuration from environment variables.
        
        Returns:
            Dictionary containing test configuration parameters.
        """
        config = {}
        
        # Test configuration - directly use LLM_* variables from environment
        config = {
            'base_url': os.getenv('LLM_BASE_URL', 'http://localhost:8888/openai/v1'),
            'model': os.getenv('LLM_MODEL', 'models/test-model'),
            'api_key': os.getenv('LLM_API_KEY', 'test-key-for-development'),
            'workspace_path': os.getenv('WORKSPACE_PATH', 'test_workspace'),
            'log_level': os.getenv('LOG_LEVEL', 'DEBUG')
        }
        
        return config
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get complete configuration combining LLM, framework, and test settings.
        
        Returns:
            Dictionary containing all configuration parameters.
        """
        config = {}
        config.update(self.get_framework_config())
        config.update(self.get_llm_config())
        
        # Add test config if in test environment
        if os.getenv('TESTING') == 'true':
            test_config = self.get_test_config()
            # Prefix test config keys to avoid conflicts
            for key, value in test_config.items():
                config[f'test_{key}'] = value
        
        return config
    
    def validate_required_config(self) -> List[str]:
        """
        Validate that all required configuration is present.
        
        Returns:
            List of validation errors. Empty list if all configuration is valid.
        """
        errors = []
        
        try:
            self.get_llm_config()
        except ConfigurationError as e:
            errors.append(f"LLM Configuration: {str(e)}")
        
        try:
            self.get_framework_config()
        except ConfigurationError as e:
            errors.append(f"Framework Configuration: {str(e)}")
        
        return errors
    
    def print_config_status(self) -> None:
        """Print current configuration status for debugging."""
        print("=== Configuration Status ===")
        
        # Check for .env file
        env_files = []
        current_dir = Path.cwd()
        for path in [current_dir] + list(current_dir.parents):
            env_path = path / ".env"
            if env_path.exists():
                env_files.append(str(env_path))
        
        if env_files:
            print(f"✓ Found .env file(s): {', '.join(env_files)}")
        else:
            print("⚠ No .env file found")
        
        # Validate configuration
        errors = self.validate_required_config()
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("✓ All required configuration is valid")
        
        # Show current environment variables (without sensitive values)
        print("\n=== Environment Variables ===")
        sensitive_vars = ['API_KEY', 'KEY', 'SECRET', 'PASSWORD', 'TOKEN']
        
        all_vars = [
            'LLM_BASE_URL', 'LLM_MODEL', 'LLM_API_KEY', 'LLM_TEMPERATURE',
            'LLM_MAX_OUTPUT_TOKENS', 'LLM_TIMEOUT_SECONDS',
            'WORKSPACE_PATH', 'LOG_LEVEL', 'LOG_FILE',
            'SHELL_TIMEOUT_SECONDS', 'SHELL_MAX_RETRIES'
        ]
        
        for var in all_vars:
            value = os.getenv(var)
            if value:
                # Hide sensitive values
                if any(sensitive in var for sensitive in sensitive_vars):
                    display_value = f"{'*' * (len(value) - 4)}{value[-4:]}" if len(value) > 4 else "****"
                else:
                    display_value = value
                print(f"  {var}={display_value}")
            else:
                print(f"  {var}=<not set>")


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager instance.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reset_config_manager() -> None:
    """Reset the global configuration manager instance (useful for testing)."""
    global _config_manager
    _config_manager = None