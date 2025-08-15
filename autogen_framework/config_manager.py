"""
Configuration Manager for the AutoGen Multi-Agent Framework.

This module provides centralized configuration management using environment variables
with validation, default values, and clear error messages. It supports loading
configuration from .env files and environment variables, as well as model-specific
configurations from JSON config files.
"""

import os
import json
import re
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


class ModelConfigurationError(ConfigurationError):
    """Raised when model configuration is invalid."""
    pass


class ConfigManager:
    """
    Centralized configuration manager for the AutoGen framework.
    
    Handles loading configuration from environment variables and .env files,
    with validation and clear error messages for missing or invalid values.
    """
    
    def __init__(self, env_file: Optional[str] = None, load_env: bool = True,
                 config_dir: Optional[str] = None, models_config_file: Optional[str] = None,
                 framework_config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            env_file: Path to .env file to load. If None, looks for .env in current directory.
            load_env: Whether to automatically load environment variables from .env file.
            config_dir: Configuration directory path (overrides CONFIG_DIR env var).
            models_config_file: Models configuration file path (overrides MODELS_CONFIG_FILE env var).
            framework_config_file: Framework configuration file path (overrides FRAMEWORK_CONFIG_FILE env var).
        """
        self.logger = logging.getLogger(__name__)
        self._config_cache: Dict[str, Any] = {}
        self.model_configs: Dict[str, Any] = {}
        self._model_patterns: List[Dict[str, Any]] = []
        self._model_defaults: Dict[str, Any] = {}
        
        # Store configuration path overrides
        self._config_dir_override = config_dir
        self._models_config_file_override = models_config_file
        self._framework_config_file_override = framework_config_file
        
        if load_env:
            self._load_env_file(env_file)
        
        # Load model configurations
        self._load_model_configurations()
    
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
        Get LLM configuration from environment variables and framework config.
        
        Connection settings come from environment variables.
        Behavior settings come from framework configuration (config files with env var fallback).
        
        Returns:
            Dictionary containing LLM configuration parameters.
            
        Raises:
            ConfigurationError: If required LLM configuration is missing or invalid.
        """
        self.logger.debug("Loading LLM configuration...")
        config = {}
        
        # Required LLM connection configuration (must be in environment variables)
        required_vars = {
            'LLM_BASE_URL': 'base_url',
            'LLM_MODEL': 'model', 
            'LLM_API_KEY': 'api_key'
        }
        
        missing_vars = []
        loaded_vars = []
        for env_var, config_key in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                missing_vars.append(env_var)
            else:
                config[config_key] = value
                loaded_vars.append(env_var)
                # Don't log API key value for security
                if 'API_KEY' in env_var:
                    self.logger.debug(f"Loaded {config_key} from environment variable {env_var}: [REDACTED]")
                else:
                    self.logger.debug(f"Loaded {config_key} from environment variable {env_var}: {value}")
        
        if missing_vars:
            error_msg = (
                f"Missing required LLM configuration environment variables: {', '.join(missing_vars)}. "
                f"Please set these variables in your .env file or environment. "
                f"See .env.example for the required format. "
                f"Successfully loaded: {', '.join(loaded_vars) if loaded_vars else 'none'}"
            )
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        self.logger.info(f"Successfully loaded LLM connection configuration from environment variables: {', '.join(loaded_vars)}")
        
        # Behavior settings come from framework configuration
        framework_config = self.get_framework_config()
        behavior_settings = {
            'temperature': framework_config.get('temperature', 0.7),
            'max_output_tokens': framework_config.get('max_output_tokens', 8192),
            'timeout': framework_config.get('timeout', 60)
        }
        
        config.update(behavior_settings)
        
        # Log behavior settings sources
        for key, value in behavior_settings.items():
            self.logger.debug(f"LLM behavior setting {key}: {value} (source: framework configuration)")
        
        # Validate configuration values
        self._validate_llm_config(config)
        
        self.logger.info("LLM configuration loaded and validated successfully")
        return config
    
    def _validate_llm_config(self, config: Dict[str, Any]) -> None:
        """
        Validate LLM configuration values with comprehensive error messages.
        
        Args:
            config: LLM configuration dictionary to validate.
            
        Raises:
            ConfigurationError: If configuration values are invalid.
        """
        self.logger.debug("Validating LLM configuration...")
        
        # Validate base_url format
        base_url = config.get('base_url', '')
        if not (base_url.startswith('http://') or base_url.startswith('https://')):
            error_msg = (
                f"Invalid LLM_BASE_URL: '{base_url}'. Must start with http:// or https://. "
                f"Examples: 'http://localhost:8888/openai/v1', 'https://api.openai.com/v1'"
            )
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        self.logger.debug(f"LLM base URL validated: {base_url}")
        
        # Validate temperature range
        temperature = config.get('temperature', 0.7)
        if not (0.0 <= temperature <= 2.0):
            error_msg = (
                f"Invalid LLM_TEMPERATURE: {temperature}. Must be between 0.0 and 2.0. "
                f"Lower values (0.1-0.3) make output more focused, higher values (0.7-1.0) more creative. "
                f"Values above 1.0 are rarely useful."
            )
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        self.logger.debug(f"LLM temperature validated: {temperature}")
        
        # Validate positive integers with helpful ranges
        validation_rules = {
            'max_output_tokens': {
                'min_value': 1,
                'recommended_range': (1024, 8192),
                'description': 'Maximum tokens in model response'
            },
            'timeout': {
                'min_value': 1,
                'recommended_range': (30, 300),
                'description': 'Request timeout in seconds'
            }
        }
        
        for key, rules in validation_rules.items():
            value = config.get(key, 1)
            min_value = rules['min_value']
            recommended_range = rules['recommended_range']
            description = rules['description']
            
            if value < min_value:
                env_var = f"LLM_{key.upper()}"
                error_msg = (
                    f"Invalid {env_var}: {value}. Must be greater than {min_value - 1}. "
                    f"{description}. Recommended range: {recommended_range[0]}-{recommended_range[1]}"
                )
                self.logger.error(error_msg)
                raise ConfigurationError(error_msg)
            
            # Log warnings for values outside recommended range
            if not (recommended_range[0] <= value <= recommended_range[1]):
                self.logger.warning(
                    f"{key} value {value} is outside recommended range {recommended_range[0]}-{recommended_range[1]}. "
                    f"This may cause performance issues or timeouts."
                )
            else:
                self.logger.debug(f"LLM {key} validated: {value}")
        
        self.logger.info("LLM configuration validation completed successfully")
    
    def get_framework_config(self) -> Dict[str, Any]:
        """
        Get framework configuration from config files and environment variables.
        
        Precedence: args > env vars > config files > built-in defaults
        
        Returns:
            Dictionary containing framework configuration parameters.
        """
        config = {}
        
        # 1. Load built-in defaults first (lowest precedence)
        default_config = self._get_framework_defaults()
        config.update(default_config)
        
        # 2. Load from config file (overrides defaults)
        framework_config_path = self._get_framework_config_path()
        if framework_config_path.exists():
            try:
                file_config = self._load_config_file(str(framework_config_path))
                flattened_config = self._flatten_framework_config(file_config)
                config.update(flattened_config)
                self.logger.info(f"Loaded framework configuration from {framework_config_path}")
                
                # Log which settings were loaded from config file
                for key, value in flattened_config.items():
                    self.logger.debug(f"Using {key} from config file: {value}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load framework config from {framework_config_path}: {e}")
                self.logger.info("Using environment variables and defaults only")
        else:
            self.logger.debug(f"No framework config file found at {framework_config_path}")
        
        # 3. Environment variables override config file values (higher precedence)
        # Connection and environment settings (should remain in env vars)
        connection_env_vars = {
            'WORKSPACE_PATH': ('workspace_path', '.', str),
            'LOG_LEVEL': ('log_level', 'INFO', str),
            'LOG_FILE': ('log_file', 'logs/framework.log', str),
        }
        
        # Behavior settings that should be deprecated from env vars
        deprecated_behavior_vars = {
            'LLM_TEMPERATURE': ('temperature', 0.7, float),
            'LLM_MAX_OUTPUT_TOKENS': ('max_output_tokens', 8192, int),
            'LLM_TIMEOUT_SECONDS': ('timeout', 60, int),
            'SHELL_TIMEOUT_SECONDS': ('shell_timeout', 30, int),
            'SHELL_MAX_RETRIES': ('shell_max_retries', 2, int)
        }
        
        # Task completion configuration with defaults
        task_completion_vars = {
            'TASK_REAL_TIME_UPDATES_ENABLED': ('task_real_time_updates_enabled', True, lambda x: x.lower() == 'true'),
            'TASK_FALLBACK_TO_BATCH_ENABLED': ('task_fallback_to_batch_enabled', True, lambda x: x.lower() == 'true'),
            'TASK_MAX_INDIVIDUAL_UPDATE_RETRIES': ('task_max_individual_update_retries', 3, int),
            'TASK_INDIVIDUAL_UPDATE_RETRY_DELAY': ('task_individual_update_retry_delay', 0.1, float),
            'TASK_FILE_LOCK_TIMEOUT': ('task_file_lock_timeout', 5, int),
            'TASK_DETAILED_LOGGING_ENABLED': ('task_detailed_logging_enabled', True, lambda x: x.lower() == 'true'),
            'TASK_RECOVERY_MECHANISM_ENABLED': ('task_recovery_mechanism_enabled', True, lambda x: x.lower() == 'true')
        }
        
        # Process connection/environment variables (these should stay in env vars)
        for env_var, (config_key, default_value, value_type) in connection_env_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    config[config_key] = value_type(value)
                    self.logger.debug(f"Using {config_key} from environment variable {env_var}")
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid value for {env_var}: '{value}'. Expected {value_type.__name__}."
                    )
        
        # Process deprecated behavior variables with migration warnings
        deprecated_vars_found = []
        for env_var, (config_key, default_value, value_type) in deprecated_behavior_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    config[config_key] = value_type(value)
                    deprecated_vars_found.append(env_var)
                    self.logger.debug(f"Using {config_key} from environment variable {env_var} (deprecated)")
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid value for {env_var}: '{value}'. Expected {value_type.__name__}."
                    )
        
        # Issue migration warnings for deprecated environment variables
        if deprecated_vars_found:
            self._issue_migration_warnings(deprecated_vars_found)
        
        # Process task completion variables
        for env_var, (config_key, default_value, value_type) in task_completion_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    config[config_key] = value_type(value)
                    self.logger.debug(f"Using {config_key} from environment variable {env_var}")
                except ValueError:
                    raise ConfigurationError(
                        f"Invalid value for {env_var}: '{value}'. Expected {value_type.__name__}."
                    )
        
        # Validate framework configuration
        self._validate_framework_config(config)
        
        return config
    
    def _get_framework_defaults(self) -> Dict[str, Any]:
        """
        Get built-in default framework configuration values.
        
        Returns:
            Dictionary containing default framework configuration.
        """
        return {
            # Connection and environment settings
            'workspace_path': '.',
            'log_level': 'INFO',
            'log_file': 'logs/framework.log',
            
            # Behavior settings (should come from config files)
            'temperature': 0.7,
            'max_output_tokens': 4096,
            'timeout': 60,
            'shell_timeout': 30,
            'shell_max_retries': 2,
            
            # Context settings
            'compression_threshold': 0.9,
            'context_size_ratio': 0.8,
            
            # Workflow settings
            'auto_approve': False,
            'verbose': False,
            'reset_session_on_start': False,
            
            # Memory settings
            'max_history_entries': 100,
            'compression_enabled': True,
            'pattern_learning_enabled': True,
            
            # Task completion settings
            'task_real_time_updates_enabled': True,
            'task_fallback_to_batch_enabled': True,
            'task_max_individual_update_retries': 3,
            'task_individual_update_retry_delay': 0.1,
            'task_file_lock_timeout': 5,
            'task_detailed_logging_enabled': True,
            'task_recovery_mechanism_enabled': True
        }
    
    def _flatten_framework_config(self, file_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten framework configuration from nested JSON structure.
        
        Args:
            file_config: Configuration loaded from JSON file.
            
        Returns:
            Flattened configuration dictionary.
        """
        flattened = {}
        
        # Map nested config structure to flat keys
        if 'llm' in file_config:
            llm_config = file_config['llm']
            if 'temperature' in llm_config:
                flattened['temperature'] = llm_config['temperature']
            if 'max_output_tokens' in llm_config:
                flattened['max_output_tokens'] = llm_config['max_output_tokens']
            if 'timeout_seconds' in llm_config:
                flattened['timeout'] = llm_config['timeout_seconds']
        
        if 'shell' in file_config:
            shell_config = file_config['shell']
            if 'timeout_seconds' in shell_config:
                flattened['shell_timeout'] = shell_config['timeout_seconds']
            if 'max_retries' in shell_config:
                flattened['shell_max_retries'] = shell_config['max_retries']
        
        if 'context' in file_config:
            context_config = file_config['context']
            if 'compression_threshold' in context_config:
                flattened['compression_threshold'] = context_config['compression_threshold']
            if 'context_size_ratio' in context_config:
                flattened['context_size_ratio'] = context_config['context_size_ratio']
        
        if 'workflow' in file_config:
            workflow_config = file_config['workflow']
            if 'auto_approve' in workflow_config:
                flattened['auto_approve'] = workflow_config['auto_approve']
            if 'verbose' in workflow_config:
                flattened['verbose'] = workflow_config['verbose']
            if 'reset_session_on_start' in workflow_config:
                flattened['reset_session_on_start'] = workflow_config['reset_session_on_start']
        
        if 'memory' in file_config:
            memory_config = file_config['memory']
            if 'max_history_entries' in memory_config:
                flattened['max_history_entries'] = memory_config['max_history_entries']
            if 'compression_enabled' in memory_config:
                flattened['compression_enabled'] = memory_config['compression_enabled']
            if 'pattern_learning_enabled' in memory_config:
                flattened['pattern_learning_enabled'] = memory_config['pattern_learning_enabled']
        
        # Handle task completion settings
        if 'task_completion' in file_config:
            task_config = file_config['task_completion']
            task_mapping = {
                'real_time_updates_enabled': 'task_real_time_updates_enabled',
                'fallback_to_batch_enabled': 'task_fallback_to_batch_enabled',
                'max_individual_update_retries': 'task_max_individual_update_retries',
                'individual_update_retry_delay': 'task_individual_update_retry_delay',
                'file_lock_timeout': 'task_file_lock_timeout',
                'detailed_logging_enabled': 'task_detailed_logging_enabled',
                'recovery_mechanism_enabled': 'task_recovery_mechanism_enabled'
            }
            
            for file_key, flat_key in task_mapping.items():
                if file_key in task_config:
                    flattened[flat_key] = task_config[file_key]
        
        return flattened
    
    def _issue_migration_warnings(self, deprecated_vars: List[str]) -> None:
        """
        Issue comprehensive migration warnings for deprecated environment variables.
        
        Args:
            deprecated_vars: List of deprecated environment variable names found.
        """
        migration_mapping = {
            'LLM_TEMPERATURE': {
                'location': 'config/framework.json under llm.temperature',
                'example_value': '0.7',
                'description': 'Controls response creativity (0.0-2.0)'
            },
            'LLM_MAX_OUTPUT_TOKENS': {
                'location': 'config/framework.json under llm.max_output_tokens',
                'example_value': '4096',
                'description': 'Maximum tokens in model response'
            },
            'LLM_TIMEOUT_SECONDS': {
                'location': 'config/framework.json under llm.timeout_seconds',
                'example_value': '60',
                'description': 'Request timeout in seconds'
            },
            'SHELL_TIMEOUT_SECONDS': {
                'location': 'config/framework.json under shell.timeout_seconds',
                'example_value': '30',
                'description': 'Shell command timeout in seconds'
            },
            'SHELL_MAX_RETRIES': {
                'location': 'config/framework.json under shell.max_retries',
                'example_value': '2',
                'description': 'Maximum shell command retry attempts'
            }
        }
        
        self.logger.warning("=" * 60)
        self.logger.warning("CONFIGURATION MIGRATION NOTICE")
        self.logger.warning("=" * 60)
        self.logger.warning("The following environment variables are DEPRECATED and should be moved to config files:")
        self.logger.warning("")
        
        for var in deprecated_vars:
            current_value = os.getenv(var, 'not set')
            migration_info = migration_mapping.get(var, {
                'location': 'config/framework.json',
                'example_value': 'unknown',
                'description': 'Configuration setting'
            })
            
            self.logger.warning(f"❌ {var} = {current_value}")
            self.logger.warning(f"   Description: {migration_info['description']}")
            self.logger.warning(f"   Move to: {migration_info['location']}")
            self.logger.warning(f"   Example: {migration_info['example_value']}")
            self.logger.warning("")
        
        # Provide migration steps
        self.logger.warning("MIGRATION STEPS:")
        self.logger.warning("1. Create or edit config/framework.json with the appropriate sections")
        self.logger.warning("2. Move the values from environment variables to the config file")
        self.logger.warning("3. Remove the environment variables from your .env file")
        self.logger.warning("4. Test that the configuration still works")
        self.logger.warning("")
        
        # Show example config structure
        framework_config_path = self._get_framework_config_path()
        self.logger.warning(f"Example config file structure for {framework_config_path}:")
        self.logger.warning("{")
        self.logger.warning('  "llm": {')
        
        for var in deprecated_vars:
            if var.startswith('LLM_'):
                migration_info = migration_mapping.get(var, {})
                config_key = var.replace('LLM_', '').lower()
                if config_key.endswith('_seconds'):
                    config_key = config_key.replace('_seconds', '')
                example_value = migration_info.get('example_value', 'value')
                # Format as number if it looks like a number
                if example_value.replace('.', '').isdigit():
                    self.logger.warning(f'    "{config_key}": {example_value},')
                else:
                    self.logger.warning(f'    "{config_key}": "{example_value}",')
        
        self.logger.warning('  },')
        self.logger.warning('  "shell": {')
        
        for var in deprecated_vars:
            if var.startswith('SHELL_'):
                migration_info = migration_mapping.get(var, {})
                config_key = var.replace('SHELL_', '').lower()
                if config_key.endswith('_seconds'):
                    config_key = config_key.replace('_seconds', '') + '_seconds'
                example_value = migration_info.get('example_value', 'value')
                # Format as number if it looks like a number
                if example_value.replace('.', '').isdigit():
                    self.logger.warning(f'    "{config_key}": {example_value},')
                else:
                    self.logger.warning(f'    "{config_key}": "{example_value}",')
        
        self.logger.warning('  }')
        self.logger.warning("}")
        self.logger.warning("")
        
        self.logger.warning("IMPORTANT:")
        self.logger.warning("• Environment variables will continue to work but may be removed in future versions")
        self.logger.warning("• Config file values are overridden by environment variables (for now)")
        self.logger.warning("• See config/MIGRATION.md for detailed migration instructions")
        self.logger.warning("• See config/examples/ for complete configuration examples")
        self.logger.warning("=" * 60)
    
    def get_configuration_sources_info(self) -> Dict[str, Any]:
        """
        Get information about configuration sources and their precedence.
        
        Returns:
            Dictionary containing configuration source information.
        """
        info = {
            "precedence_order": ["CLI arguments", "Environment variables", "Config files", "Built-in defaults"],
            "config_directory": {
                "resolved_path": str(self._get_config_directory()),
                "source": self._get_config_dir_source()
            },
            "models_config": {
                "resolved_path": str(self._get_models_config_path()),
                "source": self._get_models_config_source(),
                "exists": self._get_models_config_path().exists()
            },
            "framework_config": {
                "resolved_path": str(self._get_framework_config_path()),
                "source": self._get_framework_config_source(),
                "exists": self._get_framework_config_path().exists()
            }
        }
        return info
    
    def _get_config_dir_source(self) -> str:
        """Get the source of the config directory setting."""
        if self._config_dir_override:
            return "CLI argument (--config-dir)"
        elif os.getenv('CONFIG_DIR'):
            return "Environment variable (CONFIG_DIR)"
        else:
            return "Default (./config)"
    
    def _get_models_config_source(self) -> str:
        """Get the source of the models config file setting."""
        if self._models_config_file_override:
            return "CLI argument (--models-config)"
        elif os.getenv('MODELS_CONFIG_FILE'):
            return "Environment variable (MODELS_CONFIG_FILE)"
        else:
            return "Default (config_dir/models.json)"
    
    def _get_framework_config_source(self) -> str:
        """Get the source of the framework config file setting."""
        if self._framework_config_file_override:
            return "CLI argument (--framework-config)"
        elif os.getenv('FRAMEWORK_CONFIG_FILE'):
            return "Environment variable (FRAMEWORK_CONFIG_FILE)"
        else:
            return "Default (config_dir/framework.json)"
    
    def provide_migration_guidance(self) -> List[str]:
        """
        Provide migration guidance for deprecated or incorrectly placed settings.
        
        Returns:
            List of migration guidance messages.
        """
        guidance = []
        
        # Check for behavior settings in environment variables that should be in config files
        behavior_env_vars = [
            ('LLM_TEMPERATURE', 'config/framework.json under llm.temperature'),
            ('LLM_MAX_OUTPUT_TOKENS', 'config/framework.json under llm.max_output_tokens'),
            ('LLM_TIMEOUT_SECONDS', 'config/framework.json under llm.timeout_seconds'),
            ('SHELL_TIMEOUT_SECONDS', 'config/framework.json under shell.timeout_seconds'),
            ('SHELL_MAX_RETRIES', 'config/framework.json under shell.max_retries')
        ]
        
        deprecated_found = []
        for env_var, recommended_location in behavior_env_vars:
            if os.getenv(env_var):
                deprecated_found.append(env_var)
                guidance.append(
                    f"DEPRECATED: {env_var} should be moved to {recommended_location}. "
                    f"Environment variable support will be removed in future versions."
                )
        
        if deprecated_found:
            guidance.append(
                "To migrate these settings, add them to your config/framework.json file "
                "and remove them from your .env file. See config/MIGRATION.md for examples."
            )
        
        # Check configuration source precedence
        framework_config_path = self._get_framework_config_path()
        if framework_config_path.exists():
            if deprecated_found:
                guidance.append(
                    f"Framework config file exists at {framework_config_path}, but deprecated "
                    f"environment variables are overriding config file settings. "
                    f"Remove environment variables to use config file values."
                )
            else:
                guidance.append(
                    f"Using framework configuration from {framework_config_path}. "
                    f"This is the recommended configuration approach."
                )
        else:
            guidance.append(
                f"No framework config file found at {framework_config_path}. "
                f"Consider creating one to centralize behavior settings. "
                f"See config/examples/ for templates."
            )
        
        models_config_path = self._get_models_config_path()
        if models_config_path.exists():
            guidance.append(
                f"Using model configurations from {models_config_path} "
                f"for dynamic model family detection and token limits."
            )
        else:
            guidance.append(
                f"No models config file found at {models_config_path}. "
                f"Using built-in model configurations only."
            )
        
        return guidance
    
    def _validate_framework_config(self, config: Dict[str, Any]) -> None:
        """
        Validate framework configuration values with comprehensive error messages.
        
        Args:
            config: Framework configuration dictionary to validate.
            
        Raises:
            ConfigurationError: If configuration values are invalid.
        """
        self.logger.debug("Validating framework configuration...")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        log_level = config.get('log_level', 'INFO').upper()
        if log_level not in valid_log_levels:
            error_msg = (
                f"Invalid LOG_LEVEL: '{log_level}'. Must be one of: {', '.join(valid_log_levels)}. "
                f"DEBUG: Most verbose, INFO: Standard, WARNING: Issues only, ERROR: Errors only, CRITICAL: Fatal errors only"
            )
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        config['log_level'] = log_level
        self.logger.debug(f"Log level validated: {log_level}")
        
        # Validate framework-specific values with helpful ranges
        validation_rules = {
            'shell_timeout': {
                'min_value': 1,
                'recommended_range': (10, 300),
                'description': 'Shell command timeout in seconds'
            },
            'shell_max_retries': {
                'min_value': 0,
                'recommended_range': (0, 5),
                'description': 'Maximum shell command retry attempts'
            },
            'compression_threshold': {
                'min_value': 0.1,
                'max_value': 1.0,
                'recommended_range': (0.7, 0.95),
                'description': 'Context compression threshold (0.0-1.0)'
            },
            'context_size_ratio': {
                'min_value': 0.1,
                'max_value': 0.95,
                'recommended_range': (0.7, 0.9),
                'description': 'Ratio of token limit to use for context (0.0-1.0)'
            }
        }
        
        for key, rules in validation_rules.items():
            value = config.get(key)
            if value is None:
                continue  # Skip validation for optional values
            
            min_value = rules['min_value']
            max_value = rules.get('max_value')
            recommended_range = rules['recommended_range']
            description = rules['description']
            
            # Check minimum value
            if value < min_value:
                env_var = f"{key.upper()}"
                error_msg = (
                    f"Invalid {env_var}: {value}. Must be greater than {min_value - 1}. "
                    f"{description}. Recommended range: {recommended_range[0]}-{recommended_range[1]}"
                )
                self.logger.error(error_msg)
                raise ConfigurationError(error_msg)
            
            # Check maximum value if specified
            if max_value is not None and value > max_value:
                env_var = f"{key.upper()}"
                error_msg = (
                    f"Invalid {env_var}: {value}. Must be less than or equal to {max_value}. "
                    f"{description}. Recommended range: {recommended_range[0]}-{recommended_range[1]}"
                )
                self.logger.error(error_msg)
                raise ConfigurationError(error_msg)
            
            # Log warnings for values outside recommended range
            if not (recommended_range[0] <= value <= recommended_range[1]):
                self.logger.warning(
                    f"{key} value {value} is outside recommended range {recommended_range[0]}-{recommended_range[1]}. "
                    f"This may cause performance issues."
                )
            else:
                self.logger.debug(f"Framework {key} validated: {value}")
        
        # Validate workspace path exists and is accessible
        workspace_path = config.get('workspace_path', '.')
        workspace_path_obj = Path(workspace_path)
        
        if not workspace_path_obj.exists():
            self.logger.warning(f"Workspace path does not exist: {workspace_path}. It will be created if needed.")
        elif not workspace_path_obj.is_dir():
            error_msg = f"Workspace path is not a directory: {workspace_path}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
        elif not os.access(workspace_path, os.W_OK):
            error_msg = f"Workspace path is not writable: {workspace_path}. Check permissions."
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
        else:
            self.logger.debug(f"Workspace path validated: {workspace_path}")
        
        self.logger.info("Framework configuration validation completed successfully")
    
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

    def _get_models_config_path(self) -> Path:
        """
        Get the models configuration file path with proper precedence.
        
        Precedence: CLI args > env vars > config_dir/models.json
        
        Returns:
            Path to models configuration file.
        """
        # 1. Command-line argument takes highest precedence
        if self._models_config_file_override:
            config_path = Path(self._models_config_file_override)
            self.logger.debug(f"Using models config from CLI argument: {config_path}")
            return config_path
        
        # 2. Environment variable
        models_config_file = os.getenv('MODELS_CONFIG_FILE')
        if models_config_file:
            config_path = Path(models_config_file)
            self.logger.debug(f"Using models config from MODELS_CONFIG_FILE env var: {config_path}")
            return config_path
        
        # 3. Default fallback - models.json in config directory
        config_dir = self._get_config_directory()
        config_path = config_dir / "models.json"
        self.logger.debug(f"Using default models config path: {config_path}")
        return config_path
    
    def _get_framework_config_path(self) -> Path:
        """
        Get the framework configuration file path with proper precedence.
        
        Precedence: CLI args > env vars > config_dir/framework.json
        
        Returns:
            Path to framework configuration file.
        """
        # 1. Command-line argument takes highest precedence
        if self._framework_config_file_override:
            config_path = Path(self._framework_config_file_override)
            self.logger.debug(f"Using framework config from CLI argument: {config_path}")
            return config_path
        
        # 2. Environment variable
        framework_config_file = os.getenv('FRAMEWORK_CONFIG_FILE')
        if framework_config_file:
            config_path = Path(framework_config_file)
            self.logger.debug(f"Using framework config from FRAMEWORK_CONFIG_FILE env var: {config_path}")
            return config_path
        
        # 3. Default fallback - framework.json in config directory
        config_dir = self._get_config_directory()
        config_path = config_dir / "framework.json"
        self.logger.debug(f"Using default framework config path: {config_path}")
        return config_path

    def _load_model_configurations(self) -> None:
        """
        Load model configurations from config files with proper error handling.
        
        Loads model-specific configurations including families, token limits,
        and capabilities from JSON config files.
        """
        # Load built-in defaults first
        self._load_builtin_model_configs()
        builtin_models_count = len(self.model_configs)
        builtin_patterns_count = len(self._model_patterns)
        
        self.logger.info(f"Loaded built-in model configurations: {builtin_models_count} specific models, "
                        f"{builtin_patterns_count} detection patterns")
        
        # Get models config file path
        models_config_path = self._get_models_config_path()
        
        # Try to load custom configurations
        if models_config_path.exists():
            try:
                custom_config = self._load_config_file(str(models_config_path))
                
                # Count configurations before merging
                custom_models_count = len(custom_config.get("models", {}))
                custom_patterns_count = len(custom_config.get("patterns", []))
                has_custom_defaults = "defaults" in custom_config
                
                self._merge_model_configurations(custom_config)
                
                # Count total configurations after merging
                total_models_count = len(self.model_configs)
                total_patterns_count = len(self._model_patterns)
                
                self.logger.info(f"Successfully loaded custom model configurations from {models_config_path}")
                self.logger.info(f"Custom configurations loaded: {custom_models_count} specific models, "
                               f"{custom_patterns_count} detection patterns, "
                               f"custom defaults: {'yes' if has_custom_defaults else 'no'}")
                self.logger.info(f"Total configurations available: {total_models_count} specific models, "
                               f"{total_patterns_count} detection patterns")
                
                # Log configuration source precedence
                if custom_models_count > 0 or custom_patterns_count > 0:
                    self.logger.info("Configuration precedence: custom configurations override built-in ones")
                
            except ModelConfigurationError as e:
                self.logger.error(f"Failed to load model config from {models_config_path}: {e}")
                self.logger.error("This error prevents loading custom model configurations.")
                self.logger.info("Using built-in model configurations only")
                self._provide_config_file_fix_suggestions(models_config_path, str(e))
            except Exception as e:
                self.logger.error(f"Unexpected error loading model config from {models_config_path}: {e}")
                self.logger.error("This is likely a bug in the configuration loading code.")
                self.logger.info("Using built-in model configurations only")
        else:
            self.logger.info(f"No model config file found at {models_config_path}")
            self.logger.info("Using built-in model configurations only")
            self.logger.debug(f"To add custom model configurations, create a JSON file at: {models_config_path}")
            self.logger.debug("See config/examples/models.json for the expected format")
    
    def _get_config_directory(self) -> Path:
        """
        Get the configuration directory path with proper precedence.
        
        Precedence: CLI args > env vars > default fallback
        
        Returns:
            Path to configuration directory.
        """
        # 1. Command-line argument takes highest precedence
        if self._config_dir_override:
            config_path = Path(self._config_dir_override)
            self.logger.debug(f"Using config directory from CLI argument: {config_path}")
            return config_path
        
        # 2. Environment variable
        config_dir = os.getenv('CONFIG_DIR')
        if config_dir:
            config_path = Path(config_dir)
            self.logger.debug(f"Using config directory from CONFIG_DIR env var: {config_path}")
            return config_path
        
        # 3. Default fallback - look for config directory in current directory
        current_dir = Path.cwd()
        config_path = current_dir / "config"
        self.logger.debug(f"Using default config directory: {config_path}")
        return config_path
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file with comprehensive error handling.
        
        Args:
            config_path: Path to JSON configuration file.
            
        Returns:
            Dictionary containing configuration data.
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed.
        """
        config_path_obj = Path(config_path)
        
        try:
            # Check if file exists
            if not config_path_obj.exists():
                raise ModelConfigurationError(f"Configuration file not found: {config_path}")
            
            # Check if it's a file (not a directory)
            if not config_path_obj.is_file():
                raise ModelConfigurationError(f"Configuration path is not a file: {config_path}")
            
            # Check file permissions
            if not os.access(config_path, os.R_OK):
                raise ModelConfigurationError(f"Configuration file is not readable: {config_path}")
            
            # Load and parse JSON
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Check for empty file
                if not content:
                    raise ModelConfigurationError(f"Configuration file is empty: {config_path}")
                
                config = json.loads(content)
                
                # Validate that it's a dictionary
                if not isinstance(config, dict):
                    raise ModelConfigurationError(f"Configuration file must contain a JSON object, got {type(config).__name__}: {config_path}")
                
            self.logger.debug(f"Successfully loaded configuration from {config_path}")
            return config
            
        except json.JSONDecodeError as e:
            raise ModelConfigurationError(
                f"Invalid JSON in configuration file {config_path}: {e}. "
                f"Please check the file syntax and ensure it contains valid JSON."
            )
        except ModelConfigurationError:
            # Re-raise our own errors
            raise
        except PermissionError:
            raise ModelConfigurationError(f"Permission denied reading configuration file: {config_path}")
        except UnicodeDecodeError as e:
            raise ModelConfigurationError(f"Configuration file encoding error {config_path}: {e}. File must be UTF-8 encoded.")
        except Exception as e:
            raise ModelConfigurationError(f"Unexpected error loading configuration file {config_path}: {e}")
    
    def _load_builtin_model_configs(self) -> None:
        """Load built-in model configurations and patterns."""
        # Built-in model patterns for detection
        self._model_patterns = [
            {
                "pattern": r".*gemini-2\.0.*",
                "family": "GEMINI_2_0_FLASH",
                "token_limit": 1048576,
                "capabilities": {
                    "vision": False,
                    "function_calling": True,
                    "streaming": True
                }
            },
            {
                "pattern": r".*gemini-1\.5-pro.*",
                "family": "GEMINI_1_5_PRO", 
                "token_limit": 2097152,
                "capabilities": {
                    "vision": True,
                    "function_calling": True,
                    "streaming": True
                }
            },
            {
                "pattern": r".*gemini-1\.5.*",
                "family": "GEMINI_1_5_FLASH",
                "token_limit": 1048576,
                "capabilities": {
                    "vision": True,
                    "function_calling": True,
                    "streaming": True
                }
            },
            {
                "pattern": r".*gpt-4-turbo.*",
                "family": "GPT_4",
                "token_limit": 128000,
                "capabilities": {
                    "vision": True,
                    "function_calling": True,
                    "streaming": True
                }
            },
            {
                "pattern": r".*gpt-4.*",
                "family": "GPT_4",
                "token_limit": 8192,
                "capabilities": {
                    "vision": False,
                    "function_calling": True,
                    "streaming": True
                }
            },
            {
                "pattern": r".*claude-3-opus.*",
                "family": "CLAUDE_3_OPUS",
                "token_limit": 200000,
                "capabilities": {
                    "vision": True,
                    "function_calling": True,
                    "streaming": True
                }
            },
            {
                "pattern": r"^gpt-oss:.*",
                "family": "GPT_4",
                "token_limit": 128000,
                "capabilities": {
                    "vision": False,
                    "function_calling": True,
                    "streaming": True
                }
            }
        ]
        
        # Built-in specific model configurations
        self.model_configs = {
            "gpt-oss:20b": {
                "family": "GPT_4",
                "token_limit": 128000,
                "capabilities": {
                    "vision": False,
                    "function_calling": True,
                    "streaming": True
                }
            }
        }
        
        # Default fallback configuration
        self._model_defaults = {
            "family": "GPT_4",
            "token_limit": 8192,
            "capabilities": {
                "vision": False,
                "function_calling": False,
                "streaming": False
            }
        }
    
    def _merge_model_configurations(self, custom_config: Dict[str, Any]) -> None:
        """
        Merge custom model configurations with built-in ones with validation.
        
        Args:
            custom_config: Custom configuration dictionary from config file.
            
        Raises:
            ModelConfigurationError: If custom configuration is invalid.
        """
        self.logger.debug("Merging custom model configurations with built-in ones...")
        
        # Validate the overall structure first
        self._validate_model_config_structure(custom_config)
        
        # Merge specific model configurations (custom takes precedence)
        if "models" in custom_config:
            models_config = custom_config["models"]
            if not isinstance(models_config, dict):
                raise ModelConfigurationError("'models' section must be a dictionary")
            
            for model_name, model_config in models_config.items():
                self._validate_individual_model_config(model_name, model_config)
                self.model_configs[model_name] = model_config
                self.logger.debug(f"Added custom model configuration: {model_name}")
        
        # Merge patterns (custom patterns are added to the beginning for higher precedence)
        if "patterns" in custom_config:
            patterns_config = custom_config["patterns"]
            if not isinstance(patterns_config, list):
                raise ModelConfigurationError("'patterns' section must be a list")
            
            for i, pattern_config in enumerate(patterns_config):
                self._validate_pattern_config(pattern_config, i)
                self.logger.debug(f"Added custom pattern: {pattern_config.get('pattern', 'unknown')}")
            
            self._model_patterns = patterns_config + self._model_patterns
        
        # Update defaults if provided
        if "defaults" in custom_config:
            defaults_config = custom_config["defaults"]
            if not isinstance(defaults_config, dict):
                raise ModelConfigurationError("'defaults' section must be a dictionary")
            
            self._validate_individual_model_config("defaults", defaults_config)
            self._model_defaults.update(defaults_config)
            self.logger.debug(f"Updated default model configuration: {defaults_config}")
        
        self.logger.debug("Model configuration merging completed successfully")
    
    def _validate_model_config_structure(self, config: Dict[str, Any]) -> None:
        """
        Validate the overall structure of model configuration.
        
        Args:
            config: Model configuration dictionary to validate.
            
        Raises:
            ModelConfigurationError: If structure is invalid.
        """
        if not isinstance(config, dict):
            raise ModelConfigurationError("Model configuration must be a JSON object")
        
        valid_sections = {"models", "patterns", "defaults"}
        invalid_sections = set(config.keys()) - valid_sections
        
        if invalid_sections:
            raise ModelConfigurationError(
                f"Invalid configuration sections: {', '.join(invalid_sections)}. "
                f"Valid sections are: {', '.join(valid_sections)}"
            )
        
        if not any(section in config for section in valid_sections):
            raise ModelConfigurationError(
                "Model configuration must contain at least one of: models, patterns, or defaults"
            )
    
    def _validate_individual_model_config(self, model_name: str, config: Dict[str, Any]) -> None:
        """
        Validate an individual model configuration.
        
        Args:
            model_name: Name of the model being configured.
            config: Model configuration dictionary.
            
        Raises:
            ModelConfigurationError: If configuration is invalid.
        """
        if not isinstance(config, dict):
            raise ModelConfigurationError(f"Configuration for '{model_name}' must be a dictionary")
        
        # Required fields
        required_fields = {"family", "token_limit"}
        missing_fields = required_fields - set(config.keys())
        if missing_fields:
            raise ModelConfigurationError(
                f"Model '{model_name}' is missing required fields: {', '.join(missing_fields)}"
            )
        
        # Validate family
        family = config.get("family")
        if not isinstance(family, str) or not family:
            raise ModelConfigurationError(f"Model '{model_name}' family must be a non-empty string")
        
        # Validate token_limit
        token_limit = config.get("token_limit")
        if not isinstance(token_limit, int) or token_limit <= 0:
            raise ModelConfigurationError(
                f"Model '{model_name}' token_limit must be a positive integer, got: {token_limit}"
            )
        
        # Validate capabilities if present
        if "capabilities" in config:
            capabilities = config["capabilities"]
            if not isinstance(capabilities, dict):
                raise ModelConfigurationError(f"Model '{model_name}' capabilities must be a dictionary")
            
            # Check that all capability values are boolean
            for cap_name, cap_value in capabilities.items():
                if not isinstance(cap_value, bool):
                    raise ModelConfigurationError(
                        f"Model '{model_name}' capability '{cap_name}' must be true or false, got: {cap_value}"
                    )
    
    def _validate_pattern_config(self, pattern_config: Dict[str, Any], index: int) -> None:
        """
        Validate a pattern configuration.
        
        Args:
            pattern_config: Pattern configuration dictionary.
            index: Index of the pattern in the list (for error messages).
            
        Raises:
            ModelConfigurationError: If pattern configuration is invalid.
        """
        if not isinstance(pattern_config, dict):
            raise ModelConfigurationError(f"Pattern {index} must be a dictionary")
        
        # Required fields
        required_fields = {"pattern", "family", "token_limit"}
        missing_fields = required_fields - set(pattern_config.keys())
        if missing_fields:
            raise ModelConfigurationError(
                f"Pattern {index} is missing required fields: {', '.join(missing_fields)}"
            )
        
        # Validate pattern regex
        pattern = pattern_config.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ModelConfigurationError(f"Pattern {index} 'pattern' must be a non-empty string")
        
        try:
            re.compile(pattern)
        except re.error as e:
            raise ModelConfigurationError(f"Pattern {index} has invalid regex '{pattern}': {e}")
        
        # Validate other fields using the same logic as individual model config
        self._validate_individual_model_config(f"pattern {index}", pattern_config)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get complete model information including family, limits, capabilities.
        
        Args:
            model_name: Name of the model to get information for.
            
        Returns:
            Dictionary containing model family, token_limit, and capabilities.
        """
        self.logger.debug(f"Detecting model configuration for: {model_name}")
        
        # Check for exact model match first
        if model_name in self.model_configs:
            config = self.model_configs[model_name].copy()
            self.logger.info(f"Model configuration detected for '{model_name}': family={config['family']}, "
                           f"token_limit={config['token_limit']}, capabilities={config['capabilities']} "
                           f"(source: exact match in custom configurations)")
            return config
        
        # Try pattern matching
        attempted_patterns = []
        for pattern_config in self._model_patterns:
            pattern = pattern_config["pattern"]
            attempted_patterns.append(pattern)
            
            if re.match(pattern, model_name, re.IGNORECASE):
                config = {
                    "family": pattern_config["family"],
                    "token_limit": pattern_config["token_limit"],
                    "capabilities": pattern_config["capabilities"].copy()
                }
                self.logger.info(f"Model configuration detected for '{model_name}': family={config['family']}, "
                               f"token_limit={config['token_limit']}, capabilities={config['capabilities']} "
                               f"(source: pattern match '{pattern}')")
                return config
        
        # Fall back to defaults
        config = self._model_defaults.copy()
        self.logger.warning(f"Model '{model_name}' not recognized by any pattern. "
                          f"Attempted patterns: {attempted_patterns}. "
                          f"Using default configuration: family={config['family']}, "
                          f"token_limit={config['token_limit']}, capabilities={config['capabilities']} "
                          f"(source: built-in defaults)")
        
        # Provide helpful suggestions for unrecognized models
        self._suggest_model_configuration_fixes(model_name, attempted_patterns)
        
        return config
    
    def get_model_token_limit(self, model_name: str) -> int:
        """
        Get token limit for specific model.
        
        Args:
            model_name: Name of the model to get token limit for.
            
        Returns:
            Token limit for the model.
        """
        model_info = self.get_model_info(model_name)
        token_limit = model_info["token_limit"]
        
        # Determine the source of the token limit for logging
        source = "unknown"
        if model_name in self.model_configs:
            source = "custom configuration"
        else:
            # Check if it came from pattern matching
            for pattern_config in self._model_patterns:
                if re.match(pattern_config["pattern"], model_name, re.IGNORECASE):
                    source = f"pattern match '{pattern_config['pattern']}'"
                    break
            else:
                source = "built-in defaults"
        
        self.logger.info(f"Token limit applied for '{model_name}': {token_limit} (source: {source})")
        return token_limit
    
    def get_model_family(self, model_name: str) -> str:
        """
        Get AutoGen ModelFamily string for specific model.
        
        Args:
            model_name: Name of the model to get family for.
            
        Returns:
            ModelFamily enum name as string.
        """
        model_info = self.get_model_info(model_name)
        family = model_info["family"]
        
        # Determine the source of the model family for logging
        source = "unknown"
        if model_name in self.model_configs:
            source = "custom configuration"
        else:
            # Check if it came from pattern matching
            for pattern_config in self._model_patterns:
                if re.match(pattern_config["pattern"], model_name, re.IGNORECASE):
                    source = f"pattern match '{pattern_config['pattern']}'"
                    break
            else:
                source = "built-in defaults"
        
        self.logger.info(f"Model family applied for '{model_name}': {family} (source: {source})")
        return family
    
    def get_model_capabilities(self, model_name: str) -> Dict[str, bool]:
        """
        Get model capabilities (vision, function_calling, etc.).
        
        Args:
            model_name: Name of the model to get capabilities for.
            
        Returns:
            Dictionary of model capabilities.
        """
        model_info = self.get_model_info(model_name)
        capabilities = model_info["capabilities"]
        self.logger.debug(f"Model capabilities for {model_name}: {capabilities}")
        return capabilities
    
    def _suggest_model_configuration_fixes(self, model_name: str, attempted_patterns: List[str]) -> None:
        """
        Provide helpful suggestions for unrecognized model configurations.
        
        Args:
            model_name: The unrecognized model name.
            attempted_patterns: List of patterns that were attempted.
        """
        self.logger.info("=== MODEL CONFIGURATION SUGGESTIONS ===")
        self.logger.info(f"Model '{model_name}' was not recognized. Here are some suggestions:")
        
        # Suggest adding to custom configuration
        models_config_path = self._get_models_config_path()
        self.logger.info(f"1. Add a specific configuration for '{model_name}' in {models_config_path}:")
        self.logger.info(f'   "models": {{')
        self.logger.info(f'     "{model_name}": {{')
        self.logger.info(f'       "family": "GPT_4",')
        self.logger.info(f'       "token_limit": 8192,')
        self.logger.info(f'       "capabilities": {{"vision": false, "function_calling": true, "streaming": true}}')
        self.logger.info(f'     }}')
        self.logger.info(f'   }}')
        
        # Suggest adding a pattern
        self.logger.info(f"2. Add a pattern to match similar models:")
        self.logger.info(f'   "patterns": [')
        self.logger.info(f'     {{')
        self.logger.info(f'       "pattern": "^{re.escape(model_name.split(":")[0] if ":" in model_name else model_name.split("-")[0])}.*",')
        self.logger.info(f'       "family": "GPT_4",')
        self.logger.info(f'       "token_limit": 8192')
        self.logger.info(f'     }}')
        self.logger.info(f'   ]')
        
        # Show current defaults being used
        self.logger.info(f"3. Current defaults being used:")
        for key, value in self._model_defaults.items():
            self.logger.info(f"   {key}: {value}")
        
        self.logger.info("4. See config/examples/models.json for complete configuration examples")
        self.logger.info("==========================================")
    
    def _provide_config_file_fix_suggestions(self, config_path: Path, error_message: str) -> None:
        """
        Provide helpful suggestions for fixing configuration file issues.
        
        Args:
            config_path: Path to the problematic configuration file.
            error_message: The error message that occurred.
        """
        self.logger.info("=== CONFIGURATION FILE FIX SUGGESTIONS ===")
        self.logger.info(f"Configuration file: {config_path}")
        self.logger.info(f"Error: {error_message}")
        self.logger.info("")
        
        if "JSON" in error_message or "json" in error_message:
            self.logger.info("JSON Syntax Issues:")
            self.logger.info("1. Check for missing commas between object properties")
            self.logger.info("2. Ensure all strings are properly quoted with double quotes")
            self.logger.info("3. Check for trailing commas (not allowed in JSON)")
            self.logger.info("4. Verify all brackets and braces are properly closed")
            self.logger.info("5. Use a JSON validator online to check syntax")
        
        if "not found" in error_message.lower():
            self.logger.info("File Not Found:")
            self.logger.info(f"1. Create the configuration file at: {config_path}")
            self.logger.info("2. Ensure the directory exists and is writable")
            self.logger.info("3. Check file permissions")
        
        if "permission" in error_message.lower():
            self.logger.info("Permission Issues:")
            self.logger.info(f"1. Check file permissions: chmod 644 {config_path}")
            self.logger.info(f"2. Ensure the directory is readable: chmod 755 {config_path.parent}")
            self.logger.info("3. Verify the file is not locked by another process")
        
        if "encoding" in error_message.lower():
            self.logger.info("Encoding Issues:")
            self.logger.info("1. Ensure the file is saved with UTF-8 encoding")
            self.logger.info("2. Remove any BOM (Byte Order Mark) characters")
            self.logger.info("3. Check for non-printable characters")
        
        self.logger.info("")
        self.logger.info("Example valid configuration structure:")
        self.logger.info('{')
        self.logger.info('  "models": {')
        self.logger.info('    "your-model-name": {')
        self.logger.info('      "family": "GPT_4",')
        self.logger.info('      "token_limit": 8192,')
        self.logger.info('      "capabilities": {"vision": false, "function_calling": true}')
        self.logger.info('    }')
        self.logger.info('  },')
        self.logger.info('  "patterns": [')
        self.logger.info('    {')
        self.logger.info('      "pattern": "^gpt-.*",')
        self.logger.info('      "family": "GPT_4",')
        self.logger.info('      "token_limit": 8192')
        self.logger.info('    }')
        self.logger.info('  ]')
        self.logger.info('}')
        self.logger.info("============================================")

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
    
    def validate_config_paths(self) -> List[str]:
        """
        Validate configuration file paths and accessibility.
        
        Returns:
            List of validation errors. Empty list if all paths are valid.
        """
        errors = []
        
        # Validate config directory
        config_dir = self._get_config_directory()
        if not config_dir.exists():
            errors.append(f"Configuration directory does not exist: {config_dir}")
        elif not config_dir.is_dir():
            errors.append(f"Configuration path is not a directory: {config_dir}")
        elif not os.access(config_dir, os.R_OK):
            errors.append(f"Configuration directory is not readable: {config_dir}")
        
        # Validate models config file if it should exist
        models_config_path = self._get_models_config_path()
        if self._models_config_file_override or os.getenv('MODELS_CONFIG_FILE'):
            # Explicit path provided, must exist
            if not models_config_path.exists():
                errors.append(f"Models configuration file not found: {models_config_path}")
            elif not models_config_path.is_file():
                errors.append(f"Models configuration path is not a file: {models_config_path}")
            elif not os.access(models_config_path, os.R_OK):
                errors.append(f"Models configuration file is not readable: {models_config_path}")
        
        # Validate framework config file if it should exist
        framework_config_path = self._get_framework_config_path()
        if self._framework_config_file_override or os.getenv('FRAMEWORK_CONFIG_FILE'):
            # Explicit path provided, must exist
            if not framework_config_path.exists():
                errors.append(f"Framework configuration file not found: {framework_config_path}")
            elif not framework_config_path.is_file():
                errors.append(f"Framework configuration path is not a file: {framework_config_path}")
            elif not os.access(framework_config_path, os.R_OK):
                errors.append(f"Framework configuration file is not readable: {framework_config_path}")
        
        return errors
    
    def validate_required_config(self) -> List[str]:
        """
        Validate that all required configuration is present and accessible.
        
        Returns:
            List of validation errors. Empty list if all configuration is valid.
        """
        errors = []
        
        # First validate paths
        path_errors = self.validate_config_paths()
        errors.extend(path_errors)
        
        # Then validate configuration content
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
            'SHELL_TIMEOUT_SECONDS', 'SHELL_MAX_RETRIES',
            'TASK_REAL_TIME_UPDATES_ENABLED', 'TASK_FALLBACK_TO_BATCH_ENABLED',
            'TASK_MAX_INDIVIDUAL_UPDATE_RETRIES', 'TASK_INDIVIDUAL_UPDATE_RETRY_DELAY',
            'TASK_FILE_LOCK_TIMEOUT', 'TASK_DETAILED_LOGGING_ENABLED',
            'TASK_RECOVERY_MECHANISM_ENABLED'
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


def get_config_manager(config_dir: Optional[str] = None, 
                      models_config_file: Optional[str] = None,
                      framework_config_file: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_dir: Configuration directory path override.
        models_config_file: Models configuration file path override.
        framework_config_file: Framework configuration file path override.
    
    Returns:
        ConfigManager instance.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(
            config_dir=config_dir,
            models_config_file=models_config_file,
            framework_config_file=framework_config_file
        )
    return _config_manager


def reset_config_manager() -> None:
    """Reset the global configuration manager instance (useful for testing)."""
    global _config_manager
    _config_manager = None