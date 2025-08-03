"""
Integration test fixtures and configuration.

These fixtures are specifically designed for integration tests that use real LLM endpoints
and test actual functionality with external services.

Key Fixtures:
- real_llm_config: Real LLM configuration loaded from .env.integration
- integration_config_manager: ConfigManager with real environment loading
- llm_config: Override that uses real_llm_config for integration tests
- enable_real_services: Auto-fixture enabling real external service calls
- mock_llm_config: Raises error to prevent accidental use in integration tests

Configuration Requirements:
- .env.integration file must exist with valid LLM credentials
- LLM_BASE_URL, LLM_MODEL, and LLM_API_KEY must be set
- External services must be accessible from test environment

For detailed usage patterns, see:
- tests/TEST_PATTERNS.md - Standard test patterns and conventions
- tests/README.md - Test organization and best practices
"""

import os
import pytest
from dotenv import load_dotenv

from autogen_framework.config_manager import ConfigManager
from autogen_framework.models import LLMConfig


@pytest.fixture
def real_llm_config():
    """
    Create real LLM configuration for integration testing.
    
    This loads configuration from .env.integration file for testing with real services.
    """
    # Load integration test environment
    load_dotenv('.env.integration')
    
    return LLMConfig(
        base_url=os.getenv('LLM_BASE_URL', 'http://ctwuhome.local:8888/openai/v1'),
        model=os.getenv('LLM_MODEL', 'models/gemini-2.0-flash'),
        api_key=os.getenv('LLM_API_KEY'),
        temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
        max_output_tokens=int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '4096')),
        timeout=int(os.getenv('LLM_TIMEOUT_SECONDS', '60'))
    )


@pytest.fixture
def integration_config_manager():
    """Create a ConfigManager for integration testing with real environment loading."""
    return ConfigManager(load_env=True)


# Override the default llm_config fixture for integration tests
@pytest.fixture
def llm_config(real_llm_config):
    """Integration tests use real LLM configuration."""
    return real_llm_config


@pytest.fixture(autouse=True)
def enable_real_services():
    """
    Automatically enable real service calls in integration tests.
    
    This fixture runs automatically for all integration tests to ensure
    real external services can be called.
    """
    import os
    
    # Set environment flag to indicate integration testing
    original_integration = os.environ.get('INTEGRATION_TESTING')
    os.environ['INTEGRATION_TESTING'] = 'true'
    
    # Remove unit testing flag if present
    original_unit = os.environ.get('UNIT_TESTING')
    if 'UNIT_TESTING' in os.environ:
        del os.environ['UNIT_TESTING']
    
    yield
    
    # Restore original values
    if original_integration is None:
        os.environ.pop('INTEGRATION_TESTING', None)
    else:
        os.environ['INTEGRATION_TESTING'] = original_integration
        
    if original_unit is not None:
        os.environ['UNIT_TESTING'] = original_unit


# Ensure integration tests don't accidentally use unit test fixtures
@pytest.fixture
def mock_llm_config():
    """
    Prevent integration tests from using mock LLM configuration.
    
    This fixture raises an error if integration tests try to use mock configuration.
    """
    raise RuntimeError(
        "Integration tests should not use mock_llm_config. "
        "Use the 'real_llm_config' fixture instead for real configuration."
    )