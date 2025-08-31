"""
Tests for the main entry point module.

This module tests the main entry point functionality including:
- Command-line argument parsing
- Framework initialization
- Basic error handling
- Help and status commands
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner
from pathlib import Path
import tempfile
import os

from autogen_framework.main import main, setup_logging, print_banner, print_help
from autogen_framework.models import LLMConfig


class TestMainEntryPoint:
    """Test cases for the main entry point functionality."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            setup_logging(verbose=False, log_file=log_file)
            
            # Verify log file was created
            assert os.path.exists(log_file)
    
    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        import logging
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        setup_logging(verbose=True)
        
        # Check that DEBUG level is set
        assert root_logger.level == logging.DEBUG
    
    def test_print_banner(self, capsys):
        """Test banner printing."""
        print_banner()
        captured = capsys.readouterr()
        assert "AutoGen Multi-Agent Framework" in captured.out
    
    def test_print_help(self, capsys):
        """Test help printing."""
        print_help()
        captured = capsys.readouterr()
        assert "USAGE EXAMPLES" in captured.out
        assert "WORKFLOW PHASES" in captured.out
    
    @patch('autogen_framework.main.MainController')
    def test_main_help_examples(self, mock_controller):
        """Test --help-examples flag."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help-examples'])
        
        assert result.exit_code == 0
        assert "USAGE EXAMPLES" in result.output
        # Controller should not be initialized for help
        mock_controller.assert_not_called()
    
    @patch.dict(os.environ, {
        'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
        'LLM_MODEL': 'test-model',
        'LLM_API_KEY': 'sk-test123'
    })
    @patch('autogen_framework.main.MainController')
    def test_main_status_mode(self, mock_controller):
        """Test --status flag."""
        # Setup mock controller
        mock_instance = Mock()
        mock_instance.initialize_framework.return_value = True
        mock_instance.get_framework_status.return_value = {
            "initialized": True,
            "workspace_path": "/test/path",
            "session_id": "test-session"
        }
        mock_instance.get_session_id.return_value = "test-session"
        mock_controller.return_value = mock_instance
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(main, ['--workspace', temp_dir, '--status'])
        
        assert result.exit_code == 0
        assert "📊 Framework Status" in result.output
        mock_instance.initialize_framework.assert_called_once()
        mock_instance.get_framework_status.assert_called_once()
    
    @patch.dict(os.environ, {
        'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
        'LLM_MODEL': 'test-model',
        'LLM_API_KEY': 'sk-test123'
    })
    @patch('autogen_framework.main.MainController')
    def test_main_initialization_failure(self, mock_controller):
        """Test framework initialization failure."""
        # Setup mock controller that fails initialization
        mock_instance = Mock()
        mock_instance.initialize_framework.return_value = False
        mock_controller.return_value = mock_instance
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(main, ['--workspace', temp_dir, '--status'])
        
        assert result.exit_code == 1
        assert "❌ Framework initialization failed!" in result.output
    
    @patch.dict(os.environ, {
        'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
        'LLM_MODEL': 'test-model',
        'LLM_API_KEY': 'sk-test123'
    })
    @patch('autogen_framework.main.MainController')
    @patch('autogen_framework.main.asyncio.run')
    def test_main_specific_request(self, mock_asyncio_run, mock_controller):
        """Test processing a specific request."""
        # Setup mock controller
        mock_instance = Mock()
        mock_instance.initialize_framework.return_value = True
        mock_instance.get_session_id.return_value = "test-session"
        mock_controller.return_value = mock_instance
        
        # Setup async mock for process_user_request
        mock_instance.process_user_request = AsyncMock(return_value={
            "success": True, 
            "workflow_completed": True
        })
        
        # Mock asyncio.run to actually execute the coroutine to avoid warnings
        def mock_run(coro):
            # Create a new event loop for the test
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        
        mock_asyncio_run.side_effect = mock_run
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(main, [
                '--workspace', temp_dir,
                '--request', 'Create a test API'
            ])
        
        assert result.exit_code == 0
        assert "🔄 Processing request: Create a test API" in result.output
        mock_asyncio_run.assert_called_once()
    
    @patch.dict(os.environ, {
        'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
        'LLM_MODEL': 'test-model',
        'LLM_API_KEY': 'sk-test123'
    })
    @patch('autogen_framework.main.MainController')
    @patch('autogen_framework.main.asyncio.run')
    def test_main_continue_workflow(self, mock_asyncio_run, mock_controller):
        """Test continuing an existing workflow."""
        # Setup mock controller
        mock_instance = Mock()
        mock_instance.initialize_framework.return_value = True
        mock_instance.get_session_id.return_value = "test-session"
        mock_controller.return_value = mock_instance
        
        # Setup async mock for continue_workflow
        mock_instance.continue_workflow = AsyncMock(return_value={
            "success": True,
            "workflow_completed": False
        })
        
        # Mock asyncio.run to actually execute the coroutine to avoid warnings
        def mock_run(coro):
            # Create a new event loop for the test
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        
        mock_asyncio_run.side_effect = mock_run
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(main, [
                '--workspace', temp_dir,
                '--continue-workflow'
            ])
        
        assert result.exit_code == 0
        assert "🔄 Continuing existing workflow..." in result.output
        mock_asyncio_run.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)  # Clear environment to test CLI args override
    def test_test_llm_config_creation(self):
        """Test LLM configuration creation with custom parameters."""
        runner = CliRunner()
        
        with patch('autogen_framework.main.MainController') as mock_controller:
            mock_instance = Mock()
            mock_instance.initialize_framework.return_value = True
            mock_instance.get_framework_status.return_value = {"test": "status"}
            mock_instance.get_session_id.return_value = "test-session"
            mock_controller.return_value = mock_instance
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = runner.invoke(main, [
                    '--workspace', temp_dir,
                    '--llm-base-url', 'http://custom.url',
                    '--llm-model', 'custom-model',
                    '--llm-api-key', 'custom-key',
                    '--status'
                ])
            
            # Verify LLM config was passed to initialize_framework
            assert result.exit_code == 0
            mock_instance.initialize_framework.assert_called_once()
            
            # Get the LLMConfig that was passed
            call_args = mock_instance.initialize_framework.call_args
            test_llm_config = call_args[0][0]
            
            assert test_llm_config.base_url == 'http://custom.url'
            assert test_llm_config.model == 'custom-model'
            assert test_llm_config.api_key == 'custom-key'
    
    @patch.dict(os.environ, {
        'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
        'LLM_MODEL': 'test-model',
        'LLM_API_KEY': 'sk-test123'
    })
    def test_workspace_path_resolution(self):
        """Test workspace path resolution."""
        runner = CliRunner()
        
        with patch('autogen_framework.main.MainController') as mock_controller:
            mock_instance = Mock()
            mock_instance.initialize_framework.return_value = True
            mock_instance.get_framework_status.return_value = {"test": "status"}
            mock_instance.get_session_id.return_value = "test-session"
            mock_controller.return_value = mock_instance
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = runner.invoke(main, [
                    '--workspace', temp_dir,
                    '--status'
                ])
            
            # Verify controller was initialized with resolved path
            assert result.exit_code == 0
            mock_controller.assert_called_once()
            
            # Get the workspace path that was passed
            call_args = mock_controller.call_args
            workspace_path = call_args[0][0]
            
            # Should be absolute path
            assert os.path.isabs(workspace_path)
    
    @patch.dict(os.environ, {
        'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
        'LLM_MODEL': 'test-model',
        'LLM_API_KEY': 'sk-test123'
    })
    @patch('autogen_framework.main.MainController')
    def test_error_handling(self, mock_controller):
        """Test error handling in main function."""
        # Setup mock controller that raises exception during initialization
        mock_controller.side_effect = Exception("Test error")
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(main, ['--workspace', temp_dir, '--status'])
        
        assert result.exit_code == 1
        assert "❌ Fatal error: Test error" in result.output


if __name__ == "__main__":
    pytest.main([__file__])