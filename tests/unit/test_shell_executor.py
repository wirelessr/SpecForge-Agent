"""
Unit tests for the ShellExecutor class.

This module contains comprehensive tests for the shell command execution functionality,
including success cases, failure cases, retry mechanisms, and history tracking.
"""

import asyncio
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from autogen_framework.shell_executor import ShellExecutor
from autogen_framework.models import ExecutionResult


class TestShellExecutor:
    """Test suite for ShellExecutor class."""
    
    @pytest.fixture
    def executor(self):
        """Create a ShellExecutor instance for testing."""
        return ShellExecutor()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_execute_simple_command_success(self, executor):
        """Test successful execution of a simple command."""
        result = await executor.execute_command("echo 'Hello World'")
        
        assert result.success is True
        assert result.return_code == 0
        assert "Hello World" in result.stdout
        assert result.stderr == ""
        assert result.command == "echo 'Hello World'"
        assert result.approach_used == "direct_execution"
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_execute_command_failure(self, executor):
        """Test execution of a command that fails."""
        result = await executor.execute_command("nonexistent_command_xyz")
        
        assert result.success is False
        assert result.return_code != 0
        assert result.stdout == ""
        assert len(result.stderr) > 0
        assert result.command == "nonexistent_command_xyz"
        assert result.approach_used == "direct_execution"
    
    @pytest.mark.asyncio
    async def test_execute_command_with_working_directory(self, executor, temp_dir):
        """Test command execution with specific working directory."""
        # Create a test file in the temp directory
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        result = await executor.execute_command("ls test.txt", working_dir=temp_dir)
        
        assert result.success is True
        assert "test.txt" in result.stdout
        assert result.working_directory == temp_dir
    
    @pytest.mark.asyncio
    async def test_execute_command_timeout(self, executor):
        """Test command execution with timeout."""
        # Use a command that will definitely timeout
        result = await executor.execute_command("sleep 5", timeout=1.0)
        
        assert result.success is False
        assert result.return_code == -1
        assert "timed out" in result.stderr
        assert result.execution_time >= 1.0
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_success_after_failure(self, executor):
        """Test retry mechanism when command eventually succeeds."""
        # Mock a command that fails twice then succeeds
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            # First two calls fail, third succeeds
            async def mock_communicate_fail():
                return (b"", b"error")
            
            async def mock_communicate_success():
                return (b"success", b"")
            
            mock_process_fail = MagicMock()
            mock_process_fail.communicate = mock_communicate_fail
            mock_process_fail.returncode = 1
            
            mock_process_success = MagicMock()
            mock_process_success.communicate = mock_communicate_success
            mock_process_success.returncode = 0
            
            mock_subprocess.side_effect = [
                mock_process_fail,  # First attempt fails
                mock_process_fail,  # Second attempt fails
                mock_process_success  # Third attempt succeeds
            ]
            
            result = await executor.execute_command("test_command", max_retries=2)
            
            assert result.success is True
            assert result.stdout == "success"
            assert len(result.alternative_attempts) > 0
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_max_retries_exceeded(self, executor):
        """Test retry mechanism when max retries are exceeded."""
        # Use a command that will always fail
        result = await executor.execute_command("false", max_retries=2)
        
        assert result.success is False
        # Should have attempted 3 times total (initial + 2 retries)
        assert len(executor.execution_history) >= 3
    
    @pytest.mark.asyncio
    async def test_execution_history_tracking(self, executor):
        """Test that execution history is properly tracked."""
        initial_count = len(executor.execution_history)
        
        await executor.execute_command("echo 'test1'")
        await executor.execute_command("echo 'test2'")
        await executor.execute_command("false", max_retries=0)  # This will fail, no retries
        
        # Should have 3 commands total (retries add to history)
        assert len(executor.execution_history) >= initial_count + 3
        
        # Check that we can retrieve different types of executions
        successful = executor.get_successful_executions()
        failed = executor.get_failed_executions()
        
        assert len(successful) >= 2
        assert len(failed) >= 1
    
    def test_get_recent_executions(self, executor):
        """Test retrieval of recent executions."""
        # Add some mock executions to history
        for i in range(15):
            mock_result = ExecutionResult.create_success(
                command=f"echo 'test{i}'",
                stdout=f"test{i}",
                execution_time=0.1,
                working_directory="/tmp",
                approach_used="direct_execution"
            )
            executor.execution_history.append(mock_result)
        
        recent = executor.get_recent_executions(5)
        assert len(recent) == 5
        
        # Should get the most recent ones
        assert recent[-1].command == "echo 'test14'"
        assert recent[0].command == "echo 'test10'"
    
    def test_execution_stats(self, executor):
        """Test execution statistics calculation."""
        # Add some mock executions
        for i in range(10):
            if i < 7:  # 7 successful
                result = ExecutionResult.create_success(
                    command=f"echo 'test{i}'",
                    stdout=f"test{i}",
                    execution_time=0.1,
                    working_directory="/tmp",
                    approach_used="direct_execution"
                )
            else:  # 3 failed
                result = ExecutionResult.create_failure(
                    command=f"false{i}",
                    return_code=1,
                    stderr="error",
                    execution_time=0.1,
                    working_directory="/tmp",
                    approach_used="direct_execution"
                )
            executor.execution_history.append(result)
        
        stats = executor.get_execution_stats()
        
        assert stats["total_executions"] == 10
        assert stats["successful_executions"] == 7
        assert stats["failed_executions"] == 3
        assert stats["success_rate"] == 70.0
        assert abs(stats["average_execution_time"] - 0.1) < 0.001
    
    def test_clear_history(self, executor):
        """Test clearing execution history."""
        # Add some mock executions
        for i in range(5):
            mock_result = ExecutionResult.create_success(
                command=f"echo 'test{i}'",
                stdout=f"test{i}",
                execution_time=0.1,
                working_directory="/tmp",
                approach_used="direct_execution"
            )
            executor.execution_history.append(mock_result)
        
        assert len(executor.execution_history) == 5
        
        executor.clear_history()
        
        assert len(executor.execution_history) == 0
    
    @pytest.mark.asyncio
    async def test_execute_multiple_commands_success(self, executor):
        """Test execution of multiple commands in sequence."""
        commands = [
            "echo 'first'",
            "echo 'second'",
            "echo 'third'"
        ]
        
        results = await executor.execute_multiple_commands(commands)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert "first" in results[0].stdout
        assert "second" in results[1].stdout
        assert "third" in results[2].stdout
    
    @pytest.mark.asyncio
    async def test_execute_multiple_commands_stop_on_failure(self, executor):
        """Test multiple command execution with stop on failure."""
        commands = [
            "echo 'first'",
            "false",  # This will fail
            "echo 'third'"  # This should not execute
        ]
        
        results = await executor.execute_multiple_commands(commands, stop_on_failure=True)
        
        assert len(results) == 2  # Should stop after the failed command
        assert results[0].success is True
        assert results[1].success is False
    
    @pytest.mark.asyncio
    async def test_execute_multiple_commands_continue_on_failure(self, executor):
        """Test multiple command execution continuing after failure."""
        commands = [
            "echo 'first'",
            "false",  # This will fail
            "echo 'third'"  # This should still execute
        ]
        
        results = await executor.execute_multiple_commands(commands, stop_on_failure=False)
        
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True
    
    def test_set_default_working_dir(self, executor, temp_dir):
        """Test setting default working directory."""
        executor.set_default_working_dir(temp_dir)
        assert executor.default_working_dir == temp_dir
    
    @pytest.mark.asyncio
    async def test_working_directory_creation(self, executor, temp_dir):
        """Test that working directory is created if it doesn't exist."""
        non_existent_dir = os.path.join(temp_dir, "new_dir", "nested")
        
        result = await executor.execute_command("pwd", working_dir=non_existent_dir)
        
        assert result.success is True
        assert Path(non_existent_dir).exists()
    
    def test_empty_history_stats(self, executor):
        """Test statistics when execution history is empty."""
        executor.clear_history()
        stats = executor.get_execution_stats()
        
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        assert stats["failed_executions"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["average_execution_time"] == 0.0
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, executor):
        """Test handling of exceptions during command execution."""
        with patch('asyncio.create_subprocess_shell', side_effect=Exception("Test exception")):
            result = await executor.execute_command("test_command")
            
            assert result.success is False
            assert result.return_code == -2
            assert "Exception during command execution" in result.stderr
            assert "Test exception" in result.stderr


if __name__ == "__main__":
    pytest.main([__file__])