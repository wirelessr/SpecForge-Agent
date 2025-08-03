"""
Shell command executor with async support, history tracking, and retry mechanisms.

This module provides the ShellExecutor class which handles all shell command execution
for the AutoGen multi-agent framework. It supports asynchronous execution, maintains
execution history, and implements retry mechanisms for failed commands.
"""

import asyncio
import os
import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import ExecutionResult


class ShellExecutor:
    """
    Handles shell command execution with comprehensive logging and retry capabilities.
    
    This class provides the core shell execution functionality needed by the framework,
    including support for different working directories, execution history tracking,
    and automatic retry mechanisms for failed commands.
    """
    
    def __init__(self, default_working_dir: Optional[str] = None):
        """
        Initialize the shell executor.
        
        Args:
            default_working_dir: Default working directory for command execution.
                                If None, uses current working directory.
        """
        self.default_working_dir = default_working_dir or os.getcwd()
        self.execution_history: List[ExecutionResult] = []
        self.logger = logging.getLogger(__name__)
        
        # Configure logging if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def execute_command(
        self, 
        command: str, 
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
        retry_count: int = 0,
        max_retries: int = 2
    ) -> ExecutionResult:
        """
        Execute a shell command asynchronously with retry mechanism.
        
        Args:
            command: The shell command to execute
            working_dir: Working directory for command execution
            timeout: Command timeout in seconds (default: 30)
            retry_count: Current retry attempt number
            max_retries: Maximum number of retry attempts
            
        Returns:
            ExecutionResult containing execution details and results
        """
        if working_dir is None:
            working_dir = self.default_working_dir
        
        if timeout is None:
            timeout = 30.0
        
        # Ensure working directory exists
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Executing command: {command} (working_dir: {working_dir})")
        
        start_time = time.time()
        
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                await process.wait()
                execution_time = time.time() - start_time
                
                result = ExecutionResult.create_failure(
                    command=command,
                    return_code=-1,
                    stderr=f"Command timed out after {timeout} seconds",
                    execution_time=execution_time,
                    working_directory=working_dir,
                    approach_used="direct_execution"
                )
                
                self.execution_history.append(result)
                
                # Retry if possible
                if retry_count < max_retries:
                    self.logger.warning(f"Command timed out, retrying ({retry_count + 1}/{max_retries})")
                    return await self.execute_command(
                        command, working_dir, timeout, retry_count + 1, max_retries
                    )
                
                return result
            
            execution_time = time.time() - start_time
            return_code = process.returncode
            
            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace') if stdout else ""
            stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ""
            
            # Create result
            if return_code == 0:
                result = ExecutionResult.create_success(
                    command=command,
                    stdout=stdout_str,
                    execution_time=execution_time,
                    working_directory=working_dir,
                    approach_used="direct_execution"
                )
                self.logger.info(f"Command completed successfully in {execution_time:.2f}s")
            else:
                result = ExecutionResult.create_failure(
                    command=command,
                    return_code=return_code,
                    stderr=stderr_str,
                    execution_time=execution_time,
                    working_directory=working_dir,
                    approach_used="direct_execution"
                )
                self.logger.warning(f"Command failed with return code {return_code}")
                
                # Retry if possible
                if retry_count < max_retries:
                    self.logger.info(f"Retrying command ({retry_count + 1}/{max_retries})")
                    retry_result = await self.execute_command(
                        command, working_dir, timeout, retry_count + 1, max_retries
                    )
                    # Add this attempt as an alternative
                    retry_result.add_alternative_attempt({
                        "attempt_number": retry_count + 1,
                        "command": command,
                        "return_code": return_code,
                        "stderr": stderr_str,
                        "execution_time": execution_time
                    })
                    self.execution_history.append(result)
                    return retry_result
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Exception during command execution: {str(e)}"
            
            result = ExecutionResult.create_failure(
                command=command,
                return_code=-2,
                stderr=error_msg,
                execution_time=execution_time,
                working_directory=working_dir,
                approach_used="direct_execution"
            )
            
            self.logger.error(error_msg)
            self.execution_history.append(result)
            
            # Retry if possible
            if retry_count < max_retries:
                self.logger.info(f"Retrying after exception ({retry_count + 1}/{max_retries})")
                return await self.execute_command(
                    command, working_dir, timeout, retry_count + 1, max_retries
                )
            
            return result
    
    def get_execution_log(self) -> List[ExecutionResult]:
        """
        Get the complete execution history.
        
        Returns:
            List of ExecutionResult objects representing all executed commands
        """
        return self.execution_history.copy()
    
    def get_recent_executions(self, count: int = 10) -> List[ExecutionResult]:
        """
        Get the most recent command executions.
        
        Args:
            count: Number of recent executions to return
            
        Returns:
            List of the most recent ExecutionResult objects
        """
        return self.execution_history[-count:] if self.execution_history else []
    
    def get_failed_executions(self) -> List[ExecutionResult]:
        """
        Get all failed command executions.
        
        Returns:
            List of ExecutionResult objects for failed commands
        """
        return [result for result in self.execution_history if not result.success]
    
    def get_successful_executions(self) -> List[ExecutionResult]:
        """
        Get all successful command executions.
        
        Returns:
            List of ExecutionResult objects for successful commands
        """
        return [result for result in self.execution_history if result.success]
    
    def clear_history(self) -> None:
        """Clear the execution history."""
        self.execution_history.clear()
        self.logger.info("Execution history cleared")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about command executions.
        
        Returns:
            Dictionary containing execution statistics
        """
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0
            }
        
        successful = len(self.get_successful_executions())
        failed = len(self.get_failed_executions())
        total = len(self.execution_history)
        
        avg_time = sum(result.execution_time for result in self.execution_history) / total
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": (successful / total) * 100 if total > 0 else 0.0,
            "average_execution_time": avg_time
        }
    
    async def execute_multiple_commands(
        self, 
        commands: List[str], 
        working_dir: Optional[str] = None,
        stop_on_failure: bool = True
    ) -> List[ExecutionResult]:
        """
        Execute multiple commands in sequence.
        
        Args:
            commands: List of commands to execute
            working_dir: Working directory for all commands
            stop_on_failure: Whether to stop execution if a command fails
            
        Returns:
            List of ExecutionResult objects for all executed commands
        """
        results = []
        
        for command in commands:
            result = await self.execute_command(command, working_dir)
            results.append(result)
            
            if not result.success and stop_on_failure:
                self.logger.warning(f"Stopping execution due to failed command: {command}")
                break
        
        return results
    
    def set_default_working_dir(self, working_dir: str) -> None:
        """
        Set the default working directory for future command executions.
        
        Args:
            working_dir: New default working directory
        """
        self.default_working_dir = working_dir
        self.logger.info(f"Default working directory set to: {working_dir}")