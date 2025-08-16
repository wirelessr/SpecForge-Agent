"""
TestExecutor for integration test discovery and execution.

This module provides the TestExecutor class that handles discovery of integration
tests and executes them individually with timeout management and detailed result
collection.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import (
    TestIdentifier, ExecutionResult, TestTimingResult, 
    FileTimingResult, TestStatus
)
from .config import PerformanceConfig, EnvironmentConfig
from .utils import (
    TestDiscovery, CommandBuilder, ProcessManager, 
    StatusConverter, FileManager
)


class TestExecutor:
    """
    Executes individual integration tests with timeout and timing analysis.
    
    This class handles the discovery of integration tests and executes them
    individually to collect detailed timing and result information. It supports
    both file-level and method-level test execution with proper timeout handling.
    """
    
    def __init__(self, timeout_seconds: int = None):
        """
        Initialize TestExecutor with configuration.
        
        Args:
            timeout_seconds: Maximum execution time per test (default from config)
        """
        self.timeout_seconds = timeout_seconds or PerformanceConfig.DEFAULT_TIMEOUT_SECONDS
        self.integration_test_dir = PerformanceConfig.INTEGRATION_TEST_DIR
        self.process_manager = ProcessManager()
        self.status_converter = StatusConverter()
        
        # Apply environment timeout multiplier
        timeout_multiplier = EnvironmentConfig.get_timeout_multiplier()
        self.timeout_seconds = int(self.timeout_seconds * timeout_multiplier)
        
        # Ensure output directories exist
        FileManager.ensure_output_directories()
    
    async def execute_with_timeout(self, test_command: List[str]) -> ExecutionResult:
        """
        Execute integration test with timeout and capture results.
        
        Args:
            test_command: Command to execute as list of strings
            
        Returns:
            ExecutionResult with timing and output information
        """
        return await self.process_manager.execute_with_timeout(
            command=test_command,
            timeout_seconds=self.timeout_seconds
        )
    
    def discover_integration_tests(self) -> List[TestIdentifier]:
        """
        Discover all integration test files and methods.
        
        Scans the tests/integration/ directory to find all test files and
        extracts individual test methods for granular execution.
        
        Returns:
            List of TestIdentifier objects for all discovered tests
        """
        return TestDiscovery.discover_integration_tests()
    
    def build_pytest_command(self, test_file: str, test_method: str = None) -> List[str]:
        """
        Build pytest command for specific integration test.
        
        Args:
            test_file: Path to test file
            test_method: Optional specific test method name
            
        Returns:
            Command as list of strings ready for execution
        """
        # Create TestIdentifier for command building
        test_identifier = TestIdentifier(
            file_path=test_file,
            class_name=None,
            method_name=test_method
        )
        
        return CommandBuilder.build_pytest_command(
            test_identifier=test_identifier,
            timeout=self.timeout_seconds,
            verbose=True,
            capture_output=True
        )
    
    async def execute_single_test_file(self, test_file: str) -> FileTimingResult:
        """
        Execute all tests in a single integration test file with individual timing.
        
        Args:
            test_file: Path to the test file to execute
            
        Returns:
            FileTimingResult with timing data for all tests in the file
        """
        start_time = time.time()
        test_results = []
        
        # First, discover all test methods in this file
        all_tests = self.discover_integration_tests()
        file_tests = [test for test in all_tests if test.file_path == test_file]
        
        # If no specific methods found, run the entire file
        if not any(test.method_name for test in file_tests):
            result = await self._execute_single_test_method(test_file, None)
            test_results.append(result)
        else:
            # Execute each test method individually
            for test_identifier in file_tests:
                if test_identifier.method_name:  # Only execute method-level tests
                    result = await self._execute_single_test_method(
                        test_file, test_identifier.method_name
                    )
                    test_results.append(result)
        
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        # Determine overall file status
        file_status = self._determine_file_status(test_results)
        
        return FileTimingResult(
            file_path=test_file,
            total_execution_time=total_execution_time,
            test_results=test_results,
            file_status=file_status
        )
    
    async def execute_single_test_method(self, test_file: str, test_method: str) -> TestTimingResult:
        """
        Execute a single integration test method with timeout and timing.
        
        Args:
            test_file: Path to the test file
            test_method: Name of the test method to execute
            
        Returns:
            TestTimingResult with detailed execution information
        """
        return await self._execute_single_test_method(test_file, test_method)
    
    async def _execute_single_test_method(self, test_file: str, test_method: Optional[str]) -> TestTimingResult:
        """
        Internal method to execute a single test method or entire file.
        
        Args:
            test_file: Path to the test file
            test_method: Optional test method name (None for entire file)
            
        Returns:
            TestTimingResult with execution details
        """
        # Create test identifier
        test_identifier = TestIdentifier(
            file_path=test_file,
            class_name=None,
            method_name=test_method
        )
        
        # Build pytest command
        command = self.build_pytest_command(test_file, test_method)
        
        # Execute with timeout
        start_time = time.time()
        execution_result = await self.execute_with_timeout(command)
        end_time = time.time()
        
        # Convert execution result to test status
        test_status = self.status_converter.execution_result_to_test_status(execution_result)
        
        # Extract error message if any
        error_message = None
        if test_status in [TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT]:
            error_message = self._extract_error_message(execution_result)
        
        return TestTimingResult(
            test_identifier=test_identifier,
            execution_time=execution_result.execution_time,
            status=test_status,
            stdout=execution_result.stdout,
            stderr=execution_result.stderr,
            timeout_occurred=execution_result.timeout_occurred,
            error_message=error_message,
            start_time=start_time,
            end_time=end_time
        )
    
    def _determine_file_status(self, test_results: List[TestTimingResult]) -> TestStatus:
        """
        Determine overall file status based on individual test results.
        
        Args:
            test_results: List of individual test results
            
        Returns:
            Overall TestStatus for the file
        """
        if not test_results:
            return TestStatus.ERROR
        
        # If any test timed out, file is timeout
        if any(result.status == TestStatus.TIMEOUT for result in test_results):
            return TestStatus.TIMEOUT
        
        # If any test failed, file is failed
        if any(result.status == TestStatus.FAILED for result in test_results):
            return TestStatus.FAILED
        
        # If any test had error, file has error
        if any(result.status == TestStatus.ERROR for result in test_results):
            return TestStatus.ERROR
        
        # If all tests passed, file passed
        if all(result.status == TestStatus.PASSED for result in test_results):
            return TestStatus.PASSED
        
        # Default to error for mixed results
        return TestStatus.ERROR
    
    def _extract_error_message(self, execution_result: ExecutionResult) -> str:
        """
        Extract meaningful error message from execution result.
        
        Args:
            execution_result: Result from test execution
            
        Returns:
            Extracted error message
        """
        if execution_result.timeout_occurred:
            return f"Test timed out after {self.timeout_seconds} seconds"
        
        # Try to extract pytest error information
        stderr = execution_result.stderr.strip()
        stdout = execution_result.stdout.strip()
        
        # Look for common error patterns
        error_lines = []
        
        # Check stderr first
        if stderr:
            lines = stderr.split('\n')
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'traceback']):
                    error_lines.append(line)
        
        # Check stdout for pytest failure information
        if stdout and not error_lines:
            lines = stdout.split('\n')
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['failed', 'error', 'assert']):
                    error_lines.append(line)
        
        if error_lines:
            return '; '.join(error_lines[:3])  # Limit to first 3 error lines
        
        # Fallback to exit code information
        return f"Test failed with exit code {execution_result.exit_code}"
    
    async def execute_test_list(self, test_identifiers: List[TestIdentifier]) -> List[TestTimingResult]:
        """
        Execute a list of specific tests and return timing results.
        
        Args:
            test_identifiers: List of tests to execute
            
        Returns:
            List of TestTimingResult objects
        """
        results = []
        
        for test_identifier in test_identifiers:
            result = await self._execute_single_test_method(
                test_identifier.file_path,
                test_identifier.method_name
            )
            results.append(result)
        
        return results
    
    def get_slow_tests(self, test_results: List[TestTimingResult], 
                      threshold_seconds: Optional[float] = None) -> List[TestTimingResult]:
        """
        Filter test results to identify slow tests.
        
        Args:
            test_results: List of test timing results
            threshold_seconds: Minimum execution time to consider slow
                             (default from config)
            
        Returns:
            List of slow test results sorted by execution time
        """
        if threshold_seconds is None:
            threshold_seconds = PerformanceConfig.SLOW_TEST_THRESHOLD_SECONDS
        
        slow_tests = [
            result for result in test_results 
            if result.execution_time >= threshold_seconds
        ]
        
        return sorted(slow_tests, key=lambda x: x.execution_time, reverse=True)
    
    def get_timeout_tests(self, test_results: List[TestTimingResult]) -> List[TestTimingResult]:
        """
        Filter test results to identify tests that timed out.
        
        Args:
            test_results: List of test timing results
            
        Returns:
            List of timed out test results
        """
        return [
            result for result in test_results 
            if result.status == TestStatus.TIMEOUT
        ]
    
    def get_failed_tests(self, test_results: List[TestTimingResult]) -> List[TestTimingResult]:
        """
        Filter test results to identify failed tests.
        
        Args:
            test_results: List of test timing results
            
        Returns:
            List of failed test results
        """
        return [
            result for result in test_results 
            if result.status in [TestStatus.FAILED, TestStatus.ERROR]
        ]
    
    async def validate_test_environment(self) -> Dict[str, Any]:
        """
        Validate that the test environment is properly configured.
        
        Returns:
            Dictionary with validation results and any issues found
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'environment_info': {}
        }
        
        # Check if integration test directory exists
        integration_dir = Path(self.integration_test_dir)
        if not integration_dir.exists():
            validation_results['valid'] = False
            validation_results['issues'].append(
                f"Integration test directory not found: {self.integration_test_dir}"
            )
        else:
            # Count available tests
            test_count = len(self.discover_integration_tests())
            validation_results['environment_info']['test_count'] = test_count
            
            if test_count == 0:
                validation_results['issues'].append(
                    "No integration tests found in directory"
                )
        
        # Check pytest availability
        try:
            pytest_cmd = [EnvironmentConfig.get_pytest_executable(), '--version']
            result = await self.process_manager.execute_with_timeout(
                pytest_cmd, timeout_seconds=10
            )
            if result.exit_code == 0:
                validation_results['environment_info']['pytest_version'] = result.stdout.strip()
            else:
                validation_results['valid'] = False
                validation_results['issues'].append("pytest not available or not working")
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Error checking pytest: {str(e)}")
        
        # Check Python executable
        try:
            python_cmd = [EnvironmentConfig.get_python_executable(), '--version']
            result = await self.process_manager.execute_with_timeout(
                python_cmd, timeout_seconds=10
            )
            if result.exit_code == 0:
                validation_results['environment_info']['python_version'] = result.stdout.strip()
            else:
                validation_results['valid'] = False
                validation_results['issues'].append("Python executable not available")
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Error checking Python: {str(e)}")
        
        # Add configuration info
        validation_results['environment_info'].update({
            'timeout_seconds': self.timeout_seconds,
            'integration_test_dir': self.integration_test_dir,
            'output_dir': str(PerformanceConfig.get_output_dir())
        })
        
        return validation_results