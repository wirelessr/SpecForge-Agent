"""
TestTimingAnalyzer for comprehensive timing analysis of integration tests.

This module provides the TestTimingAnalyzer class that orchestrates the timing
analysis of all integration tests, running them individually with timeout handling
and generating comprehensive timing reports to identify slow tests.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .models import (
    TestIdentifier, TimingReport, FileTimingResult, TestTimingResult,
    TestStatus, ExecutionResult
)
from .config import PerformanceConfig, EnvironmentConfig
from .test_executor import TestExecutor
from .utils import FileManager


class TestTimingAnalyzer:
    """
    Orchestrates comprehensive timing analysis of integration tests.
    
    This class manages the execution of all integration tests with individual
    timing measurements, timeout handling, and generates detailed reports
    identifying the slowest tests for further profiling analysis.
    """
    
    def __init__(self, test_directory: str = None, timeout_seconds: int = None):
        """
        Initialize TestTimingAnalyzer with configuration.
        
        Args:
            test_directory: Directory containing integration tests 
                          (default: from PerformanceConfig)
            timeout_seconds: Maximum execution time per test 
                           (default: from PerformanceConfig)
        """
        self.test_directory = test_directory or PerformanceConfig.INTEGRATION_TEST_DIR
        self.timeout_seconds = timeout_seconds or PerformanceConfig.DEFAULT_TIMEOUT_SECONDS
        
        # Initialize TestExecutor with our timeout
        self.test_executor = TestExecutor(timeout_seconds=self.timeout_seconds)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.results: List[FileTimingResult] = []
        
        # Ensure output directories exist
        FileManager.ensure_output_directories()
    
    async def analyze_all_integration_tests(self) -> TimingReport:
        """
        Run all integration tests with comprehensive timing analysis.
        
        This method discovers all integration tests, executes them individually
        with timeout handling, and generates a comprehensive timing report
        identifying the slowest tests.
        
        Returns:
            TimingReport with detailed timing analysis for all tests
        """
        self.logger.info("Starting comprehensive integration test timing analysis")
        start_time = time.time()
        
        # Validate test environment first
        validation_result = await self.test_executor.validate_test_environment()
        if not validation_result['valid']:
            raise RuntimeError(
                f"Test environment validation failed: {validation_result['issues']}"
            )
        
        # Discover all integration tests
        all_tests = self.test_executor.discover_integration_tests()
        if not all_tests:
            self.logger.warning("No integration tests found for analysis")
            return TimingReport(
                total_execution_time=0.0,
                file_results=[],
                timeout_threshold=self.timeout_seconds
            )
        
        self.logger.info(f"Discovered {len(all_tests)} integration tests")
        
        # Group tests by file for organized execution
        tests_by_file = self._group_tests_by_file(all_tests)
        
        # Execute each test method individually
        file_results = []
        for file_path, file_tests in tests_by_file.items():
            self.logger.info(f"Analyzing tests in {file_path}")
            
            try:
                # Get only method-level tests (skip file-level tests)
                method_tests = [test for test in file_tests if test.method_name]
                
                if not method_tests:
                    # If no method tests found, run the whole file
                    file_result = await self.run_single_test_file(file_path)
                else:
                    # Run each test method individually
                    test_results = []
                    file_start_time = time.time()
                    
                    for test_identifier in method_tests:
                        self.logger.debug(f"Running {test_identifier.full_name}")
                        
                        try:
                            # Execute using the full test identifier
                            result = await self._execute_single_test_with_identifier(test_identifier)
                            test_results.append(result)
                            
                        except Exception as e:
                            self.logger.error(f"Error running {test_identifier.full_name}: {e}")
                            # Create error result for this test
                            error_result = TestTimingResult(
                                test_identifier=test_identifier,
                                execution_time=0.0,
                                status=TestStatus.ERROR,
                                error_message=str(e)
                            )
                            test_results.append(error_result)
                    
                    file_end_time = time.time()
                    total_file_time = file_end_time - file_start_time
                    
                    # Determine overall file status
                    file_status = self._determine_file_status(test_results)
                    
                    file_result = FileTimingResult(
                        file_path=file_path,
                        total_execution_time=total_file_time,
                        test_results=test_results,
                        file_status=file_status
                    )
                
                file_results.append(file_result)
                
                # Log progress
                self._log_file_progress(file_result)
                
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {str(e)}")
                # Create error result for this file
                error_result = FileTimingResult(
                    file_path=file_path,
                    total_execution_time=0.0,
                    test_results=[],
                    file_status=TestStatus.ERROR
                )
                file_results.append(error_result)
        
        # Calculate total execution time
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        # Create comprehensive timing report
        timing_report = TimingReport(
            total_execution_time=total_execution_time,
            file_results=file_results,
            timeout_threshold=self.timeout_seconds
        )
        
        # Log summary
        self._log_analysis_summary(timing_report)
        
        # Store results for later access
        self.results = file_results
        
        return timing_report
    
    async def run_single_test_file(self, test_file: str) -> FileTimingResult:
        """
        Run all tests in a single integration test file with individual timing.
        
        Args:
            test_file: Path to the test file to execute
            
        Returns:
            FileTimingResult with timing data for all tests in the file
        """
        self.logger.debug(f"Running tests in file: {test_file}")
        
        try:
            # Delegate to TestExecutor for actual execution
            file_result = await self.test_executor.execute_single_test_file(test_file)
            
            # Add any additional analysis here if needed
            self._analyze_file_result(file_result)
            
            return file_result
            
        except Exception as e:
            self.logger.error(f"Error executing tests in {test_file}: {str(e)}")
            
            # Return error result
            return FileTimingResult(
                file_path=test_file,
                total_execution_time=0.0,
                test_results=[],
                file_status=TestStatus.ERROR
            )
    
    async def run_single_test_method(self, test_file: str, test_method: str) -> TestTimingResult:
        """
        Run a single integration test method with timeout and timing.
        
        Args:
            test_file: Path to the test file
            test_method: Name of the test method to execute
            
        Returns:
            TestTimingResult with detailed execution information
        """
        self.logger.debug(f"Running test method: {test_file}::{test_method}")
        
        try:
            # Find the test identifier with class information
            all_tests = self.test_executor.discover_integration_tests()
            target_test = None
            
            for test in all_tests:
                if (test.file_path == test_file and 
                    test.method_name == test_method):
                    target_test = test
                    break
            
            if target_test:
                # Use the full test identifier
                return await self._execute_single_test_with_identifier(target_test)
            else:
                # Fallback: create basic identifier
                test_identifier = TestIdentifier(
                    file_path=test_file,
                    method_name=test_method
                )
                return await self._execute_single_test_with_identifier(test_identifier)
            
        except Exception as e:
            self.logger.error(f"Error executing {test_file}::{test_method}: {str(e)}")
            
            # Create error result
            test_identifier = TestIdentifier(
                file_path=test_file,
                method_name=test_method
            )
            
            return TestTimingResult(
                test_identifier=test_identifier,
                execution_time=0.0,
                status=TestStatus.ERROR,
                error_message=str(e)
            )
    
    def generate_timing_report(self, 
                             results: Optional[List[FileTimingResult]] = None) -> TimingReport:
        """
        Generate comprehensive timing report from analysis results.
        
        Args:
            results: Optional list of file results (uses stored results if None)
            
        Returns:
            TimingReport with comprehensive analysis
        """
        if results is None:
            results = self.results
        
        if not results:
            return TimingReport(
                total_execution_time=0.0,
                file_results=[],
                timeout_threshold=self.timeout_seconds
            )
        
        # Calculate total execution time
        total_time = sum(result.total_execution_time for result in results)
        
        return TimingReport(
            total_execution_time=total_time,
            file_results=results,
            timeout_threshold=self.timeout_seconds
        )
    
    def identify_slow_tests(self, 
                          timing_report: TimingReport,
                          threshold_seconds: Optional[float] = None) -> List[TestTimingResult]:
        """
        Identify slow tests from timing report for profiling analysis.
        
        Args:
            timing_report: Comprehensive timing report
            threshold_seconds: Minimum execution time to consider slow
                             (default from config)
            
        Returns:
            List of slow test results sorted by execution time (slowest first)
        """
        if threshold_seconds is None:
            threshold_seconds = PerformanceConfig.SLOW_TEST_THRESHOLD_SECONDS
        
        slow_tests = []
        
        for file_result in timing_report.file_results:
            for test_result in file_result.test_results:
                if (test_result.execution_time >= threshold_seconds and 
                    test_result.status == TestStatus.PASSED):
                    slow_tests.append(test_result)
        
        # Sort by execution time (slowest first)
        return sorted(slow_tests, key=lambda x: x.execution_time, reverse=True)
    
    def identify_timeout_tests(self, timing_report: TimingReport) -> List[TestTimingResult]:
        """
        Identify tests that timed out during execution.
        
        Args:
            timing_report: Comprehensive timing report
            
        Returns:
            List of timed out test results
        """
        timeout_tests = []
        
        for file_result in timing_report.file_results:
            for test_result in file_result.test_results:
                if test_result.status == TestStatus.TIMEOUT:
                    timeout_tests.append(test_result)
        
        return timeout_tests
    
    def identify_failed_tests(self, timing_report: TimingReport) -> List[TestTimingResult]:
        """
        Identify tests that failed during execution.
        
        Args:
            timing_report: Comprehensive timing report
            
        Returns:
            List of failed test results
        """
        failed_tests = []
        
        for file_result in timing_report.file_results:
            for test_result in file_result.test_results:
                if test_result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    failed_tests.append(test_result)
        
        return failed_tests
    
    async def analyze_specific_tests(self, 
                                   test_identifiers: List[TestIdentifier]) -> List[TestTimingResult]:
        """
        Analyze timing for a specific list of tests.
        
        Args:
            test_identifiers: List of specific tests to analyze
            
        Returns:
            List of TestTimingResult objects for the specified tests
        """
        self.logger.info(f"Analyzing {len(test_identifiers)} specific tests")
        
        results = []
        
        for test_identifier in test_identifiers:
            try:
                result = await self._execute_single_test_with_identifier(test_identifier)
                results.append(result)
                
            except Exception as e:
                self.logger.error(
                    f"Error analyzing {test_identifier.full_name}: {str(e)}"
                )
                
                # Create error result
                error_result = TestTimingResult(
                    test_identifier=test_identifier,
                    execution_time=0.0,
                    status=TestStatus.ERROR,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def get_analysis_summary(self, timing_report: TimingReport) -> Dict[str, Any]:
        """
        Generate summary statistics from timing analysis.
        
        Args:
            timing_report: Comprehensive timing report
            
        Returns:
            Dictionary with summary statistics
        """
        all_tests = []
        for file_result in timing_report.file_results:
            all_tests.extend(file_result.test_results)
        
        if not all_tests:
            return {
                'total_tests': 0,
                'total_execution_time': 0.0,
                'average_execution_time': 0.0,
                'slowest_test_time': 0.0,
                'fastest_test_time': 0.0,
                'timeout_count': 0,
                'failed_count': 0,
                'passed_count': 0
            }
        
        execution_times = [test.execution_time for test in all_tests]
        
        return {
            'total_tests': len(all_tests),
            'total_execution_time': timing_report.total_execution_time,
            'average_execution_time': sum(execution_times) / len(execution_times),
            'slowest_test_time': max(execution_times),
            'fastest_test_time': min(execution_times),
            'timeout_count': len([t for t in all_tests if t.status == TestStatus.TIMEOUT]),
            'failed_count': len([t for t in all_tests if t.status in [TestStatus.FAILED, TestStatus.ERROR]]),
            'passed_count': len([t for t in all_tests if t.status == TestStatus.PASSED])
        }
    
    def _group_tests_by_file(self, tests: List[TestIdentifier]) -> Dict[str, List[TestIdentifier]]:
        """Group test identifiers by file path."""
        tests_by_file = {}
        
        for test in tests:
            file_path = test.file_path
            if file_path not in tests_by_file:
                tests_by_file[file_path] = []
            tests_by_file[file_path].append(test)
        
        return tests_by_file
    
    def _analyze_file_result(self, file_result: FileTimingResult):
        """Perform additional analysis on file result."""
        # Add any file-level analysis here
        # For now, just log some basic info
        self.logger.debug(
            f"File {file_result.file_path}: "
            f"{file_result.test_count} tests, "
            f"{file_result.total_execution_time:.2f}s total"
        )
    
    def _log_file_progress(self, file_result: FileTimingResult):
        """Log progress information for a completed file."""
        self.logger.info(
            f"Completed {file_result.file_path}: "
            f"{file_result.passed_count}/{file_result.test_count} passed, "
            f"{file_result.total_execution_time:.2f}s"
        )
        
        if file_result.timeout_count > 0:
            self.logger.warning(
                f"  {file_result.timeout_count} tests timed out"
            )
        
        if file_result.failed_count > 0:
            self.logger.warning(
                f"  {file_result.failed_count} tests failed"
            )
    
    async def _execute_single_test_with_identifier(self, test_identifier: TestIdentifier) -> TestTimingResult:
        """
        Execute a single test using the full TestIdentifier.
        
        Args:
            test_identifier: Complete test identifier with class and method info
            
        Returns:
            TestTimingResult with execution details
        """
        # Build pytest command using the full test identifier
        from .utils import CommandBuilder
        command = CommandBuilder.build_pytest_command(
            test_identifier=test_identifier,
            timeout=self.timeout_seconds,
            verbose=True,
            capture_output=True
        )
        
        # Execute with timeout
        start_time = time.time()
        execution_result = await self.test_executor.execute_with_timeout(command)
        end_time = time.time()
        
        # Convert execution result to test status
        test_status = self.test_executor.status_converter.execution_result_to_test_status(execution_result)
        
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
    
    def _determine_file_status(self, test_results: List[TestTimingResult]) -> TestStatus:
        """Determine overall file status based on individual test results."""
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
    
    def _log_analysis_summary(self, timing_report: TimingReport):
        """Log summary of the complete analysis."""
        summary = self.get_analysis_summary(timing_report)
        
        self.logger.info("=== Timing Analysis Summary ===")
        self.logger.info(f"Total tests analyzed: {summary['total_tests']}")
        self.logger.info(f"Total execution time: {summary['total_execution_time']:.2f}s")
        self.logger.info(f"Average test time: {summary['average_execution_time']:.2f}s")
        self.logger.info(f"Slowest test: {summary['slowest_test_time']:.2f}s")
        self.logger.info(f"Fastest test: {summary['fastest_test_time']:.2f}s")
        self.logger.info(f"Passed: {summary['passed_count']}")
        self.logger.info(f"Failed: {summary['failed_count']}")
        self.logger.info(f"Timeouts: {summary['timeout_count']}")
        
        # Identify slow tests for profiling
        slow_tests = self.identify_slow_tests(timing_report)
        if slow_tests:
            self.logger.info(f"Slow tests identified for profiling: {len(slow_tests)}")
            for i, test in enumerate(slow_tests[:5]):  # Show top 5
                self.logger.info(
                    f"  {i+1}. {test.test_identifier.full_name}: "
                    f"{test.execution_time:.2f}s"
                )
        
        self.logger.info("=== End Summary ===")
    
    async def save_timing_report(self, 
                               timing_report: TimingReport,
                               output_path: Optional[Path] = None) -> Path:
        """
        Save timing report to file for later analysis.
        
        Args:
            timing_report: Timing report to save
            output_path: Optional output path (default: auto-generated)
            
        Returns:
            Path where report was saved
        """
        if output_path is None:
            timestamp = int(time.time())
            filename = f"timing_report_{timestamp}.json"
            output_path = PerformanceConfig.get_output_dir("reports") / filename
        
        # Convert to JSON-serializable format
        report_data = {
            'total_execution_time': timing_report.total_execution_time,
            'analysis_timestamp': timing_report.analysis_timestamp,
            'timeout_threshold': timing_report.timeout_threshold,
            'total_test_count': timing_report.total_test_count,
            'file_results': []
        }
        
        for file_result in timing_report.file_results:
            file_data = {
                'file_path': file_result.file_path,
                'total_execution_time': file_result.total_execution_time,
                'file_status': file_result.file_status.value,
                'test_count': file_result.test_count,
                'passed_count': file_result.passed_count,
                'failed_count': file_result.failed_count,
                'timeout_count': file_result.timeout_count,
                'test_results': []
            }
            
            for test_result in file_result.test_results:
                test_data = {
                    'test_identifier': {
                        'file_path': test_result.test_identifier.file_path,
                        'class_name': test_result.test_identifier.class_name,
                        'method_name': test_result.test_identifier.method_name,
                        'full_name': test_result.test_identifier.full_name
                    },
                    'execution_time': test_result.execution_time,
                    'status': test_result.status.value,
                    'timeout_occurred': test_result.timeout_occurred,
                    'error_message': test_result.error_message,
                    'start_time': test_result.start_time,
                    'end_time': test_result.end_time
                }
                file_data['test_results'].append(test_data)
            
            report_data['file_results'].append(file_data)
        
        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Timing report saved to: {output_path}")
        return output_path