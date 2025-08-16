"""
Unit tests for TestExecutor class.

This module tests the TestExecutor functionality for integration test
discovery and execution with proper mocking of external dependencies.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from tests.perf.test_executor import TestExecutor
from tests.perf.models import (
    TestIdentifier, ExecutionResult, TestTimingResult, 
    FileTimingResult, TestStatus
)


class TestTestExecutor:
    """Unit tests for TestExecutor class."""
    
    @pytest.fixture
    def test_executor(self):
        """Create TestExecutor instance for testing."""
        return TestExecutor(timeout_seconds=30)
    
    @pytest.fixture
    def mock_execution_result(self):
        """Mock ExecutionResult for testing."""
        return ExecutionResult(
            command="pytest test_file.py::test_method",
            exit_code=0,
            stdout="test output",
            stderr="",
            execution_time=5.0,
            timeout_occurred=False,
            start_time=1000.0,
            end_time=1005.0
        )
    
    def test_init_default_timeout(self):
        """Test TestExecutor initialization with default timeout."""
        executor = TestExecutor()
        assert executor.timeout_seconds == 60  # Default from config
        assert executor.integration_test_dir == "tests/integration"
    
    def test_init_custom_timeout(self):
        """Test TestExecutor initialization with custom timeout."""
        executor = TestExecutor(timeout_seconds=120)
        assert executor.timeout_seconds == 120
    
    def test_build_pytest_command_file_only(self, test_executor):
        """Test building pytest command for entire file."""
        command = test_executor.build_pytest_command("tests/integration/test_example.py")
        
        assert "pytest" in command[0]
        assert "tests/integration/test_example.py" in command
        assert "-v" in command
        assert "--timeout" in command
        assert "30" in command  # timeout value
    
    def test_build_pytest_command_with_method(self, test_executor):
        """Test building pytest command for specific method."""
        command = test_executor.build_pytest_command(
            "tests/integration/test_example.py", 
            "test_method"
        )
        
        assert "pytest" in command[0]
        assert "tests/integration/test_example.py::test_method" in command
        assert "-v" in command
    
    @patch('tests.perf.test_executor.TestDiscovery.discover_integration_tests')
    def test_discover_integration_tests(self, mock_discover, test_executor):
        """Test integration test discovery."""
        # Mock the discovery result
        mock_tests = [
            TestIdentifier("tests/integration/test_a.py", None, None),
            TestIdentifier("tests/integration/test_a.py", "TestClass", "test_method1"),
            TestIdentifier("tests/integration/test_b.py", None, "test_function")
        ]
        mock_discover.return_value = mock_tests
        
        result = test_executor.discover_integration_tests()
        
        assert len(result) == 3
        assert all(isinstance(test, TestIdentifier) for test in result)
        mock_discover.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_with_timeout_success(self, test_executor, mock_execution_result):
        """Test successful test execution with timeout."""
        with patch.object(test_executor.process_manager, 'execute_with_timeout', 
                         return_value=mock_execution_result) as mock_execute:
            
            result = await test_executor.execute_with_timeout(["pytest", "test.py"])
            
            assert result == mock_execution_result
            mock_execute.assert_called_once_with(
                command=["pytest", "test.py"],
                timeout_seconds=30
            )
    
    @pytest.mark.asyncio
    async def test_execute_single_test_method(self, test_executor, mock_execution_result):
        """Test executing a single test method."""
        with patch.object(test_executor, 'execute_with_timeout', 
                         return_value=mock_execution_result) as mock_execute:
            
            result = await test_executor.execute_single_test_method(
                "tests/integration/test_example.py", 
                "test_method"
            )
            
            assert isinstance(result, TestTimingResult)
            assert result.test_identifier.file_path == "tests/integration/test_example.py"
            assert result.test_identifier.method_name == "test_method"
            assert result.execution_time == 5.0
            assert result.status == TestStatus.PASSED
            mock_execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_single_test_method_timeout(self, test_executor):
        """Test executing a test method that times out."""
        timeout_result = ExecutionResult(
            command="pytest test.py",
            exit_code=-1,
            stdout="",
            stderr="Process timed out",
            execution_time=30.0,
            timeout_occurred=True
        )
        
        with patch.object(test_executor, 'execute_with_timeout', 
                         return_value=timeout_result):
            
            result = await test_executor.execute_single_test_method(
                "tests/integration/test_example.py", 
                "test_slow_method"
            )
            
            assert result.status == TestStatus.TIMEOUT
            assert result.timeout_occurred is True
            assert "timed out" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_execute_single_test_file(self, test_executor):
        """Test executing all tests in a single file."""
        # Mock discovery to return some test methods
        mock_tests = [
            TestIdentifier("tests/integration/test_example.py", "TestClass", "test_method1"),
            TestIdentifier("tests/integration/test_example.py", "TestClass", "test_method2")
        ]
        
        # Mock individual test execution results
        mock_result1 = TestTimingResult(
            test_identifier=mock_tests[0],
            execution_time=3.0,
            status=TestStatus.PASSED,
            start_time=1000.0,
            end_time=1003.0
        )
        mock_result2 = TestTimingResult(
            test_identifier=mock_tests[1],
            execution_time=2.0,
            status=TestStatus.PASSED,
            start_time=1003.0,
            end_time=1005.0
        )
        
        with patch.object(test_executor, 'discover_integration_tests', 
                         return_value=mock_tests), \
             patch.object(test_executor, '_execute_single_test_method', 
                         side_effect=[mock_result1, mock_result2]):
            
            result = await test_executor.execute_single_test_file(
                "tests/integration/test_example.py"
            )
            
            assert isinstance(result, FileTimingResult)
            assert result.file_path == "tests/integration/test_example.py"
            assert len(result.test_results) == 2
            assert result.file_status == TestStatus.PASSED
            assert result.test_count == 2
            assert result.passed_count == 2
    
    def test_get_slow_tests(self, test_executor):
        """Test filtering slow tests from results."""
        test_results = [
            TestTimingResult(
                test_identifier=TestIdentifier("test1.py", None, "fast_test"),
                execution_time=2.0,
                status=TestStatus.PASSED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier("test2.py", None, "slow_test"),
                execution_time=15.0,
                status=TestStatus.PASSED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier("test3.py", None, "very_slow_test"),
                execution_time=25.0,
                status=TestStatus.PASSED
            )
        ]
        
        slow_tests = test_executor.get_slow_tests(test_results, threshold_seconds=10.0)
        
        assert len(slow_tests) == 2
        assert slow_tests[0].execution_time == 25.0  # Sorted by time descending
        assert slow_tests[1].execution_time == 15.0
    
    def test_get_timeout_tests(self, test_executor):
        """Test filtering timeout tests from results."""
        test_results = [
            TestTimingResult(
                test_identifier=TestIdentifier("test1.py", None, "normal_test"),
                execution_time=5.0,
                status=TestStatus.PASSED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier("test2.py", None, "timeout_test"),
                execution_time=60.0,
                status=TestStatus.TIMEOUT,
                timeout_occurred=True
            )
        ]
        
        timeout_tests = test_executor.get_timeout_tests(test_results)
        
        assert len(timeout_tests) == 1
        assert timeout_tests[0].status == TestStatus.TIMEOUT
    
    def test_get_failed_tests(self, test_executor):
        """Test filtering failed tests from results."""
        test_results = [
            TestTimingResult(
                test_identifier=TestIdentifier("test1.py", None, "passing_test"),
                execution_time=5.0,
                status=TestStatus.PASSED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier("test2.py", None, "failed_test"),
                execution_time=3.0,
                status=TestStatus.FAILED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier("test3.py", None, "error_test"),
                execution_time=1.0,
                status=TestStatus.ERROR
            )
        ]
        
        failed_tests = test_executor.get_failed_tests(test_results)
        
        assert len(failed_tests) == 2
        assert TestStatus.FAILED in [test.status for test in failed_tests]
        assert TestStatus.ERROR in [test.status for test in failed_tests]
    
    def test_determine_file_status_all_passed(self, test_executor):
        """Test file status determination when all tests pass."""
        test_results = [
            TestTimingResult(
                test_identifier=TestIdentifier("test.py", None, "test1"),
                execution_time=1.0,
                status=TestStatus.PASSED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier("test.py", None, "test2"),
                execution_time=2.0,
                status=TestStatus.PASSED
            )
        ]
        
        status = test_executor._determine_file_status(test_results)
        assert status == TestStatus.PASSED
    
    def test_determine_file_status_with_failures(self, test_executor):
        """Test file status determination with some failures."""
        test_results = [
            TestTimingResult(
                test_identifier=TestIdentifier("test.py", None, "test1"),
                execution_time=1.0,
                status=TestStatus.PASSED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier("test.py", None, "test2"),
                execution_time=2.0,
                status=TestStatus.FAILED
            )
        ]
        
        status = test_executor._determine_file_status(test_results)
        assert status == TestStatus.FAILED
    
    def test_determine_file_status_with_timeout(self, test_executor):
        """Test file status determination with timeout."""
        test_results = [
            TestTimingResult(
                test_identifier=TestIdentifier("test.py", None, "test1"),
                execution_time=1.0,
                status=TestStatus.PASSED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier("test.py", None, "test2"),
                execution_time=60.0,
                status=TestStatus.TIMEOUT
            )
        ]
        
        status = test_executor._determine_file_status(test_results)
        assert status == TestStatus.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_validate_test_environment(self, test_executor):
        """Test test environment validation."""
        # Mock successful validation
        mock_pytest_result = ExecutionResult(
            command="pytest --version",
            exit_code=0,
            stdout="pytest 7.0.0",
            stderr="",
            execution_time=0.1
        )
        
        mock_python_result = ExecutionResult(
            command="python --version", 
            exit_code=0,
            stdout="Python 3.9.0",
            stderr="",
            execution_time=0.1
        )
        
        with patch.object(test_executor.process_manager, 'execute_with_timeout',
                         side_effect=[mock_pytest_result, mock_python_result]), \
             patch.object(test_executor, 'discover_integration_tests',
                         return_value=[TestIdentifier("test.py", None, None)]), \
             patch('pathlib.Path.exists', return_value=True):
            
            result = await test_executor.validate_test_environment()
            
            assert result['valid'] is True
            assert len(result['issues']) == 0
            assert 'pytest_version' in result['environment_info']
            assert 'python_version' in result['environment_info']
            assert result['environment_info']['test_count'] == 1