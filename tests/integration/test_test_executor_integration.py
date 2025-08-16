"""
Integration tests for TestExecutor with real test discovery and execution.

This module tests the TestExecutor functionality against the actual
integration test directory to verify real-world behavior.
"""

import pytest
import asyncio
from pathlib import Path

from tests.perf.test_executor import TestExecutor
from tests.perf.models import TestStatus, TestIdentifier


class TestTestExecutorIntegration:
    """Integration tests for TestExecutor class."""
    
    @pytest.fixture
    def test_executor(self):
        """Create TestExecutor instance for integration testing."""
        return TestExecutor(timeout_seconds=30)  # Shorter timeout for testing
    
    @pytest.mark.integration
    def test_discover_real_integration_tests(self, test_executor):
        """Test discovery of actual integration tests."""
        tests = test_executor.discover_integration_tests()
        
        # Should find some integration tests
        assert len(tests) > 0
        
        # All should be TestIdentifier objects
        assert all(isinstance(test, TestIdentifier) for test in tests)
        
        # Should have tests from the integration directory
        assert all("tests/integration" in test.file_path for test in tests)
        
        # Should have some file-level and some method-level tests
        file_level_tests = [test for test in tests if test.method_name is None]
        method_level_tests = [test for test in tests if test.method_name is not None]
        
        # We expect to have both types
        assert len(file_level_tests) > 0
        print(f"Found {len(file_level_tests)} file-level tests")
        print(f"Found {len(method_level_tests)} method-level tests")
    
    @pytest.mark.integration
    async def test_validate_test_environment_real(self, test_executor):
        """Test environment validation with real environment."""
        validation = await test_executor.validate_test_environment()
        
        # Environment should be valid
        assert validation['valid'] is True
        assert len(validation['issues']) == 0
        
        # Should have environment info
        env_info = validation['environment_info']
        assert 'test_count' in env_info
        assert 'pytest_version' in env_info
        assert 'python_version' in env_info
        assert 'timeout_seconds' in env_info
        
        # Test count should match discovery
        discovered_tests = test_executor.discover_integration_tests()
        assert env_info['test_count'] == len(discovered_tests)
        
        print(f"Environment validation: {validation}")
    
    @pytest.mark.integration
    def test_build_real_pytest_commands(self, test_executor):
        """Test building pytest commands for real integration tests."""
        tests = test_executor.discover_integration_tests()
        
        # Take first few tests to build commands for
        sample_tests = tests[:3] if len(tests) >= 3 else tests
        
        for test in sample_tests:
            if test.method_name:
                # Test method-specific command
                command = test_executor.build_pytest_command(
                    test.file_path, test.method_name
                )
            else:
                # Test file-level command
                command = test_executor.build_pytest_command(test.file_path)
            
            # Verify command structure
            assert "pytest" in command[0]
            assert test.file_path in ' '.join(command)
            assert "-v" in command
            assert "--timeout" in command
            
            print(f"Command for {test.full_name}: {' '.join(command)}")
    
    @pytest.mark.integration
    async def test_execute_simple_integration_test(self, test_executor):
        """Test executing a simple integration test (if available)."""
        tests = test_executor.discover_integration_tests()
        
        if not tests:
            pytest.skip("No integration tests found")
        
        # Find a simple test to execute (prefer file-level tests for speed)
        simple_test = None
        for test in tests:
            if test.method_name is None:  # File-level test
                simple_test = test
                break
        
        if not simple_test:
            # Fallback to first method-level test
            simple_test = tests[0]
        
        print(f"Executing test: {simple_test.full_name}")
        
        # Execute the test
        if simple_test.method_name:
            result = await test_executor.execute_single_test_method(
                simple_test.file_path, simple_test.method_name
            )
        else:
            file_result = await test_executor.execute_single_test_file(
                simple_test.file_path
            )
            result = file_result.test_results[0] if file_result.test_results else None
        
        # Verify result structure
        assert result is not None
        assert hasattr(result, 'execution_time')
        assert hasattr(result, 'status')
        assert result.status in [TestStatus.PASSED, TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT]
        
        print(f"Test result: {result.status}, time: {result.execution_time:.2f}s")
        
        # If test failed, print some debug info
        if result.status != TestStatus.PASSED:
            print(f"Test failed - stdout: {result.stdout[:200]}...")
            print(f"Test failed - stderr: {result.stderr[:200]}...")
            if result.error_message:
                print(f"Error message: {result.error_message}")
    
    @pytest.mark.integration
    def test_filter_methods_work(self, test_executor):
        """Test that filtering methods work with real test data."""
        # Create some mock results to test filtering
        from tests.perf.models import TestTimingResult
        
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
                test_identifier=TestIdentifier("test3.py", None, "failed_test"),
                execution_time=5.0,
                status=TestStatus.FAILED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier("test4.py", None, "timeout_test"),
                execution_time=60.0,
                status=TestStatus.TIMEOUT,
                timeout_occurred=True
            )
        ]
        
        # Test slow test filtering
        slow_tests = test_executor.get_slow_tests(test_results, threshold_seconds=10.0)
        assert len(slow_tests) == 2  # slow_test and timeout_test
        
        # Test timeout filtering
        timeout_tests = test_executor.get_timeout_tests(test_results)
        assert len(timeout_tests) == 1
        assert timeout_tests[0].status == TestStatus.TIMEOUT
        
        # Test failed test filtering
        failed_tests = test_executor.get_failed_tests(test_results)
        assert len(failed_tests) == 1
        assert failed_tests[0].status == TestStatus.FAILED
        
        print(f"Filtering tests: {len(slow_tests)} slow, {len(timeout_tests)} timeout, {len(failed_tests)} failed")