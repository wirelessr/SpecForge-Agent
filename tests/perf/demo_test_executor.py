#!/usr/bin/env python3
"""
Demonstration script for TestExecutor functionality.

This script shows how to use the TestExecutor to discover and analyze
integration tests for performance optimization.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.perf.test_executor import TestExecutor
from tests.perf.models import TestStatus


async def main():
    """Demonstrate TestExecutor functionality."""
    print("=== TestExecutor Demonstration ===\n")
    
    # Create TestExecutor instance
    executor = TestExecutor(timeout_seconds=30)
    
    # 1. Validate environment
    print("1. Validating test environment...")
    validation = await executor.validate_test_environment()
    
    if validation['valid']:
        print("✅ Environment is valid")
        env_info = validation['environment_info']
        print(f"   - Found {env_info['test_count']} integration tests")
        print(f"   - Python: {env_info['python_version']}")
        print(f"   - Pytest: {env_info['pytest_version']}")
        print(f"   - Timeout: {env_info['timeout_seconds']} seconds")
    else:
        print("❌ Environment validation failed:")
        for issue in validation['issues']:
            print(f"   - {issue}")
        return
    
    print()
    
    # 2. Discover integration tests
    print("2. Discovering integration tests...")
    tests = executor.discover_integration_tests()
    
    file_tests = [t for t in tests if t.method_name is None]
    method_tests = [t for t in tests if t.method_name is not None]
    
    print(f"✅ Discovered {len(tests)} total tests:")
    print(f"   - {len(file_tests)} file-level tests")
    print(f"   - {len(method_tests)} method-level tests")
    
    # Show some examples
    print("\n   Example file-level tests:")
    for test in file_tests[:3]:
        print(f"     - {test.file_path}")
    
    print("\n   Example method-level tests:")
    for test in method_tests[:3]:
        print(f"     - {test.full_name}")
    
    print()
    
    # 3. Build sample commands
    print("3. Building pytest commands...")
    sample_tests = tests[:2]
    
    for test in sample_tests:
        if test.method_name:
            command = executor.build_pytest_command(test.file_path, test.method_name)
        else:
            command = executor.build_pytest_command(test.file_path)
        
        print(f"   Test: {test.full_name}")
        print(f"   Command: {' '.join(command)}")
        print()
    
    # 4. Execute a simple test (optional - can be slow)
    print("4. Test execution capability...")
    
    # Find a simple test file to execute
    simple_test = None
    for test in file_tests:
        # Look for a test that might be quick
        if any(keyword in test.file_path.lower() for keyword in ['config', 'model', 'util']):
            simple_test = test
            break
    
    if not simple_test and file_tests:
        simple_test = file_tests[0]  # Fallback to first file test
    
    if simple_test:
        print(f"   Executing sample test: {simple_test.file_path}")
        print("   (This may take a moment...)")
        
        try:
            result = await executor.execute_single_test_file(simple_test.file_path)
            
            print(f"   ✅ Execution completed:")
            print(f"      - Status: {result.file_status}")
            print(f"      - Total time: {result.total_execution_time:.2f}s")
            print(f"      - Tests run: {result.test_count}")
            print(f"      - Passed: {result.passed_count}")
            print(f"      - Failed: {result.failed_count}")
            print(f"      - Timeouts: {result.timeout_count}")
            
            if result.test_results:
                slowest = max(result.test_results, key=lambda x: x.execution_time)
                print(f"      - Slowest test: {slowest.execution_time:.2f}s")
        
        except Exception as e:
            print(f"   ❌ Execution failed: {str(e)}")
    else:
        print("   No suitable test found for demonstration")
    
    print()
    
    # 5. Demonstrate filtering capabilities
    print("5. Demonstrating filtering capabilities...")
    
    # Create some sample results for filtering demo
    from tests.perf.models import TestTimingResult, TestIdentifier
    
    sample_results = [
        TestTimingResult(
            test_identifier=TestIdentifier("test_fast.py", None, "test_quick"),
            execution_time=1.5,
            status=TestStatus.PASSED
        ),
        TestTimingResult(
            test_identifier=TestIdentifier("test_slow.py", None, "test_heavy"),
            execution_time=25.0,
            status=TestStatus.PASSED
        ),
        TestTimingResult(
            test_identifier=TestIdentifier("test_broken.py", None, "test_failing"),
            execution_time=3.0,
            status=TestStatus.FAILED
        ),
        TestTimingResult(
            test_identifier=TestIdentifier("test_timeout.py", None, "test_hanging"),
            execution_time=60.0,
            status=TestStatus.TIMEOUT,
            timeout_occurred=True
        )
    ]
    
    slow_tests = executor.get_slow_tests(sample_results, threshold_seconds=10.0)
    timeout_tests = executor.get_timeout_tests(sample_results)
    failed_tests = executor.get_failed_tests(sample_results)
    
    print(f"   Sample results analysis:")
    print(f"   - Slow tests (>10s): {len(slow_tests)}")
    for test in slow_tests:
        print(f"     * {test.test_identifier.full_name}: {test.execution_time:.1f}s")
    
    print(f"   - Timeout tests: {len(timeout_tests)}")
    for test in timeout_tests:
        print(f"     * {test.test_identifier.full_name}")
    
    print(f"   - Failed tests: {len(failed_tests)}")
    for test in failed_tests:
        print(f"     * {test.test_identifier.full_name}: {test.status}")
    
    print("\n=== TestExecutor demonstration complete! ===")
    print("\nThe TestExecutor is ready for:")
    print("- Discovering all integration tests")
    print("- Executing tests individually with timeout")
    print("- Collecting detailed timing information")
    print("- Identifying slow, failed, and timeout tests")
    print("- Building proper pytest commands")
    print("\nNext step: Implement TestTimingAnalyzer to orchestrate full timing analysis")


if __name__ == "__main__":
    asyncio.run(main())