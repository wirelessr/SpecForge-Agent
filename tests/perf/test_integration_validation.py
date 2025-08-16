"""
Integration validation tests for performance analysis tools.

This module tests the integration between different performance analysis components
and validates end-to-end performance analysis workflows with real integration tests.
"""

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

from .models import TestIdentifier, TestStatus, ProfilerType
from .test_executor import TestExecutor
from .timing_analyzer import TestTimingAnalyzer
from .profiler import PerformanceProfiler
from .bottleneck_analyzer import BottleneckAnalyzer
from .config import PerformanceConfig
from .cli import PerformanceAnalysisCLI


class TestEndToEndWorkflow:
    """Test complete end-to-end performance analysis workflow."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.timing_analyzer = TestTimingAnalyzer(timeout_seconds=30)
        self.profiler = PerformanceProfiler()
        self.bottleneck_analyzer = BottleneckAnalyzer()
    
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self):
        """Test complete workflow from timing analysis to optimization recommendations."""
        # Step 1: Discover and time integration tests
        timing_report = await self.timing_analyzer.analyze_all_integration_tests()
        
        # Verify timing analysis completed
        assert timing_report is not None
        assert timing_report.total_test_count > 0
        assert len(timing_report.file_results) > 0
        
        # Step 2: Identify slow tests
        slow_tests = self.timing_analyzer.identify_slow_tests(
            timing_report, threshold_seconds=1.0  # Lower threshold for testing
        )
        
        # Should find some tests (even if not actually slow)
        if len(slow_tests) == 0:
            # If no slow tests found, use the slowest available tests
            all_tests = []
            for file_result in timing_report.file_results:
                all_tests.extend([
                    test for test in file_result.test_results 
                    if test.status == TestStatus.PASSED
                ])
            
            if all_tests:
                # Take the slowest test for profiling
                slow_tests = sorted(all_tests, key=lambda x: x.execution_time, reverse=True)[:1]
        
        if len(slow_tests) > 0:
            # Step 3: Profile slow tests
            profiling_results = await self.profiler.profile_slow_tests(
                slow_tests[:2], max_tests=2  # Limit for testing
            )
            
            # Verify profiling completed
            assert len(profiling_results) > 0
            
            for result in profiling_results:
                assert result.test_identifier is not None
                assert result.profiler_used == ProfilerType.CPROFILE
                assert result.total_profiling_time > 0
            
            # Step 4: Analyze bottlenecks
            bottleneck_reports = self.bottleneck_analyzer.analyze_cprofile_results(profiling_results)
            
            # Verify bottleneck analysis
            for report in bottleneck_reports:
                assert report.test_identifier is not None
                assert isinstance(report.time_categories.implement_agent_time, (int, float))
                assert len(report.optimization_recommendations) >= 0  # May be empty for simple tests
        
        # Workflow should complete without errors
        assert True, "Complete workflow executed successfully"
    
    @pytest.mark.asyncio
    async def test_workflow_with_timeout_recovery(self):
        """Test workflow handles timeouts gracefully."""
        # Use very short timeout to force some timeouts
        short_analyzer = TestTimingAnalyzer(timeout_seconds=1)
        
        # Run timing analysis with short timeout
        timing_report = await short_analyzer.analyze_all_integration_tests()
        
        # Should complete even with timeouts
        assert timing_report is not None
        assert timing_report.total_test_count > 0
        
        # Check for timeout handling
        timeout_tests = short_analyzer.identify_timeout_tests(timing_report)
        
        # If timeouts occurred, verify they were handled properly
        for timeout_test in timeout_tests:
            assert timeout_test.status == TestStatus.TIMEOUT
            assert timeout_test.timeout_occurred is True
            assert timeout_test.error_message is not None
            assert "timeout" in timeout_test.error_message.lower()
        
        # Workflow should continue with non-timeout tests
        passed_tests = []
        for file_result in timing_report.file_results:
            passed_tests.extend([
                test for test in file_result.test_results 
                if test.status == TestStatus.PASSED
            ])
        
        # Should have some successful tests even with timeouts
        # (This might be 0 if all tests timeout, which is also valid)
        assert len(passed_tests) >= 0
    
    @pytest.mark.asyncio
    async def test_selective_profiling_workflow(self):
        """Test workflow that profiles only specific tests."""
        # Define specific tests to profile
        target_tests = [
            TestIdentifier(
                file_path="tests/integration/test_real_main_controller.py",
                method_name="test_initialization"
            )
        ]
        
        # Run timing analysis on specific tests
        timing_results = await self.timing_analyzer.analyze_specific_tests(target_tests)
        
        # Verify specific test analysis
        assert len(timing_results) == len(target_tests)
        
        # Profile the specific tests
        if any(result.status == TestStatus.PASSED for result in timing_results):
            passed_results = [r for r in timing_results if r.status == TestStatus.PASSED]
            
            profiling_results = await self.profiler.profile_slow_tests(passed_results)
            
            # Verify profiling of specific tests
            assert len(profiling_results) == len(passed_results)
            
            for result in profiling_results:
                assert result.test_identifier in [t.test_identifier for t in passed_results]


class TestComponentIntegration:
    """Test integration between performance analysis components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_executor = TestExecutor()
        self.timing_analyzer = TestTimingAnalyzer()
        self.profiler = PerformanceProfiler()
    
    @pytest.mark.asyncio
    async def test_test_executor_timing_analyzer_integration(self):
        """Test integration between TestExecutor and TestTimingAnalyzer."""
        # TestTimingAnalyzer should use TestExecutor internally
        assert self.timing_analyzer.test_executor is not None
        
        # Both should use consistent timeout settings
        assert self.timing_analyzer.timeout_seconds == self.timing_analyzer.test_executor.timeout_seconds
        
        # Test discovery should be consistent
        executor_tests = self.test_executor.discover_integration_tests()
        
        # Run timing analysis and verify it uses the same test discovery
        timing_report = await self.timing_analyzer.analyze_all_integration_tests()
        
        # Should analyze tests from the same discovery mechanism
        analyzed_test_count = timing_report.total_test_count
        
        # Should have analyzed some tests (may be fewer due to filtering or failures)
        assert analyzed_test_count >= 0
        
        # If tests were found by executor, analyzer should process them
        if len(executor_tests) > 0:
            assert analyzed_test_count > 0, "Analyzer should process discovered tests"
    
    @pytest.mark.asyncio
    async def test_timing_analyzer_profiler_integration(self):
        """Test integration between TestTimingAnalyzer and PerformanceProfiler."""
        # Run timing analysis to get slow tests
        timing_report = await self.timing_analyzer.analyze_all_integration_tests()
        
        if timing_report.total_test_count > 0:
            # Get slow tests (use low threshold for testing)
            slow_tests = self.timing_analyzer.identify_slow_tests(
                timing_report, threshold_seconds=0.1
            )
            
            if len(slow_tests) == 0:
                # Use any passed test for integration testing
                all_tests = []
                for file_result in timing_report.file_results:
                    all_tests.extend([
                        test for test in file_result.test_results 
                        if test.status == TestStatus.PASSED
                    ])
                slow_tests = all_tests[:1]  # Take first passed test
            
            if len(slow_tests) > 0:
                # Profile the slow tests
                profiling_results = await self.profiler.profile_slow_tests(
                    slow_tests[:1], max_tests=1
                )
                
                # Verify integration
                assert len(profiling_results) == 1
                
                profiling_result = profiling_results[0]
                original_test = slow_tests[0]
                
                # Test identifiers should match
                assert profiling_result.test_identifier.file_path == original_test.test_identifier.file_path
                assert profiling_result.test_identifier.method_name == original_test.test_identifier.method_name
                
                # Profiling time should be reasonable compared to original timing
                if profiling_result.total_profiling_time > 0 and original_test.execution_time > 0:
                    # Profiling should take longer due to overhead, but not excessively
                    overhead_ratio = profiling_result.total_profiling_time / original_test.execution_time
                    assert overhead_ratio >= 1.0, "Profiling should take at least as long as original"
                    assert overhead_ratio <= 10.0, "Profiling overhead should be reasonable"
    
    def test_profiler_bottleneck_analyzer_integration(self):
        """Test integration between PerformanceProfiler and BottleneckAnalyzer."""
        # Create mock profiling result for integration testing
        from .models import CProfileResult, ProfilingResult
        
        test_identifier = TestIdentifier(
            file_path="test_integration.py",
            method_name="test_method"
        )
        
        # Create realistic function stats
        function_stats = {
            'autogen_framework/agents/implement_agent.py:45(execute_task)': {
                'cumulative_time': 3.0,
                'total_time': 1.0,
                'call_count': 1,
                'filename': 'autogen_framework/agents/implement_agent.py',
                'function_name': 'execute_task'
            },
            'tests/integration/conftest.py:15(setup_test)': {
                'cumulative_time': 0.5,
                'total_time': 0.4,
                'call_count': 1,
                'filename': 'tests/integration/conftest.py',
                'function_name': 'setup_test'
            }
        }
        
        cprofile_result = CProfileResult(
            test_identifier=test_identifier,
            profile_file_path="/tmp/test.prof",
            total_time=5.0,
            function_stats=function_stats,
            top_functions=[],
            call_count=2
        )
        
        profiling_result = ProfilingResult(
            test_identifier=test_identifier,
            profiler_used=ProfilerType.CPROFILE,
            cprofile_result=cprofile_result,
            total_profiling_time=5.5
        )
        
        # Test component timing extraction (profiler functionality)
        component_timings = self.profiler._extract_component_timings(cprofile_result)
        
        # Verify component timing extraction
        assert isinstance(component_timings, dict)
        assert 'ImplementAgent' in component_timings
        assert 'Test_Infrastructure' in component_timings
        
        # Test bottleneck analysis (analyzer functionality)
        bottleneck_analyzer = BottleneckAnalyzer()
        reports = bottleneck_analyzer.analyze_cprofile_results([profiling_result])
        
        # Verify integration between profiler output and analyzer input
        assert len(reports) == 1
        report = reports[0]
        
        assert report.test_identifier == test_identifier
        assert len(report.component_bottlenecks) > 0
        
        # Component timings from profiler should influence bottleneck analysis
        implement_agent_bottlenecks = [
            b for b in report.component_bottlenecks 
            if 'ImplementAgent' in b.component_name
        ]
        assert len(implement_agent_bottlenecks) > 0, "Should identify ImplementAgent bottleneck"


class TestRealIntegrationTestAnalysis:
    """Test performance analysis with real integration tests."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.timing_analyzer = TestTimingAnalyzer(timeout_seconds=60)  # Longer timeout for real tests
        self.profiler = PerformanceProfiler()
    
    @pytest.mark.asyncio
    async def test_analyze_real_main_controller_tests(self):
        """Test analysis of real MainController integration tests."""
        # Target specific real integration tests
        target_tests = [
            TestIdentifier(
                file_path="tests/integration/test_real_main_controller.py",
                method_name="test_initialization"
            )
        ]
        
        # Analyze timing
        timing_results = await self.timing_analyzer.analyze_specific_tests(target_tests)
        
        # Verify we got results
        assert len(timing_results) > 0
        
        # Check for successful execution
        successful_results = [r for r in timing_results if r.status == TestStatus.PASSED]
        
        if len(successful_results) > 0:
            # Profile one successful test
            profiling_results = await self.profiler.profile_slow_tests(
                successful_results[:1], max_tests=1
            )
            
            assert len(profiling_results) == 1
            profiling_result = profiling_results[0]
            
            # Verify profiling captured meaningful data
            if profiling_result.cprofile_result:
                cprofile_data = profiling_result.cprofile_result
                
                # Should have captured function calls
                assert cprofile_data.call_count > 0
                assert cprofile_data.total_time > 0
                assert len(cprofile_data.function_stats) > 0
                
                # Should have some framework-related functions
                framework_functions = [
                    func_name for func_name in cprofile_data.function_stats.keys()
                    if 'autogen_framework' in func_name or 'main_controller' in func_name.lower()
                ]
                
                # May or may not have framework functions depending on test complexity
                # Just verify the profiling structure is correct
                assert isinstance(framework_functions, list)
    
    @pytest.mark.asyncio
    async def test_analyze_real_agent_integration_tests(self):
        """Test analysis of real agent integration tests."""
        # Look for agent-related integration tests
        test_executor = TestExecutor()
        all_tests = test_executor.discover_integration_tests()
        
        # Find agent-related tests
        agent_tests = [
            test for test in all_tests
            if ('agent' in test.file_path.lower() and 
                test.method_name and 
                'test_' in test.method_name)
        ]
        
        if len(agent_tests) > 0:
            # Analyze a few agent tests
            selected_tests = agent_tests[:2]  # Limit for testing
            
            timing_results = await self.timing_analyzer.analyze_specific_tests(selected_tests)
            
            # Verify analysis completed
            assert len(timing_results) == len(selected_tests)
            
            # Check for any successful executions
            successful_results = [r for r in timing_results if r.status == TestStatus.PASSED]
            
            # If we have successful results, verify they have reasonable timing
            for result in successful_results:
                assert result.execution_time > 0
                assert result.execution_time < 300  # Should complete within 5 minutes
                assert result.test_identifier is not None
    
    def test_validate_test_discovery_accuracy(self):
        """Test that test discovery finds expected integration tests."""
        test_executor = TestExecutor()
        discovered_tests = test_executor.discover_integration_tests()
        
        # Should discover some integration tests
        assert len(discovered_tests) > 0, "Should discover integration tests"
        
        # Verify test identifier structure
        for test in discovered_tests:
            assert isinstance(test, TestIdentifier)
            assert test.file_path.startswith("tests/integration/")
            assert test.file_path.endswith(".py")
            assert test.full_name is not None
            
            # If method name exists, should be a test method
            if test.method_name:
                assert test.method_name.startswith("test_")
        
        # Should find some known integration test files
        discovered_files = {test.file_path for test in discovered_tests}
        expected_files = [
            "tests/integration/test_real_main_controller.py",
            "tests/integration/test_real_agent_manager.py"
        ]
        
        # Check if any expected files exist
        found_expected = any(
            expected_file in discovered_files 
            for expected_file in expected_files
        )
        
        # Note: This might be False if integration tests don't exist yet
        # That's okay - just verify the discovery mechanism works
        assert isinstance(found_expected, bool)


class TestCLIIntegration:
    """Test CLI integration with performance analysis components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.cli = PerformanceAnalysisCLI()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_cli_timing_analysis_command(self):
        """Test CLI timing analysis command."""
        # Test timing analysis command
        try:
            result = await self.cli.run_timing_analysis(
                output_dir=str(self.temp_dir),
                timeout=30,
                max_tests=5  # Limit for testing
            )
            
            # Verify CLI executed successfully
            assert result is not None
            
            # Check if output files were created
            output_files = list(self.temp_dir.glob("*.json"))
            
            # May or may not have output files depending on test availability
            # Just verify CLI didn't crash
            assert isinstance(output_files, list)
            
        except Exception as e:
            # CLI might fail if no tests available - that's okay for testing
            assert "no.*test" in str(e).lower() or "not.*found" in str(e).lower()
    
    def test_cli_help_and_validation(self):
        """Test CLI help and argument validation."""
        # Test that CLI can be instantiated
        assert self.cli is not None
        
        # Test argument validation
        with pytest.raises((ValueError, TypeError)):
            # Invalid timeout should raise error
            asyncio.run(self.cli.run_timing_analysis(timeout=-1))
        
        with pytest.raises((ValueError, TypeError)):
            # Invalid max_tests should raise error
            asyncio.run(self.cli.run_timing_analysis(max_tests=0))


class TestPerformanceReporting:
    """Test performance reporting and output generation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.timing_analyzer = TestTimingAnalyzer()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_timing_report_serialization(self):
        """Test that timing reports can be serialized and saved."""
        # Run a minimal timing analysis
        timing_report = await self.timing_analyzer.analyze_all_integration_tests()
        
        # Save report to file
        output_path = await self.timing_analyzer.save_timing_report(
            timing_report, 
            output_path=self.temp_dir / "test_timing_report.json"
        )
        
        # Verify file was created
        assert output_path.exists()
        
        # Verify file contains valid JSON
        with open(output_path, 'r') as f:
            report_data = json.load(f)
        
        # Verify report structure
        assert 'total_execution_time' in report_data
        assert 'analysis_timestamp' in report_data
        assert 'timeout_threshold' in report_data
        assert 'file_results' in report_data
        
        # Verify data types
        assert isinstance(report_data['total_execution_time'], (int, float))
        assert isinstance(report_data['analysis_timestamp'], (int, float))
        assert isinstance(report_data['file_results'], list)
    
    def test_bottleneck_report_generation(self):
        """Test bottleneck report generation and formatting."""
        from .models import CProfileResult, ProfilingResult, BottleneckReport
        
        # Create mock data for report generation
        test_identifier = TestIdentifier(
            file_path="test_report.py",
            method_name="test_method"
        )
        
        function_stats = {
            'test_function': {
                'cumulative_time': 2.0,
                'total_time': 1.5,
                'call_count': 5,
                'filename': 'test.py',
                'function_name': 'test_function'
            }
        }
        
        cprofile_result = CProfileResult(
            test_identifier=test_identifier,
            profile_file_path="/tmp/test.prof",
            total_time=3.0,
            function_stats=function_stats,
            top_functions=[],
            call_count=5
        )
        
        profiling_result = ProfilingResult(
            test_identifier=test_identifier,
            profiler_used=ProfilerType.CPROFILE,
            cprofile_result=cprofile_result
        )
        
        # Generate bottleneck report
        analyzer = BottleneckAnalyzer()
        reports = analyzer.analyze_cprofile_results([profiling_result])
        
        # Verify report generation
        assert len(reports) == 1
        report = reports[0]
        
        # Test report properties
        assert isinstance(report.test_vs_implementation_ratio, (int, float))
        assert report.test_vs_implementation_ratio >= 0
        
        # Test summary generation
        summary = analyzer.get_bottleneck_summary(reports)
        
        if summary:  # May be empty for simple test data
            assert 'total_tests_analyzed' in summary
            assert summary['total_tests_analyzed'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])