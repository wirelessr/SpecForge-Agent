"""
Comprehensive validation tests for performance analysis tools.

This module tests and validates the accuracy and reliability of the performance
analysis infrastructure including timing accuracy, profiling data collection,
bottleneck identification, timeout handling, and optimization recommendations.
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from .models import (
    TestIdentifier, TestTimingResult, TestStatus, CProfileResult,
    ProfilingResult, ProfilerType, ComponentBottleneck, OptimizationRecommendation,
    TimeCategories, ComponentTimings, BottleneckReport
)
from .test_executor import TestExecutor
from .timing_analyzer import TestTimingAnalyzer
from .profiler import PerformanceProfiler
from .bottleneck_analyzer import BottleneckAnalyzer
from .config import PerformanceConfig


class TestTimingAccuracy:
    """Test timing accuracy and measurement precision."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_executor = TestExecutor(timeout_seconds=30)
        self.timing_analyzer = TestTimingAnalyzer(timeout_seconds=30)
    
    @pytest.mark.asyncio
    async def test_timing_measurement_accuracy(self):
        """Test that timing measurements are accurate within acceptable tolerance."""
        # Create a simple test that should take a predictable amount of time
        test_identifier = TestIdentifier(
            file_path="tests/integration/test_real_main_controller.py",
            class_name="TestRealMainController",
            method_name="test_initialization"
        )
        
        # Measure execution time manually
        start_time = time.time()
        
        # Execute the test
        result = await self.test_executor.execute_single_test_method(
            test_identifier.file_path, test_identifier.method_name
        )
        
        end_time = time.time()
        manual_timing = end_time - start_time
        
        # Verify timing accuracy (should be within 10% tolerance)
        timing_difference = abs(result.execution_time - manual_timing)
        tolerance = manual_timing * 0.1  # 10% tolerance
        
        assert timing_difference <= tolerance, (
            f"Timing measurement inaccurate: measured {result.execution_time:.3f}s, "
            f"expected ~{manual_timing:.3f}s, difference {timing_difference:.3f}s"
        )
        
        # Verify timing is reasonable (not zero or negative)
        assert result.execution_time > 0, "Execution time should be positive"
        assert result.execution_time < 60, "Test should complete within timeout"
    
    @pytest.mark.asyncio
    async def test_timing_consistency_across_runs(self):
        """Test that timing measurements are consistent across multiple runs."""
        test_identifier = TestIdentifier(
            file_path="tests/integration/test_real_main_controller.py",
            method_name="test_initialization"
        )
        
        # Run the same test multiple times
        execution_times = []
        num_runs = 3
        
        for i in range(num_runs):
            result = await self.test_executor.execute_single_test_method(
                test_identifier.file_path, test_identifier.method_name
            )
            
            if result.status == TestStatus.PASSED:
                execution_times.append(result.execution_time)
        
        # Should have at least 2 successful runs to compare
        assert len(execution_times) >= 2, "Need at least 2 successful runs for consistency check"
        
        # Calculate coefficient of variation (std dev / mean)
        import statistics
        mean_time = statistics.mean(execution_times)
        std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 0
        
        # Timing should be reasonably consistent (CV < 50%)
        assert coefficient_of_variation < 0.5, (
            f"Timing inconsistent across runs: CV={coefficient_of_variation:.2f}, "
            f"times={execution_times}"
        )
    
    @pytest.mark.asyncio
    async def test_timeout_accuracy(self):
        """Test that timeout handling is accurate and reliable."""
        # Create a test that should timeout
        timeout_seconds = 2  # Very short timeout
        test_executor = TestExecutor(timeout_seconds=timeout_seconds)
        
        # Use a test that might take longer than timeout
        test_identifier = TestIdentifier(
            file_path="tests/integration/test_real_main_controller.py",
            method_name="test_complete_workflow_execution"
        )
        
        start_time = time.time()
        result = await test_executor.execute_single_test_method(
            test_identifier.file_path, test_identifier.method_name
        )
        end_time = time.time()
        
        actual_duration = end_time - start_time
        
        # If test timed out, verify timeout was enforced accurately
        if result.status == TestStatus.TIMEOUT:
            # Should timeout within reasonable margin of the specified timeout
            timeout_tolerance = 2.0  # 2 second tolerance for process cleanup
            assert actual_duration <= (timeout_seconds + timeout_tolerance), (
                f"Timeout not enforced: took {actual_duration:.2f}s, "
                f"expected ~{timeout_seconds}s"
            )
            
            assert result.timeout_occurred is True
            assert "timed out" in result.error_message.lower()
        else:
            # If test completed, it should be within timeout
            assert actual_duration < timeout_seconds, (
                f"Test completed but took longer than timeout: {actual_duration:.2f}s"
            )


class TestProfilingDataCollection:
    """Test profiling data collection accuracy and completeness."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.profiler = PerformanceProfiler()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_cprofile_data_collection(self):
        """Test that cProfile data collection works correctly."""
        test_identifier = TestIdentifier(
            file_path="tests/integration/test_real_main_controller.py",
            method_name="test_initialization"
        )
        
        # Profile the test
        profiling_result = await self.profiler.profile_with_cprofile(test_identifier)
        
        # Verify profiling result structure
        assert profiling_result is not None
        assert profiling_result.test_identifier == test_identifier
        assert profiling_result.profiler_used == ProfilerType.CPROFILE
        assert profiling_result.total_profiling_time > 0
        
        # Verify cProfile data
        cprofile_result = profiling_result.cprofile_result
        if cprofile_result:  # May be None if test failed
            assert cprofile_result.total_time > 0
            assert cprofile_result.call_count > 0
            assert len(cprofile_result.function_stats) > 0
            assert len(cprofile_result.top_functions) > 0
            
            # Verify function statistics have required fields
            for func_name, func_data in cprofile_result.function_stats.items():
                assert 'cumulative_time' in func_data
                assert 'total_time' in func_data
                assert 'call_count' in func_data
                assert 'filename' in func_data
                assert 'function_name' in func_data
                
                # Verify data types and ranges
                assert isinstance(func_data['cumulative_time'], (int, float))
                assert isinstance(func_data['total_time'], (int, float))
                assert isinstance(func_data['call_count'], int)
                assert func_data['cumulative_time'] >= 0
                assert func_data['total_time'] >= 0
                assert func_data['call_count'] >= 0
    
    @pytest.mark.asyncio
    async def test_component_timing_extraction(self):
        """Test extraction of component timings from profiling data."""
        # Create mock cProfile data with known component functions
        test_identifier = TestIdentifier(
            file_path="test_file.py",
            method_name="test_method"
        )
        
        mock_function_stats = {
            'autogen_framework/agents/implement_agent.py:45(execute_task)': {
                'cumulative_time': 5.0,
                'total_time': 1.0,
                'call_count': 1,
                'filename': 'autogen_framework/agents/implement_agent.py',
                'function_name': 'execute_task'
            },
            'autogen_framework/agents/task_decomposer.py:30(decompose_task)': {
                'cumulative_time': 3.0,
                'total_time': 2.0,
                'call_count': 5,
                'filename': 'autogen_framework/agents/task_decomposer.py',
                'function_name': 'decompose_task'
            },
            'autogen_framework/shell_executor.py:20(execute_command)': {
                'cumulative_time': 2.0,
                'total_time': 1.5,
                'call_count': 10,
                'filename': 'autogen_framework/shell_executor.py',
                'function_name': 'execute_command'
            }
        }
        
        cprofile_result = CProfileResult(
            test_identifier=test_identifier,
            profile_file_path="/tmp/test.prof",
            total_time=15.0,
            function_stats=mock_function_stats,
            top_functions=[],
            call_count=16
        )
        
        # Extract component timings
        component_timings = self.profiler._extract_component_timings(cprofile_result)
        
        # Verify component timing extraction
        assert isinstance(component_timings, dict)
        assert 'TaskDecomposer' in component_timings
        assert 'ShellExecutor' in component_timings
        assert 'ImplementAgent' in component_timings
        
        # Verify that known components have non-zero times
        assert component_timings['TaskDecomposer'] > 0
        assert component_timings['ShellExecutor'] > 0
        
        # Verify total timing makes sense
        total_component_time = sum(component_timings.values())
        assert total_component_time <= cprofile_result.total_time * 1.1  # Allow some overhead
    
    def test_profiling_overhead_calculation(self):
        """Test calculation of profiling overhead."""
        # Create profiling result with known timing
        test_identifier = TestIdentifier(file_path="test.py", method_name="test")
        
        profiling_result = ProfilingResult(
            test_identifier=test_identifier,
            profiler_used=ProfilerType.CPROFILE,
            total_profiling_time=10.0,
            profiling_overhead=2.0
        )
        
        # Verify overhead calculation
        assert profiling_result.profiling_overhead >= 0
        assert profiling_result.profiling_overhead <= profiling_result.total_profiling_time
        
        # Overhead should be reasonable (< 50% of total time for most cases)
        overhead_percentage = (profiling_result.profiling_overhead / 
                             profiling_result.total_profiling_time * 100)
        assert overhead_percentage < 100, "Overhead cannot exceed total profiling time"


class TestBottleneckIdentification:
    """Test bottleneck identification with known slow integration tests."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = BottleneckAnalyzer()
        self.timing_analyzer = TestTimingAnalyzer()
    
    def create_slow_test_scenario(self) -> List[ProfilingResult]:
        """Create a scenario with known bottlenecks for testing."""
        # Create test identifiers for known slow tests
        slow_tests = [
            TestIdentifier(
                file_path="tests/integration/test_real_main_controller.py",
                method_name="test_complete_workflow_execution"
            ),
            TestIdentifier(
                file_path="tests/integration/test_implement_agent_tasks_execution.py",
                method_name="test_complex_task_execution"
            )
        ]
        
        profiling_results = []
        
        for test_id in slow_tests:
            # Create mock profiling data with realistic bottlenecks
            function_stats = {
                'autogen_framework/agents/implement_agent.py:45(execute_task)': {
                    'cumulative_time': 8.0,  # High time - should be identified as bottleneck
                    'total_time': 2.0,
                    'call_count': 1,
                    'filename': 'autogen_framework/agents/implement_agent.py',
                    'function_name': 'execute_task'
                },
                'autogen_framework/agents/task_decomposer.py:30(decompose_task)': {
                    'cumulative_time': 5.0,  # Moderate time
                    'total_time': 3.0,
                    'call_count': 10,  # High call count - should be flagged
                    'filename': 'autogen_framework/agents/task_decomposer.py',
                    'function_name': 'decompose_task'
                },
                'openai/api.py:100(chat_completion)': {
                    'cumulative_time': 6.0,  # LLM API calls - should be identified
                    'total_time': 5.5,
                    'call_count': 3,
                    'filename': 'openai/api.py',
                    'function_name': 'chat_completion'
                },
                'tests/integration/conftest.py:15(setup_test)': {
                    'cumulative_time': 1.0,  # Test overhead
                    'total_time': 0.8,
                    'call_count': 1,
                    'filename': 'tests/integration/conftest.py',
                    'function_name': 'setup_test'
                }
            }
            
            cprofile_result = CProfileResult(
                test_identifier=test_id,
                profile_file_path="/tmp/test.prof",
                total_time=20.0,
                function_stats=function_stats,
                top_functions=[
                    {
                        'name': 'autogen_framework/agents/implement_agent.py:45(execute_task)',
                        'cumulative_time': 8.0,
                        'total_time': 2.0,
                        'call_count': 1,
                        'percentage': 40.0
                    }
                ],
                call_count=15
            )
            
            profiling_result = ProfilingResult(
                test_identifier=test_id,
                profiler_used=ProfilerType.CPROFILE,
                cprofile_result=cprofile_result,
                total_profiling_time=20.5
            )
            
            profiling_results.append(profiling_result)
        
        return profiling_results
    
    def test_bottleneck_identification_accuracy(self):
        """Test that bottlenecks are correctly identified from profiling data."""
        profiling_results = self.create_slow_test_scenario()
        
        # Analyze bottlenecks
        bottleneck_reports = self.analyzer.analyze_cprofile_results(profiling_results)
        
        assert len(bottleneck_reports) > 0, "Should generate bottleneck reports"
        
        for report in bottleneck_reports:
            # Verify report structure
            assert isinstance(report, BottleneckReport)
            assert len(report.component_bottlenecks) > 0
            
            # Check that significant bottlenecks are identified
            significant_bottlenecks = [b for b in report.component_bottlenecks if b.is_significant]
            assert len(significant_bottlenecks) > 0, "Should identify significant bottlenecks"
            
            # Verify bottlenecks are sorted by time spent
            bottleneck_times = [b.time_spent for b in report.component_bottlenecks]
            assert bottleneck_times == sorted(bottleneck_times, reverse=True), (
                "Bottlenecks should be sorted by time spent"
            )
            
            # Check for expected bottleneck components
            bottleneck_components = {b.component_name for b in report.component_bottlenecks}
            expected_components = {'ImplementAgent', 'TaskDecomposer', 'LLM_API_Calls'}
            
            # Should identify at least some expected components
            found_expected = bottleneck_components.intersection(expected_components)
            assert len(found_expected) > 0, (
                f"Should identify expected components. Found: {bottleneck_components}, "
                f"Expected: {expected_components}"
            )
    
    def test_component_vs_test_overhead_categorization(self):
        """Test accurate categorization of component time vs test overhead."""
        profiling_results = self.create_slow_test_scenario()
        
        for profiling_result in profiling_results:
            time_categories = self.analyzer.categorize_time_spent(profiling_result.cprofile_result)
            
            # Verify time categorization
            assert isinstance(time_categories, TimeCategories)
            assert isinstance(time_categories.component_timings, ComponentTimings)
            
            # Should identify both component time and test overhead
            assert time_categories.implement_agent_time > 0, "Should identify ImplementAgent time"
            assert time_categories.component_timings.test_overhead > 0, "Should identify test overhead"
            
            # Component time should be significant portion for these tests
            total_time = time_categories.total_time
            if total_time > 0:
                component_percentage = (time_categories.implement_agent_time / total_time) * 100
                assert component_percentage > 20, (
                    f"ImplementAgent should be significant portion of time: {component_percentage:.1f}%"
                )
    
    def test_optimization_recommendation_generation(self):
        """Test that accurate optimization recommendations are generated."""
        profiling_results = self.create_slow_test_scenario()
        bottleneck_reports = self.analyzer.analyze_cprofile_results(profiling_results)
        
        for report in bottleneck_reports:
            recommendations = report.optimization_recommendations
            
            # Should generate some recommendations
            assert len(recommendations) > 0, "Should generate optimization recommendations"
            
            # Verify recommendation structure
            for rec in recommendations:
                assert isinstance(rec, OptimizationRecommendation)
                assert rec.component
                assert rec.issue_description
                assert rec.recommendation
                assert rec.expected_impact in ['high', 'medium', 'low']
                assert rec.implementation_effort in ['high', 'medium', 'low']
                assert rec.priority_score > 0
            
            # Recommendations should be sorted by priority
            priority_scores = [rec.priority_score for rec in recommendations]
            assert priority_scores == sorted(priority_scores, reverse=True), (
                "Recommendations should be sorted by priority score"
            )
            
            # Should have high-priority recommendations for significant bottlenecks
            high_priority_recs = [rec for rec in recommendations if rec.expected_impact == 'high']
            significant_bottlenecks = [b for b in report.component_bottlenecks if b.is_significant]
            
            if len(significant_bottlenecks) > 0:
                assert len(high_priority_recs) > 0, (
                    "Should generate high-priority recommendations for significant bottlenecks"
                )


class TestTimeoutHandling:
    """Test timeout handling and error recovery in performance tools."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.short_timeout = 2  # Very short timeout for testing
        self.test_executor = TestExecutor(timeout_seconds=self.short_timeout)
        self.timing_analyzer = TestTimingAnalyzer(timeout_seconds=self.short_timeout)
    
    @pytest.mark.asyncio
    async def test_timeout_detection_and_handling(self):
        """Test that timeouts are properly detected and handled."""
        # Use a test that might take longer than our short timeout
        test_identifier = TestIdentifier(
            file_path="tests/integration/test_real_main_controller.py",
            method_name="test_complete_workflow_execution"
        )
        
        start_time = time.time()
        result = await self.test_executor.execute_single_test_method(
            test_identifier.file_path, test_identifier.method_name
        )
        end_time = time.time()
        
        actual_duration = end_time - start_time
        
        # If test timed out, verify proper handling
        if result.status == TestStatus.TIMEOUT:
            # Verify timeout was enforced
            assert actual_duration <= (self.short_timeout + 3), (
                f"Timeout not enforced properly: {actual_duration:.2f}s > {self.short_timeout + 3}s"
            )
            
            # Verify timeout indicators
            assert result.timeout_occurred is True
            assert result.error_message is not None
            assert "timeout" in result.error_message.lower()
            
            # Should still have some output captured
            assert isinstance(result.stdout, str)
            assert isinstance(result.stderr, str)
    
    @pytest.mark.asyncio
    async def test_timeout_recovery_in_batch_analysis(self):
        """Test that timeout in one test doesn't break batch analysis."""
        # Create a mix of tests - some that might timeout, some that should pass quickly
        test_identifiers = [
            TestIdentifier(
                file_path="tests/integration/test_real_main_controller.py",
                method_name="test_initialization"  # Should be quick
            ),
            TestIdentifier(
                file_path="tests/integration/test_real_main_controller.py", 
                method_name="test_complete_workflow_execution"  # Might timeout
            )
        ]
        
        # Analyze all tests
        results = await self.timing_analyzer.analyze_specific_tests(test_identifiers)
        
        # Should get results for all tests, even if some timed out
        assert len(results) == len(test_identifiers)
        
        # Verify that timeout in one test doesn't affect others
        passed_tests = [r for r in results if r.status == TestStatus.PASSED]
        timeout_tests = [r for r in results if r.status == TestStatus.TIMEOUT]
        
        # Should have at least one result of each type (or all passed if system is fast)
        assert len(passed_tests) + len(timeout_tests) == len(results)
        
        # Each result should have proper error handling
        for result in results:
            assert result.test_identifier is not None
            assert result.execution_time >= 0
            assert isinstance(result.status, TestStatus)
            
            if result.status == TestStatus.TIMEOUT:
                assert result.timeout_occurred is True
                assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_profiler_timeout_handling(self):
        """Test timeout handling in profiler components."""
        profiler = PerformanceProfiler()
        
        # Create test that might timeout during profiling
        test_identifier = TestIdentifier(
            file_path="tests/integration/test_real_main_controller.py",
            method_name="test_complete_workflow_execution"
        )
        
        # Profile with short timeout (profiler uses 2x timeout internally)
        try:
            profiling_result = await profiler.profile_with_cprofile(test_identifier)
            
            # If profiling completed, verify result structure
            assert profiling_result is not None
            assert profiling_result.test_identifier == test_identifier
            assert profiling_result.total_profiling_time >= 0
            
            # Even if test timed out, should have some profiling data structure
            assert profiling_result.profiler_used == ProfilerType.CPROFILE
            
        except Exception as e:
            # If profiling failed due to timeout, should get meaningful error
            assert "timeout" in str(e).lower() or "failed" in str(e).lower()
    
    def test_error_recovery_in_analysis_pipeline(self):
        """Test error recovery when analysis components fail."""
        analyzer = BottleneckAnalyzer()
        
        # Create invalid profiling result to test error handling
        invalid_result = ProfilingResult(
            test_identifier=TestIdentifier(file_path="invalid.py"),
            profiler_used=ProfilerType.CPROFILE,
            cprofile_result=None  # Invalid - should be handled gracefully
        )
        
        # Analyze should handle invalid data gracefully
        reports = analyzer.analyze_cprofile_results([invalid_result])
        
        # Should return empty results rather than crashing
        assert isinstance(reports, list)
        assert len(reports) == 0  # No valid data to analyze
        
        # Test with mixed valid and invalid data
        valid_function_stats = {
            'test_function': {
                'cumulative_time': 1.0,
                'total_time': 1.0,
                'call_count': 1,
                'filename': 'test.py',
                'function_name': 'test_function'
            }
        }
        
        valid_cprofile = CProfileResult(
            test_identifier=TestIdentifier(file_path="valid.py"),
            profile_file_path="/tmp/valid.prof",
            total_time=1.0,
            function_stats=valid_function_stats,
            top_functions=[],
            call_count=1
        )
        
        valid_result = ProfilingResult(
            test_identifier=TestIdentifier(file_path="valid.py"),
            profiler_used=ProfilerType.CPROFILE,
            cprofile_result=valid_cprofile
        )
        
        # Mix valid and invalid results
        mixed_results = [invalid_result, valid_result]
        reports = analyzer.analyze_cprofile_results(mixed_results)
        
        # Should process valid results and skip invalid ones
        assert len(reports) == 1  # Only the valid result
        assert reports[0].test_identifier.file_path == "valid.py"


class TestOptimizationRecommendationAccuracy:
    """Test accuracy and relevance of optimization recommendations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = BottleneckAnalyzer()
    
    def create_high_api_call_scenario(self) -> CProfileResult:
        """Create scenario with high LLM API call frequency."""
        function_stats = {}
        
        # Create many LLM API calls
        for i in range(30):  # High number of API calls
            function_stats[f'openai/api.py:{100+i}(chat_completion_{i})'] = {
                'cumulative_time': 2.0,
                'total_time': 1.8,
                'call_count': 1,
                'filename': 'openai/api.py',
                'function_name': f'chat_completion_{i}'
            }
        
        return CProfileResult(
            test_identifier=TestIdentifier(file_path="high_api_test.py"),
            profile_file_path="/tmp/test.prof",
            total_time=60.0,
            function_stats=function_stats,
            top_functions=[],
            call_count=30
        )
    
    def create_high_call_count_scenario(self) -> CProfileResult:
        """Create scenario with high call count in specific component."""
        function_stats = {
            'autogen_framework/shell_executor.py:20(execute_command)': {
                'cumulative_time': 15.0,
                'total_time': 12.0,
                'call_count': 100,  # Very high call count
                'filename': 'autogen_framework/shell_executor.py',
                'function_name': 'execute_command'
            },
            'subprocess.py:500(run)': {
                'cumulative_time': 10.0,
                'total_time': 8.0,
                'call_count': 100,
                'filename': 'subprocess.py',
                'function_name': 'run'
            }
        }
        
        return CProfileResult(
            test_identifier=TestIdentifier(file_path="high_calls_test.py"),
            profile_file_path="/tmp/test.prof",
            total_time=25.0,
            function_stats=function_stats,
            top_functions=[],
            call_count=200
        )
    
    def create_context_loading_bottleneck_scenario(self) -> CProfileResult:
        """Create scenario with context loading bottleneck."""
        function_stats = {
            'autogen_framework/context_manager.py:100(load_context)': {
                'cumulative_time': 8.0,
                'total_time': 7.0,
                'call_count': 5,
                'filename': 'autogen_framework/context_manager.py',
                'function_name': 'load_context'
            },
            'autogen_framework/context_manager.py:150(get_implementation_context)': {
                'cumulative_time': 5.0,
                'total_time': 4.0,
                'call_count': 3,
                'filename': 'autogen_framework/context_manager.py',
                'function_name': 'get_implementation_context'
            }
        }
        
        return CProfileResult(
            test_identifier=TestIdentifier(file_path="context_bottleneck_test.py"),
            profile_file_path="/tmp/test.prof",
            total_time=15.0,
            function_stats=function_stats,
            top_functions=[],
            call_count=8
        )
    
    def test_high_api_call_frequency_recommendations(self):
        """Test recommendations for high LLM API call frequency."""
        cprofile_result = self.create_high_api_call_scenario()
        
        # Analyze bottlenecks
        bottlenecks = self.analyzer.identify_implement_agent_bottlenecks(cprofile_result)
        time_categories = self.analyzer.categorize_time_spent(cprofile_result)
        
        # Generate recommendations
        recommendations = self.analyzer.generate_optimization_recommendations(
            bottlenecks, time_categories
        )
        
        # Should identify LLM API calls as bottleneck
        llm_bottlenecks = [b for b in bottlenecks if 'LLM' in b.component_name]
        assert len(llm_bottlenecks) > 0, "Should identify LLM API calls as bottleneck"
        
        # Should recommend reducing API call frequency
        api_recommendations = [
            rec for rec in recommendations 
            if 'api call' in rec.issue_description.lower() or 'llm' in rec.component.lower()
        ]
        assert len(api_recommendations) > 0, "Should recommend optimizing LLM API calls"
        
        # Recommendations should be high priority for significant bottlenecks
        high_priority_api_recs = [
            rec for rec in api_recommendations 
            if rec.expected_impact == 'high'
        ]
        assert len(high_priority_api_recs) > 0, "Should have high-priority API optimization recommendations"
    
    def test_high_call_count_recommendations(self):
        """Test recommendations for components with high call counts."""
        cprofile_result = self.create_high_call_count_scenario()
        
        bottlenecks = self.analyzer.identify_implement_agent_bottlenecks(cprofile_result)
        time_categories = self.analyzer.categorize_time_spent(cprofile_result)
        
        recommendations = self.analyzer.generate_optimization_recommendations(
            bottlenecks, time_categories
        )
        
        # Should identify ShellExecutor bottleneck
        shell_bottlenecks = [b for b in bottlenecks if 'Shell' in b.component_name]
        assert len(shell_bottlenecks) > 0, "Should identify ShellExecutor bottleneck"
        
        # Should recommend optimizing shell execution
        shell_recommendations = [
            rec for rec in recommendations 
            if 'shell' in rec.component.lower() or 'command' in rec.issue_description.lower()
        ]
        assert len(shell_recommendations) > 0, "Should recommend optimizing shell execution"
        
        # Should suggest specific optimizations for high call counts
        call_optimization_recs = [
            rec for rec in shell_recommendations
            if 'call' in rec.issue_description.lower() or 'batch' in rec.recommendation.lower()
        ]
        assert len(call_optimization_recs) > 0, "Should suggest call count optimizations"
    
    def test_context_loading_recommendations(self):
        """Test recommendations for context loading bottlenecks."""
        cprofile_result = self.create_context_loading_bottleneck_scenario()
        
        bottlenecks = self.analyzer.identify_implement_agent_bottlenecks(cprofile_result)
        time_categories = self.analyzer.categorize_time_spent(cprofile_result)
        
        recommendations = self.analyzer.generate_optimization_recommendations(
            bottlenecks, time_categories
        )
        
        # Should identify ContextManager bottleneck
        context_bottlenecks = [b for b in bottlenecks if 'Context' in b.component_name]
        assert len(context_bottlenecks) > 0, "Should identify ContextManager bottleneck"
        
        # Should recommend context optimization
        context_recommendations = [
            rec for rec in recommendations 
            if 'context' in rec.component.lower()
        ]
        assert len(context_recommendations) > 0, "Should recommend context optimizations"
        
        # Should suggest specific context optimizations
        context_optimization_types = [
            rec.recommendation.lower() for rec in context_recommendations
        ]
        
        # Should suggest caching, lazy loading, or compression
        optimization_keywords = ['cach', 'lazy', 'compress', 'reduc']
        has_relevant_optimization = any(
            any(keyword in rec for keyword in optimization_keywords)
            for rec in context_optimization_types
        )
        assert has_relevant_optimization, (
            f"Should suggest relevant context optimizations. Got: {context_optimization_types}"
        )
    
    def test_recommendation_priority_scoring(self):
        """Test that recommendation priority scoring is accurate."""
        # Create scenario with mixed bottlenecks
        function_stats = {
            # High impact, low effort (should be highest priority)
            'autogen_framework/agents/task_decomposer.py:30(decompose_task)': {
                'cumulative_time': 10.0,  # High time
                'total_time': 8.0,
                'call_count': 50,  # High calls - easy to optimize
                'filename': 'autogen_framework/agents/task_decomposer.py',
                'function_name': 'decompose_task'
            },
            # Medium impact, medium effort
            'autogen_framework/context_manager.py:100(load_context)': {
                'cumulative_time': 5.0,  # Medium time
                'total_time': 4.0,
                'call_count': 5,  # Low calls - harder to optimize
                'filename': 'autogen_framework/context_manager.py',
                'function_name': 'load_context'
            },
            # Low impact (should be lowest priority)
            'tests/integration/conftest.py:15(setup_test)': {
                'cumulative_time': 1.0,  # Low time
                'total_time': 0.8,
                'call_count': 1,
                'filename': 'tests/integration/conftest.py',
                'function_name': 'setup_test'
            }
        }
        
        cprofile_result = CProfileResult(
            test_identifier=TestIdentifier(file_path="priority_test.py"),
            profile_file_path="/tmp/test.prof",
            total_time=16.0,
            function_stats=function_stats,
            top_functions=[],
            call_count=56
        )
        
        bottlenecks = self.analyzer.identify_implement_agent_bottlenecks(cprofile_result)
        time_categories = self.analyzer.categorize_time_spent(cprofile_result)
        
        recommendations = self.analyzer.generate_optimization_recommendations(
            bottlenecks, time_categories
        )
        
        # Verify recommendations are sorted by priority
        priority_scores = [rec.priority_score for rec in recommendations]
        assert priority_scores == sorted(priority_scores, reverse=True), (
            "Recommendations should be sorted by priority score"
        )
        
        # High-impact components should have higher priority
        if len(recommendations) >= 2:
            # TaskDecomposer (high impact) should have higher priority than test setup (low impact)
            task_decomposer_recs = [
                rec for rec in recommendations 
                if 'taskdecomposer' in rec.component.lower()
            ]
            test_setup_recs = [
                rec for rec in recommendations 
                if 'test' in rec.component.lower()
            ]
            
            if task_decomposer_recs and test_setup_recs:
                max_task_priority = max(rec.priority_score for rec in task_decomposer_recs)
                max_test_priority = max(rec.priority_score for rec in test_setup_recs)
                
                assert max_task_priority > max_test_priority, (
                    "High-impact components should have higher priority than low-impact ones"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])