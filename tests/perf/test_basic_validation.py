"""
Basic validation tests for performance analysis tools.

This module provides basic validation tests that verify the core functionality
of the performance analysis infrastructure without complex integration requirements.
"""

import pytest
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any

from .models import TestIdentifier, TestStatus, ProfilerType
from .test_executor import TestExecutor
from .timing_analyzer import TestTimingAnalyzer
from .profiler import PerformanceProfiler
from .bottleneck_analyzer import BottleneckAnalyzer


class TestBasicFunctionality:
    """Test basic functionality of performance analysis components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_executor = TestExecutor(timeout_seconds=30)
        self.timing_analyzer = TestTimingAnalyzer(timeout_seconds=30)
        self.profiler = PerformanceProfiler()
        self.bottleneck_analyzer = BottleneckAnalyzer()
    
    def test_component_initialization(self):
        """Test that all components can be initialized."""
        # All components should initialize without errors
        assert self.test_executor is not None
        assert self.timing_analyzer is not None
        assert self.profiler is not None
        assert self.bottleneck_analyzer is not None
        
        # Check that components have expected attributes
        assert hasattr(self.test_executor, 'timeout_seconds')
        assert hasattr(self.timing_analyzer, 'test_executor')
        assert hasattr(self.profiler, 'primary_profiler')
        assert hasattr(self.bottleneck_analyzer, 'analyze_cprofile_results')
    
    def test_test_discovery_basic(self):
        """Test basic test discovery functionality."""
        discovered_tests = self.test_executor.discover_integration_tests()
        
        # Should return a list (may be empty if no tests exist)
        assert isinstance(discovered_tests, list)
        
        # If tests are found, they should have proper structure
        for test in discovered_tests:
            assert isinstance(test, TestIdentifier)
            assert test.file_path is not None
            assert test.full_name is not None
            assert test.file_path.endswith('.py')
    
    @pytest.mark.asyncio
    async def test_timing_analysis_basic(self):
        """Test basic timing analysis functionality."""
        # Should be able to run timing analysis without crashing
        timing_report = await self.timing_analyzer.analyze_all_integration_tests()
        
        # Should return a timing report object
        assert timing_report is not None
        assert hasattr(timing_report, 'total_execution_time')
        assert hasattr(timing_report, 'file_results')
        assert hasattr(timing_report, 'total_test_count')
        
        # Timing values should be reasonable
        assert timing_report.total_execution_time >= 0
        assert timing_report.total_test_count >= 0
        assert isinstance(timing_report.file_results, list)
    
    def test_bottleneck_analyzer_with_mock_data(self):
        """Test bottleneck analyzer with mock profiling data."""
        from .models import CProfileResult, ProfilingResult
        
        # Create mock profiling data
        test_identifier = TestIdentifier(
            file_path="mock_test.py",
            method_name="test_mock"
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
            profile_file_path="/tmp/mock.prof",
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
        
        # Analyze bottlenecks
        reports = self.bottleneck_analyzer.analyze_cprofile_results([profiling_result])
        
        # Should generate reports without crashing
        assert isinstance(reports, list)
        
        # If reports are generated, they should have proper structure
        for report in reports:
            assert hasattr(report, 'test_identifier')
            assert hasattr(report, 'component_bottlenecks')
            assert hasattr(report, 'optimization_recommendations')
            assert report.test_identifier == test_identifier
    
    def test_profiler_component_timing_extraction(self):
        """Test component timing extraction functionality."""
        from .models import CProfileResult
        
        # Create mock cProfile data with framework components
        test_identifier = TestIdentifier(
            file_path="component_test.py",
            method_name="test_components"
        )
        
        function_stats = {
            'autogen_framework/agents/implement_agent.py:45(execute_task)': {
                'cumulative_time': 3.0,
                'total_time': 1.0,
                'call_count': 1,
                'filename': 'autogen_framework/agents/implement_agent.py',
                'function_name': 'execute_task'
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
            profile_file_path="/tmp/components.prof",
            total_time=5.0,
            function_stats=function_stats,
            top_functions=[],
            call_count=11
        )
        
        # Extract component timings
        component_timings = self.profiler._extract_component_timings(cprofile_result)
        
        # Should return a dictionary with component names
        assert isinstance(component_timings, dict)
        
        # Should have expected component categories
        expected_components = [
            'TaskDecomposer', 'ShellExecutor', 'ErrorRecovery', 
            'ContextManager', 'ImplementAgent', 'LLM_API_Calls', 
            'Test_Infrastructure', 'Other'
        ]
        
        for component in expected_components:
            assert component in component_timings
            assert isinstance(component_timings[component], (int, float))
            assert component_timings[component] >= 0
    
    def test_optimization_recommendation_generation(self):
        """Test optimization recommendation generation."""
        from .models import ComponentBottleneck, TimeCategories, ComponentTimings
        
        # Create mock bottlenecks
        bottlenecks = [
            ComponentBottleneck(
                component_name='TaskDecomposer',
                time_spent=5.0,
                percentage_of_total=25.0,
                function_calls=[{'function': 'decompose_task', 'calls': 50, 'time': 3.0}],
                optimization_potential='high'
            )
        ]
        
        time_categories = TimeCategories(
            implement_agent_time=10.0,
            component_timings=ComponentTimings(
                task_decomposer=5.0,
                test_overhead=2.0
            )
        )
        
        # Generate recommendations
        recommendations = self.bottleneck_analyzer.generate_optimization_recommendations(
            bottlenecks, time_categories
        )
        
        # Should generate recommendations
        assert isinstance(recommendations, list)
        
        # If recommendations are generated, they should have proper structure
        for rec in recommendations:
            assert hasattr(rec, 'component')
            assert hasattr(rec, 'issue_description')
            assert hasattr(rec, 'recommendation')
            assert hasattr(rec, 'expected_impact')
            assert hasattr(rec, 'implementation_effort')
            assert hasattr(rec, 'priority_score')
            
            # Check valid values
            assert rec.expected_impact in ['high', 'medium', 'low']
            assert rec.implementation_effort in ['high', 'medium', 'low']
            assert rec.priority_score >= 0
    
    def test_error_handling_with_invalid_data(self):
        """Test error handling with invalid data."""
        from .models import ProfilingResult
        
        # Test with None cProfile result
        invalid_result = ProfilingResult(
            test_identifier=TestIdentifier(file_path="invalid.py"),
            profiler_used=ProfilerType.CPROFILE,
            cprofile_result=None
        )
        
        # Should handle invalid data gracefully
        reports = self.bottleneck_analyzer.analyze_cprofile_results([invalid_result])
        
        # Should return empty list rather than crashing
        assert isinstance(reports, list)
        assert len(reports) == 0
        
        # Test with empty list
        empty_reports = self.bottleneck_analyzer.analyze_cprofile_results([])
        assert isinstance(empty_reports, list)
        assert len(empty_reports) == 0
    
    def test_timing_report_properties(self):
        """Test timing report calculated properties."""
        from .models import TimingReport, FileTimingResult, TestTimingResult
        
        # Create mock timing data
        test_result1 = TestTimingResult(
            test_identifier=TestIdentifier(file_path="test1.py", method_name="test_fast"),
            execution_time=1.0,
            status=TestStatus.PASSED
        )
        
        test_result2 = TestTimingResult(
            test_identifier=TestIdentifier(file_path="test2.py", method_name="test_slow"),
            execution_time=5.0,
            status=TestStatus.PASSED
        )
        
        file_result = FileTimingResult(
            file_path="test_file.py",
            total_execution_time=6.0,
            test_results=[test_result1, test_result2],
            file_status=TestStatus.PASSED
        )
        
        timing_report = TimingReport(
            total_execution_time=6.0,
            file_results=[file_result]
        )
        
        # Test calculated properties
        assert timing_report.total_test_count == 2
        
        slowest_tests = timing_report.slowest_tests
        assert len(slowest_tests) == 2
        assert slowest_tests[0].execution_time == 5.0  # Slowest first
        assert slowest_tests[1].execution_time == 1.0
        
        # Test file result properties
        assert file_result.test_count == 2
        assert file_result.passed_count == 2
        assert file_result.failed_count == 0
        assert file_result.timeout_count == 0


class TestValidationAccuracy:
    """Test validation accuracy of performance measurements."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_executor = TestExecutor(timeout_seconds=10)
    
    @pytest.mark.asyncio
    async def test_timeout_enforcement(self):
        """Test that timeouts are properly enforced."""
        # Use a very short timeout
        short_executor = TestExecutor(timeout_seconds=1)
        
        # Try to find any integration test
        discovered_tests = short_executor.discover_integration_tests()
        
        if discovered_tests:
            test_identifier = discovered_tests[0]
            
            start_time = time.time()
            result = await short_executor.execute_single_test_method(
                test_identifier.file_path, test_identifier.method_name
            )
            end_time = time.time()
            
            actual_duration = end_time - start_time
            
            # If test timed out, verify timeout was enforced
            if result.status == TestStatus.TIMEOUT:
                # Should timeout within reasonable margin (3 seconds tolerance)
                assert actual_duration <= 4, f"Timeout not enforced: {actual_duration:.2f}s"
                assert result.timeout_occurred is True
                assert result.error_message is not None
            else:
                # If test completed, it should be within timeout
                assert actual_duration <= 1.5, f"Test took too long: {actual_duration:.2f}s"
    
    def test_data_model_validation(self):
        """Test that data models have proper validation."""
        from .models import TestIdentifier, TestTimingResult, ComponentBottleneck
        
        # Test TestIdentifier
        test_id = TestIdentifier(
            file_path="test.py",
            class_name="TestClass",
            method_name="test_method"
        )
        
        assert test_id.full_name == "test.py::TestClass::test_method"
        
        # Test without class name
        test_id2 = TestIdentifier(
            file_path="test.py",
            method_name="test_method"
        )
        
        assert test_id2.full_name == "test.py::test_method"
        
        # Test TestTimingResult
        timing_result = TestTimingResult(
            test_identifier=test_id,
            execution_time=2.5,
            status=TestStatus.PASSED
        )
        
        assert timing_result.duration_ms == 2500.0
        
        # Test ComponentBottleneck
        bottleneck = ComponentBottleneck(
            component_name='TestComponent',
            time_spent=5.0,
            percentage_of_total=25.0
        )
        
        assert bottleneck.is_significant is True  # > 10%
        
        bottleneck2 = ComponentBottleneck(
            component_name='SmallComponent',
            time_spent=1.0,
            percentage_of_total=5.0
        )
        
        assert bottleneck2.is_significant is False  # <= 10%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])