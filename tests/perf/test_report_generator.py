"""
Tests for comprehensive performance reporting and visualization.

This module tests the PerformanceReportGenerator and ReportComparisonTool
to ensure proper HTML report generation, flame graph integration, and
comparison functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

try:
    from .report_generator import PerformanceReportGenerator, ReportComparisonTool
    from .models import (
        TimingReport, ProfilingResult, BottleneckReport, TestIdentifier,
        TestTimingResult, FileTimingResult, CProfileResult, PySpyResult,
        ComponentBottleneck, OptimizationRecommendation, TimeCategories,
        ComponentTimings, TestStatus, ProfilerType
    )
    from .llm_profiler import LLMProfilingResult, LLMAPICall
except ImportError:
    # Handle direct execution
    from report_generator import PerformanceReportGenerator, ReportComparisonTool
    from models import (
        TimingReport, ProfilingResult, BottleneckReport, TestIdentifier,
        TestTimingResult, FileTimingResult, CProfileResult, PySpyResult,
        ComponentBottleneck, OptimizationRecommendation, TimeCategories,
        ComponentTimings, TestStatus, ProfilerType
    )
    from llm_profiler import LLMProfilingResult, LLMAPICall


class TestPerformanceReportGenerator:
    """Test cases for PerformanceReportGenerator."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def report_generator(self, temp_output_dir):
        """Create PerformanceReportGenerator instance for testing."""
        return PerformanceReportGenerator(temp_output_dir)
    
    @pytest.fixture
    def sample_timing_report(self):
        """Create sample timing report for testing."""
        test_id = TestIdentifier(
            file_path="tests/integration/test_sample.py",
            class_name="TestSample",
            method_name="test_method"
        )
        
        test_result = TestTimingResult(
            test_identifier=test_id,
            execution_time=5.0,
            status=TestStatus.PASSED
        )
        
        file_result = FileTimingResult(
            file_path="tests/integration/test_sample.py",
            total_execution_time=5.0,
            test_results=[test_result]
        )
        
        return TimingReport(
            total_execution_time=5.0,
            file_results=[file_result]
        )
    
    @pytest.fixture
    def sample_profiling_results(self):
        """Create sample profiling results for testing."""
        test_id = TestIdentifier(
            file_path="tests/integration/test_sample.py",
            class_name="TestSample",
            method_name="test_method"
        )
        
        cprofile_result = CProfileResult(
            test_identifier=test_id,
            profile_file_path="profile.prof",
            total_time=5.0,
            function_stats={
                "function_1": {
                    'cumulative_time': 2.0,
                    'total_time': 1.5,
                    'call_count': 10,
                    'function_name': 'test_function',
                    'filename': 'test_file.py'
                }
            },
            call_count=100
        )
        
        pyspy_result = PySpyResult(
            test_identifier=test_id,
            flamegraph_path="flamegraph.svg",
            sampling_duration=5.0,
            sample_count=500
        )
        
        llm_call = LLMAPICall(
            timestamp=1234567890.0,
            component="TaskDecomposer",
            request_url="http://localhost:8888/v1/chat/completions",
            request_method="POST",
            prompt_text="Sample prompt",
            prompt_size_chars=13,
            response_text="Sample response",
            response_size_chars=15,
            total_time=2.0,
            network_time=0.5,
            processing_time=1.5,
            status_code=200,
            success=True
        )
        
        llm_result = LLMProfilingResult(
            test_identifier=test_id,
            api_calls=[llm_call]
        )
        
        profiling_result = ProfilingResult(
            test_identifier=test_id,
            profiler_used=ProfilerType.CPROFILE,
            cprofile_result=cprofile_result,
            pyspy_result=pyspy_result,
            llm_profiling_result=llm_result,
            component_timings={
                'TaskDecomposer': 1.0,
                'ShellExecutor': 2.0,
                'ErrorRecovery': 1.0,
                'ContextManager': 1.0
            },
            total_profiling_time=5.0
        )
        
        return [profiling_result]
    
    @pytest.fixture
    def sample_bottleneck_reports(self):
        """Create sample bottleneck reports for testing."""
        test_id = TestIdentifier(
            file_path="tests/integration/test_sample.py",
            class_name="TestSample",
            method_name="test_method"
        )
        
        component_timings = ComponentTimings(
            task_decomposer=1.0,
            shell_executor=2.0,
            error_recovery=1.0,
            context_manager=1.0,
            test_overhead=0.5
        )
        
        time_categories = TimeCategories(
            test_setup_time=0.2,
            test_teardown_time=0.3,
            implement_agent_time=5.0,
            component_timings=component_timings
        )
        
        bottleneck = ComponentBottleneck(
            component_name="ShellExecutor",
            time_spent=2.0,
            percentage_of_total=40.0,
            optimization_potential="high"
        )
        
        recommendation = OptimizationRecommendation(
            component="ShellExecutor",
            issue_description="High subprocess overhead",
            recommendation="Batch commands where possible",
            expected_impact="high",
            implementation_effort="medium"
        )
        
        report = BottleneckReport(
            test_identifier=test_id,
            time_categories=time_categories,
            component_bottlenecks=[bottleneck],
            optimization_recommendations=[recommendation]
        )
        
        return [report]
    
    def test_generate_comprehensive_report(
        self,
        report_generator,
        sample_timing_report,
        sample_profiling_results,
        sample_bottleneck_reports,
        temp_output_dir
    ):
        """Test comprehensive HTML report generation."""
        # Generate report
        report_path = report_generator.generate_comprehensive_report(
            timing_report=sample_timing_report,
            profiling_results=sample_profiling_results,
            bottleneck_reports=sample_bottleneck_reports,
            report_name="test_report"
        )
        
        # Verify report file exists
        assert Path(report_path).exists()
        assert report_path.endswith("test_report.html")
        
        # Verify JSON data file exists
        json_path = Path(temp_output_dir) / "test_report_data.json"
        assert json_path.exists()
        
        # Verify HTML content
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for key sections
        assert "Performance Analysis Report" in html_content
        assert "Timing Analysis" in html_content
        assert "Profiling Results" in html_content
        assert "Bottleneck Analysis" in html_content
        assert "Optimization Recommendations" in html_content
        assert "LLM Analysis" in html_content
        
        # Check for specific data
        assert "test_sample.py" in html_content
        assert "ShellExecutor" in html_content
        assert "TaskDecomposer" in html_content
    
    def test_prepare_timing_data(
        self,
        report_generator,
        sample_timing_report
    ):
        """Test timing data preparation for reports."""
        timing_data = report_generator._prepare_timing_data(sample_timing_report)
        
        # Verify structure
        assert 'summary' in timing_data
        assert 'slowest_tests' in timing_data
        assert 'timeout_tests' in timing_data
        assert 'file_breakdown' in timing_data
        
        # Verify summary data
        summary = timing_data['summary']
        assert summary['total_tests'] == 1
        assert summary['total_time'] == 5.0
        assert summary['average_test_time'] == 5.0
        assert summary['timeout_count'] == 0
        
        # Verify slowest tests
        slowest = timing_data['slowest_tests']
        assert len(slowest) == 1
        assert slowest[0]['test_name'] == "tests/integration/test_sample.py::TestSample::test_method"
        assert slowest[0]['execution_time'] == 5.0
    
    def test_prepare_profiling_data(
        self,
        report_generator,
        sample_profiling_results
    ):
        """Test profiling data preparation for reports."""
        profiling_data = report_generator._prepare_profiling_data(sample_profiling_results)
        
        # Verify structure
        assert 'summary' in profiling_data
        assert 'results' in profiling_data
        
        # Verify summary
        summary = profiling_data['summary']
        assert summary['profiled_tests'] == 1
        assert summary['total_profiling_time'] == 5.0
        assert summary['cprofile_results'] == 1
        assert summary['pyspy_results'] == 1
        
        # Verify results
        results = profiling_data['results']
        assert len(results) == 1
        assert results[0]['profiler_used'] == 'cProfile'
        assert results[0]['has_cprofile'] is True
        assert results[0]['has_pyspy'] is True
        assert results[0]['has_llm_data'] is True
    
    def test_prepare_bottleneck_data(
        self,
        report_generator,
        sample_bottleneck_reports
    ):
        """Test bottleneck data preparation for reports."""
        bottleneck_data = report_generator._prepare_bottleneck_data(sample_bottleneck_reports)
        
        # Verify structure
        assert 'summary' in bottleneck_data
        assert 'component_summary' in bottleneck_data
        assert 'reports' in bottleneck_data
        
        # Verify summary
        summary = bottleneck_data['summary']
        assert summary['total_reports'] == 1
        assert summary['unique_components'] == 1
        assert summary['total_bottlenecks'] == 1
        
        # Verify component summary
        component_summary = bottleneck_data['component_summary']
        assert 'ShellExecutor' in component_summary
        assert component_summary['ShellExecutor']['total_time'] == 2.0
    
    def test_prepare_llm_analysis_data(
        self,
        report_generator,
        sample_profiling_results
    ):
        """Test LLM analysis data preparation for reports."""
        llm_data = report_generator._prepare_llm_analysis_data(sample_profiling_results)
        
        # Verify structure
        assert 'summary' in llm_data
        assert 'component_breakdown' in llm_data
        assert 'slow_calls' in llm_data
        assert 'results' in llm_data
        
        # Verify summary
        summary = llm_data['summary']
        assert summary['tests_with_llm_data'] == 1
        assert summary['total_api_calls'] == 1
        assert summary['total_llm_time'] == 2.0
        
        # Verify component breakdown
        component_breakdown = llm_data['component_breakdown']
        assert 'TaskDecomposer' in component_breakdown
        assert component_breakdown['TaskDecomposer']['calls'] == 1
    
    def test_flame_graph_integration(
        self,
        report_generator,
        sample_profiling_results,
        temp_output_dir
    ):
        """Test flame graph integration into reports."""
        # Create mock SVG file
        svg_path = Path(temp_output_dir) / "flamegraph.svg"
        svg_content = '<svg><rect width="100" height="50"/></svg>'
        with open(svg_path, 'w') as f:
            f.write(svg_content)
        
        # Update profiling result with correct path
        sample_profiling_results[0].pyspy_result.flamegraph_path = str(svg_path)
        
        flame_graph_data = report_generator._prepare_flame_graph_data(sample_profiling_results)
        
        # Verify flame graph data
        assert flame_graph_data['available_graphs'] == 1
        graphs = flame_graph_data['graphs']
        assert len(graphs) == 1
        assert graphs[0]['svg_content'] == svg_content
        assert graphs[0]['test_name'] == "tests/integration/test_sample.py::TestSample::test_method"
    
    def test_empty_data_handling(self, report_generator):
        """Test handling of empty data sets."""
        # Test with empty profiling results
        profiling_data = report_generator._prepare_profiling_data([])
        assert profiling_data['summary'] == {}
        assert profiling_data['results'] == []
        
        # Test with empty bottleneck reports
        bottleneck_data = report_generator._prepare_bottleneck_data([])
        assert bottleneck_data['summary'] == {}
        assert bottleneck_data['reports'] == []
        
        # Test with empty LLM data
        llm_data = report_generator._prepare_llm_analysis_data([])
        assert llm_data['summary'] == {}
        assert llm_data['results'] == []


class TestReportComparisonTool:
    """Test cases for ReportComparisonTool."""
    
    @pytest.fixture
    def temp_reports_dir(self):
        """Create temporary reports directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def comparison_tool(self, temp_reports_dir):
        """Create ReportComparisonTool instance for testing."""
        return ReportComparisonTool(temp_reports_dir)
    
    @pytest.fixture
    def sample_report_data(self, temp_reports_dir):
        """Create sample report data files for comparison testing."""
        baseline_data = {
            'metadata': {
                'generation_time': '2024-01-01T10:00:00',
                'total_tests_analyzed': 10,
                'total_execution_time': 50.0
            },
            'timing_analysis': {
                'summary': {
                    'total_tests': 10,
                    'total_time': 50.0,
                    'average_test_time': 5.0,
                    'timeout_count': 1
                }
            },
            'bottleneck_analysis': {
                'component_summary': {
                    'TaskDecomposer': {'total_time': 10.0, 'occurrences': 2},
                    'ShellExecutor': {'total_time': 20.0, 'occurrences': 3}
                }
            }
        }
        
        current_data = {
            'metadata': {
                'generation_time': '2024-01-02T10:00:00',
                'total_tests_analyzed': 10,
                'total_execution_time': 45.0
            },
            'timing_analysis': {
                'summary': {
                    'total_tests': 10,
                    'total_time': 45.0,
                    'average_test_time': 4.5,
                    'timeout_count': 0
                }
            },
            'bottleneck_analysis': {
                'component_summary': {
                    'TaskDecomposer': {'total_time': 9.0, 'occurrences': 2},
                    'ShellExecutor': {'total_time': 18.0, 'occurrences': 3}
                }
            }
        }
        
        # Save baseline report
        baseline_path = Path(temp_reports_dir) / "baseline_data.json"
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f)
        
        # Save current report
        current_path = Path(temp_reports_dir) / "current_data.json"
        with open(current_path, 'w') as f:
            json.dump(current_data, f)
        
        return str(baseline_path), str(current_path)
    
    def test_generate_comparison_report(
        self,
        comparison_tool,
        sample_report_data,
        temp_reports_dir
    ):
        """Test comparison report generation."""
        baseline_path, current_path = sample_report_data
        
        # Generate comparison report using PerformanceReportGenerator
        generator = PerformanceReportGenerator(temp_reports_dir)
        comparison_path = generator.generate_comparison_report(
            baseline_report_path=baseline_path,
            current_report_path=current_path,
            comparison_name="test_comparison"
        )
        
        # Verify report exists
        assert Path(comparison_path).exists()
        assert comparison_path.endswith("test_comparison.html")
        
        # Verify HTML content
        with open(comparison_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert "Performance Comparison Report" in html_content
        assert "Performance Trends" in html_content
    
    def test_compare_timing_data(self, comparison_tool):
        """Test timing data comparison logic."""
        generator = PerformanceReportGenerator()
        
        baseline_data = {
            'timing_analysis': {
                'summary': {
                    'total_time': 50.0,
                    'average_test_time': 5.0,
                    'timeout_count': 1
                }
            }
        }
        
        current_data = {
            'timing_analysis': {
                'summary': {
                    'total_time': 45.0,
                    'average_test_time': 4.5,
                    'timeout_count': 0
                }
            }
        }
        
        comparison = generator._compare_timing_data(baseline_data, current_data)
        
        # Verify improvements are detected
        assert comparison['total_time_change'] == -10.0  # 10% improvement
        assert comparison['average_time_change'] == -10.0  # 10% improvement
        assert comparison['timeout_count_change'] == -100.0  # 100% improvement (0 timeouts)
    
    def test_compare_bottleneck_data(self, comparison_tool):
        """Test bottleneck data comparison logic."""
        generator = PerformanceReportGenerator()
        
        baseline_data = {
            'bottleneck_analysis': {
                'component_summary': {
                    'TaskDecomposer': {'total_time': 10.0},
                    'ShellExecutor': {'total_time': 20.0}
                }
            }
        }
        
        current_data = {
            'bottleneck_analysis': {
                'component_summary': {
                    'TaskDecomposer': {'total_time': 9.0},
                    'ShellExecutor': {'total_time': 18.0}
                }
            }
        }
        
        comparison = generator._compare_bottleneck_data(baseline_data, current_data)
        
        # Verify component improvements
        component_changes = comparison['component_changes']
        assert component_changes['TaskDecomposer']['improvement'] is True
        assert component_changes['ShellExecutor']['improvement'] is True
        
        improved_components = comparison['improved_components']
        assert 'TaskDecomposer' in improved_components
        assert 'ShellExecutor' in improved_components
    
    def test_multi_report_comparison(
        self,
        comparison_tool,
        temp_reports_dir
    ):
        """Test multi-report trend analysis."""
        # Create multiple report files
        report_paths = []
        
        for i in range(3):
            report_data = {
                'metadata': {
                    'generation_time': f'2024-01-{i+1:02d}T10:00:00',
                    'total_tests_analyzed': 10,
                    'total_execution_time': 50.0 - (i * 5.0)  # Improving trend
                },
                'timing_analysis': {
                    'summary': {
                        'total_time': 50.0 - (i * 5.0),
                        'total_tests': 10
                    }
                },
                'bottleneck_analysis': {
                    'component_summary': {
                        'TaskDecomposer': {'total_time': 10.0 - i}
                    }
                }
            }
            
            report_path = Path(temp_reports_dir) / f"report_{i}_data.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f)
            
            report_paths.append(str(report_path))
        
        # Generate multi-comparison report
        comparison_path = comparison_tool.compare_multiple_reports(
            report_paths=report_paths,
            output_name="test_multi_comparison"
        )
        
        # Verify report exists
        assert Path(comparison_path).exists()
        
        # Verify HTML content
        with open(comparison_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert "Performance Trends Analysis" in html_content
        assert "improving" in html_content.lower() or "stable" in html_content.lower()
    
    def test_trend_calculation(self, comparison_tool):
        """Test trend direction calculation."""
        # Test improving trend
        improving_values = [10.0, 9.0, 8.0, 7.0]
        trend = comparison_tool._calculate_trend_direction(improving_values)
        assert trend == 'improving'
        
        # Test degrading trend
        degrading_values = [7.0, 8.0, 9.0, 10.0]
        trend = comparison_tool._calculate_trend_direction(degrading_values)
        assert trend == 'degrading'
        
        # Test stable trend
        stable_values = [8.0, 8.1, 7.9, 8.0]
        trend = comparison_tool._calculate_trend_direction(stable_values)
        assert trend == 'stable'
        
        # Test insufficient data
        insufficient_values = [8.0]
        trend = comparison_tool._calculate_trend_direction(insufficient_values)
        assert trend == 'insufficient_data'
    
    def test_improvement_percentage_calculation(self, comparison_tool):
        """Test improvement percentage calculation."""
        # Test improvement
        values = [10.0, 8.0]
        improvement = comparison_tool._calculate_improvement_percentage(values)
        assert improvement == 20.0  # 20% improvement
        
        # Test regression
        values = [8.0, 10.0]
        improvement = comparison_tool._calculate_improvement_percentage(values)
        assert improvement == -25.0  # 25% regression
        
        # Test no change
        values = [8.0, 8.0]
        improvement = comparison_tool._calculate_improvement_percentage(values)
        assert improvement == 0.0
        
        # Test edge cases
        values = [0.0, 5.0]
        improvement = comparison_tool._calculate_improvement_percentage(values)
        assert improvement == 0.0  # Division by zero protection
        
        values = [5.0]
        improvement = comparison_tool._calculate_improvement_percentage(values)
        assert improvement == 0.0  # Insufficient data


@pytest.mark.integration
class TestReportGeneratorIntegration:
    """Integration tests for report generator with real-like data."""
    
    def test_end_to_end_report_generation(self):
        """Test complete end-to-end report generation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create report generator
            generator = PerformanceReportGenerator(temp_dir)
            
            # Create realistic test data
            timing_report = self._create_realistic_timing_report()
            profiling_results = self._create_realistic_profiling_results()
            bottleneck_reports = self._create_realistic_bottleneck_reports()
            
            # Generate comprehensive report
            report_path = generator.generate_comprehensive_report(
                timing_report=timing_report,
                profiling_results=profiling_results,
                bottleneck_reports=bottleneck_reports,
                report_name="integration_test_report"
            )
            
            # Verify report generation
            assert Path(report_path).exists()
            
            # Verify HTML structure
            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check for all major sections
            required_sections = [
                "Performance Analysis Report",
                "Executive Summary",
                "Timing Analysis",
                "Profiling Results",
                "Bottleneck Analysis",
                "Optimization Recommendations",
                "LLM Analysis"
            ]
            
            for section in required_sections:
                assert section in html_content, f"Missing section: {section}"
            
            # Verify interactive elements
            assert "chart.js" in html_content
            assert "createTimingChart" in html_content
            assert "createComponentChart" in html_content
            
            # Verify data integrity
            json_path = Path(temp_dir) / "integration_test_report_data.json"
            assert json_path.exists()
            
            with open(json_path, 'r') as f:
                report_data = json.load(f)
            
            assert 'metadata' in report_data
            assert 'timing_analysis' in report_data
            assert 'profiling_analysis' in report_data
            assert 'bottleneck_analysis' in report_data
    
    def _create_realistic_timing_report(self) -> TimingReport:
        """Create realistic timing report for integration testing."""
        test_results = []
        
        # Create various test scenarios
        test_scenarios = [
            ("fast", 0.5, TestStatus.PASSED),
            ("medium", 3.0, TestStatus.PASSED),
            ("slow", 15.0, TestStatus.PASSED),
            ("timeout", 60.0, TestStatus.TIMEOUT)
        ]
        
        for i, (category, base_time, status) in enumerate(test_scenarios):
            for j in range(3):  # 3 tests per category
                test_id = TestIdentifier(
                    file_path=f"tests/integration/test_{category}_{j}.py",
                    class_name=f"Test{category.title()}{j}",
                    method_name=f"test_method_{j}"
                )
                
                result = TestTimingResult(
                    test_identifier=test_id,
                    execution_time=base_time + (j * 0.5),
                    status=status,
                    timeout_occurred=(status == TestStatus.TIMEOUT)
                )
                test_results.append(result)
        
        # Group by file
        file_results = []
        files = {}
        
        for result in test_results:
            file_path = result.test_identifier.file_path
            if file_path not in files:
                files[file_path] = []
            files[file_path].append(result)
        
        for file_path, tests in files.items():
            file_result = FileTimingResult(
                file_path=file_path,
                total_execution_time=sum(t.execution_time for t in tests),
                test_results=tests
            )
            file_results.append(file_result)
        
        return TimingReport(
            total_execution_time=sum(r.execution_time for r in test_results),
            file_results=file_results
        )
    
    def _create_realistic_profiling_results(self) -> List[ProfilingResult]:
        """Create realistic profiling results for integration testing."""
        # Only profile the slow tests
        slow_tests = [
            "tests/integration/test_slow_0.py::TestSlow0::test_method_0",
            "tests/integration/test_slow_1.py::TestSlow1::test_method_1"
        ]
        
        results = []
        
        for test_name in slow_tests:
            parts = test_name.split("::")
            test_id = TestIdentifier(
                file_path=parts[0],
                class_name=parts[1],
                method_name=parts[2]
            )
            
            # Create comprehensive profiling result
            result = ProfilingResult(
                test_identifier=test_id,
                profiler_used=ProfilerType.CPROFILE,
                component_timings={
                    'TaskDecomposer': 2.0,
                    'ShellExecutor': 8.0,
                    'ErrorRecovery': 3.0,
                    'ContextManager': 2.0
                },
                total_profiling_time=15.0
            )
            
            results.append(result)
        
        return results
    
    def _create_realistic_bottleneck_reports(self) -> List[BottleneckReport]:
        """Create realistic bottleneck reports for integration testing."""
        reports = []
        
        test_names = [
            "tests/integration/test_slow_0.py::TestSlow0::test_method_0",
            "tests/integration/test_slow_1.py::TestSlow1::test_method_1"
        ]
        
        for test_name in test_names:
            parts = test_name.split("::")
            test_id = TestIdentifier(
                file_path=parts[0],
                class_name=parts[1],
                method_name=parts[2]
            )
            
            # Create realistic bottleneck report
            bottlenecks = [
                ComponentBottleneck(
                    component_name="ShellExecutor",
                    time_spent=8.0,
                    percentage_of_total=53.3,
                    optimization_potential="high"
                ),
                ComponentBottleneck(
                    component_name="ErrorRecovery",
                    time_spent=3.0,
                    percentage_of_total=20.0,
                    optimization_potential="medium"
                )
            ]
            
            recommendations = [
                OptimizationRecommendation(
                    component="ShellExecutor",
                    issue_description="High subprocess creation overhead detected",
                    recommendation="Implement command batching and shell command chaining",
                    expected_impact="high",
                    implementation_effort="medium"
                )
            ]
            
            report = BottleneckReport(
                test_identifier=test_id,
                time_categories=TimeCategories(),
                component_bottlenecks=bottlenecks,
                optimization_recommendations=recommendations
            )
            
            reports.append(report)
        
        return reports


if __name__ == "__main__":
    pytest.main([__file__, "-v"])