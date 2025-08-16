#!/usr/bin/env python3
"""
Demo script for comprehensive performance reporting and visualization.

This script demonstrates the complete reporting functionality including:
- HTML report generation with timing data and profiling results
- Flame graph integration into reports
- Comparison tools for analyzing multiple profiling runs
- Actionable optimization recommendations in report format
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import time

# Add the parent directory to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from report_generator import PerformanceReportGenerator, ReportComparisonTool
    from models import (
        TimingReport, ProfilingResult, BottleneckReport, TestIdentifier,
        TestTimingResult, FileTimingResult, CProfileResult, PySpyResult,
        ComponentBottleneck, OptimizationRecommendation, TimeCategories,
        ComponentTimings, TestStatus, ProfilerType
    )
    from llm_profiler import LLMProfilingResult, LLMAPICall
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)


class ReportGeneratorDemo:
    """
    Demonstration of comprehensive performance reporting capabilities.
    
    This class creates sample data and demonstrates all reporting features
    including HTML generation, flame graph integration, and comparison tools.
    """
    
    def __init__(self):
        """Initialize demo with logging and output directory."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = Path("artifacts/performance/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize report generator
        self.report_generator = PerformanceReportGenerator(str(self.output_dir))
        self.comparison_tool = ReportComparisonTool(str(self.output_dir))
    
    async def run_comprehensive_demo(self):
        """Run complete demonstration of reporting capabilities."""
        self.logger.info("Starting comprehensive performance reporting demo")
        
        try:
            # Generate sample data
            timing_report = self._create_sample_timing_report()
            profiling_results = self._create_sample_profiling_results()
            bottleneck_reports = self._create_sample_bottleneck_reports()
            
            # Generate comprehensive report
            report_path = self.report_generator.generate_comprehensive_report(
                timing_report=timing_report,
                profiling_results=profiling_results,
                bottleneck_reports=bottleneck_reports,
                report_name="demo_comprehensive_report"
            )
            
            self.logger.info(f"Generated comprehensive report: {report_path}")
            
            # Create baseline and current reports for comparison
            baseline_report = self._create_baseline_report()
            current_report = self._create_current_report()
            
            # Generate comparison report
            comparison_path = self.report_generator.generate_comparison_report(
                baseline_report_path=baseline_report,
                current_report_path=current_report,
                comparison_name="demo_performance_comparison"
            )
            
            self.logger.info(f"Generated comparison report: {comparison_path}")
            
            # Demonstrate multi-report comparison
            multi_reports = self._create_multiple_reports()
            multi_comparison_path = self.comparison_tool.compare_multiple_reports(
                report_paths=multi_reports,
                output_name="demo_multi_comparison"
            )
            
            self.logger.info(f"Generated multi-comparison report: {multi_comparison_path}")
            
            # Print summary
            self._print_demo_summary([report_path, comparison_path, multi_comparison_path])
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            raise
    
    def _create_sample_timing_report(self) -> TimingReport:
        """Create sample timing report with realistic test data."""
        # Create sample test results
        test_results = []
        
        # Fast tests
        for i in range(15):
            test_id = TestIdentifier(
                file_path=f"tests/integration/test_fast_{i}.py",
                class_name=f"TestFast{i}",
                method_name=f"test_method_{i}"
            )
            result = TestTimingResult(
                test_identifier=test_id,
                execution_time=0.5 + (i * 0.1),  # 0.5 to 2.0 seconds
                status=TestStatus.PASSED
            )
            test_results.append(result)
        
        # Medium tests
        for i in range(10):
            test_id = TestIdentifier(
                file_path=f"tests/integration/test_medium_{i}.py",
                class_name=f"TestMedium{i}",
                method_name=f"test_method_{i}"
            )
            result = TestTimingResult(
                test_identifier=test_id,
                execution_time=3.0 + (i * 0.5),  # 3.0 to 7.5 seconds
                status=TestStatus.PASSED
            )
            test_results.append(result)
        
        # Slow tests
        for i in range(5):
            test_id = TestIdentifier(
                file_path=f"tests/integration/test_slow_{i}.py",
                class_name=f"TestSlow{i}",
                method_name=f"test_method_{i}"
            )
            result = TestTimingResult(
                test_identifier=test_id,
                execution_time=10.0 + (i * 5.0),  # 10.0 to 30.0 seconds
                status=TestStatus.PASSED
            )
            test_results.append(result)
        
        # Timeout tests
        for i in range(2):
            test_id = TestIdentifier(
                file_path=f"tests/integration/test_timeout_{i}.py",
                class_name=f"TestTimeout{i}",
                method_name=f"test_method_{i}"
            )
            result = TestTimingResult(
                test_identifier=test_id,
                execution_time=60.0,  # Timeout threshold
                status=TestStatus.TIMEOUT,
                timeout_occurred=True,
                error_message="Test exceeded 60 second timeout"
            )
            test_results.append(result)
        
        # Group tests by file
        file_results = []
        current_file = None
        current_tests = []
        
        for result in sorted(test_results, key=lambda x: x.test_identifier.file_path):
            if current_file != result.test_identifier.file_path:
                if current_file is not None:
                    file_result = FileTimingResult(
                        file_path=current_file,
                        total_execution_time=sum(t.execution_time for t in current_tests),
                        test_results=current_tests
                    )
                    file_results.append(file_result)
                
                current_file = result.test_identifier.file_path
                current_tests = []
            
            current_tests.append(result)
        
        # Add final file
        if current_file is not None:
            file_result = FileTimingResult(
                file_path=current_file,
                total_execution_time=sum(t.execution_time for t in current_tests),
                test_results=current_tests
            )
            file_results.append(file_result)
        
        return TimingReport(
            total_execution_time=sum(r.execution_time for r in test_results),
            file_results=file_results,
            timeout_threshold=60.0
        )
    
    def _create_sample_profiling_results(self) -> List[ProfilingResult]:
        """Create sample profiling results with cProfile and py-spy data."""
        profiling_results = []
        
        # Create profiling results for slow tests
        slow_test_names = [
            "tests/integration/test_slow_0.py::TestSlow0::test_method_0",
            "tests/integration/test_slow_1.py::TestSlow1::test_method_1",
            "tests/integration/test_slow_2.py::TestSlow2::test_method_2"
        ]
        
        for i, test_name in enumerate(slow_test_names):
            parts = test_name.split("::")
            test_id = TestIdentifier(
                file_path=parts[0],
                class_name=parts[1],
                method_name=parts[2]
            )
            
            # Create cProfile result
            cprofile_result = CProfileResult(
                test_identifier=test_id,
                profile_file_path=f"profile_{i}.prof",
                total_time=10.0 + (i * 5.0),
                function_stats={
                    f"function_{j}": {
                        'cumulative_time': 1.0 + (j * 0.5),
                        'total_time': 0.8 + (j * 0.3),
                        'call_count': 10 + j,
                        'function_name': f"test_function_{j}",
                        'filename': f"test_file_{j}.py"
                    }
                    for j in range(20)
                },
                call_count=200 + (i * 50)
            )
            
            # Create py-spy result
            pyspy_result = PySpyResult(
                test_identifier=test_id,
                flamegraph_path=f"flamegraph_{i}.svg",
                sampling_duration=10.0 + (i * 5.0),
                sample_count=1000 + (i * 200)
            )
            
            # Create LLM profiling result
            llm_calls = []
            for j in range(5 + i):
                llm_call = LLMAPICall(
                    timestamp=time.time() + j,
                    component=["TaskDecomposer", "ErrorRecovery", "ImplementAgent"][j % 3],
                    request_url="http://localhost:8888/v1/chat/completions",
                    request_method="POST",
                    prompt_text=f"Sample prompt {j} for test {i}" * (10 + j),
                    prompt_size_chars=len(f"Sample prompt {j} for test {i}" * (10 + j)),
                    response_text=f"Sample response {j}" * (5 + j),
                    response_size_chars=len(f"Sample response {j}" * (5 + j)),
                    total_time=2.0 + (j * 0.5),
                    network_time=0.5 + (j * 0.1),
                    processing_time=1.5 + (j * 0.4),
                    status_code=200,
                    success=True
                )
                llm_calls.append(llm_call)
            
            llm_result = LLMProfilingResult(
                test_identifier=test_id,
                api_calls=llm_calls
            )
            
            # Create combined profiling result
            profiling_result = ProfilingResult(
                test_identifier=test_id,
                profiler_used=ProfilerType.CPROFILE,
                cprofile_result=cprofile_result,
                pyspy_result=pyspy_result,
                llm_profiling_result=llm_result,
                component_timings={
                    'TaskDecomposer': 2.0 + i,
                    'ShellExecutor': 3.0 + i,
                    'ErrorRecovery': 1.5 + i,
                    'ContextManager': 1.0 + i
                },
                total_profiling_time=10.0 + (i * 5.0)
            )
            
            profiling_results.append(profiling_result)
        
        return profiling_results
    
    def _create_sample_bottleneck_reports(self) -> List[BottleneckReport]:
        """Create sample bottleneck analysis reports."""
        bottleneck_reports = []
        
        # Create bottleneck reports for profiled tests
        test_names = [
            "tests/integration/test_slow_0.py::TestSlow0::test_method_0",
            "tests/integration/test_slow_1.py::TestSlow1::test_method_1",
            "tests/integration/test_slow_2.py::TestSlow2::test_method_2"
        ]
        
        for i, test_name in enumerate(test_names):
            parts = test_name.split("::")
            test_id = TestIdentifier(
                file_path=parts[0],
                class_name=parts[1],
                method_name=parts[2]
            )
            
            # Create component timings
            component_timings = ComponentTimings(
                task_decomposer=2.0 + i,
                shell_executor=3.0 + i,
                error_recovery=1.5 + i,
                context_manager=1.0 + i,
                llm_api_calls=2.5 + i,
                test_overhead=1.0,
                other=0.5
            )
            
            time_categories = TimeCategories(
                test_setup_time=0.5,
                test_teardown_time=0.5,
                implement_agent_time=component_timings.total_component_time,
                component_timings=component_timings
            )
            
            # Create component bottlenecks
            bottlenecks = [
                ComponentBottleneck(
                    component_name="ShellExecutor",
                    time_spent=3.0 + i,
                    percentage_of_total=30.0 + (i * 5),
                    optimization_potential="high"
                ),
                ComponentBottleneck(
                    component_name="TaskDecomposer",
                    time_spent=2.0 + i,
                    percentage_of_total=20.0 + (i * 3),
                    optimization_potential="medium"
                ),
                ComponentBottleneck(
                    component_name="ErrorRecovery",
                    time_spent=1.5 + i,
                    percentage_of_total=15.0 + (i * 2),
                    optimization_potential="low"
                )
            ]
            
            # Create optimization recommendations
            recommendations = [
                OptimizationRecommendation(
                    component="ShellExecutor",
                    issue_description="High subprocess creation overhead",
                    recommendation="Batch commands where possible, use shell command chaining",
                    expected_impact="high",
                    implementation_effort="medium"
                ),
                OptimizationRecommendation(
                    component="TaskDecomposer",
                    issue_description="Redundant task decomposition calls",
                    recommendation="Implement memoization for repeated task patterns",
                    expected_impact="medium",
                    implementation_effort="low"
                )
            ]
            
            # Create bottleneck report
            report = BottleneckReport(
                test_identifier=test_id,
                time_categories=time_categories,
                component_bottlenecks=bottlenecks,
                optimization_recommendations=recommendations
            )
            
            bottleneck_reports.append(report)
        
        return bottleneck_reports
    
    def _create_baseline_report(self) -> str:
        """Create baseline report data for comparison."""
        baseline_data = {
            'metadata': {
                'generation_time': '2024-01-01T10:00:00',
                'total_tests_analyzed': 30,
                'total_execution_time': 150.0
            },
            'timing_analysis': {
                'summary': {
                    'total_tests': 30,
                    'total_time': 150.0,
                    'average_test_time': 5.0,
                    'timeout_count': 1
                }
            },
            'bottleneck_analysis': {
                'component_summary': {
                    'TaskDecomposer': {'total_time': 20.0, 'occurrences': 5},
                    'ShellExecutor': {'total_time': 35.0, 'occurrences': 5},
                    'ErrorRecovery': {'total_time': 15.0, 'occurrences': 3}
                }
            },
            'optimization_recommendations': {
                'total_recommendations': 8,
                'high_priority_count': 3,
                'medium_priority_count': 3,
                'low_priority_count': 2
            }
        }
        
        baseline_path = self.output_dir / "baseline_report_data.json"
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        return str(baseline_path)
    
    def _create_current_report(self) -> str:
        """Create current report data for comparison."""
        current_data = {
            'metadata': {
                'generation_time': '2024-01-15T10:00:00',
                'total_tests_analyzed': 32,
                'total_execution_time': 140.0
            },
            'timing_analysis': {
                'summary': {
                    'total_tests': 32,
                    'total_time': 140.0,
                    'average_test_time': 4.375,
                    'timeout_count': 0
                }
            },
            'bottleneck_analysis': {
                'component_summary': {
                    'TaskDecomposer': {'total_time': 18.0, 'occurrences': 5},
                    'ShellExecutor': {'total_time': 30.0, 'occurrences': 5},
                    'ErrorRecovery': {'total_time': 12.0, 'occurrences': 2}
                }
            },
            'optimization_recommendations': {
                'total_recommendations': 6,
                'high_priority_count': 2,
                'medium_priority_count': 3,
                'low_priority_count': 1
            }
        }
        
        current_path = self.output_dir / "current_report_data.json"
        with open(current_path, 'w') as f:
            json.dump(current_data, f, indent=2)
        
        return str(current_path)
    
    def _create_multiple_reports(self) -> List[str]:
        """Create multiple report data files for trend analysis."""
        reports = []
        
        # Create 5 reports showing improvement trend
        for i in range(5):
            report_data = {
                'metadata': {
                    'generation_time': f'2024-01-{i+1:02d}T10:00:00',
                    'total_tests_analyzed': 30 + i,
                    'total_execution_time': 150.0 - (i * 5.0)  # Improving trend
                },
                'timing_analysis': {
                    'summary': {
                        'total_tests': 30 + i,
                        'total_time': 150.0 - (i * 5.0),
                        'average_test_time': (150.0 - (i * 5.0)) / (30 + i),
                        'timeout_count': max(0, 2 - i)
                    }
                },
                'bottleneck_analysis': {
                    'component_summary': {
                        'TaskDecomposer': {'total_time': 20.0 - i, 'occurrences': 5},
                        'ShellExecutor': {'total_time': 35.0 - (i * 2), 'occurrences': 5},
                        'ErrorRecovery': {'total_time': 15.0 - i, 'occurrences': max(1, 3 - i)}
                    }
                }
            }
            
            report_path = self.output_dir / f"trend_report_{i}_data.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            reports.append(str(report_path))
        
        return reports
    
    def _print_demo_summary(self, report_paths: List[str]):
        """Print summary of generated reports."""
        print("\n" + "="*60)
        print("PERFORMANCE REPORTING DEMO COMPLETED")
        print("="*60)
        print("\nGenerated Reports:")
        
        for path in report_paths:
            print(f"  • {Path(path).name}")
            print(f"    {path}")
        
        print(f"\nAll reports saved to: {self.output_dir}")
        print("\nFeatures Demonstrated:")
        print("  ✓ Comprehensive HTML report generation")
        print("  ✓ Timing analysis with slowest tests identification")
        print("  ✓ Component-level bottleneck analysis")
        print("  ✓ LLM API call profiling integration")
        print("  ✓ Flame graph integration (simulated)")
        print("  ✓ Optimization recommendations")
        print("  ✓ Performance comparison between runs")
        print("  ✓ Multi-report trend analysis")
        print("  ✓ Interactive HTML with charts and tables")
        
        print("\nNext Steps:")
        print("  1. Open the HTML reports in a web browser")
        print("  2. Review the optimization recommendations")
        print("  3. Use comparison reports to track improvements")
        print("  4. Integrate with CI/CD for automated performance monitoring")


async def main():
    """Main demo function."""
    demo = ReportGeneratorDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())