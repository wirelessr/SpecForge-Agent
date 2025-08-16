"""
Comprehensive reporting and visualization for performance analysis results.

This module provides HTML report generation with timing data, profiling results,
flame graph integration, and comparison tools for analyzing multiple profiling runs.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import logging
import base64

try:
    from .models import (
        TimingReport, ProfilingResult, BottleneckReport, TestIdentifier,
        OptimizationRecommendation, ComponentBottleneck, TestTimingResult
    )
    from .llm_profiler import LLMProfilingResult, LLMAPICall
except ImportError:
    # Handle direct execution
    from models import (
        TimingReport, ProfilingResult, BottleneckReport, TestIdentifier,
        OptimizationRecommendation, ComponentBottleneck, TestTimingResult
    )
    from llm_profiler import LLMProfilingResult, LLMAPICall


class PerformanceReportGenerator:
    """
    Generates comprehensive HTML reports with timing data and profiling results.
    
    This class creates detailed performance analysis reports that include:
    - Test timing analysis with slowest tests identification
    - Component-level bottleneck analysis
    - LLM API call profiling results
    - Flame graph integration
    - Optimization recommendations
    - Comparison between multiple profiling runs
    """
    
    def __init__(self, output_dir: str = "artifacts/performance/reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Template for HTML reports
        self.html_template = self._load_html_template()
    
    def generate_comprehensive_report(
        self,
        timing_report: TimingReport,
        profiling_results: List[ProfilingResult],
        bottleneck_reports: List[BottleneckReport],
        report_name: str = None
    ) -> str:
        """
        Generate comprehensive HTML report with all performance data.
        
        Args:
            timing_report: Test timing analysis results
            profiling_results: Detailed profiling results
            bottleneck_reports: Bottleneck analysis results
            report_name: Optional custom name for the report
            
        Returns:
            Path to generated HTML report file
        """
        if not report_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"performance_report_{timestamp}"
        
        self.logger.info(f"Generating comprehensive performance report: {report_name}")
        
        # Prepare report data
        report_data = {
            'metadata': self._generate_report_metadata(timing_report, profiling_results),
            'timing_analysis': self._prepare_timing_data(timing_report),
            'profiling_analysis': self._prepare_profiling_data(profiling_results),
            'bottleneck_analysis': self._prepare_bottleneck_data(bottleneck_reports),
            'optimization_recommendations': self._prepare_recommendations_data(bottleneck_reports),
            'flame_graphs': self._prepare_flame_graph_data(profiling_results),
            'llm_analysis': self._prepare_llm_analysis_data(profiling_results)
        }
        
        # Generate HTML content
        html_content = self._generate_html_report(report_data)
        
        # Save report
        report_path = self.output_dir / f"{report_name}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save raw data as JSON for comparison tools
        json_path = self.output_dir / f"{report_name}_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def generate_comparison_report(
        self,
        baseline_report_path: str,
        current_report_path: str,
        comparison_name: str = None
    ) -> str:
        """
        Generate comparison report between two performance analysis runs.
        
        Args:
            baseline_report_path: Path to baseline report JSON data
            current_report_path: Path to current report JSON data
            comparison_name: Optional name for comparison report
            
        Returns:
            Path to generated comparison HTML report
        """
        if not comparison_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_name = f"performance_comparison_{timestamp}"
        
        self.logger.info(f"Generating performance comparison report: {comparison_name}")
        
        # Load report data
        baseline_data = self._load_report_data(baseline_report_path)
        current_data = self._load_report_data(current_report_path)
        
        # Generate comparison analysis
        comparison_data = {
            'metadata': self._generate_comparison_metadata(baseline_data, current_data),
            'timing_comparison': self._compare_timing_data(baseline_data, current_data),
            'profiling_comparison': self._compare_profiling_data(baseline_data, current_data),
            'bottleneck_comparison': self._compare_bottleneck_data(baseline_data, current_data),
            'recommendation_comparison': self._compare_recommendations(baseline_data, current_data),
            'performance_trends': self._analyze_performance_trends(baseline_data, current_data)
        }
        
        # Generate comparison HTML
        html_content = self._generate_comparison_html(comparison_data)
        
        # Save comparison report
        report_path = self.output_dir / f"{comparison_name}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save comparison data
        json_path = self.output_dir / f"{comparison_name}_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        self.logger.info(f"Comparison report generated: {report_path}")
        return str(report_path)
    
    def _generate_report_metadata(
        self,
        timing_report: TimingReport,
        profiling_results: List[ProfilingResult]
    ) -> Dict[str, Any]:
        """Generate metadata for the performance report."""
        return {
            'generation_time': datetime.now().isoformat(),
            'total_tests_analyzed': timing_report.total_test_count,
            'total_execution_time': timing_report.total_execution_time,
            'profiled_tests_count': len(profiling_results),
            'timeout_threshold': timing_report.timeout_threshold,
            'analysis_timestamp': timing_report.analysis_timestamp
        }
    
    def _prepare_timing_data(self, timing_report: TimingReport) -> Dict[str, Any]:
        """Prepare timing analysis data for report."""
        slowest_tests = timing_report.slowest_tests[:10]  # Top 10 slowest
        timeout_tests = timing_report.timeout_tests
        
        return {
            'summary': {
                'total_tests': timing_report.total_test_count,
                'total_time': timing_report.total_execution_time,
                'average_test_time': timing_report.total_execution_time / max(1, timing_report.total_test_count),
                'timeout_count': len(timeout_tests)
            },
            'slowest_tests': [
                {
                    'test_name': test.test_identifier.full_name,
                    'execution_time': test.execution_time,
                    'status': test.status.value,
                    'timeout_occurred': test.timeout_occurred
                }
                for test in slowest_tests
            ],
            'timeout_tests': [
                {
                    'test_name': test.test_identifier.full_name,
                    'execution_time': test.execution_time,
                    'error_message': test.error_message
                }
                for test in timeout_tests
            ],
            'file_breakdown': [
                {
                    'file_path': file_result.file_path,
                    'total_time': file_result.total_execution_time,
                    'test_count': file_result.test_count,
                    'passed_count': file_result.passed_count,
                    'failed_count': file_result.failed_count,
                    'timeout_count': file_result.timeout_count
                }
                for file_result in timing_report.file_results
            ]
        }
    
    def _prepare_profiling_data(self, profiling_results: List[ProfilingResult]) -> Dict[str, Any]:
        """Prepare profiling analysis data for report."""
        if not profiling_results:
            return {'summary': {}, 'results': []}
        
        total_profiling_time = sum(result.total_profiling_time for result in profiling_results)
        
        return {
            'summary': {
                'profiled_tests': len(profiling_results),
                'total_profiling_time': total_profiling_time,
                'average_profiling_time': total_profiling_time / len(profiling_results),
                'cprofile_results': len([r for r in profiling_results if r.cprofile_result]),
                'pyspy_results': len([r for r in profiling_results if r.pyspy_result])
            },
            'results': [
                {
                    'test_name': result.test_identifier.full_name,
                    'profiler_used': result.profiler_used.value,
                    'total_time': result.total_profiling_time,
                    'component_timings': result.component_timings,
                    'has_cprofile': result.cprofile_result is not None,
                    'has_pyspy': result.pyspy_result is not None,
                    'has_llm_data': result.has_llm_data,
                    'flamegraph_path': result.pyspy_result.flamegraph_path if result.pyspy_result else None
                }
                for result in profiling_results
            ]
        }
    
    def _prepare_bottleneck_data(self, bottleneck_reports: List[BottleneckReport]) -> Dict[str, Any]:
        """Prepare bottleneck analysis data for report."""
        if not bottleneck_reports:
            return {'summary': {}, 'reports': []}
        
        # Aggregate bottleneck data across all reports
        all_bottlenecks = []
        for report in bottleneck_reports:
            all_bottlenecks.extend(report.component_bottlenecks)
        
        # Group bottlenecks by component
        component_bottlenecks = {}
        for bottleneck in all_bottlenecks:
            component = bottleneck.component_name
            if component not in component_bottlenecks:
                component_bottlenecks[component] = {
                    'total_time': 0.0,
                    'occurrences': 0,
                    'avg_percentage': 0.0
                }
            component_bottlenecks[component]['total_time'] += bottleneck.time_spent
            component_bottlenecks[component]['occurrences'] += 1
            component_bottlenecks[component]['avg_percentage'] += bottleneck.percentage_of_total
        
        # Calculate averages
        for component_data in component_bottlenecks.values():
            if component_data['occurrences'] > 0:
                component_data['avg_percentage'] /= component_data['occurrences']
        
        return {
            'summary': {
                'total_reports': len(bottleneck_reports),
                'unique_components': len(component_bottlenecks),
                'total_bottlenecks': len(all_bottlenecks)
            },
            'component_summary': component_bottlenecks,
            'reports': [
                {
                    'test_name': report.test_identifier.full_name,
                    'test_vs_implementation_ratio': report.test_vs_implementation_ratio,
                    'implement_agent_percentage': report.time_categories.component_timings.implement_agent_percentage,
                    'context_loading_performance': report.context_loading_performance,
                    'component_efficiency_metrics': report.component_efficiency_metrics,
                    'bottlenecks': [
                        {
                            'component': bottleneck.component_name,
                            'time_spent': bottleneck.time_spent,
                            'percentage': bottleneck.percentage_of_total,
                            'optimization_potential': bottleneck.optimization_potential,
                            'is_significant': bottleneck.is_significant
                        }
                        for bottleneck in report.top_bottlenecks[:5]  # Top 5 bottlenecks
                    ]
                }
                for report in bottleneck_reports
            ]
        }
    
    def _prepare_recommendations_data(self, bottleneck_reports: List[BottleneckReport]) -> Dict[str, Any]:
        """Prepare optimization recommendations data for report."""
        if not bottleneck_reports:
            return {'summary': {}, 'recommendations': []}
        
        # Collect all recommendations
        all_recommendations = []
        for report in bottleneck_reports:
            all_recommendations.extend(report.optimization_recommendations)
        
        # Group by priority and component
        high_priority = [rec for rec in all_recommendations if rec.expected_impact == "high"]
        medium_priority = [rec for rec in all_recommendations if rec.expected_impact == "medium"]
        low_priority = [rec for rec in all_recommendations if rec.expected_impact == "low"]
        
        # Sort by priority score
        high_priority.sort(key=lambda x: x.priority_score, reverse=True)
        medium_priority.sort(key=lambda x: x.priority_score, reverse=True)
        low_priority.sort(key=lambda x: x.priority_score, reverse=True)
        
        return {
            'summary': {
                'total_recommendations': len(all_recommendations),
                'high_priority_count': len(high_priority),
                'medium_priority_count': len(medium_priority),
                'low_priority_count': len(low_priority)
            },
            'high_priority': [
                {
                    'component': rec.component,
                    'issue': rec.issue_description,
                    'recommendation': rec.recommendation,
                    'expected_impact': rec.expected_impact,
                    'implementation_effort': rec.implementation_effort,
                    'priority_score': rec.priority_score
                }
                for rec in high_priority[:10]  # Top 10 high priority
            ],
            'medium_priority': [
                {
                    'component': rec.component,
                    'issue': rec.issue_description,
                    'recommendation': rec.recommendation,
                    'expected_impact': rec.expected_impact,
                    'implementation_effort': rec.implementation_effort,
                    'priority_score': rec.priority_score
                }
                for rec in medium_priority[:10]  # Top 10 medium priority
            ],
            'low_priority': [
                {
                    'component': rec.component,
                    'issue': rec.issue_description,
                    'recommendation': rec.recommendation,
                    'expected_impact': rec.expected_impact,
                    'implementation_effort': rec.implementation_effort,
                    'priority_score': rec.priority_score
                }
                for rec in low_priority[:5]  # Top 5 low priority
            ]
        }
    
    def _prepare_flame_graph_data(self, profiling_results: List[ProfilingResult]) -> Dict[str, Any]:
        """Prepare flame graph data for report integration."""
        flame_graphs = []
        
        for result in profiling_results:
            if result.pyspy_result and result.pyspy_result.flamegraph_path:
                flamegraph_path = Path(result.pyspy_result.flamegraph_path)
                if flamegraph_path.exists():
                    # Read SVG content for embedding
                    try:
                        with open(flamegraph_path, 'r', encoding='utf-8') as f:
                            svg_content = f.read()
                        
                        flame_graphs.append({
                            'test_name': result.test_identifier.full_name,
                            'svg_content': svg_content,
                            'file_path': str(flamegraph_path),
                            'sampling_duration': result.pyspy_result.sampling_duration,
                            'sample_count': result.pyspy_result.sample_count
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to read flame graph {flamegraph_path}: {e}")
        
        return {
            'available_graphs': len(flame_graphs),
            'graphs': flame_graphs
        }
    
    def _prepare_llm_analysis_data(self, profiling_results: List[ProfilingResult]) -> Dict[str, Any]:
        """Prepare LLM API call analysis data for report."""
        llm_results = [result for result in profiling_results if result.has_llm_data]
        
        if not llm_results:
            return {'summary': {}, 'results': []}
        
        # Aggregate LLM statistics
        total_api_calls = sum(result.llm_profiling_result.total_api_calls for result in llm_results)
        total_llm_time = sum(result.llm_profiling_result.total_llm_time for result in llm_results)
        
        # Collect all API calls for analysis
        all_api_calls = []
        for result in llm_results:
            all_api_calls.extend(result.llm_profiling_result.api_calls)
        
        # Analyze slow calls
        slow_calls = [call for call in all_api_calls if call.is_slow_call]
        
        # Component breakdown
        component_stats = {}
        for result in llm_results:
            for component, count in result.llm_profiling_result.component_call_counts.items():
                if component not in component_stats:
                    component_stats[component] = {'calls': 0, 'time': 0.0}
                component_stats[component]['calls'] += count
                component_stats[component]['time'] += result.llm_profiling_result.component_total_times.get(component, 0.0)
        
        return {
            'summary': {
                'tests_with_llm_data': len(llm_results),
                'total_api_calls': total_api_calls,
                'total_llm_time': total_llm_time,
                'average_call_time': total_llm_time / max(1, total_api_calls),
                'slow_calls_count': len(slow_calls)
            },
            'component_breakdown': component_stats,
            'slow_calls': [
                {
                    'component': call.component,
                    'time': call.total_time,
                    'prompt_size': call.prompt_size_chars,
                    'response_size': call.response_size_chars,
                    'network_time': call.network_time,
                    'processing_time': call.processing_time,
                    'url': call.request_url,
                    'error': call.error_message
                }
                for call in slow_calls[:10]  # Top 10 slow calls
            ],
            'results': [
                {
                    'test_name': result.test_identifier.full_name,
                    'total_calls': result.llm_profiling_result.total_api_calls,
                    'total_time': result.llm_profiling_result.total_llm_time,
                    'average_call_time': result.llm_profiling_result.average_call_time,
                    'slowest_call': result.llm_profiling_result.slowest_call_time,
                    'average_prompt_size': result.llm_profiling_result.average_prompt_size,
                    'network_percentage': result.llm_profiling_result.network_percentage,
                    'component_calls': result.llm_profiling_result.component_call_counts
                }
                for result in llm_results
            ]
        }
    
    def _load_report_data(self, report_path: str) -> Dict[str, Any]:
        """Load report data from JSON file."""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load report data from {report_path}: {e}")
            return {}
    
    def _generate_comparison_metadata(
        self,
        baseline_data: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate metadata for comparison report."""
        return {
            'comparison_time': datetime.now().isoformat(),
            'baseline_timestamp': baseline_data.get('metadata', {}).get('generation_time', 'Unknown'),
            'current_timestamp': current_data.get('metadata', {}).get('generation_time', 'Unknown'),
            'baseline_tests': baseline_data.get('metadata', {}).get('total_tests_analyzed', 0),
            'current_tests': current_data.get('metadata', {}).get('total_tests_analyzed', 0)
        }
    
    def _compare_timing_data(
        self,
        baseline_data: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare timing data between baseline and current runs."""
        baseline_timing = baseline_data.get('timing_analysis', {}).get('summary', {})
        current_timing = current_data.get('timing_analysis', {}).get('summary', {})
        
        # Calculate percentage changes
        def calc_change(baseline_val, current_val):
            if baseline_val == 0:
                return 0.0 if current_val == 0 else float('inf')
            return ((current_val - baseline_val) / baseline_val) * 100
        
        return {
            'total_time_change': calc_change(
                baseline_timing.get('total_time', 0),
                current_timing.get('total_time', 0)
            ),
            'average_time_change': calc_change(
                baseline_timing.get('average_test_time', 0),
                current_timing.get('average_test_time', 0)
            ),
            'timeout_count_change': calc_change(
                baseline_timing.get('timeout_count', 0),
                current_timing.get('timeout_count', 0)
            ),
            'baseline_summary': baseline_timing,
            'current_summary': current_timing
        }
    
    def _compare_profiling_data(
        self,
        baseline_data: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare profiling data between runs."""
        baseline_profiling = baseline_data.get('profiling_analysis', {}).get('summary', {})
        current_profiling = current_data.get('profiling_analysis', {}).get('summary', {})
        
        def calc_change(baseline_val, current_val):
            if baseline_val == 0:
                return 0.0 if current_val == 0 else float('inf')
            return ((current_val - baseline_val) / baseline_val) * 100
        
        return {
            'profiling_time_change': calc_change(
                baseline_profiling.get('total_profiling_time', 0),
                current_profiling.get('total_profiling_time', 0)
            ),
            'average_profiling_time_change': calc_change(
                baseline_profiling.get('average_profiling_time', 0),
                current_profiling.get('average_profiling_time', 0)
            ),
            'baseline_summary': baseline_profiling,
            'current_summary': current_profiling
        }
    
    def _compare_bottleneck_data(
        self,
        baseline_data: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare bottleneck analysis between runs."""
        baseline_bottlenecks = baseline_data.get('bottleneck_analysis', {}).get('component_summary', {})
        current_bottlenecks = current_data.get('bottleneck_analysis', {}).get('component_summary', {})
        
        # Compare component performance
        component_changes = {}
        all_components = set(baseline_bottlenecks.keys()) | set(current_bottlenecks.keys())
        
        for component in all_components:
            baseline_time = baseline_bottlenecks.get(component, {}).get('total_time', 0)
            current_time = current_bottlenecks.get(component, {}).get('total_time', 0)
            
            if baseline_time == 0:
                change = 0.0 if current_time == 0 else float('inf')
            else:
                change = ((current_time - baseline_time) / baseline_time) * 100
            
            component_changes[component] = {
                'baseline_time': baseline_time,
                'current_time': current_time,
                'percentage_change': change,
                'improvement': change < 0
            }
        
        return {
            'component_changes': component_changes,
            'improved_components': [
                comp for comp, data in component_changes.items() 
                if data['improvement'] and abs(data['percentage_change']) > 5
            ],
            'regressed_components': [
                comp for comp, data in component_changes.items() 
                if not data['improvement'] and abs(data['percentage_change']) > 5
            ]
        }
    
    def _compare_recommendations(
        self,
        baseline_data: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare optimization recommendations between runs."""
        baseline_recs = baseline_data.get('optimization_recommendations', {})
        current_recs = current_data.get('optimization_recommendations', {})
        
        return {
            'baseline_high_priority': baseline_recs.get('high_priority_count', 0),
            'current_high_priority': current_recs.get('high_priority_count', 0),
            'baseline_total': baseline_recs.get('total_recommendations', 0),
            'current_total': current_recs.get('total_recommendations', 0),
            'new_recommendations': self._find_new_recommendations(baseline_recs, current_recs),
            'resolved_recommendations': self._find_resolved_recommendations(baseline_recs, current_recs)
        }
    
    def _find_new_recommendations(
        self,
        baseline_recs: Dict[str, Any],
        current_recs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find recommendations that appear in current but not baseline."""
        # This is a simplified implementation - in practice would need more sophisticated matching
        baseline_issues = set()
        for priority in ['high_priority', 'medium_priority', 'low_priority']:
            for rec in baseline_recs.get(priority, []):
                baseline_issues.add(rec.get('issue', ''))
        
        new_recommendations = []
        for priority in ['high_priority', 'medium_priority', 'low_priority']:
            for rec in current_recs.get(priority, []):
                if rec.get('issue', '') not in baseline_issues:
                    new_recommendations.append(rec)
        
        return new_recommendations[:10]  # Limit to 10 new recommendations
    
    def _find_resolved_recommendations(
        self,
        baseline_recs: Dict[str, Any],
        current_recs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find recommendations that appear in baseline but not current."""
        current_issues = set()
        for priority in ['high_priority', 'medium_priority', 'low_priority']:
            for rec in current_recs.get(priority, []):
                current_issues.add(rec.get('issue', ''))
        
        resolved_recommendations = []
        for priority in ['high_priority', 'medium_priority', 'low_priority']:
            for rec in baseline_recs.get(priority, []):
                if rec.get('issue', '') not in current_issues:
                    resolved_recommendations.append(rec)
        
        return resolved_recommendations[:10]  # Limit to 10 resolved recommendations
    
    def _analyze_performance_trends(
        self,
        baseline_data: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance trends between runs."""
        trends = {
            'overall_performance': 'stable',
            'key_improvements': [],
            'key_regressions': [],
            'recommendations': []
        }
        
        # Analyze timing trends
        timing_comparison = self._compare_timing_data(baseline_data, current_data)
        total_time_change = timing_comparison.get('total_time_change', 0)
        
        if total_time_change < -10:
            trends['overall_performance'] = 'improved'
            trends['key_improvements'].append(f"Total execution time improved by {abs(total_time_change):.1f}%")
        elif total_time_change > 10:
            trends['overall_performance'] = 'regressed'
            trends['key_regressions'].append(f"Total execution time regressed by {total_time_change:.1f}%")
        
        # Analyze component trends
        bottleneck_comparison = self._compare_bottleneck_data(baseline_data, current_data)
        improved_components = bottleneck_comparison.get('improved_components', [])
        regressed_components = bottleneck_comparison.get('regressed_components', [])
        
        for component in improved_components[:3]:  # Top 3 improvements
            change = bottleneck_comparison['component_changes'][component]['percentage_change']
            trends['key_improvements'].append(f"{component} performance improved by {abs(change):.1f}%")
        
        for component in regressed_components[:3]:  # Top 3 regressions
            change = bottleneck_comparison['component_changes'][component]['percentage_change']
            trends['key_regressions'].append(f"{component} performance regressed by {change:.1f}%")
        
        # Generate trend-based recommendations
        if len(regressed_components) > len(improved_components):
            trends['recommendations'].append("Focus on addressing component regressions")
        if total_time_change > 5:
            trends['recommendations'].append("Investigate overall performance regression")
        if len(improved_components) > 0:
            trends['recommendations'].append("Document and preserve successful optimizations")
        
        return trends
    
    def _load_html_template(self) -> str:
        """Load HTML template for reports."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #eee; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        .section h3 {{ color: #555; margin-top: 20px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .table {{ width: 100%%; border-collapse: collapse; margin: 15px 0; }}
        .table th, .table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .table th {{ background-color: #f8f9fa; font-weight: bold; }}
        .table tr:hover {{ background-color: #f5f5f5; }}
        .recommendation {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 15px; margin: 10px 0; }}
        .recommendation.high {{ border-left: 4px solid #dc3545; }}
        .recommendation.medium {{ border-left: 4px solid #ffc107; }}
        .recommendation.low {{ border-left: 4px solid #28a745; }}
        .flame-graph {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }}
        .flame-graph-header {{ background: #f8f9fa; padding: 10px; font-weight: bold; }}
        .progress-bar {{ width: 100%%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%%; background: linear-gradient(90deg, #28a745, #ffc107, #dc3545); transition: width 0.3s ease; }}
        .comparison-change {{ font-weight: bold; }}
        .improvement {{ color: #28a745; }}
        .regression {{ color: #dc3545; }}
        .stable {{ color: #6c757d; }}
        .tabs {{ display: flex; border-bottom: 1px solid #ddd; margin-bottom: 20px; }}
        .tab {{ padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; }}
        .tab.active {{ border-bottom-color: #007bff; background: #f8f9fa; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        {content}
    </div>
    
    <script>
        // Tab functionality
        function showTab(tabName) {{
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            document.querySelector(`[onclick="showTab('${{tabName}})"]`).classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }}
        
        // Initialize first tab as active
        document.addEventListener('DOMContentLoaded', function() {{
            const firstTab = document.querySelector('.tab');
            if (firstTab) {{
                firstTab.click();
            }}
        }});
        
        // Chart generation functions
        function createTimingChart(canvasId, data) {{
            const ctx = document.getElementById(canvasId).getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: data.labels,
                    datasets: [{{
                        label: 'Execution Time (seconds)',
                        data: data.values,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{ beginAtZero: true }}
                    }}
                }}
            }});
        }}
        
        function createComponentChart(canvasId, data) {{
            const ctx = document.getElementById(canvasId).getContext('2d');
            new Chart(ctx, {{
                type: 'pie',
                data: {{
                    labels: data.labels,
                    datasets: [{{
                        data: data.values,
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'
                        ]
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ position: 'bottom' }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
        '''
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML content for performance report."""
        metadata = report_data['metadata']
        timing_data = report_data['timing_analysis']
        profiling_data = report_data['profiling_analysis']
        bottleneck_data = report_data['bottleneck_analysis']
        recommendations_data = report_data['optimization_recommendations']
        flame_graph_data = report_data['flame_graphs']
        llm_data = report_data['llm_analysis']
        
        content = f'''
        <div class="header">
            <h1>Performance Analysis Report</h1>
            <p>Generated: {metadata['generation_time']}</p>
            <p>Tests Analyzed: {metadata['total_tests_analyzed']} | Execution Time: {metadata['total_execution_time']:.2f}s</p>
        </div>
        
        <div class="tabs">
            <div class="tab" onclick="showTab('overview')">Overview</div>
            <div class="tab" onclick="showTab('timing')">Timing Analysis</div>
            <div class="tab" onclick="showTab('profiling')">Profiling Results</div>
            <div class="tab" onclick="showTab('bottlenecks')">Bottleneck Analysis</div>
            <div class="tab" onclick="showTab('recommendations')">Recommendations</div>
            <div class="tab" onclick="showTab('flamegraphs')">Flame Graphs</div>
            <div class="tab" onclick="showTab('llm')">LLM Analysis</div>
        </div>
        
        <div id="overview" class="tab-content">
            {self._generate_overview_section(metadata, timing_data, profiling_data)}
        </div>
        
        <div id="timing" class="tab-content">
            {self._generate_timing_section(timing_data)}
        </div>
        
        <div id="profiling" class="tab-content">
            {self._generate_profiling_section(profiling_data)}
        </div>
        
        <div id="bottlenecks" class="tab-content">
            {self._generate_bottleneck_section(bottleneck_data)}
        </div>
        
        <div id="recommendations" class="tab-content">
            {self._generate_recommendations_section(recommendations_data)}
        </div>
        
        <div id="flamegraphs" class="tab-content">
            {self._generate_flamegraph_section(flame_graph_data)}
        </div>
        
        <div id="llm" class="tab-content">
            {self._generate_llm_section(llm_data)}
        </div>
        '''
        
        return self.html_template.format(content=content)
    
    def _generate_comparison_html(self, comparison_data: Dict[str, Any]) -> str:
        """Generate HTML content for comparison report."""
        metadata = comparison_data['metadata']
        timing_comparison = comparison_data['timing_comparison']
        bottleneck_comparison = comparison_data['bottleneck_comparison']
        trends = comparison_data['performance_trends']
        
        content = f'''
        <div class="header">
            <h1>Performance Comparison Report</h1>
            <p>Generated: {metadata['comparison_time']}</p>
            <p>Baseline: {metadata['baseline_timestamp']} | Current: {metadata['current_timestamp']}</p>
        </div>
        
        <div class="section">
            <h2>Performance Trends</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value {self._get_trend_class(trends['overall_performance'])}">{trends['overall_performance'].title()}</div>
                    <div class="metric-label">Overall Performance</div>
                </div>
            </div>
            
            <h3>Key Improvements</h3>
            <ul>
                {self._generate_list_items(trends['key_improvements'])}
            </ul>
            
            <h3>Key Regressions</h3>
            <ul>
                {self._generate_list_items(trends['key_regressions'])}
            </ul>
            
            <h3>Recommendations</h3>
            <ul>
                {self._generate_list_items(trends['recommendations'])}
            </ul>
        </div>
        
        <div class="section">
            <h2>Timing Comparison</h2>
            {self._generate_timing_comparison_section(timing_comparison)}
        </div>
        
        <div class="section">
            <h2>Component Performance Changes</h2>
            {self._generate_component_comparison_section(bottleneck_comparison)}
        </div>
        '''
        
        return self.html_template.format(content=content)
    
    def _generate_overview_section(
        self,
        metadata: Dict[str, Any],
        timing_data: Dict[str, Any],
        profiling_data: Dict[str, Any]
    ) -> str:
        """Generate overview section HTML."""
        timing_summary = timing_data.get('summary', {})
        profiling_summary = profiling_data.get('summary', {})
        
        return f'''
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{timing_summary.get('total_tests', 0)}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{timing_summary.get('total_time', 0):.1f}s</div>
                    <div class="metric-label">Total Execution Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{timing_summary.get('average_test_time', 0):.2f}s</div>
                    <div class="metric-label">Average Test Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{timing_summary.get('timeout_count', 0)}</div>
                    <div class="metric-label">Timeout Tests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{profiling_summary.get('profiled_tests', 0)}</div>
                    <div class="metric-label">Profiled Tests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{profiling_summary.get('total_profiling_time', 0):.1f}s</div>
                    <div class="metric-label">Total Profiling Time</div>
                </div>
            </div>
        </div>
        '''    

    def _generate_timing_section(self, timing_data: Dict[str, Any]) -> str:
        """Generate timing analysis section HTML."""
        slowest_tests = timing_data.get('slowest_tests', [])
        timeout_tests = timing_data.get('timeout_tests', [])
        
        slowest_table = self._generate_table(
            ['Test Name', 'Execution Time', 'Status'],
            [[test['test_name'], f"{test['execution_time']:.2f}s", test['status']] for test in slowest_tests[:10]]
        )
        
        timeout_table = ""
        if timeout_tests:
            timeout_table = f'''
            <h3>Timeout Tests</h3>
            {self._generate_table(
                ['Test Name', 'Execution Time', 'Error Message'],
                [[test['test_name'], f"{test['execution_time']:.2f}s", test.get('error_message', 'N/A')] for test in timeout_tests]
            )}
            '''
        
        return f'''
        <div class="section">
            <h2>Timing Analysis</h2>
            <canvas id="timingChart" width="400" height="200"></canvas>
            
            <h3>Slowest Tests</h3>
            {slowest_table}
            
            {timeout_table}
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const timingData = {{
                    labels: """ + json.dumps([test['test_name'].split('::')[-1] for test in slowest_tests[:10]]) + """,
                    values: """ + json.dumps([test['execution_time'] for test in slowest_tests[:10]]) + """
                }};
                createTimingChart('timingChart', timingData);
            }});
        </script>
        '''
    
    def _generate_profiling_section(self, profiling_data: Dict[str, Any]) -> str:
        """Generate profiling results section HTML."""
        summary = profiling_data.get('summary', {})
        results = profiling_data.get('results', [])
        
        results_table = self._generate_table(
            ['Test Name', 'Profiler', 'Total Time', 'Has cProfile', 'Has py-spy', 'Has LLM Data'],
            [
                [
                    result['test_name'],
                    result['profiler_used'],
                    f"{result['total_time']:.2f}s",
                    '✓' if result['has_cprofile'] else '✗',
                    '✓' if result['has_pyspy'] else '✗',
                    '✓' if result['has_llm_data'] else '✗'
                ]
                for result in results
            ]
        )
        
        return f'''
        <div class="section">
            <h2>Profiling Results</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary.get('profiled_tests', 0)}</div>
                    <div class="metric-label">Profiled Tests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_profiling_time', 0):.1f}s</div>
                    <div class="metric-label">Total Profiling Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('cprofile_results', 0)}</div>
                    <div class="metric-label">cProfile Results</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('pyspy_results', 0)}</div>
                    <div class="metric-label">py-spy Results</div>
                </div>
            </div>
            
            <h3>Profiling Results by Test</h3>
            {results_table}
        </div>
        '''
    
    def _generate_bottleneck_section(self, bottleneck_data: Dict[str, Any]) -> str:
        """Generate bottleneck analysis section HTML."""
        summary = bottleneck_data.get('summary', {})
        component_summary = bottleneck_data.get('component_summary', {})
        reports = bottleneck_data.get('reports', [])
        
        # Component summary table
        component_table = self._generate_table(
            ['Component', 'Total Time', 'Occurrences', 'Avg Percentage'],
            [
                [
                    component,
                    f"{data['total_time']:.2f}s",
                    str(data['occurrences']),
                    f"{data['avg_percentage']:.1f}%"
                ]
                for component, data in component_summary.items()
            ]
        )
        
        # Individual test bottlenecks
        test_bottlenecks = ""
        for report in reports[:5]:  # Show top 5 tests
            bottlenecks_list = ""
            for bottleneck in report['bottlenecks']:
                significance = "🔴" if bottleneck['is_significant'] else "🟡"
                bottlenecks_list += f'''
                <li>{significance} {bottleneck['component']}: {bottleneck['time_spent']:.2f}s ({bottleneck['percentage']:.1f}%)</li>
                '''
            
            test_bottlenecks += f'''
            <div class="recommendation">
                <h4>{report['test_name']}</h4>
                <p>ImplementAgent: {report['implement_agent_percentage']:.1f}% | Test Ratio: {report['test_vs_implementation_ratio']:.2f}</p>
                <ul>{bottlenecks_list}</ul>
            </div>
            '''
        
        return f'''
        <div class="section">
            <h2>Bottleneck Analysis</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_reports', 0)}</div>
                    <div class="metric-label">Analyzed Tests</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('unique_components', 0)}</div>
                    <div class="metric-label">Components Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_bottlenecks', 0)}</div>
                    <div class="metric-label">Total Bottlenecks</div>
                </div>
            </div>
            
            <h3>Component Performance Summary</h3>
            <canvas id="componentChart" width="400" height="200"></canvas>
            {component_table}
            
            <h3>Test-Specific Bottlenecks</h3>
            {test_bottlenecks}
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const componentData = {{
                    labels: """ + json.dumps(list(component_summary.keys())) + """,
                    values: """ + json.dumps([data['total_time'] for data in component_summary.values()]) + """
                }};
                createComponentChart('componentChart', componentData);
            }});
        </script>
        '''
    
    def _generate_recommendations_section(self, recommendations_data: Dict[str, Any]) -> str:
        """Generate optimization recommendations section HTML."""
        summary = recommendations_data.get('summary', {})
        high_priority = recommendations_data.get('high_priority', [])
        medium_priority = recommendations_data.get('medium_priority', [])
        low_priority = recommendations_data.get('low_priority', [])
        
        def format_recommendations(recs, priority_class):
            html = ""
            for rec in recs:
                html += f'''
                <div class="recommendation {priority_class}">
                    <h4>{rec['component']} - {rec['expected_impact'].title()} Impact</h4>
                    <p><strong>Issue:</strong> {rec['issue']}</p>
                    <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                    <p><strong>Effort:</strong> {rec['implementation_effort'].title()} | <strong>Priority Score:</strong> {rec['priority_score']:.1f}</p>
                </div>
                '''
            return html
        
        return f'''
        <div class="section">
            <h2>Optimization Recommendations</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_recommendations', 0)}</div>
                    <div class="metric-label">Total Recommendations</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('high_priority_count', 0)}</div>
                    <div class="metric-label">High Priority</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('medium_priority_count', 0)}</div>
                    <div class="metric-label">Medium Priority</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('low_priority_count', 0)}</div>
                    <div class="metric-label">Low Priority</div>
                </div>
            </div>
            
            <h3>High Priority Recommendations</h3>
            {format_recommendations(high_priority, 'high')}
            
            <h3>Medium Priority Recommendations</h3>
            {format_recommendations(medium_priority, 'medium')}
            
            <h3>Low Priority Recommendations</h3>
            {format_recommendations(low_priority, 'low')}
        </div>
        '''
    
    def _generate_flamegraph_section(self, flame_graph_data: Dict[str, Any]) -> str:
        """Generate flame graph section HTML."""
        graphs = flame_graph_data.get('graphs', [])
        
        if not graphs:
            return '''
            <div class="section">
                <h2>Flame Graphs</h2>
                <p>No flame graphs available. Ensure py-spy profiling is enabled and successful.</p>
            </div>
            '''
        
        flame_graphs_html = ""
        for graph in graphs:
            flame_graphs_html += f'''
            <div class="flame-graph">
                <div class="flame-graph-header">
                    {graph['test_name']} - Duration: {graph['sampling_duration']:.1f}s, Samples: {graph['sample_count']}
                </div>
                <div style="max-height: 600px; overflow: auto;">
                    {graph['svg_content']}
                </div>
            </div>
            '''
        
        return f'''
        <div class="section">
            <h2>Flame Graphs</h2>
            <p>Interactive flame graphs showing call stack sampling data. Wider bars indicate more time spent in functions.</p>
            {flame_graphs_html}
        </div>
        '''
    
    def _generate_llm_section(self, llm_data: Dict[str, Any]) -> str:
        """Generate LLM analysis section HTML."""
        summary = llm_data.get('summary', {})
        component_breakdown = llm_data.get('component_breakdown', {})
        slow_calls = llm_data.get('slow_calls', [])
        results = llm_data.get('results', [])
        
        if summary.get('tests_with_llm_data', 0) == 0:
            return '''
            <div class="section">
                <h2>LLM API Analysis</h2>
                <p>No LLM API call data available. Ensure LLM profiling is enabled during test execution.</p>
            </div>
            '''
        
        # Component breakdown table
        component_table = self._generate_table(
            ['Component', 'API Calls', 'Total Time', 'Avg Time per Call'],
            [
                [
                    component,
                    str(data['calls']),
                    f"{data['time']:.2f}s",
                    f"{data['time'] / max(1, data['calls']):.2f}s"
                ]
                for component, data in component_breakdown.items()
            ]
        )
        
        # Slow calls table
        slow_calls_table = ""
        if slow_calls:
            slow_calls_table = f'''
            <h3>Slow API Calls (>5s)</h3>
            {self._generate_table(
                ['Component', 'Total Time', 'Network Time', 'Processing Time', 'Prompt Size'],
                [
                    [
                        call['component'],
                        f"{call['time']:.2f}s",
                        f"{call['network_time']:.2f}s",
                        f"{call['processing_time']:.2f}s",
                        f"{call['prompt_size']} chars"
                    ]
                    for call in slow_calls
                ]
            )}
            '''
        
        return f'''
        <div class="section">
            <h2>LLM API Analysis</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_api_calls', 0)}</div>
                    <div class="metric-label">Total API Calls</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('total_llm_time', 0):.1f}s</div>
                    <div class="metric-label">Total LLM Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('average_call_time', 0):.2f}s</div>
                    <div class="metric-label">Average Call Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary.get('slow_calls_count', 0)}</div>
                    <div class="metric-label">Slow Calls</div>
                </div>
            </div>
            
            <h3>Component Breakdown</h3>
            {component_table}
            
            {slow_calls_table}
        </div>
        '''
    
    def _generate_timing_comparison_section(self, timing_comparison: Dict[str, Any]) -> str:
        """Generate timing comparison section HTML."""
        baseline = timing_comparison.get('baseline_summary', {})
        current = timing_comparison.get('current_summary', {})
        
        def format_change(change_pct):
            if change_pct == 0:
                return '<span class="stable">No change</span>'
            elif change_pct < 0:
                return f'<span class="improvement">↓ {abs(change_pct):.1f}%</span>'
            else:
                return f'<span class="regression">↑ {change_pct:.1f}%</span>'
        
        return f'''
        <table class="table">
            <tr>
                <th>Metric</th>
                <th>Baseline</th>
                <th>Current</th>
                <th>Change</th>
            </tr>
            <tr>
                <td>Total Time</td>
                <td>{baseline.get('total_time', 0):.2f}s</td>
                <td>{current.get('total_time', 0):.2f}s</td>
                <td>{format_change(timing_comparison.get('total_time_change', 0))}</td>
            </tr>
            <tr>
                <td>Average Test Time</td>
                <td>{baseline.get('average_test_time', 0):.2f}s</td>
                <td>{current.get('average_test_time', 0):.2f}s</td>
                <td>{format_change(timing_comparison.get('average_time_change', 0))}</td>
            </tr>
            <tr>
                <td>Timeout Count</td>
                <td>{baseline.get('timeout_count', 0)}</td>
                <td>{current.get('timeout_count', 0)}</td>
                <td>{format_change(timing_comparison.get('timeout_count_change', 0))}</td>
            </tr>
        </table>
        '''
    
    def _generate_component_comparison_section(self, bottleneck_comparison: Dict[str, Any]) -> str:
        """Generate component comparison section HTML."""
        component_changes = bottleneck_comparison.get('component_changes', {})
        
        def format_change(change_pct):
            if abs(change_pct) < 5:
                return '<span class="stable">Stable</span>'
            elif change_pct < 0:
                return f'<span class="improvement">↓ {abs(change_pct):.1f}%</span>'
            else:
                return f'<span class="regression">↑ {change_pct:.1f}%</span>'
        
        rows = []
        for component, data in component_changes.items():
            rows.append([
                component,
                f"{data['baseline_time']:.2f}s",
                f"{data['current_time']:.2f}s",
                format_change(data['percentage_change'])
            ])
        
        return self._generate_table(
            ['Component', 'Baseline Time', 'Current Time', 'Change'],
            rows
        )
    
    def _generate_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Generate HTML table from headers and rows."""
        header_html = ''.join(f'<th>{header}</th>' for header in headers)
        rows_html = ''
        
        for row in rows:
            row_html = ''.join(f'<td>{cell}</td>' for cell in row)
            rows_html += f'<tr>{row_html}</tr>'
        
        return f'''
        <table class="table">
            <tr>{header_html}</tr>
            {rows_html}
        </table>
        '''
    
    def _generate_list_items(self, items: List[str]) -> str:
        """Generate HTML list items from list of strings."""
        if not items:
            return '<li>None</li>'
        return ''.join(f'<li>{item}</li>' for item in items)
    
    def _get_trend_class(self, trend: str) -> str:
        """Get CSS class for trend indication."""
        if trend == 'improved':
            return 'improvement'
        elif trend == 'regressed':
            return 'regression'
        else:
            return 'stable'


class ReportComparisonTool:
    """
    Tool for comparing multiple performance analysis runs.
    
    This class provides utilities for analyzing performance trends across
    multiple profiling runs and identifying patterns in optimization efforts.
    """
    
    def __init__(self, reports_dir: str = "artifacts/performance/reports"):
        """
        Initialize comparison tool.
        
        Args:
            reports_dir: Directory containing performance reports
        """
        self.reports_dir = Path(reports_dir)
        self.logger = logging.getLogger(__name__)
    
    def compare_multiple_reports(
        self,
        report_paths: List[str],
        output_name: str = None
    ) -> str:
        """
        Compare multiple performance reports to identify trends.
        
        Args:
            report_paths: List of paths to report JSON data files
            output_name: Optional name for output comparison report
            
        Returns:
            Path to generated multi-comparison report
        """
        if len(report_paths) < 2:
            raise ValueError("At least 2 reports required for comparison")
        
        if not output_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"multi_comparison_{timestamp}"
        
        self.logger.info(f"Comparing {len(report_paths)} performance reports")
        
        # Load all report data
        reports_data = []
        for path in report_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_source_path'] = path
                    reports_data.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to load report {path}: {e}")
        
        if len(reports_data) < 2:
            raise ValueError("Failed to load sufficient reports for comparison")
        
        # Analyze trends across all reports
        trend_analysis = self._analyze_multi_report_trends(reports_data)
        
        # Generate comparison HTML
        html_content = self._generate_multi_comparison_html(trend_analysis)
        
        # Save report
        output_path = self.reports_dir / f"{output_name}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save trend data
        json_path = self.reports_dir / f"{output_name}_trends.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(trend_analysis, f, indent=2, default=str)
        
        self.logger.info(f"Multi-comparison report generated: {output_path}")
        return str(output_path)
    
    def _analyze_multi_report_trends(self, reports_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends across multiple reports."""
        # Sort reports by timestamp
        reports_data.sort(key=lambda x: x.get('metadata', {}).get('generation_time', ''))
        
        # Extract time series data
        timestamps = []
        total_times = []
        test_counts = []
        component_trends = {}
        
        for report in reports_data:
            metadata = report.get('metadata', {})
            timing = report.get('timing_analysis', {}).get('summary', {})
            bottlenecks = report.get('bottleneck_analysis', {}).get('component_summary', {})
            
            timestamps.append(metadata.get('generation_time', ''))
            total_times.append(timing.get('total_time', 0))
            test_counts.append(timing.get('total_tests', 0))
            
            # Track component performance over time
            for component, data in bottlenecks.items():
                if component not in component_trends:
                    component_trends[component] = []
                component_trends[component].append(data.get('total_time', 0))
        
        # Calculate trend statistics
        trend_stats = {
            'report_count': len(reports_data),
            'time_range': {
                'start': timestamps[0] if timestamps else '',
                'end': timestamps[-1] if timestamps else ''
            },
            'performance_trend': self._calculate_trend_direction(total_times),
            'test_count_trend': self._calculate_trend_direction(test_counts),
            'component_trends': {}
        }
        
        # Analyze component trends
        for component, values in component_trends.items():
            trend_stats['component_trends'][component] = {
                'trend_direction': self._calculate_trend_direction(values),
                'values': values,
                'improvement_percentage': self._calculate_improvement_percentage(values)
            }
        
        return {
            'metadata': {
                'analysis_time': datetime.now().isoformat(),
                'reports_analyzed': len(reports_data),
                'source_reports': [report['_source_path'] for report in reports_data]
            },
            'trend_statistics': trend_stats,
            'time_series_data': {
                'timestamps': timestamps,
                'total_times': total_times,
                'test_counts': test_counts,
                'component_trends': component_trends
            }
        }
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate overall trend direction from list of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        first_half_avg = sum(values[:len(values)//2]) / max(1, len(values)//2)
        second_half_avg = sum(values[len(values)//2:]) / max(1, len(values) - len(values)//2)
        
        change_pct = ((second_half_avg - first_half_avg) / max(0.001, first_half_avg)) * 100
        
        if abs(change_pct) < 5:
            return 'stable'
        elif change_pct < 0:
            return 'improving'
        else:
            return 'degrading'
    
    def _calculate_improvement_percentage(self, values: List[float]) -> float:
        """Calculate improvement percentage from first to last value."""
        if len(values) < 2 or values[0] == 0:
            return 0.0
        
        return ((values[0] - values[-1]) / values[0]) * 100
    
    def _generate_multi_comparison_html(self, trend_analysis: Dict[str, Any]) -> str:
        """Generate HTML for multi-report comparison."""
        # This would generate a comprehensive HTML report with trend charts
        # For brevity, returning a simplified version
        metadata = trend_analysis['metadata']
        trend_stats = trend_analysis['trend_statistics']
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Report Performance Trends</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .trend-improving {{ color: #28a745; }}
                .trend-degrading {{ color: #dc3545; }}
                .trend-stable {{ color: #6c757d; }}
            </style>
        </head>
        <body>
            <h1>Performance Trends Analysis</h1>
            <p>Analysis Time: {metadata['analysis_time']}</p>
            <p>Reports Analyzed: {metadata['reports_analyzed']}</p>
            
            <h2>Overall Trends</h2>
            <p>Performance Trend: <span class="trend-{trend_stats['performance_trend']}">{trend_stats['performance_trend'].title()}</span></p>
            <p>Test Count Trend: <span class="trend-{trend_stats['test_count_trend']}">{trend_stats['test_count_trend'].title()}</span></p>
            
            <h2>Component Trends</h2>
            <ul>
        '''
        
        for component, data in trend_stats['component_trends'].items():
            trend_class = f"trend-{data['trend_direction']}"
            html_content += f'''
                <li>{component}: <span class="{trend_class}">{data['trend_direction'].title()}</span> 
                    ({data['improvement_percentage']:.1f}% change)</li>
            '''
        
        html_content += '''
            </ul>
        </body>
        </html>
        '''
        
        return html_content