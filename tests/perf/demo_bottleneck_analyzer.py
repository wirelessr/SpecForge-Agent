#!/usr/bin/env python3
"""
Demo script for BottleneckAnalyzer functionality.

This script demonstrates how to use the BottleneckAnalyzer to analyze
profiling results and generate optimization recommendations.
"""

import asyncio
import logging
from pathlib import Path

from .bottleneck_analyzer import BottleneckAnalyzer
from .models import (
    TestIdentifier, CProfileResult, ProfilingResult, ProfilerType
)


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_profiling_data():
    """Create sample profiling data for demonstration."""
    # Create test identifier
    test_identifier = TestIdentifier(
        file_path="tests/integration/test_implement_agent_integration.py",
        class_name="TestImplementAgentIntegration", 
        method_name="test_execute_complex_task"
    )
    
    # Create mock cProfile data representing a slow ImplementAgent execution
    function_stats = {
        # ImplementAgent main execution
        'autogen_framework/agents/implement_agent.py:120(execute_task)': {
            'cumulative_time': 8.5,
            'total_time': 0.5,
            'call_count': 1,
            'filename': 'autogen_framework/agents/implement_agent.py',
            'line_number': 120,
            'function_name': 'execute_task'
        },
        
        # TaskDecomposer - significant bottleneck
        'autogen_framework/agents/task_decomposer.py:45(decompose_task)': {
            'cumulative_time': 6.2,
            'total_time': 2.1,
            'call_count': 3,
            'filename': 'autogen_framework/agents/task_decomposer.py',
            'line_number': 45,
            'function_name': 'decompose_task'
        },
        
        # LLM API calls - major bottleneck
        'openai/api_resources/chat_completion.py:200(create)': {
            'cumulative_time': 12.3,
            'total_time': 11.8,
            'call_count': 8,
            'filename': 'openai/api_resources/chat_completion.py',
            'line_number': 200,
            'function_name': 'create'
        },
        
        # ShellExecutor - moderate usage
        'autogen_framework/shell_executor.py:80(execute_command)': {
            'cumulative_time': 3.4,
            'total_time': 2.1,
            'call_count': 15,
            'filename': 'autogen_framework/shell_executor.py',
            'line_number': 80,
            'function_name': 'execute_command'
        },
        
        # ErrorRecovery - triggered multiple times
        'autogen_framework/agents/error_recovery.py:60(recover_from_error)': {
            'cumulative_time': 4.7,
            'total_time': 1.2,
            'call_count': 5,
            'filename': 'autogen_framework/agents/error_recovery.py',
            'line_number': 60,
            'function_name': 'recover_from_error'
        },
        
        # ContextManager - loading context
        'autogen_framework/context_manager.py:150(get_implementation_context)': {
            'cumulative_time': 2.8,
            'total_time': 1.9,
            'call_count': 2,
            'filename': 'autogen_framework/context_manager.py',
            'line_number': 150,
            'function_name': 'get_implementation_context'
        },
        
        # Additional ContextManager functions for Task 7 demonstration
        'autogen_framework/context_manager.py:80(load_context)': {
            'cumulative_time': 1.5,
            'total_time': 1.2,
            'call_count': 3,
            'filename': 'autogen_framework/context_manager.py',
            'line_number': 80,
            'function_name': 'load_context'
        },
        
        'autogen_framework/context_manager.py:200(compress_context)': {
            'cumulative_time': 0.8,
            'total_time': 0.6,
            'call_count': 2,
            'filename': 'autogen_framework/context_manager.py',
            'line_number': 200,
            'function_name': 'compress_context'
        },
        
        'autogen_framework/context_manager.py:120(initialize_context)': {
            'cumulative_time': 0.5,
            'total_time': 0.4,
            'call_count': 1,
            'filename': 'autogen_framework/context_manager.py',
            'line_number': 120,
            'function_name': 'initialize_context'
        },
        
        # Test infrastructure overhead
        'tests/integration/conftest.py:25(setup_integration_test)': {
            'cumulative_time': 1.5,
            'total_time': 1.2,
            'call_count': 1,
            'filename': 'tests/integration/conftest.py',
            'line_number': 25,
            'function_name': 'setup_integration_test'
        },
        
        'pytest/_pytest/fixtures.py:500(call_fixture_func)': {
            'cumulative_time': 0.8,
            'total_time': 0.3,
            'call_count': 12,
            'filename': 'pytest/_pytest/fixtures.py',
            'line_number': 500,
            'function_name': 'call_fixture_func'
        },
        
        # Subprocess calls from ShellExecutor
        'subprocess.py:350(run)': {
            'cumulative_time': 2.1,
            'total_time': 1.8,
            'call_count': 15,
            'filename': 'subprocess.py',
            'line_number': 350,
            'function_name': 'run'
        }
    }
    
    # Create top functions list
    top_functions = [
        {
            'name': 'openai/api_resources/chat_completion.py:200(create)',
            'cumulative_time': 12.3,
            'total_time': 11.8,
            'call_count': 8,
            'percentage': 29.8
        },
        {
            'name': 'autogen_framework/agents/implement_agent.py:120(execute_task)',
            'cumulative_time': 8.5,
            'total_time': 0.5,
            'call_count': 1,
            'percentage': 20.6
        },
        {
            'name': 'autogen_framework/agents/task_decomposer.py:45(decompose_task)',
            'cumulative_time': 6.2,
            'total_time': 2.1,
            'call_count': 3,
            'percentage': 15.0
        }
    ]
    
    # Create cProfile result
    cprofile_result = CProfileResult(
        test_identifier=test_identifier,
        profile_file_path="/tmp/demo_profile.prof",
        total_time=41.3,  # Total execution time
        function_stats=function_stats,
        top_functions=top_functions,
        call_count=65
    )
    
    # Create profiling result
    profiling_result = ProfilingResult(
        test_identifier=test_identifier,
        profiler_used=ProfilerType.CPROFILE,
        cprofile_result=cprofile_result,
        total_profiling_time=42.1,
        profiling_overhead=0.8
    )
    
    return [profiling_result]


def print_bottleneck_report(report):
    """Print a formatted bottleneck report."""
    print(f"\n{'='*80}")
    print(f"BOTTLENECK ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"Test: {report.test_identifier.full_name}")
    print(f"Analysis Time: {report.analysis_timestamp}")
    
    # Time categories
    print(f"\n{'Time Categories':<30}")
    print(f"{'-'*50}")
    tc = report.time_categories
    print(f"{'Test Setup Time:':<30} {tc.test_setup_time:.2f}s")
    print(f"{'Test Teardown Time:':<30} {tc.test_teardown_time:.2f}s")
    print(f"{'ImplementAgent Time:':<30} {tc.implement_agent_time:.2f}s")
    print(f"{'Total Time:':<30} {tc.total_time:.2f}s")
    
    # Component timings
    print(f"\n{'Component Breakdown':<30}")
    print(f"{'-'*50}")
    ct = tc.component_timings
    print(f"{'TaskDecomposer:':<30} {ct.task_decomposer:.2f}s")
    print(f"{'ShellExecutor:':<30} {ct.shell_executor:.2f}s")
    print(f"{'ErrorRecovery:':<30} {ct.error_recovery:.2f}s")
    print(f"{'ContextManager:':<30} {ct.context_manager:.2f}s")
    print(f"{'LLM API Calls:':<30} {ct.llm_api_calls:.2f}s")
    print(f"{'Test Overhead:':<30} {ct.test_overhead:.2f}s")
    print(f"{'Other:':<30} {ct.other:.2f}s")
    print(f"{'ImplementAgent %:':<30} {ct.implement_agent_percentage:.1f}%")
    
    # Task 7 Enhancement: Detailed Component Analysis
    if hasattr(report, 'detailed_component_timings') and report.detailed_component_timings:
        print(f"\n{'DETAILED COMPONENT ANALYSIS (Task 7)':<30}")
        print(f"{'='*80}")
        
        for component, data in report.detailed_component_timings.items():
            if data['total_time'] > 0:
                print(f"\n{component}:")
                print(f"  Total Time: {data['total_time']:.3f}s")
                print(f"  Call Count: {data['call_count']}")
                print(f"  Avg Call Time: {data['avg_call_time']:.3f}s")
                print(f"  Max Single Call: {data['max_single_call']:.3f}s")
                
                # Show phase breakdown
                if component == 'TaskDecomposer' and data['decomposition_phases']:
                    print(f"  Decomposition Phases:")
                    for phase, time in data['decomposition_phases'].items():
                        print(f"    {phase}: {time:.3f}s")
                elif component == 'ShellExecutor' and data['execution_phases']:
                    print(f"  Execution Phases:")
                    for phase, time in data['execution_phases'].items():
                        print(f"    {phase}: {time:.3f}s")
                elif component == 'ErrorRecovery' and data['recovery_phases']:
                    print(f"  Recovery Phases:")
                    for phase, time in data['recovery_phases'].items():
                        print(f"    {phase}: {time:.3f}s")
                elif component == 'ContextManager' and data['context_phases']:
                    print(f"  Context Phases:")
                    for phase, time in data['context_phases'].items():
                        print(f"    {phase}: {time:.3f}s")
                    print(f"  Context Loading Time: {data['context_loading_time']:.3f}s")
                
                # Show top functions
                if data['functions']:
                    print(f"  Top Functions:")
                    sorted_functions = sorted(
                        data['functions'].items(),
                        key=lambda x: x[1]['cumulative_time'],
                        reverse=True
                    )[:3]
                    for func_name, func_data in sorted_functions:
                        print(f"    {func_data['function_name']}: {func_data['cumulative_time']:.3f}s ({func_data['call_count']} calls)")
    
    # Context Loading Performance (Task 7 requirement)
    if hasattr(report, 'context_loading_performance'):
        context_perf = report.context_loading_performance
        print(f"\n{'Context Loading Performance':<30}")
        print(f"{'-'*50}")
        print(f"{'Context Loading Time:':<30} {context_perf['context_loading_time']:.3f}s")
        print(f"{'% of ContextManager:':<30} {context_perf['percentage_of_context_manager']:.1f}%")
        print(f"{'Total ContextManager Time:':<30} {context_perf['total_context_manager_time']:.3f}s")
    
    # Component Efficiency Metrics (Task 7 requirement)
    if hasattr(report, 'component_efficiency_metrics'):
        efficiency = report.component_efficiency_metrics
        if efficiency:
            print(f"\n{'Component Efficiency Metrics':<30}")
            print(f"{'-'*50}")
            for component, metrics in efficiency.items():
                print(f"{component}:")
                print(f"  Calls/sec: {metrics['calls_per_second']:.2f}")
                print(f"  Avg Call Efficiency: {metrics['avg_call_efficiency']:.3f}s")
                print(f"  Efficiency Ratio: {metrics['efficiency_ratio']:.2f}")
    
    # Performance Summary (Task 7 requirement)
    if hasattr(report, 'component_performance_summary') and report.component_performance_summary:
        summary = report.component_performance_summary
        print(f"\n{'Performance Summary':<30}")
        print(f"{'-'*50}")
        print(f"Total Component Time: {summary['total_component_time']:.3f}s")
        
        if summary['performance_insights']:
            print(f"\nPerformance Insights:")
            for insight in summary['performance_insights']:
                print(f"  • {insight}")
        
        if summary['optimization_priorities']:
            print(f"\nOptimization Priorities:")
            for i, priority in enumerate(summary['optimization_priorities'][:3], 1):
                print(f"  {i}. {priority['component']} ({priority['priority']} priority)")
                print(f"     Reason: {priority['reason']}")
    
    # Component bottlenecks
    print(f"\n{'Component Bottlenecks':<30}")
    print(f"{'-'*80}")
    for i, bottleneck in enumerate(report.top_bottlenecks[:5], 1):
        print(f"{i}. {bottleneck.component_name}")
        print(f"   Time Spent: {bottleneck.time_spent:.2f}s ({bottleneck.percentage_of_total:.1f}%)")
        print(f"   Optimization Potential: {bottleneck.optimization_potential}")
        print(f"   Top Functions:")
        for func in bottleneck.function_calls[:3]:
            print(f"     - {func['function']}: {func['time']:.2f}s ({func['calls']} calls)")
        print()
    
    # Optimization recommendations
    print(f"{'High Priority Recommendations':<30}")
    print(f"{'-'*80}")
    for i, rec in enumerate(report.high_priority_recommendations[:5], 1):
        print(f"{i}. Component: {rec.component}")
        print(f"   Issue: {rec.issue_description}")
        print(f"   Recommendation: {rec.recommendation}")
        print(f"   Impact: {rec.expected_impact} | Effort: {rec.implementation_effort}")
        print(f"   Priority Score: {rec.priority_score:.1f}")
        print()
    
    # Test vs Implementation ratio
    print(f"{'Performance Analysis':<30}")
    print(f"{'-'*50}")
    ratio = report.test_vs_implementation_ratio
    if ratio == float('inf'):
        print(f"Test/Implementation Ratio: ∞ (no implementation time detected)")
    else:
        print(f"Test/Implementation Ratio: {ratio:.2f}")
    
    if ratio > 1.0:
        print("⚠️  Test overhead exceeds implementation time - focus on test optimization")
    elif ratio > 0.5:
        print("⚠️  Significant test overhead detected - consider test optimization")
    else:
        print("✅ Implementation time dominates - focus on ImplementAgent optimization")


def main():
    """Main demo function."""
    setup_logging()
    
    print("BottleneckAnalyzer Demo")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample profiling data...")
    profiling_results = create_sample_profiling_data()
    
    # Initialize analyzer
    print("Initializing BottleneckAnalyzer...")
    analyzer = BottleneckAnalyzer()
    
    # Analyze bottlenecks
    print("Analyzing bottlenecks...")
    bottleneck_reports = analyzer.analyze_cprofile_results(profiling_results)
    
    # Print results
    for report in bottleneck_reports:
        print_bottleneck_report(report)
    
    # Generate summary
    if bottleneck_reports:
        print(f"\n{'SUMMARY ACROSS ALL TESTS':<30}")
        print(f"{'='*80}")
        summary = analyzer.get_bottleneck_summary(bottleneck_reports)
        
        print(f"Total Tests Analyzed: {summary['total_tests_analyzed']}")
        print(f"Average Test/Implementation Ratio: {summary['avg_test_vs_implementation_ratio']:.2f}")
        
        print(f"\nMost Common Recommendations:")
        for i, rec in enumerate(summary['most_common_recommendations'][:3], 1):
            print(f"{i}. {rec['component']}: {rec['issue']} (appears {rec['count']} times)")
            print(f"   Recommendation: {rec['recommendation']}")
            print(f"   Impact: {rec['impact']}")
            print()


if __name__ == "__main__":
    main()