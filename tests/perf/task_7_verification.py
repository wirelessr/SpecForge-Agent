#!/usr/bin/env python3
"""
Task 7 Verification Script

This script verifies that all Task 7 requirements have been implemented:
- Function pattern matching to identify TaskDecomposer, ShellExecutor, and ErrorRecovery calls
- Timing extraction for each ImplementAgent component from cProfile data
- Context loading performance measurement
"""

import sys
from pathlib import Path

# Add the current directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from bottleneck_analyzer import BottleneckAnalyzer
from models import TestIdentifier, CProfileResult, ProfilingResult, ProfilerType


def create_test_data():
    """Create test data with all ImplementAgent components."""
    test_identifier = TestIdentifier(
        file_path="tests/integration/test_implement_agent.py",
        class_name="TestImplementAgent",
        method_name="test_task_execution"
    )
    
    # Mock function stats with all component types
    function_stats = {
        # TaskDecomposer functions
        'autogen_framework/agents/task_decomposer.py:45(decompose_task)': {
            'cumulative_time': 3.0,
            'total_time': 2.0,
            'call_count': 5,
            'filename': 'autogen_framework/agents/task_decomposer.py',
            'line_number': 45,
            'function_name': 'decompose_task'
        },
        'autogen_framework/agents/task_decomposer.py:80(analyze_task_complexity)': {
            'cumulative_time': 1.5,
            'total_time': 1.2,
            'call_count': 3,
            'filename': 'autogen_framework/agents/task_decomposer.py',
            'line_number': 80,
            'function_name': 'analyze_task_complexity'
        },
        
        # ShellExecutor functions
        'autogen_framework/shell_executor.py:20(execute_command)': {
            'cumulative_time': 2.0,
            'total_time': 1.5,
            'call_count': 10,
            'filename': 'autogen_framework/shell_executor.py',
            'line_number': 20,
            'function_name': 'execute_command'
        },
        'subprocess.py:350(run)': {
            'cumulative_time': 1.8,
            'total_time': 1.6,
            'call_count': 10,
            'filename': 'subprocess.py',
            'line_number': 350,
            'function_name': 'run'
        },
        
        # ErrorRecovery functions
        'autogen_framework/agents/error_recovery.py:60(recover_from_error)': {
            'cumulative_time': 2.5,
            'total_time': 1.8,
            'call_count': 3,
            'filename': 'autogen_framework/agents/error_recovery.py',
            'line_number': 60,
            'function_name': 'recover_from_error'
        },
        'autogen_framework/agents/error_recovery.py:100(analyze_error)': {
            'cumulative_time': 1.2,
            'total_time': 0.9,
            'call_count': 4,
            'filename': 'autogen_framework/agents/error_recovery.py',
            'line_number': 100,
            'function_name': 'analyze_error'
        },
        
        # ContextManager functions (including context loading)
        'autogen_framework/context_manager.py:150(get_implementation_context)': {
            'cumulative_time': 2.0,
            'total_time': 1.5,
            'call_count': 2,
            'filename': 'autogen_framework/context_manager.py',
            'line_number': 150,
            'function_name': 'get_implementation_context'
        },
        'autogen_framework/context_manager.py:80(load_context)': {
            'cumulative_time': 1.5,
            'total_time': 1.2,
            'call_count': 3,
            'filename': 'autogen_framework/context_manager.py',
            'line_number': 80,
            'function_name': 'load_context'
        }
    }
    
    cprofile_result = CProfileResult(
        test_identifier=test_identifier,
        profile_file_path="/tmp/test_profile.prof",
        total_time=15.0,
        function_stats=function_stats,
        top_functions=[],
        call_count=40
    )
    
    return ProfilingResult(
        test_identifier=test_identifier,
        profiler_used=ProfilerType.CPROFILE,
        cprofile_result=cprofile_result,
        total_profiling_time=15.5
    )


def verify_task_7_requirements():
    """Verify all Task 7 requirements are implemented."""
    print("Task 7 Verification: ImplementAgent Component Timing Extraction")
    print("=" * 70)
    
    analyzer = BottleneckAnalyzer()
    profiling_result = create_test_data()
    
    # Requirement 4.1: Function pattern matching to identify components
    print("\n1. Testing function pattern matching...")
    
    # Test TaskDecomposer patterns
    assert analyzer._categorize_function('decompose_task', 'task_decomposer.py') == 'TaskDecomposer'
    assert analyzer._categorize_function('analyze_task_complexity', 'task_decomposer.py') == 'TaskDecomposer'
    print("   ✓ TaskDecomposer patterns working")
    
    # Test ShellExecutor patterns
    assert analyzer._categorize_function('execute_command', 'shell_executor.py') == 'ShellExecutor'
    assert analyzer._categorize_function('run', 'subprocess.py') == 'ShellExecutor'
    print("   ✓ ShellExecutor patterns working")
    
    # Test ErrorRecovery patterns
    assert analyzer._categorize_function('recover_from_error', 'error_recovery.py') == 'ErrorRecovery'
    assert analyzer._categorize_function('analyze_error', 'error_recovery.py') == 'ErrorRecovery'
    print("   ✓ ErrorRecovery patterns working")
    
    # Requirement 4.2: Timing extraction for each component
    print("\n2. Testing component timing extraction...")
    
    component_timings = analyzer.extract_component_timings(profiling_result.cprofile_result)
    
    # Verify structure
    expected_components = ['TaskDecomposer', 'ShellExecutor', 'ErrorRecovery', 'ContextManager']
    for component in expected_components:
        assert component in component_timings
        data = component_timings[component]
        assert 'total_time' in data
        assert 'call_count' in data
        assert 'functions' in data
        assert 'avg_call_time' in data
        assert 'max_single_call' in data
    
    # Verify timing data is extracted
    assert component_timings['TaskDecomposer']['total_time'] > 0
    assert component_timings['ShellExecutor']['total_time'] > 0
    assert component_timings['ErrorRecovery']['total_time'] > 0
    assert component_timings['ContextManager']['total_time'] > 0
    
    print("   ✓ Component timing extraction working")
    print(f"     - TaskDecomposer: {component_timings['TaskDecomposer']['total_time']:.2f}s")
    print(f"     - ShellExecutor: {component_timings['ShellExecutor']['total_time']:.2f}s")
    print(f"     - ErrorRecovery: {component_timings['ErrorRecovery']['total_time']:.2f}s")
    print(f"     - ContextManager: {component_timings['ContextManager']['total_time']:.2f}s")
    
    # Requirement 4.4: Context loading performance measurement
    print("\n3. Testing context loading performance measurement...")
    
    context_data = component_timings['ContextManager']
    assert 'context_loading_time' in context_data
    assert context_data['context_loading_time'] > 0
    
    # Test context loading calculation
    context_loading_time = analyzer._calculate_context_loading_time(context_data['functions'])
    assert context_loading_time > 0
    
    print("   ✓ Context loading performance measurement working")
    print(f"     - Context loading time: {context_data['context_loading_time']:.2f}s")
    print(f"     - Total context manager time: {context_data['total_time']:.2f}s")
    
    # Requirement 4.5: Phase-specific timing breakdown
    print("\n4. Testing phase-specific timing breakdown...")
    
    # Check phase breakdown for each component
    assert 'decomposition_phases' in component_timings['TaskDecomposer']
    assert 'execution_phases' in component_timings['ShellExecutor']
    assert 'recovery_phases' in component_timings['ErrorRecovery']
    assert 'context_phases' in component_timings['ContextManager']
    
    print("   ✓ Phase-specific timing breakdown working")
    
    # Test performance summary generation
    print("\n5. Testing performance summary generation...")
    
    summary = analyzer.get_component_performance_summary(component_timings)
    assert 'total_component_time' in summary
    assert 'component_breakdown' in summary
    assert 'performance_insights' in summary
    assert 'optimization_priorities' in summary
    
    print("   ✓ Performance summary generation working")
    
    # Test integration with bottleneck report
    print("\n6. Testing integration with bottleneck analysis...")
    
    reports = analyzer.analyze_cprofile_results([profiling_result])
    assert len(reports) == 1
    
    report = reports[0]
    assert hasattr(report, 'detailed_component_timings')
    assert hasattr(report, 'component_performance_summary')
    assert report.detailed_component_timings is not None
    assert report.component_performance_summary is not None
    
    # Test new report properties
    context_perf = report.context_loading_performance
    assert 'context_loading_time' in context_perf
    assert 'percentage_of_context_manager' in context_perf
    
    efficiency_metrics = report.component_efficiency_metrics
    assert len(efficiency_metrics) > 0
    
    print("   ✓ Integration with bottleneck analysis working")
    
    print("\n" + "=" * 70)
    print("✅ ALL TASK 7 REQUIREMENTS SUCCESSFULLY IMPLEMENTED!")
    print("=" * 70)
    
    print("\nImplemented Features:")
    print("• Enhanced function pattern matching for ImplementAgent components")
    print("• Detailed timing extraction for TaskDecomposer, ShellExecutor, ErrorRecovery")
    print("• Context loading performance measurement and analysis")
    print("• Phase-specific timing breakdown for each component")
    print("• Component efficiency metrics and optimization insights")
    print("• Integration with existing bottleneck analysis framework")
    
    return True


if __name__ == "__main__":
    try:
        verify_task_7_requirements()
        print("\n🎉 Task 7 verification completed successfully!")
    except Exception as e:
        print(f"\n❌ Task 7 verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)