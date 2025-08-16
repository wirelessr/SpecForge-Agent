"""
Tests for BottleneckAnalyzer functionality.

This module tests the bottleneck analysis capabilities including time categorization,
component bottleneck identification, and optimization recommendation generation.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from .bottleneck_analyzer import BottleneckAnalyzer
from .models import (
    TestIdentifier, CProfileResult, ProfilingResult, ProfilerType,
    ComponentBottleneck, OptimizationRecommendation, TimeCategories, ComponentTimings
)


class TestBottleneckAnalyzer:
    """Test cases for BottleneckAnalyzer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = BottleneckAnalyzer()
        
        # Create test data
        self.test_identifier = TestIdentifier(
            file_path="tests/integration/test_implement_agent.py",
            class_name="TestImplementAgent",
            method_name="test_execute_task"
        )
        
        # Mock cProfile data with ImplementAgent components
        self.mock_function_stats = {
            'autogen_framework/agents/implement_agent.py:45(execute_task)': {
                'cumulative_time': 5.0,
                'total_time': 1.0,
                'call_count': 1,
                'filename': 'autogen_framework/agents/implement_agent.py',
                'line_number': 45,
                'function_name': 'execute_task'
            },
            'autogen_framework/agents/task_decomposer.py:30(decompose_task)': {
                'cumulative_time': 3.0,
                'total_time': 2.0,
                'call_count': 5,
                'filename': 'autogen_framework/agents/task_decomposer.py',
                'line_number': 30,
                'function_name': 'decompose_task'
            },
            'autogen_framework/shell_executor.py:20(execute_command)': {
                'cumulative_time': 2.0,
                'total_time': 1.5,
                'call_count': 10,
                'filename': 'autogen_framework/shell_executor.py',
                'line_number': 20,
                'function_name': 'execute_command'
            },
            'tests/integration/conftest.py:15(setup_test)': {
                'cumulative_time': 1.0,
                'total_time': 0.8,
                'call_count': 1,
                'filename': 'tests/integration/conftest.py',
                'line_number': 15,
                'function_name': 'setup_test'
            },
            'openai/api.py:100(chat_completion)': {
                'cumulative_time': 4.0,
                'total_time': 3.5,
                'call_count': 3,
                'filename': 'openai/api.py',
                'line_number': 100,
                'function_name': 'chat_completion'
            }
        }
        
        self.cprofile_result = CProfileResult(
            test_identifier=self.test_identifier,
            profile_file_path="/tmp/test_profile.prof",
            total_time=15.0,
            function_stats=self.mock_function_stats,
            top_functions=[
                {
                    'name': 'autogen_framework/agents/implement_agent.py:45(execute_task)',
                    'cumulative_time': 5.0,
                    'total_time': 1.0,
                    'call_count': 1,
                    'percentage': 33.3
                }
            ],
            call_count=20
        )
        
        self.profiling_result = ProfilingResult(
            test_identifier=self.test_identifier,
            profiler_used=ProfilerType.CPROFILE,
            cprofile_result=self.cprofile_result,
            total_profiling_time=15.5
        )
    
    def test_categorize_time_spent(self):
        """Test time categorization functionality."""
        time_categories = self.analyzer.categorize_time_spent(self.cprofile_result)
        
        # Verify time categories structure
        assert isinstance(time_categories, TimeCategories)
        assert isinstance(time_categories.component_timings, ComponentTimings)
        
        # Check that ImplementAgent components are identified
        assert time_categories.component_timings.task_decomposer > 0
        assert time_categories.component_timings.shell_executor > 0
        assert time_categories.component_timings.llm_api_calls > 0
        assert time_categories.component_timings.test_overhead > 0
        
        # Verify implement_agent_time includes component times
        assert time_categories.implement_agent_time > 0
    
    def test_extract_component_timings(self):
        """Test detailed component timing extraction (Task 7 requirement)."""
        component_timings = self.analyzer.extract_component_timings(self.cprofile_result)
        
        # Verify structure
        assert isinstance(component_timings, dict)
        expected_components = ['TaskDecomposer', 'ShellExecutor', 'ErrorRecovery', 'ContextManager']
        
        for component in expected_components:
            assert component in component_timings
            data = component_timings[component]
            
            # Check required fields
            assert 'total_time' in data
            assert 'call_count' in data
            assert 'functions' in data
            assert 'avg_call_time' in data
            assert 'max_single_call' in data
            
            # Check component-specific fields
            if component == 'TaskDecomposer':
                assert 'decomposition_phases' in data
            elif component == 'ShellExecutor':
                assert 'execution_phases' in data
            elif component == 'ErrorRecovery':
                assert 'recovery_phases' in data
            elif component == 'ContextManager':
                assert 'context_phases' in data
                assert 'context_loading_time' in data
        
        # Verify that components with data in mock have non-zero times
        assert component_timings['TaskDecomposer']['total_time'] > 0
        assert component_timings['ShellExecutor']['total_time'] > 0
        
        # Verify function-level data
        task_decomposer_functions = component_timings['TaskDecomposer']['functions']
        assert len(task_decomposer_functions) > 0
        
        for func_name, func_data in task_decomposer_functions.items():
            assert 'cumulative_time' in func_data
            assert 'total_time' in func_data
            assert 'call_count' in func_data
            assert 'avg_time_per_call' in func_data
    
    def test_component_performance_summary(self):
        """Test component performance summary generation (Task 7 requirement)."""
        component_timings = self.analyzer.extract_component_timings(self.cprofile_result)
        summary = self.analyzer.get_component_performance_summary(component_timings)
        
        # Verify summary structure
        assert 'total_component_time' in summary
        assert 'component_breakdown' in summary
        assert 'performance_insights' in summary
        assert 'optimization_priorities' in summary
        
        # Check component breakdown
        breakdown = summary['component_breakdown']
        for component in ['TaskDecomposer', 'ShellExecutor', 'ErrorRecovery', 'ContextManager']:
            if component in breakdown:
                comp_data = breakdown[component]
                assert 'time' in comp_data
                assert 'percentage' in comp_data
                assert 'call_count' in comp_data
                assert 'avg_call_time' in comp_data
                assert 'efficiency_ratio' in comp_data
        
        # Check insights and priorities
        assert isinstance(summary['performance_insights'], list)
        assert isinstance(summary['optimization_priorities'], list)
        
        # Verify priorities are sorted correctly
        priorities = summary['optimization_priorities']
        if len(priorities) > 1:
            priority_values = {'high': 3, 'medium': 2, 'low': 1}
            for i in range(len(priorities) - 1):
                current_priority = priority_values.get(priorities[i]['priority'], 0)
                next_priority = priority_values.get(priorities[i + 1]['priority'], 0)
                assert current_priority >= next_priority
    
    def test_context_loading_performance_measurement(self):
        """Test context loading performance measurement (Task 7 requirement)."""
        # Add context loading functions to mock data
        context_functions = {
            'autogen_framework/context_manager.py:100(load_context)': {
                'cumulative_time': 1.5,
                'total_time': 1.2,
                'call_count': 2,
                'filename': 'autogen_framework/context_manager.py',
                'line_number': 100,
                'function_name': 'load_context'
            },
            'autogen_framework/context_manager.py:150(get_implementation_context)': {
                'cumulative_time': 2.0,
                'total_time': 1.8,
                'call_count': 1,
                'filename': 'autogen_framework/context_manager.py',
                'line_number': 150,
                'function_name': 'get_implementation_context'
            }
        }
        
        # Update mock data
        enhanced_function_stats = {**self.mock_function_stats, **context_functions}
        enhanced_cprofile_result = CProfileResult(
            test_identifier=self.test_identifier,
            profile_file_path="/tmp/test_profile.prof",
            total_time=18.5,  # Updated total
            function_stats=enhanced_function_stats,
            top_functions=self.cprofile_result.top_functions,
            call_count=22
        )
        
        # Extract component timings
        component_timings = self.analyzer.extract_component_timings(enhanced_cprofile_result)
        
        # Verify context loading time calculation
        context_data = component_timings['ContextManager']
        assert context_data['context_loading_time'] > 0
        assert context_data['total_time'] > 0
        
        # Verify context phases
        context_phases = context_data['context_phases']
        assert 'context_loading' in context_phases
        assert context_phases['context_loading'] > 0
        
        # Test context loading calculation method directly
        context_loading_time = self.analyzer._calculate_context_loading_time(context_data['functions'])
        assert context_loading_time > 0
    
    def test_identify_implement_agent_bottlenecks(self):
        """Test component bottleneck identification."""
        bottlenecks = self.analyzer.identify_implement_agent_bottlenecks(self.cprofile_result)
        
        # Should identify bottlenecks for components present in mock data
        assert len(bottlenecks) > 0
        
        # Check that bottlenecks are ComponentBottleneck objects
        for bottleneck in bottlenecks:
            assert isinstance(bottleneck, ComponentBottleneck)
            assert bottleneck.component_name in [
                'ImplementAgent', 'TaskDecomposer', 'ShellExecutor', 'LLM_API_Calls'
            ]
            assert bottleneck.time_spent > 0
            assert bottleneck.percentage_of_total > 0
        
        # Should be sorted by time spent (highest first)
        if len(bottlenecks) > 1:
            assert bottlenecks[0].time_spent >= bottlenecks[1].time_spent
    
    def test_generate_optimization_recommendations(self):
        """Test optimization recommendation generation."""
        # First get bottlenecks
        bottlenecks = self.analyzer.identify_implement_agent_bottlenecks(self.cprofile_result)
        time_categories = self.analyzer.categorize_time_spent(self.cprofile_result)
        
        # Generate recommendations
        recommendations = self.analyzer.generate_optimization_recommendations(
            bottlenecks, time_categories
        )
        
        # Should generate some recommendations
        assert len(recommendations) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert isinstance(rec, OptimizationRecommendation)
            assert rec.component
            assert rec.issue_description
            assert rec.recommendation
            assert rec.expected_impact in ['high', 'medium', 'low']
            assert rec.implementation_effort in ['high', 'medium', 'low']
            assert rec.priority_score > 0
        
        # Should be sorted by priority score (highest first)
        if len(recommendations) > 1:
            assert recommendations[0].priority_score >= recommendations[1].priority_score
    
    def test_analyze_cprofile_results(self):
        """Test complete cProfile results analysis."""
        results = [self.profiling_result]
        
        bottleneck_reports = self.analyzer.analyze_cprofile_results(results)
        
        # Should generate one report
        assert len(bottleneck_reports) == 1
        
        report = bottleneck_reports[0]
        
        # Verify report structure
        assert report.test_identifier == self.test_identifier
        assert isinstance(report.time_categories, TimeCategories)
        assert len(report.component_bottlenecks) > 0
        assert len(report.optimization_recommendations) > 0
        
        # Check that report has meaningful data
        assert report.time_categories.implement_agent_time > 0
        assert any(b.is_significant for b in report.component_bottlenecks)
    
    def test_categorize_function_patterns(self):
        """Test function categorization patterns."""
        # Test ImplementAgent component patterns
        assert self.analyzer._categorize_function('execute_task', 'implement_agent.py') == 'ImplementAgent'
        assert self.analyzer._categorize_function('decompose_task', 'task_decomposer.py') == 'TaskDecomposer'
        assert self.analyzer._categorize_function('execute_command', 'shell_executor.py') == 'ShellExecutor'
        assert self.analyzer._categorize_function('recover_from_error', 'error_recovery.py') == 'ErrorRecovery'
        assert self.analyzer._categorize_function('get_context', 'context_manager.py') == 'ContextManager'
        
        # Test LLM API patterns
        assert self.analyzer._categorize_function('chat_completion', 'openai/api.py') == 'LLM_API_Calls'
        assert self.analyzer._categorize_function('generate_response', 'anthropic.py') == 'LLM_API_Calls'
        
        # Test infrastructure patterns - setup/teardown are detected within Test_Infrastructure
        setup_result = self.analyzer._categorize_function('setup_test', 'conftest.py')
        assert setup_result in ['Test_Setup', 'Test_Infrastructure']  # Either is acceptable
        
        assert self.analyzer._categorize_function('test_something', 'test_file.py') == 'Test_Infrastructure'
        
        teardown_result = self.analyzer._categorize_function('teardown_fixture', 'conftest.py')
        assert teardown_result in ['Test_Teardown', 'Test_Infrastructure']  # Either is acceptable
        
        # Test unknown functions
        assert self.analyzer._categorize_function('unknown_function', 'random_file.py') == 'Other'
    
    def test_assess_optimization_potential(self):
        """Test optimization potential assessment."""
        # High potential - high percentage
        potential = self.analyzer._assess_optimization_potential(
            'TaskDecomposer', 5.0, 25.0, [{'calls': 50}]
        )
        assert potential == 'high'
        
        # High potential - many calls
        potential = self.analyzer._assess_optimization_potential(
            'ShellExecutor', 2.0, 15.0, [{'calls': 150}]
        )
        assert potential == 'high'
        
        # Medium potential - moderate percentage
        potential = self.analyzer._assess_optimization_potential(
            'ContextManager', 2.0, 15.0, [{'calls': 10}]
        )
        assert potential == 'medium'
        
        # Medium potential - LLM calls
        potential = self.analyzer._assess_optimization_potential(
            'LLM_API_Calls', 1.0, 8.0, [{'calls': 5}]
        )
        assert potential == 'medium'
        
        # Low potential - low percentage
        potential = self.analyzer._assess_optimization_potential(
            'ErrorRecovery', 0.5, 3.0, [{'calls': 2}]
        )
        assert potential == 'low'
    
    def test_get_bottleneck_summary(self):
        """Test bottleneck summary generation."""
        # Create multiple reports
        reports = []
        for i in range(3):
            result = ProfilingResult(
                test_identifier=TestIdentifier(
                    file_path=f"test_file_{i}.py",
                    method_name=f"test_method_{i}"
                ),
                profiler_used=ProfilerType.CPROFILE,
                cprofile_result=self.cprofile_result
            )
            reports.append(self.analyzer._analyze_single_test(result))
        
        summary = self.analyzer.get_bottleneck_summary(reports)
        
        # Verify summary structure
        assert 'total_tests_analyzed' in summary
        assert 'component_bottlenecks' in summary
        assert 'most_common_recommendations' in summary
        assert 'avg_test_vs_implementation_ratio' in summary
        
        assert summary['total_tests_analyzed'] == 3
        assert len(summary['component_bottlenecks']) > 0
        assert isinstance(summary['avg_test_vs_implementation_ratio'], float)
    
    def test_empty_results_handling(self):
        """Test handling of empty or invalid results."""
        # Empty results list
        reports = self.analyzer.analyze_cprofile_results([])
        assert reports == []
        
        # Result without cProfile data
        empty_result = ProfilingResult(
            test_identifier=self.test_identifier,
            profiler_used=ProfilerType.CPROFILE,
            cprofile_result=None
        )
        reports = self.analyzer.analyze_cprofile_results([empty_result])
        assert reports == []
        
        # Empty summary
        summary = self.analyzer.get_bottleneck_summary([])
        assert summary == {}
    
    def test_component_specific_recommendations(self):
        """Test component-specific recommendation generation."""
        # Create bottleneck with high call count for ShellExecutor
        shell_bottleneck = ComponentBottleneck(
            component_name='ShellExecutor',
            time_spent=5.0,
            percentage_of_total=25.0,
            function_calls=[
                {'function': 'subprocess.run', 'calls': 50, 'time': 3.0},
                {'function': 'execute_command', 'calls': 20, 'time': 2.0}
            ],
            optimization_potential='high'
        )
        
        recommendations = self.analyzer._generate_component_recommendations(shell_bottleneck)
        
        # Should generate recommendations for ShellExecutor
        assert len(recommendations) > 0
        
        # Should generate some recommendation since it's a significant bottleneck
        assert len(recommendations) > 0
        
        # Check recommendation quality
        for rec in recommendations:
            assert rec.component == 'ShellExecutor'
            assert rec.expected_impact in ['high', 'medium', 'low']
            assert rec.implementation_effort in ['high', 'medium', 'low']
            assert rec.priority_score > 0