"""
BottleneckAnalyzer for analyzing profiling results and identifying optimization targets.

This module provides the BottleneckAnalyzer class that analyzes cProfile results to
categorize time spent in test infrastructure vs ImplementAgent code, identifies
component-level bottlenecks, and generates actionable optimization recommendations.
"""

import logging
import pstats
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from pathlib import Path
import re

try:
    from .models import (
        TestIdentifier, CProfileResult, ProfilingResult, BottleneckReport,
        TimeCategories, ComponentTimings, ComponentBottleneck, OptimizationRecommendation
    )
    from .utils import PatternMatcher
except ImportError:
    # Handle direct execution
    from models import (
        TestIdentifier, CProfileResult, ProfilingResult, BottleneckReport,
        TimeCategories, ComponentTimings, ComponentBottleneck, OptimizationRecommendation
    )
    from utils import PatternMatcher

# Forward reference to avoid circular imports
if TYPE_CHECKING:
    from .llm_profiler import LLMProfilingResult


class BottleneckAnalyzer:
    """
    Analyzes cProfile results to identify ImplementAgent optimization targets.
    
    This class focuses on distinguishing between test overhead and actual ImplementAgent
    execution time, identifying specific component bottlenecks, and generating actionable
    optimization recommendations based on profiling data analysis.
    """
    
    def __init__(self):
        """Initialize BottleneckAnalyzer with pattern matching capabilities."""
        self.logger = logging.getLogger(__name__)
        self.pattern_matcher = PatternMatcher()
        
        # Define component patterns for ImplementAgent analysis
        self._setup_component_patterns()
        
        # Define optimization recommendation templates
        self._setup_recommendation_templates()
    
    def _setup_component_patterns(self):
        """Setup patterns for identifying ImplementAgent components."""
        self.component_patterns = {
            'TaskDecomposer': [
                r'task_decomposer',
                r'decompose_task',
                r'TaskDecomposer',
                r'_decompose_.*',
                r'.*task.*decompos.*',
                r'analyze_task_complexity',
                r'generate_command_sequence',
                r'create_execution_plan',
                r'validate_decomposition'
            ],
            'ShellExecutor': [
                r'shell_executor',
                r'execute_command',
                r'ShellExecutor',
                r'_execute_.*',
                r'subprocess.*',
                r'.*shell.*exec.*',
                r'execute_plan',
                r'run_command',
                r'capture_output',
                r'handle_timeout',
                r'process_control'
            ],
            'ErrorRecovery': [
                r'error_recovery',
                r'recover_from_error',
                r'ErrorRecovery',
                r'_recover_.*',
                r'.*error.*recover.*',
                r'.*retry.*',
                r'analyze_error',
                r'generate_strategies',
                r'try_strategy',
                r'categorize_error',
                r'recovery_attempt'
            ],
            'ContextManager': [
                r'context_manager',
                r'get_context',
                r'ContextManager',
                r'_get_context.*',
                r'.*context.*manag.*',
                r'load_context',
                r'get_implementation_context',
                r'get_plan_context',
                r'update_execution_history',
                r'compress_context',
                r'initialize_context'
            ],
            'ImplementAgent': [
                r'implement_agent',
                r'ImplementAgent',
                r'execute_task',
                r'_process_task.*',
                r'.*implement.*agent.*',
                r'coordinate_execution',
                r'manage_workflow',
                r'handle_task_result'
            ]
        }
        
        # LLM API call patterns
        self.llm_patterns = [
            r'openai.*',
            r'.*api.*call.*',
            r'.*llm.*',
            r'.*chat.*completion.*',
            r'generate_response',
            r'.*anthropic.*',
            r'.*claude.*',
            r'aiohttp.*request.*',
            r'httpx.*'
        ]
        
        # Test infrastructure patterns
        self.test_patterns = [
            r'pytest.*',
            r'test_.*',
            r'.*test.*setup.*',
            r'.*test.*teardown.*',
            r'.*fixture.*',
            r'conftest.*',
            r'.*mock.*',
            r'unittest.*',
            r'.*assert.*'
        ]
    
    def _setup_recommendation_templates(self):
        """Setup templates for generating optimization recommendations."""
        self.recommendation_templates = {
            'TaskDecomposer': {
                'high_time': {
                    'issue': "TaskDecomposer is consuming significant execution time",
                    'recommendation': "Optimize task decomposition logic, consider caching decomposition results for similar tasks",
                    'impact': "high",
                    'effort': "medium"
                },
                'high_calls': {
                    'issue': "TaskDecomposer has excessive function calls",
                    'recommendation': "Reduce redundant decomposition calls, implement memoization for repeated task patterns",
                    'impact': "medium",
                    'effort': "low"
                }
            },
            'ShellExecutor': {
                'high_time': {
                    'issue': "ShellExecutor is the primary bottleneck",
                    'recommendation': "Optimize command execution, consider parallel execution for independent commands",
                    'impact': "high",
                    'effort': "high"
                },
                'subprocess_overhead': {
                    'issue': "High subprocess creation overhead",
                    'recommendation': "Batch commands where possible, use shell command chaining to reduce process creation",
                    'impact': "medium",
                    'effort': "medium"
                }
            },
            'ErrorRecovery': {
                'high_time': {
                    'issue': "ErrorRecovery is consuming excessive time",
                    'recommendation': "Optimize error analysis logic, implement faster error pattern matching",
                    'impact': "high",
                    'effort': "medium"
                },
                'frequent_recovery': {
                    'issue': "Error recovery is triggered too frequently",
                    'recommendation': "Improve initial command generation to reduce error rates, implement better error prevention",
                    'impact': "high",
                    'effort': "high"
                }
            },
            'ContextManager': {
                'high_time': {
                    'issue': "ContextManager is slow to load context",
                    'recommendation': "Implement context caching, optimize context loading and compression",
                    'impact': "medium",
                    'effort': "medium"
                },
                'memory_overhead': {
                    'issue': "High memory usage in context management",
                    'recommendation': "Implement more aggressive context compression, use lazy loading for large contexts",
                    'impact': "medium",
                    'effort': "low"
                }
            },
            'LLM_API_Calls': {
                'high_time': {
                    'issue': "LLM API calls are the primary bottleneck",
                    'recommendation': "Optimize prompt sizes, implement response caching, consider faster model alternatives",
                    'impact': "high",
                    'effort': "low"
                },
                'frequent_calls': {
                    'issue': "Too many LLM API calls",
                    'recommendation': "Batch multiple requests, implement intelligent caching, reduce redundant calls",
                    'impact': "high",
                    'effort': "medium"
                }
            },
            'Test_Infrastructure': {
                'high_overhead': {
                    'issue': "Test infrastructure overhead is significant",
                    'recommendation': "Optimize test setup/teardown, use faster fixtures, reduce mock complexity",
                    'impact': "low",
                    'effort': "medium"
                }
            }
        }
    
    def analyze_cprofile_results(self, results: List[ProfilingResult]) -> List[BottleneckReport]:
        """
        Analyze cProfile data to identify bottlenecks in ImplementAgent.
        
        Args:
            results: List of ProfilingResult objects from performance profiling
            
        Returns:
            List of BottleneckReport objects with detailed bottleneck analysis
        """
        if not results:
            self.logger.warning("No profiling results provided for analysis")
            return []
        
        self.logger.info(f"Analyzing {len(results)} profiling results for bottlenecks")
        
        bottleneck_reports = []
        
        for result in results:
            if not result.cprofile_result:
                self.logger.warning(
                    f"No cProfile data available for {result.test_identifier.full_name}"
                )
                continue
            
            try:
                # Analyze this specific test's profiling data
                report = self._analyze_single_test(result)
                bottleneck_reports.append(report)
                
                self.logger.debug(
                    f"Analyzed bottlenecks for {result.test_identifier.full_name}: "
                    f"{len(report.component_bottlenecks)} bottlenecks, "
                    f"{len(report.optimization_recommendations)} recommendations"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to analyze bottlenecks for {result.test_identifier.full_name}: {e}"
                )
        
        self.logger.info(f"Completed bottleneck analysis for {len(bottleneck_reports)} tests")
        return bottleneck_reports
    
    def analyze_profiling_results_with_llm(self, results: List[ProfilingResult]) -> List[BottleneckReport]:
        """
        Analyze profiling results including both cProfile and LLM API call data.
        
        Args:
            results: List of ProfilingResult objects with cProfile and LLM data
            
        Returns:
            List of BottleneckReport objects with comprehensive analysis
        """
        if not results:
            self.logger.warning("No profiling results provided for analysis")
            return []
        
        self.logger.info(f"Analyzing {len(results)} profiling results with LLM data for bottlenecks")
        
        bottleneck_reports = []
        
        for result in results:
            try:
                # Start with standard cProfile analysis
                if result.cprofile_result:
                    report = self._analyze_single_test(result)
                else:
                    # Create minimal report if no cProfile data
                    report = BottleneckReport(
                        test_identifier=result.test_identifier,
                        time_categories=TimeCategories()
                    )
                
                # Add LLM API call analysis if available
                if result.llm_profiling_result and result.llm_profiling_result.total_api_calls > 0:
                    llm_bottlenecks = self.analyze_llm_api_bottlenecks(result.llm_profiling_result)
                    llm_recommendations = self.generate_llm_optimization_recommendations(result.llm_profiling_result)
                    
                    # Merge LLM bottlenecks with existing ones
                    report.component_bottlenecks.extend(llm_bottlenecks)
                    report.optimization_recommendations.extend(llm_recommendations)
                    
                    # Update time categories to include LLM time
                    report.time_categories.component_timings.llm_api_calls = result.llm_profiling_result.total_llm_time
                    report.time_categories.implement_agent_time += result.llm_profiling_result.total_llm_time
                    
                    self.logger.debug(
                        f"Added LLM analysis: {len(llm_bottlenecks)} LLM bottlenecks, "
                        f"{len(llm_recommendations)} LLM recommendations"
                    )
                
                bottleneck_reports.append(report)
                
                self.logger.debug(
                    f"Analyzed bottlenecks for {result.test_identifier.full_name}: "
                    f"{len(report.component_bottlenecks)} total bottlenecks, "
                    f"{len(report.optimization_recommendations)} total recommendations"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to analyze bottlenecks for {result.test_identifier.full_name}: {e}"
                )
        
        self.logger.info(f"Completed comprehensive bottleneck analysis for {len(bottleneck_reports)} tests")
        return bottleneck_reports
    
    def _analyze_single_test(self, profiling_result: ProfilingResult) -> BottleneckReport:
        """
        Analyze bottlenecks for a single test's profiling data.
        
        Args:
            profiling_result: ProfilingResult with cProfile data
            
        Returns:
            BottleneckReport with detailed analysis
        """
        cprofile_result = profiling_result.cprofile_result
        
        # Extract detailed component timings (Task 7 requirement)
        component_timings = self.extract_component_timings(cprofile_result)
        
        # Categorize time spent across different areas
        time_categories = self.categorize_time_spent(cprofile_result)
        
        # Update component timings in time categories with extracted data
        time_categories.component_timings.task_decomposer = component_timings['TaskDecomposer']['total_time']
        time_categories.component_timings.shell_executor = component_timings['ShellExecutor']['total_time']
        time_categories.component_timings.error_recovery = component_timings['ErrorRecovery']['total_time']
        time_categories.component_timings.context_manager = component_timings['ContextManager']['total_time']
        
        # Identify component-level bottlenecks
        component_bottlenecks = self.identify_implement_agent_bottlenecks(cprofile_result)
        
        # Generate optimization recommendations
        optimization_recommendations = self.generate_optimization_recommendations(
            component_bottlenecks, time_categories
        )
        
        # Create enhanced bottleneck report with component timing details
        report = BottleneckReport(
            test_identifier=profiling_result.test_identifier,
            time_categories=time_categories,
            component_bottlenecks=component_bottlenecks,
            optimization_recommendations=optimization_recommendations
        )
        
        # Add component timing details to the report (extend the dataclass)
        report.detailed_component_timings = component_timings
        report.component_performance_summary = self.get_component_performance_summary(component_timings)
        
        return report
    
    def categorize_time_spent(self, cprofile_result: CProfileResult) -> TimeCategories:
        """
        Categorize time into test overhead vs ImplementAgent execution using function patterns.
        
        Args:
            cprofile_result: Analyzed cProfile data
            
        Returns:
            TimeCategories with breakdown of time spent
        """
        # Initialize component timings
        component_timings = ComponentTimings()
        
        # Track test infrastructure time
        test_setup_time = 0.0
        test_teardown_time = 0.0
        implement_agent_time = 0.0
        
        # Analyze each function in the profile
        for func_name, func_data in cprofile_result.function_stats.items():
            cumulative_time = func_data['cumulative_time']
            function_name = func_data['function_name']
            filename = func_data['filename']
            
            # Categorize the function
            category = self._categorize_function(function_name, filename)
            
            if category == 'TaskDecomposer':
                component_timings.task_decomposer += cumulative_time
                implement_agent_time += cumulative_time
            elif category == 'ShellExecutor':
                component_timings.shell_executor += cumulative_time
                implement_agent_time += cumulative_time
            elif category == 'ErrorRecovery':
                component_timings.error_recovery += cumulative_time
                implement_agent_time += cumulative_time
            elif category == 'ContextManager':
                component_timings.context_manager += cumulative_time
                implement_agent_time += cumulative_time
            elif category == 'LLM_API_Calls':
                component_timings.llm_api_calls += cumulative_time
                implement_agent_time += cumulative_time
            elif category == 'Test_Setup':
                test_setup_time += cumulative_time
                component_timings.test_overhead += cumulative_time
            elif category == 'Test_Teardown':
                test_teardown_time += cumulative_time
                component_timings.test_overhead += cumulative_time
            elif category == 'Test_Infrastructure':
                component_timings.test_overhead += cumulative_time
            else:
                component_timings.other += cumulative_time
        
        return TimeCategories(
            test_setup_time=test_setup_time,
            test_teardown_time=test_teardown_time,
            implement_agent_time=implement_agent_time,
            component_timings=component_timings
        )
    
    def _categorize_function(self, function_name: str, filename: str) -> str:
        """
        Categorize a function based on its name and filename.
        
        Args:
            function_name: Name of the function
            filename: File containing the function
            
        Returns:
            Category string for the function
        """
        # Check ImplementAgent components first
        for component, patterns in self.component_patterns.items():
            if any(re.search(pattern, function_name, re.IGNORECASE) for pattern in patterns):
                return component
            if any(re.search(pattern, filename, re.IGNORECASE) for pattern in patterns):
                return component
        
        # Check for LLM API calls
        if any(re.search(pattern, function_name, re.IGNORECASE) for pattern in self.llm_patterns):
            return 'LLM_API_Calls'
        
        # Check for test infrastructure
        if any(re.search(pattern, function_name, re.IGNORECASE) for pattern in self.test_patterns):
            if 'setup' in function_name.lower():
                return 'Test_Setup'
            elif 'teardown' in function_name.lower():
                return 'Test_Teardown'
            else:
                return 'Test_Infrastructure'
        
        # Check filename for test patterns
        if any(re.search(pattern, filename, re.IGNORECASE) for pattern in self.test_patterns):
            return 'Test_Infrastructure'
        
        return 'Other'
    
    def extract_component_timings(self, cprofile_result: CProfileResult) -> Dict[str, Dict[str, Any]]:
        """
        Extract detailed timing information for each ImplementAgent component.
        
        This method implements the core functionality for task 7: extracting timing
        data for TaskDecomposer, ShellExecutor, ErrorRecovery, and ContextManager
        components from cProfile data.
        
        Args:
            cprofile_result: Analyzed cProfile data
            
        Returns:
            Dictionary with detailed timing breakdown per component
        """
        component_timings = {
            'TaskDecomposer': {
                'total_time': 0.0,
                'call_count': 0,
                'functions': {},
                'avg_call_time': 0.0,
                'max_single_call': 0.0,
                'decomposition_phases': {}
            },
            'ShellExecutor': {
                'total_time': 0.0,
                'call_count': 0,
                'functions': {},
                'avg_call_time': 0.0,
                'max_single_call': 0.0,
                'execution_phases': {}
            },
            'ErrorRecovery': {
                'total_time': 0.0,
                'call_count': 0,
                'functions': {},
                'avg_call_time': 0.0,
                'max_single_call': 0.0,
                'recovery_phases': {}
            },
            'ContextManager': {
                'total_time': 0.0,
                'call_count': 0,
                'functions': {},
                'avg_call_time': 0.0,
                'max_single_call': 0.0,
                'context_loading_time': 0.0,
                'context_phases': {}
            }
        }
        
        # Analyze each function in the profile
        for func_name, func_data in cprofile_result.function_stats.items():
            cumulative_time = func_data['cumulative_time']
            total_time = func_data['total_time']
            call_count = func_data['call_count']
            function_name = func_data['function_name']
            filename = func_data['filename']
            
            # Determine which component this function belongs to
            component = self._categorize_function(function_name, filename)
            
            if component in component_timings:
                # Update component totals
                component_timings[component]['total_time'] += cumulative_time
                component_timings[component]['call_count'] += call_count
                
                # Track individual function performance
                component_timings[component]['functions'][func_name] = {
                    'cumulative_time': cumulative_time,
                    'total_time': total_time,
                    'call_count': call_count,
                    'avg_time_per_call': total_time / call_count if call_count > 0 else 0.0,
                    'function_name': function_name,
                    'filename': filename
                }
                
                # Update max single call time
                avg_call_time = total_time / call_count if call_count > 0 else 0.0
                if avg_call_time > component_timings[component]['max_single_call']:
                    component_timings[component]['max_single_call'] = avg_call_time
                
                # Extract phase-specific timing for each component
                self._extract_component_phases(component, function_name, cumulative_time, component_timings[component])
        
        # Calculate averages and additional metrics
        for component, data in component_timings.items():
            if data['call_count'] > 0:
                data['avg_call_time'] = data['total_time'] / data['call_count']
            
            # Calculate context loading performance for ContextManager
            if component == 'ContextManager':
                data['context_loading_time'] = self._calculate_context_loading_time(data['functions'])
        
        return component_timings
    
    def _extract_component_phases(self, component: str, function_name: str, 
                                cumulative_time: float, component_data: Dict[str, Any]):
        """
        Extract phase-specific timing information for component functions.
        
        Args:
            component: Component name (TaskDecomposer, ShellExecutor, etc.)
            function_name: Name of the function being analyzed
            cumulative_time: Time spent in this function
            component_data: Component timing data dictionary to update
        """
        if component == 'TaskDecomposer':
            phases = component_data['decomposition_phases']
            if 'analyze' in function_name.lower() or 'complexity' in function_name.lower():
                phases['analysis'] = phases.get('analysis', 0.0) + cumulative_time
            elif 'generate' in function_name.lower() or 'command' in function_name.lower():
                phases['command_generation'] = phases.get('command_generation', 0.0) + cumulative_time
            elif 'plan' in function_name.lower() or 'execution' in function_name.lower():
                phases['plan_creation'] = phases.get('plan_creation', 0.0) + cumulative_time
            elif 'validate' in function_name.lower():
                phases['validation'] = phases.get('validation', 0.0) + cumulative_time
            else:
                phases['other'] = phases.get('other', 0.0) + cumulative_time
        
        elif component == 'ShellExecutor':
            phases = component_data['execution_phases']
            if 'subprocess' in function_name.lower() or 'run' in function_name.lower():
                phases['process_execution'] = phases.get('process_execution', 0.0) + cumulative_time
            elif 'capture' in function_name.lower() or 'output' in function_name.lower():
                phases['output_capture'] = phases.get('output_capture', 0.0) + cumulative_time
            elif 'timeout' in function_name.lower() or 'wait' in function_name.lower():
                phases['timeout_handling'] = phases.get('timeout_handling', 0.0) + cumulative_time
            elif 'command' in function_name.lower() and 'prepare' in function_name.lower():
                phases['command_preparation'] = phases.get('command_preparation', 0.0) + cumulative_time
            else:
                phases['other'] = phases.get('other', 0.0) + cumulative_time
        
        elif component == 'ErrorRecovery':
            phases = component_data['recovery_phases']
            if 'analyze' in function_name.lower() or 'categorize' in function_name.lower():
                phases['error_analysis'] = phases.get('error_analysis', 0.0) + cumulative_time
            elif 'generate' in function_name.lower() or 'strategy' in function_name.lower():
                phases['strategy_generation'] = phases.get('strategy_generation', 0.0) + cumulative_time
            elif 'try' in function_name.lower() or 'attempt' in function_name.lower():
                phases['strategy_execution'] = phases.get('strategy_execution', 0.0) + cumulative_time
            elif 'record' in function_name.lower() or 'learn' in function_name.lower():
                phases['pattern_learning'] = phases.get('pattern_learning', 0.0) + cumulative_time
            else:
                phases['other'] = phases.get('other', 0.0) + cumulative_time
        
        elif component == 'ContextManager':
            phases = component_data['context_phases']
            if 'load' in function_name.lower() or 'get' in function_name.lower():
                phases['context_loading'] = phases.get('context_loading', 0.0) + cumulative_time
            elif 'compress' in function_name.lower():
                phases['context_compression'] = phases.get('context_compression', 0.0) + cumulative_time
            elif 'update' in function_name.lower() or 'history' in function_name.lower():
                phases['history_update'] = phases.get('history_update', 0.0) + cumulative_time
            elif 'initialize' in function_name.lower():
                phases['initialization'] = phases.get('initialization', 0.0) + cumulative_time
            else:
                phases['other'] = phases.get('other', 0.0) + cumulative_time
    
    def _calculate_context_loading_time(self, functions: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate specific context loading performance metrics.
        
        Args:
            functions: Dictionary of function timing data
            
        Returns:
            Total time spent in context loading operations
        """
        context_loading_time = 0.0
        
        for func_name, func_data in functions.items():
            function_name = func_data['function_name'].lower()
            
            # Identify context loading specific functions
            if any(keyword in function_name for keyword in [
                'load_context', 'get_context', 'get_implementation_context',
                'get_plan_context', 'initialize_context'
            ]):
                context_loading_time += func_data['cumulative_time']
        
        return context_loading_time
    
    def get_component_performance_summary(self, component_timings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a performance summary for all ImplementAgent components.
        
        Args:
            component_timings: Component timing data from extract_component_timings
            
        Returns:
            Dictionary with performance summary and optimization insights
        """
        total_component_time = sum(data['total_time'] for data in component_timings.values())
        
        summary = {
            'total_component_time': total_component_time,
            'component_breakdown': {},
            'performance_insights': [],
            'optimization_priorities': []
        }
        
        # Calculate component breakdown
        for component, data in component_timings.items():
            percentage = (data['total_time'] / total_component_time * 100) if total_component_time > 0 else 0.0
            
            summary['component_breakdown'][component] = {
                'time': data['total_time'],
                'percentage': percentage,
                'call_count': data['call_count'],
                'avg_call_time': data['avg_call_time'],
                'max_single_call': data['max_single_call'],
                'efficiency_ratio': data['avg_call_time'] / data['max_single_call'] if data['max_single_call'] > 0 else 0.0
            }
            
            # Generate performance insights
            if percentage > 30:
                summary['performance_insights'].append(
                    f"{component} dominates execution time ({percentage:.1f}%) - primary optimization target"
                )
                summary['optimization_priorities'].append({
                    'component': component,
                    'priority': 'high',
                    'reason': f'Consumes {percentage:.1f}% of total component time'
                })
            elif percentage > 15:
                summary['performance_insights'].append(
                    f"{component} is a significant contributor ({percentage:.1f}%) - secondary optimization target"
                )
                summary['optimization_priorities'].append({
                    'component': component,
                    'priority': 'medium',
                    'reason': f'Consumes {percentage:.1f}% of total component time'
                })
            
            # Check for inefficient call patterns
            if data['call_count'] > 50 and data['avg_call_time'] > 0.1:
                summary['performance_insights'].append(
                    f"{component} has many slow calls ({data['call_count']} calls, avg {data['avg_call_time']:.3f}s)"
                )
                summary['optimization_priorities'].append({
                    'component': component,
                    'priority': 'medium',
                    'reason': f'High call frequency with slow average time'
                })
        
        # Sort optimization priorities by priority level
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        summary['optimization_priorities'].sort(
            key=lambda x: priority_order.get(x['priority'], 0), 
            reverse=True
        )
        
        return summary
    
    def analyze_llm_api_bottlenecks(self, llm_result: 'LLMProfilingResult') -> List[ComponentBottleneck]:
        """
        Analyze LLM API call patterns to identify bottlenecks and optimization opportunities.
        
        Args:
            llm_result: LLM profiling results from LLMAPIProfiler
            
        Returns:
            List of ComponentBottleneck objects for LLM API usage
        """
        bottlenecks = []
        
        if not llm_result or llm_result.total_api_calls == 0:
            return bottlenecks
        
        # Analyze overall LLM usage
        if llm_result.total_llm_time > 5.0:  # Significant LLM time
            llm_bottleneck = ComponentBottleneck(
                component_name='LLM_API_Calls',
                time_spent=llm_result.total_llm_time,
                percentage_of_total=100.0,  # Will be recalculated in context
                optimization_potential='high' if llm_result.total_llm_time > 10.0 else 'medium'
            )
            
            # Add function call details for LLM bottleneck
            llm_bottleneck.function_calls = [
                {
                    'name': f'{call.component}::{call.request_method}',
                    'time': call.total_time,
                    'calls': 1,
                    'avg_time': call.total_time,
                    'prompt_size': call.prompt_size_chars,
                    'response_size': call.response_size_chars,
                    'network_time': call.network_time,
                    'processing_time': call.processing_time
                }
                for call in llm_result.api_calls
            ]
            
            bottlenecks.append(llm_bottleneck)
        
        # Analyze component-specific LLM usage
        for component, total_time in llm_result.component_total_times.items():
            if total_time > 2.0:  # Significant component LLM time
                call_count = llm_result.component_call_counts.get(component, 0)
                
                component_bottleneck = ComponentBottleneck(
                    component_name=f'{component}_LLM_Usage',
                    time_spent=total_time,
                    percentage_of_total=(total_time / llm_result.total_llm_time * 100) if llm_result.total_llm_time > 0 else 0.0,
                    optimization_potential='high' if call_count > 5 else 'medium'
                )
                
                # Add component-specific call details
                component_calls = [call for call in llm_result.api_calls if call.component == component]
                component_bottleneck.function_calls = [
                    {
                        'name': f'{call.component}::API_Call',
                        'time': call.total_time,
                        'calls': 1,
                        'avg_time': call.total_time,
                        'prompt_size': call.prompt_size_chars,
                        'network_percentage': (call.network_time / call.total_time * 100) if call.total_time > 0 else 0.0
                    }
                    for call in component_calls
                ]
                
                bottlenecks.append(component_bottleneck)
        
        return bottlenecks
    
    def generate_llm_optimization_recommendations(self, llm_result: 'LLMProfilingResult') -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on LLM API usage patterns.
        
        Args:
            llm_result: LLM profiling results
            
        Returns:
            List of OptimizationRecommendation objects for LLM usage
        """
        recommendations = []
        
        if not llm_result or llm_result.total_api_calls == 0:
            return recommendations
        
        # High frequency API calls
        if llm_result.total_api_calls > 15:
            recommendations.append(OptimizationRecommendation(
                component='LLM_API_Calls',
                issue_description=f'High number of LLM API calls ({llm_result.total_api_calls} calls)',
                recommendation='Implement request batching, response caching, or reduce redundant calls',
                expected_impact='high',
                implementation_effort='medium'
            ))
        
        # Large prompt sizes
        if llm_result.average_prompt_size > 4000:
            recommendations.append(OptimizationRecommendation(
                component='LLM_API_Calls',
                issue_description=f'Large average prompt size ({llm_result.average_prompt_size:.0f} characters)',
                recommendation='Optimize prompt templates, implement context compression, or use prompt caching',
                expected_impact='medium',
                implementation_effort='low'
            ))
        
        # High network latency
        if llm_result.network_percentage > 35:
            recommendations.append(OptimizationRecommendation(
                component='LLM_API_Calls',
                issue_description=f'High network overhead ({llm_result.network_percentage:.1f}% of LLM time)',
                recommendation='Use closer API endpoints, implement connection pooling, or async request batching',
                expected_impact='medium',
                implementation_effort='medium'
            ))
        
        # Slow API calls
        if len(llm_result.slow_calls) > 0:
            avg_slow_time = sum(call.total_time for call in llm_result.slow_calls) / len(llm_result.slow_calls)
            recommendations.append(OptimizationRecommendation(
                component='LLM_API_Calls',
                issue_description=f'{len(llm_result.slow_calls)} slow API calls detected (avg {avg_slow_time:.1f}s)',
                recommendation='Investigate slow calls for prompt optimization, model selection, or timeout handling',
                expected_impact='high',
                implementation_effort='low'
            ))
        
        # Strong correlation between prompt size and response time
        if abs(llm_result.prompt_size_correlation) > 0.7:
            recommendations.append(OptimizationRecommendation(
                component='LLM_API_Calls',
                issue_description=f'Strong correlation between prompt size and response time (r={llm_result.prompt_size_correlation:.2f})',
                recommendation='Focus on prompt size optimization as primary performance improvement strategy',
                expected_impact='high',
                implementation_effort='low'
            ))
        
        # Component-specific recommendations
        for component, call_count in llm_result.component_call_counts.items():
            total_time = llm_result.component_total_times.get(component, 0.0)
            
            if call_count > 5 and total_time > 5.0:
                avg_time = total_time / call_count
                recommendations.append(OptimizationRecommendation(
                    component=component,
                    issue_description=f'{component} makes frequent LLM calls ({call_count} calls, avg {avg_time:.1f}s)',
                    recommendation=f'Optimize {component} to reduce LLM dependency or implement component-specific caching',
                    expected_impact='medium',
                    implementation_effort='medium'
                ))
        
        return recommendations

    def identify_implement_agent_bottlenecks(self, cprofile_result: CProfileResult) -> List[ComponentBottleneck]:
        """
        Identify specific ImplementAgent component bottlenecks from call patterns.
        
        Args:
            cprofile_result: Analyzed cProfile data
            
        Returns:
            List of ComponentBottleneck objects for ImplementAgent components
        """
        # Aggregate time by component
        component_times = {}
        component_calls = {}
        
        total_time = cprofile_result.total_time
        
        # Analyze functions to identify component bottlenecks
        for func_name, func_data in cprofile_result.function_stats.items():
            cumulative_time = func_data['cumulative_time']
            call_count = func_data['call_count']
            function_name = func_data['function_name']
            filename = func_data['filename']
            
            # Categorize function
            category = self._categorize_function(function_name, filename)
            
            # Only track ImplementAgent components and LLM calls
            if category in ['TaskDecomposer', 'ShellExecutor', 'ErrorRecovery', 
                          'ContextManager', 'ImplementAgent', 'LLM_API_Calls']:
                
                if category not in component_times:
                    component_times[category] = 0.0
                    component_calls[category] = []
                
                component_times[category] += cumulative_time
                component_calls[category].append({
                    'function': func_name,
                    'time': cumulative_time,
                    'calls': call_count,
                    'percentage': (cumulative_time / total_time * 100) if total_time > 0 else 0
                })
        
        # Create ComponentBottleneck objects
        bottlenecks = []
        
        for component, total_component_time in component_times.items():
            percentage = (total_component_time / total_time * 100) if total_time > 0 else 0
            
            # Sort function calls by time spent
            function_calls = sorted(
                component_calls[component],
                key=lambda x: x['time'],
                reverse=True
            )
            
            # Determine optimization potential
            optimization_potential = self._assess_optimization_potential(
                component, total_component_time, percentage, function_calls
            )
            
            bottleneck = ComponentBottleneck(
                component_name=component,
                time_spent=total_component_time,
                percentage_of_total=percentage,
                function_calls=function_calls[:10],  # Top 10 functions
                optimization_potential=optimization_potential
            )
            
            bottlenecks.append(bottleneck)
        
        # Sort by time spent (most significant first)
        return sorted(bottlenecks, key=lambda x: x.time_spent, reverse=True)
    
    def _assess_optimization_potential(self, component: str, time_spent: float, 
                                    percentage: float, function_calls: List[Dict]) -> str:
        """
        Assess optimization potential for a component bottleneck.
        
        Args:
            component: Component name
            time_spent: Total time spent in component
            percentage: Percentage of total execution time
            function_calls: List of function call data
            
        Returns:
            Optimization potential: "high", "medium", or "low"
        """
        # High potential if component takes >20% of total time
        if percentage > 20.0:
            return "high"
        
        # High potential if there are many calls to the same function
        if function_calls:
            top_function = function_calls[0]
            if top_function['calls'] > 100:
                return "high"
        
        # Medium potential if component takes >10% of total time
        if percentage > 10.0:
            return "medium"
        
        # Medium potential for LLM calls (always optimizable)
        if component == 'LLM_API_Calls' and percentage > 5.0:
            return "medium"
        
        # Low potential otherwise
        return "low"
    
    def generate_optimization_recommendations(self, 
                                           bottlenecks: List[ComponentBottleneck],
                                           time_categories: TimeCategories) -> List[OptimizationRecommendation]:
        """
        Generate specific optimization recommendations based on identified bottlenecks.
        
        Args:
            bottlenecks: List of identified component bottlenecks
            time_categories: Time categorization data
            
        Returns:
            List of OptimizationRecommendation objects
        """
        recommendations = []
        
        # Analyze each bottleneck
        for bottleneck in bottlenecks:
            component_recommendations = self._generate_component_recommendations(bottleneck)
            recommendations.extend(component_recommendations)
        
        # Add general recommendations based on time categories
        general_recommendations = self._generate_general_recommendations(time_categories)
        recommendations.extend(general_recommendations)
        
        # Sort by priority score (highest first)
        return sorted(recommendations, key=lambda x: x.priority_score, reverse=True)
    
    def _generate_component_recommendations(self, bottleneck: ComponentBottleneck) -> List[OptimizationRecommendation]:
        """
        Generate recommendations for a specific component bottleneck.
        
        Args:
            bottleneck: ComponentBottleneck to analyze
            
        Returns:
            List of OptimizationRecommendation objects
        """
        recommendations = []
        component = bottleneck.component_name
        
        # Get templates for this component
        if component not in self.recommendation_templates:
            return recommendations
        
        templates = self.recommendation_templates[component]
        
        # High time recommendation
        if bottleneck.is_significant:
            if 'high_time' in templates:
                template = templates['high_time']
                rec = OptimizationRecommendation(
                    component=component,
                    issue_description=f"{template['issue']} ({bottleneck.percentage_of_total:.1f}% of total time)",
                    recommendation=template['recommendation'],
                    expected_impact=template['impact'],
                    implementation_effort=template['effort']
                )
                recommendations.append(rec)
        
        # High call count recommendation
        if bottleneck.function_calls:
            top_function = bottleneck.function_calls[0]
            if top_function['calls'] > 50:
                if 'high_calls' in templates:
                    template = templates['high_calls']
                    rec = OptimizationRecommendation(
                        component=component,
                        issue_description=f"{template['issue']} ({top_function['calls']} calls to {top_function['function']})",
                        recommendation=template['recommendation'],
                        expected_impact=template.get('impact', 'medium'),
                        implementation_effort=template.get('effort', 'medium')
                    )
                    recommendations.append(rec)
        
        # Component-specific recommendations
        if component == 'ShellExecutor':
            # Check for subprocess overhead
            subprocess_calls = sum(1 for call in bottleneck.function_calls 
                                 if 'subprocess' in call['function'].lower())
            if subprocess_calls > 10:
                template = templates.get('subprocess_overhead', {})
                if template:
                    rec = OptimizationRecommendation(
                        component=component,
                        issue_description=f"{template['issue']} ({subprocess_calls} subprocess calls)",
                        recommendation=template['recommendation'],
                        expected_impact=template['impact'],
                        implementation_effort=template['effort']
                    )
                    recommendations.append(rec)
        
        elif component == 'ErrorRecovery':
            # Check for frequent recovery
            recovery_calls = sum(call['calls'] for call in bottleneck.function_calls 
                               if 'recover' in call['function'].lower())
            if recovery_calls > 20:
                template = templates.get('frequent_recovery', {})
                if template:
                    rec = OptimizationRecommendation(
                        component=component,
                        issue_description=f"{template['issue']} ({recovery_calls} recovery attempts)",
                        recommendation=template['recommendation'],
                        expected_impact=template['impact'],
                        implementation_effort=template['effort']
                    )
                    recommendations.append(rec)
        
        elif component == 'LLM_API_Calls':
            # Check for frequent API calls
            api_calls = sum(call['calls'] for call in bottleneck.function_calls)
            if api_calls > 10:
                template = templates.get('frequent_calls', {})
                if template:
                    rec = OptimizationRecommendation(
                        component=component,
                        issue_description=f"{template['issue']} ({api_calls} API calls)",
                        recommendation=template['recommendation'],
                        expected_impact=template['impact'],
                        implementation_effort=template['effort']
                    )
                    recommendations.append(rec)
        
        return recommendations
    
    def _generate_general_recommendations(self, time_categories: TimeCategories) -> List[OptimizationRecommendation]:
        """
        Generate general recommendations based on time category analysis.
        
        Args:
            time_categories: TimeCategories with time breakdown
            
        Returns:
            List of general OptimizationRecommendation objects
        """
        recommendations = []
        
        # Check test infrastructure overhead
        total_time = time_categories.total_time
        test_overhead_percentage = 0.0
        
        if total_time > 0:
            test_time = time_categories.test_setup_time + time_categories.test_teardown_time
            test_overhead_percentage = (test_time / total_time) * 100
        
        if test_overhead_percentage > 30.0:
            template = self.recommendation_templates['Test_Infrastructure']['high_overhead']
            rec = OptimizationRecommendation(
                component='Test_Infrastructure',
                issue_description=f"{template['issue']} ({test_overhead_percentage:.1f}% of total time)",
                recommendation=template['recommendation'],
                expected_impact=template['impact'],
                implementation_effort=template['effort']
            )
            recommendations.append(rec)
        
        # Check ImplementAgent vs test ratio
        component_timings = time_categories.component_timings
        impl_percentage = component_timings.implement_agent_percentage
        
        if impl_percentage < 50.0 and total_time > 5.0:  # Only for tests taking >5 seconds
            rec = OptimizationRecommendation(
                component='General',
                issue_description=f"ImplementAgent execution is only {impl_percentage:.1f}% of total time",
                recommendation="Focus on reducing test infrastructure overhead rather than optimizing ImplementAgent components",
                expected_impact="medium",
                implementation_effort="medium"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def get_bottleneck_summary(self, reports: List[BottleneckReport]) -> Dict[str, Any]:
        """
        Generate summary statistics across multiple bottleneck reports.
        
        Args:
            reports: List of BottleneckReport objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not reports:
            return {}
        
        # Aggregate component times across all reports
        component_totals = {}
        total_tests = len(reports)
        
        for report in reports:
            for bottleneck in report.component_bottlenecks:
                component = bottleneck.component_name
                if component not in component_totals:
                    component_totals[component] = {
                        'total_time': 0.0,
                        'test_count': 0,
                        'avg_percentage': 0.0,
                        'max_percentage': 0.0
                    }
                
                component_totals[component]['total_time'] += bottleneck.time_spent
                component_totals[component]['test_count'] += 1
                component_totals[component]['avg_percentage'] += bottleneck.percentage_of_total
                component_totals[component]['max_percentage'] = max(
                    component_totals[component]['max_percentage'],
                    bottleneck.percentage_of_total
                )
        
        # Calculate averages
        for component_data in component_totals.values():
            if component_data['test_count'] > 0:
                component_data['avg_percentage'] /= component_data['test_count']
        
        # Get most common recommendations
        all_recommendations = []
        for report in reports:
            all_recommendations.extend(report.optimization_recommendations)
        
        recommendation_counts = {}
        for rec in all_recommendations:
            key = f"{rec.component}:{rec.issue_description}"
            if key not in recommendation_counts:
                recommendation_counts[key] = {
                    'count': 0,
                    'recommendation': rec
                }
            recommendation_counts[key]['count'] += 1
        
        # Sort by frequency
        common_recommendations = sorted(
            recommendation_counts.values(),
            key=lambda x: x['count'],
            reverse=True
        )[:5]  # Top 5 most common
        
        return {
            'total_tests_analyzed': total_tests,
            'component_bottlenecks': component_totals,
            'most_common_recommendations': [
                {
                    'count': item['count'],
                    'component': item['recommendation'].component,
                    'issue': item['recommendation'].issue_description,
                    'recommendation': item['recommendation'].recommendation,
                    'impact': item['recommendation'].expected_impact
                }
                for item in common_recommendations
            ],
            'avg_test_vs_implementation_ratio': sum(
                report.test_vs_implementation_ratio for report in reports
            ) / total_tests if total_tests > 0 else 0.0
        }