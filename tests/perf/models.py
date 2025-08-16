"""
Data models for performance analysis components.

This module defines all data structures used throughout the performance
analysis infrastructure for timing results, profiling data, and reports.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from enum import Enum
import time

# Forward reference for LLMProfilingResult to avoid circular imports
if TYPE_CHECKING:
    from .llm_profiler import LLMProfilingResult


class TestStatus(Enum):
    """Test execution status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


class ProfilerType(Enum):
    """Available profiler types."""
    CPROFILE = "cProfile"
    PYSPY = "py-spy"
    PERF = "perf"


@dataclass
class TestIdentifier:
    """Identifies a specific test for execution and analysis."""
    file_path: str
    class_name: Optional[str] = None
    method_name: Optional[str] = None
    full_name: str = field(init=False)
    
    def __post_init__(self):
        """Generate full test name from components."""
        if self.method_name and self.class_name:
            # Full method specification: file::class::method
            self.full_name = f"{self.file_path}::{self.class_name}::{self.method_name}"
        elif self.method_name:
            # Method without class: file::method
            self.full_name = f"{self.file_path}::{self.method_name}"
        else:
            # File-level test
            self.full_name = self.file_path


@dataclass
class ExecutionResult:
    """Result of executing a shell command or test."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    timeout_occurred: bool = False
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.exit_code == 0 and not self.timeout_occurred


@dataclass
class TestTimingResult:
    """Timing result for a single test method execution."""
    test_identifier: TestIdentifier
    execution_time: float
    status: TestStatus
    stdout: str = ""
    stderr: str = ""
    timeout_occurred: bool = False
    error_message: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def duration_ms(self) -> float:
        """Get execution time in milliseconds."""
        return self.execution_time * 1000


@dataclass
class FileTimingResult:
    """Timing results for all tests in a single file."""
    file_path: str
    total_execution_time: float
    test_results: List[TestTimingResult] = field(default_factory=list)
    file_status: TestStatus = TestStatus.PASSED
    
    @property
    def test_count(self) -> int:
        """Total number of tests in file."""
        return len(self.test_results)
    
    @property
    def passed_count(self) -> int:
        """Number of passed tests."""
        return sum(1 for result in self.test_results if result.status == TestStatus.PASSED)
    
    @property
    def failed_count(self) -> int:
        """Number of failed tests."""
        return sum(1 for result in self.test_results if result.status == TestStatus.FAILED)
    
    @property
    def timeout_count(self) -> int:
        """Number of timed out tests."""
        return sum(1 for result in self.test_results if result.status == TestStatus.TIMEOUT)


@dataclass
class TimingReport:
    """Comprehensive timing report for all analyzed tests."""
    total_execution_time: float
    file_results: List[FileTimingResult] = field(default_factory=list)
    analysis_timestamp: float = field(default_factory=time.time)
    timeout_threshold: float = 60.0
    
    @property
    def total_test_count(self) -> int:
        """Total number of tests analyzed."""
        return sum(file_result.test_count for file_result in self.file_results)
    
    @property
    def slowest_tests(self) -> List[TestTimingResult]:
        """Get slowest tests across all files, sorted by execution time."""
        all_tests = []
        for file_result in self.file_results:
            all_tests.extend(file_result.test_results)
        return sorted(all_tests, key=lambda x: x.execution_time, reverse=True)
    
    @property
    def timeout_tests(self) -> List[TestTimingResult]:
        """Get all tests that timed out."""
        timeout_tests = []
        for file_result in self.file_results:
            timeout_tests.extend([
                test for test in file_result.test_results 
                if test.status == TestStatus.TIMEOUT
            ])
        return timeout_tests


@dataclass
class CProfileResult:
    """Result from cProfile profiling analysis."""
    test_identifier: TestIdentifier
    profile_file_path: str
    total_time: float
    function_stats: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=dict)
    top_functions: List[Dict[str, Any]] = field(default_factory=list)
    call_count: int = 0
    
    @property
    def stats_summary(self) -> Dict[str, Any]:
        """Get summary statistics from profiling data."""
        return {
            'total_calls': self.call_count,
            'total_time': self.total_time,
            'top_function_count': len(self.top_functions),
            'profile_file': self.profile_file_path
        }


@dataclass
class PySpyResult:
    """Result from py-spy profiling analysis."""
    test_identifier: TestIdentifier
    flamegraph_path: str
    sampling_duration: float
    sample_count: int = 0
    process_id: Optional[int] = None
    
    @property
    def samples_per_second(self) -> float:
        """Calculate sampling rate."""
        if self.sampling_duration > 0:
            return self.sample_count / self.sampling_duration
        return 0.0


@dataclass
class ProfilingResult:
    """Combined profiling result from multiple profilers."""
    test_identifier: TestIdentifier
    profiler_used: ProfilerType
    cprofile_result: Optional[CProfileResult] = None
    pyspy_result: Optional[PySpyResult] = None
    component_timings: Dict[str, float] = field(default_factory=dict)
    total_profiling_time: float = 0.0
    profiling_overhead: float = 0.0
    # Task 8: LLM API call profiling results
    llm_profiling_result: Optional['LLMProfilingResult'] = None
    
    @property
    def primary_result(self) -> Union[CProfileResult, PySpyResult, None]:
        """Get the primary profiling result based on profiler type."""
        if self.profiler_used == ProfilerType.CPROFILE and self.cprofile_result:
            return self.cprofile_result
        elif self.profiler_used == ProfilerType.PYSPY and self.pyspy_result:
            return self.pyspy_result
        return None
    
    @property
    def has_llm_data(self) -> bool:
        """Check if LLM profiling data is available."""
        return (self.llm_profiling_result is not None and 
                self.llm_profiling_result.total_api_calls > 0)


@dataclass
class ComponentTimings:
    """Timing breakdown for ImplementAgent components."""
    task_decomposer: float = 0.0
    shell_executor: float = 0.0
    error_recovery: float = 0.0
    context_manager: float = 0.0
    llm_api_calls: float = 0.0
    test_overhead: float = 0.0
    other: float = 0.0
    
    @property
    def total_component_time(self) -> float:
        """Total time spent in ImplementAgent components."""
        return (self.task_decomposer + self.shell_executor + 
                self.error_recovery + self.context_manager + self.llm_api_calls)
    
    @property
    def implement_agent_percentage(self) -> float:
        """Percentage of time spent in ImplementAgent vs test overhead."""
        total = self.total_component_time + self.test_overhead + self.other
        if total > 0:
            return (self.total_component_time / total) * 100
        return 0.0


@dataclass
class TimeCategories:
    """Categorization of time spent during test execution."""
    test_setup_time: float = 0.0
    test_teardown_time: float = 0.0
    implement_agent_time: float = 0.0
    component_timings: ComponentTimings = field(default_factory=ComponentTimings)
    
    @property
    def total_time(self) -> float:
        """Total categorized time."""
        return (self.test_setup_time + self.test_teardown_time + 
                self.implement_agent_time)


@dataclass
class ComponentBottleneck:
    """Identified bottleneck in a specific component."""
    component_name: str
    time_spent: float
    percentage_of_total: float
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    optimization_potential: str = "unknown"  # "high", "medium", "low"
    
    @property
    def is_significant(self) -> bool:
        """Check if this bottleneck is significant (>10% of total time)."""
        return self.percentage_of_total > 10.0


@dataclass
class OptimizationRecommendation:
    """Specific optimization recommendation based on analysis."""
    component: str
    issue_description: str
    recommendation: str
    expected_impact: str  # "high", "medium", "low"
    implementation_effort: str  # "high", "medium", "low"
    priority_score: float = 0.0
    
    def __post_init__(self):
        """Calculate priority score based on impact and effort."""
        impact_scores = {"high": 3, "medium": 2, "low": 1}
        effort_scores = {"low": 3, "medium": 2, "high": 1}
        
        impact = impact_scores.get(self.expected_impact.lower(), 1)
        effort = effort_scores.get(self.implementation_effort.lower(), 1)
        self.priority_score = impact * effort


@dataclass
class BottleneckReport:
    """Comprehensive bottleneck analysis report."""
    test_identifier: TestIdentifier
    time_categories: TimeCategories
    component_bottlenecks: List[ComponentBottleneck] = field(default_factory=list)
    optimization_recommendations: List[OptimizationRecommendation] = field(default_factory=list)
    analysis_timestamp: float = field(default_factory=time.time)
    # Task 7 enhancements: detailed component timing extraction
    detailed_component_timings: Optional[Dict[str, Dict[str, Any]]] = None
    component_performance_summary: Optional[Dict[str, Any]] = None
    
    @property
    def top_bottlenecks(self) -> List[ComponentBottleneck]:
        """Get top bottlenecks sorted by time spent."""
        return sorted(self.component_bottlenecks, 
                     key=lambda x: x.time_spent, reverse=True)
    
    @property
    def high_priority_recommendations(self) -> List[OptimizationRecommendation]:
        """Get high priority optimization recommendations."""
        return sorted([rec for rec in self.optimization_recommendations 
                      if rec.expected_impact == "high"],
                     key=lambda x: x.priority_score, reverse=True)
    
    @property
    def test_vs_implementation_ratio(self) -> float:
        """Ratio of test overhead to ImplementAgent execution time."""
        test_time = (self.time_categories.test_setup_time + 
                    self.time_categories.test_teardown_time)
        impl_time = self.time_categories.implement_agent_time
        
        if impl_time > 0:
            return test_time / impl_time
        return float('inf')
    
    @property
    def context_loading_performance(self) -> Dict[str, float]:
        """Get context loading performance metrics (Task 7 requirement)."""
        if not self.detailed_component_timings:
            return {'context_loading_time': 0.0, 'percentage_of_context_manager': 0.0}
        
        context_data = self.detailed_component_timings.get('ContextManager', {})
        context_loading_time = context_data.get('context_loading_time', 0.0)
        total_context_time = context_data.get('total_time', 0.0)
        
        percentage = (context_loading_time / total_context_time * 100) if total_context_time > 0 else 0.0
        
        return {
            'context_loading_time': context_loading_time,
            'percentage_of_context_manager': percentage,
            'total_context_manager_time': total_context_time
        }
    
    @property
    def component_efficiency_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get efficiency metrics for each component (Task 7 requirement)."""
        if not self.detailed_component_timings:
            return {}
        
        efficiency_metrics = {}
        
        for component, data in self.detailed_component_timings.items():
            if data['call_count'] > 0:
                efficiency_metrics[component] = {
                    'calls_per_second': data['call_count'] / data['total_time'] if data['total_time'] > 0 else 0.0,
                    'avg_call_efficiency': data['avg_call_time'],
                    'max_call_overhead': data['max_single_call'],
                    'efficiency_ratio': data['avg_call_time'] / data['max_single_call'] if data['max_single_call'] > 0 else 1.0
                }
        
        return efficiency_metrics