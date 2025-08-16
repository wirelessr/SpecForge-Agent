"""
Comprehensive validation test runner for performance analysis tools.

This module provides a comprehensive test suite that validates all aspects
of the performance analysis infrastructure including accuracy, reliability,
integration, and error handling.
"""

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .models import TestIdentifier, TestStatus, ProfilerType
from .test_executor import TestExecutor
from .timing_analyzer import TestTimingAnalyzer
from .profiler import PerformanceProfiler
from .bottleneck_analyzer import BottleneckAnalyzer
from .config import PerformanceConfig


class ComprehensiveValidationSuite:
    """
    Comprehensive validation suite for performance analysis tools.
    
    This class orchestrates a complete validation of the performance analysis
    infrastructure, testing accuracy, reliability, integration, and error handling.
    """
    
    def __init__(self):
        """Initialize validation suite."""
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Initialize components
        self.test_executor = TestExecutor(timeout_seconds=30)
        self.timing_analyzer = TestTimingAnalyzer(timeout_seconds=30)
        self.profiler = PerformanceProfiler()
        self.bottleneck_analyzer = BottleneckAnalyzer()
    
    def cleanup(self):
        """Cleanup validation suite resources."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation of all performance analysis tools.
        
        Returns:
            Dictionary with validation results for each component
        """
        self.logger.info("Starting comprehensive performance analysis validation")
        
        validation_results = {
            'test_discovery': await self._validate_test_discovery(),
            'timing_accuracy': await self._validate_timing_accuracy(),
            'timeout_handling': await self._validate_timeout_handling(),
            'profiling_data_collection': await self._validate_profiling_data_collection(),
            'bottleneck_identification': await self._validate_bottleneck_identification(),
            'optimization_recommendations': await self._validate_optimization_recommendations(),
            'integration_workflow': await self._validate_integration_workflow(),
            'error_recovery': await self._validate_error_recovery(),
            'performance_overhead': await self._validate_performance_overhead(),
            'report_generation': await self._validate_report_generation()
        }
        
        # Calculate overall validation score
        validation_results['overall_score'] = self._calculate_overall_score(validation_results)
        validation_results['validation_timestamp'] = time.time()
        
        self.logger.info(f"Comprehensive validation completed. Overall score: {validation_results['overall_score']:.2f}")
        
        return validation_results
    
    async def _validate_test_discovery(self) -> Dict[str, Any]:
        """Validate test discovery functionality."""
        self.logger.info("Validating test discovery...")
        
        try:
            # Test discovery
            discovered_tests = self.test_executor.discover_integration_tests()
            
            # Validate discovery results
            validation_score = 0.0
            issues = []
            
            if len(discovered_tests) > 0:
                validation_score += 40  # Found tests
                
                # Validate test identifier structure
                valid_identifiers = 0
                for test in discovered_tests:
                    if (isinstance(test, TestIdentifier) and 
                        test.file_path and 
                        test.full_name):
                        valid_identifiers += 1
                
                if valid_identifiers == len(discovered_tests):
                    validation_score += 30  # All identifiers valid
                else:
                    issues.append(f"Invalid identifiers: {len(discovered_tests) - valid_identifiers}")
                
                # Check for expected test patterns
                test_files = {test.file_path for test in discovered_tests}
                integration_files = [f for f in test_files if 'integration' in f]
                
                if len(integration_files) > 0:
                    validation_score += 30  # Found integration tests
                else:
                    issues.append("No integration test files found")
            else:
                issues.append("No tests discovered")
            
            return {
                'score': validation_score,
                'max_score': 100,
                'tests_discovered': len(discovered_tests),
                'issues': issues,
                'status': 'passed' if validation_score >= 70 else 'failed'
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 100,
                'tests_discovered': 0,
                'issues': [f"Discovery failed: {str(e)}"],
                'status': 'error'
            }
    
    async def _validate_timing_accuracy(self) -> Dict[str, Any]:
        """Validate timing measurement accuracy."""
        self.logger.info("Validating timing accuracy...")
        
        try:
            # Find a simple test to measure
            discovered_tests = self.test_executor.discover_integration_tests()
            
            if not discovered_tests:
                return {
                    'score': 0,
                    'max_score': 100,
                    'issues': ['No tests available for timing validation'],
                    'status': 'skipped'
                }
            
            # Use first available test
            test_identifier = discovered_tests[0]
            
            # Measure timing multiple times
            timing_measurements = []
            for i in range(3):  # 3 measurements for consistency check
                start_time = time.time()
                result = await self.test_executor.execute_single_test_method(
                    test_identifier.file_path, test_identifier.method_name
                )
                end_time = time.time()
                
                if result.status == TestStatus.PASSED:
                    manual_timing = end_time - start_time
                    timing_measurements.append({
                        'measured_time': result.execution_time,
                        'manual_time': manual_timing,
                        'difference': abs(result.execution_time - manual_timing)
                    })
            
            validation_score = 0.0
            issues = []
            
            if len(timing_measurements) > 0:
                validation_score += 30  # Got timing measurements
                
                # Check accuracy (within 20% tolerance)
                accurate_measurements = 0
                for measurement in timing_measurements:
                    tolerance = measurement['manual_time'] * 0.2
                    if measurement['difference'] <= tolerance:
                        accurate_measurements += 1
                
                accuracy_percentage = (accurate_measurements / len(timing_measurements)) * 100
                validation_score += (accuracy_percentage / 100) * 40  # Up to 40 points for accuracy
                
                # Check consistency (coefficient of variation < 50%)
                measured_times = [m['measured_time'] for m in timing_measurements]
                if len(measured_times) > 1:
                    import statistics
                    mean_time = statistics.mean(measured_times)
                    std_dev = statistics.stdev(measured_times)
                    cv = std_dev / mean_time if mean_time > 0 else 0
                    
                    if cv < 0.5:
                        validation_score += 30  # Consistent measurements
                    else:
                        issues.append(f"Inconsistent timing: CV={cv:.2f}")
                
                if accuracy_percentage < 80:
                    issues.append(f"Low accuracy: {accuracy_percentage:.1f}%")
            else:
                issues.append("No successful timing measurements")
            
            return {
                'score': validation_score,
                'max_score': 100,
                'measurements': len(timing_measurements),
                'accuracy_percentage': accuracy_percentage if 'accuracy_percentage' in locals() else 0,
                'issues': issues,
                'status': 'passed' if validation_score >= 70 else 'failed'
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 100,
                'measurements': 0,
                'issues': [f"Timing validation failed: {str(e)}"],
                'status': 'error'
            }
    
    async def _validate_timeout_handling(self) -> Dict[str, Any]:
        """Validate timeout handling functionality."""
        self.logger.info("Validating timeout handling...")
        
        try:
            # Use very short timeout to test timeout handling
            short_executor = TestExecutor(timeout_seconds=1)
            
            # Find a test that might timeout
            discovered_tests = self.test_executor.discover_integration_tests()
            
            if not discovered_tests:
                return {
                    'score': 0,
                    'max_score': 100,
                    'issues': ['No tests available for timeout validation'],
                    'status': 'skipped'
                }
            
            # Try to execute a test with short timeout
            test_identifier = discovered_tests[0]
            
            start_time = time.time()
            result = await short_executor.execute_single_test_method(
                test_identifier.file_path, test_identifier.method_name
            )
            end_time = time.time()
            
            actual_duration = end_time - start_time
            
            validation_score = 0.0
            issues = []
            
            # Check timeout enforcement
            if result.status == TestStatus.TIMEOUT:
                validation_score += 40  # Timeout detected
                
                # Check timeout was enforced within reasonable time
                if actual_duration <= 3:  # 1s timeout + 2s tolerance
                    validation_score += 30  # Timeout enforced accurately
                else:
                    issues.append(f"Timeout not enforced: took {actual_duration:.2f}s")
                
                # Check timeout indicators
                if result.timeout_occurred:
                    validation_score += 15  # Timeout flag set
                
                if result.error_message and 'timeout' in result.error_message.lower():
                    validation_score += 15  # Error message indicates timeout
                else:
                    issues.append("Timeout error message missing or unclear")
            
            elif result.status == TestStatus.PASSED:
                # Test completed within timeout - also valid
                if actual_duration < 1:
                    validation_score += 70  # Fast test, timeout handling not needed
                else:
                    issues.append("Test should have timed out but didn't")
            
            else:
                # Test failed for other reasons
                validation_score += 20  # At least executed without crashing
                issues.append(f"Test failed with status: {result.status}")
            
            return {
                'score': validation_score,
                'max_score': 100,
                'timeout_occurred': result.status == TestStatus.TIMEOUT,
                'actual_duration': actual_duration,
                'issues': issues,
                'status': 'passed' if validation_score >= 70 else 'failed'
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 100,
                'timeout_occurred': False,
                'issues': [f"Timeout validation failed: {str(e)}"],
                'status': 'error'
            }
    
    async def _validate_profiling_data_collection(self) -> Dict[str, Any]:
        """Validate profiling data collection functionality."""
        self.logger.info("Validating profiling data collection...")
        
        try:
            # Find a test to profile
            discovered_tests = self.test_executor.discover_integration_tests()
            
            if not discovered_tests:
                return {
                    'score': 0,
                    'max_score': 100,
                    'issues': ['No tests available for profiling validation'],
                    'status': 'skipped'
                }
            
            test_identifier = discovered_tests[0]
            
            # Profile the test
            profiling_result = await self.profiler.profile_with_cprofile(test_identifier)
            
            validation_score = 0.0
            issues = []
            
            if profiling_result:
                validation_score += 20  # Got profiling result
                
                # Check profiling result structure
                if profiling_result.test_identifier == test_identifier:
                    validation_score += 10  # Correct test identifier
                
                if profiling_result.profiler_used == ProfilerType.CPROFILE:
                    validation_score += 10  # Correct profiler type
                
                if profiling_result.total_profiling_time > 0:
                    validation_score += 10  # Reasonable profiling time
                
                # Check cProfile data
                if profiling_result.cprofile_result:
                    validation_score += 20  # Got cProfile data
                    
                    cprofile_data = profiling_result.cprofile_result
                    
                    if cprofile_data.total_time > 0:
                        validation_score += 10  # Reasonable total time
                    
                    if cprofile_data.call_count > 0:
                        validation_score += 10  # Got function calls
                    
                    if len(cprofile_data.function_stats) > 0:
                        validation_score += 10  # Got function statistics
                        
                        # Check function stats structure
                        valid_stats = 0
                        for func_name, func_data in cprofile_data.function_stats.items():
                            required_fields = ['cumulative_time', 'total_time', 'call_count', 'filename', 'function_name']
                            if all(field in func_data for field in required_fields):
                                valid_stats += 1
                        
                        if valid_stats == len(cprofile_data.function_stats):
                            validation_score += 10  # All function stats valid
                        else:
                            issues.append(f"Invalid function stats: {len(cprofile_data.function_stats) - valid_stats}")
                    else:
                        issues.append("No function statistics collected")
                else:
                    issues.append("No cProfile data collected")
            else:
                issues.append("No profiling result generated")
            
            return {
                'score': validation_score,
                'max_score': 100,
                'profiling_completed': profiling_result is not None,
                'cprofile_data_available': profiling_result.cprofile_result is not None if profiling_result else False,
                'issues': issues,
                'status': 'passed' if validation_score >= 70 else 'failed'
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 100,
                'profiling_completed': False,
                'issues': [f"Profiling validation failed: {str(e)}"],
                'status': 'error'
            }
    
    async def _validate_bottleneck_identification(self) -> Dict[str, Any]:
        """Validate bottleneck identification functionality."""
        self.logger.info("Validating bottleneck identification...")
        
        try:
            # Create mock profiling data with known bottlenecks
            from .models import CProfileResult, ProfilingResult
            
            test_identifier = TestIdentifier(
                file_path="validation_test.py",
                method_name="test_bottleneck"
            )
            
            # Create function stats with clear bottlenecks
            function_stats = {
                'autogen_framework/agents/implement_agent.py:45(execute_task)': {
                    'cumulative_time': 8.0,  # High time - should be bottleneck
                    'total_time': 2.0,
                    'call_count': 1,
                    'filename': 'autogen_framework/agents/implement_agent.py',
                    'function_name': 'execute_task'
                },
                'autogen_framework/agents/task_decomposer.py:30(decompose_task)': {
                    'cumulative_time': 5.0,  # Moderate time
                    'total_time': 3.0,
                    'call_count': 10,  # High call count
                    'filename': 'autogen_framework/agents/task_decomposer.py',
                    'function_name': 'decompose_task'
                },
                'openai/api.py:100(chat_completion)': {
                    'cumulative_time': 6.0,  # LLM API calls
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
                test_identifier=test_identifier,
                profile_file_path="/tmp/validation.prof",
                total_time=20.0,
                function_stats=function_stats,
                top_functions=[],
                call_count=15
            )
            
            profiling_result = ProfilingResult(
                test_identifier=test_identifier,
                profiler_used=ProfilerType.CPROFILE,
                cprofile_result=cprofile_result
            )
            
            # Analyze bottlenecks
            bottleneck_reports = self.bottleneck_analyzer.analyze_cprofile_results([profiling_result])
            
            validation_score = 0.0
            issues = []
            
            if len(bottleneck_reports) > 0:
                validation_score += 20  # Generated bottleneck report
                
                report = bottleneck_reports[0]
                
                # Check report structure
                if hasattr(report, 'component_bottlenecks') and len(report.component_bottlenecks) > 0:
                    validation_score += 20  # Identified bottlenecks
                    
                    # Check for expected bottleneck components
                    bottleneck_components = {b.component_name for b in report.component_bottlenecks}
                    expected_components = {'ImplementAgent', 'TaskDecomposer', 'LLM_API_Calls'}
                    
                    found_expected = bottleneck_components.intersection(expected_components)
                    if len(found_expected) > 0:
                        validation_score += 20  # Found expected components
                    else:
                        issues.append(f"Expected components not found: {expected_components}")
                    
                    # Check bottleneck significance
                    significant_bottlenecks = [b for b in report.component_bottlenecks if b.is_significant]
                    if len(significant_bottlenecks) > 0:
                        validation_score += 15  # Found significant bottlenecks
                    
                    # Check bottleneck sorting
                    bottleneck_times = [b.time_spent for b in report.component_bottlenecks]
                    if bottleneck_times == sorted(bottleneck_times, reverse=True):
                        validation_score += 10  # Properly sorted
                    else:
                        issues.append("Bottlenecks not sorted by time spent")
                
                # Check time categorization
                if hasattr(report, 'time_categories'):
                    validation_score += 15  # Has time categorization
                    
                    if report.time_categories.implement_agent_time > 0:
                        validation_score += 10  # Identified ImplementAgent time
                    else:
                        issues.append("No ImplementAgent time identified")
                else:
                    issues.append("No time categorization")
            else:
                issues.append("No bottleneck reports generated")
            
            return {
                'score': validation_score,
                'max_score': 100,
                'bottlenecks_identified': len(bottleneck_reports[0].component_bottlenecks) if bottleneck_reports else 0,
                'significant_bottlenecks': len([b for b in bottleneck_reports[0].component_bottlenecks if b.is_significant]) if bottleneck_reports else 0,
                'issues': issues,
                'status': 'passed' if validation_score >= 70 else 'failed'
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 100,
                'bottlenecks_identified': 0,
                'issues': [f"Bottleneck validation failed: {str(e)}"],
                'status': 'error'
            }
    
    async def _validate_optimization_recommendations(self) -> Dict[str, Any]:
        """Validate optimization recommendation generation."""
        self.logger.info("Validating optimization recommendations...")
        
        try:
            # Use the same mock data as bottleneck identification
            from .models import CProfileResult, ProfilingResult, ComponentBottleneck
            
            test_identifier = TestIdentifier(
                file_path="validation_test.py",
                method_name="test_optimization"
            )
            
            # Create bottlenecks that should trigger recommendations
            bottlenecks = [
                ComponentBottleneck(
                    component_name='TaskDecomposer',
                    time_spent=5.0,
                    percentage_of_total=25.0,
                    function_calls=[{'function': 'decompose_task', 'calls': 50, 'time': 3.0}],
                    optimization_potential='high'
                ),
                ComponentBottleneck(
                    component_name='LLM_API_Calls',
                    time_spent=6.0,
                    percentage_of_total=30.0,
                    function_calls=[{'function': 'chat_completion', 'calls': 20, 'time': 5.5}],
                    optimization_potential='high'
                )
            ]
            
            from .models import TimeCategories, ComponentTimings
            time_categories = TimeCategories(
                implement_agent_time=11.0,
                component_timings=ComponentTimings(
                    task_decomposer=5.0,
                    llm_api_calls=6.0,
                    test_overhead=1.0
                )
            )
            
            # Generate recommendations
            recommendations = self.bottleneck_analyzer.generate_optimization_recommendations(
                bottlenecks, time_categories
            )
            
            validation_score = 0.0
            issues = []
            
            if len(recommendations) > 0:
                validation_score += 30  # Generated recommendations
                
                # Check recommendation structure
                valid_recommendations = 0
                for rec in recommendations:
                    if (rec.component and rec.issue_description and rec.recommendation and
                        rec.expected_impact in ['high', 'medium', 'low'] and
                        rec.implementation_effort in ['high', 'medium', 'low'] and
                        rec.priority_score > 0):
                        valid_recommendations += 1
                
                if valid_recommendations == len(recommendations):
                    validation_score += 20  # All recommendations valid
                else:
                    issues.append(f"Invalid recommendations: {len(recommendations) - valid_recommendations}")
                
                # Check recommendation sorting
                priority_scores = [rec.priority_score for rec in recommendations]
                if priority_scores == sorted(priority_scores, reverse=True):
                    validation_score += 15  # Properly sorted by priority
                else:
                    issues.append("Recommendations not sorted by priority")
                
                # Check for high-priority recommendations for significant bottlenecks
                high_priority_recs = [rec for rec in recommendations if rec.expected_impact == 'high']
                if len(high_priority_recs) > 0:
                    validation_score += 20  # Has high-priority recommendations
                
                # Check recommendation relevance
                rec_components = {rec.component.lower() for rec in recommendations}
                expected_components = {'taskdecomposer', 'llm', 'api'}
                
                if any(comp in ' '.join(rec_components) for comp in expected_components):
                    validation_score += 15  # Relevant recommendations
                else:
                    issues.append("Recommendations not relevant to bottlenecks")
            else:
                issues.append("No optimization recommendations generated")
            
            return {
                'score': validation_score,
                'max_score': 100,
                'recommendations_generated': len(recommendations),
                'high_priority_recommendations': len([r for r in recommendations if r.expected_impact == 'high']),
                'issues': issues,
                'status': 'passed' if validation_score >= 70 else 'failed'
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 100,
                'recommendations_generated': 0,
                'issues': [f"Optimization validation failed: {str(e)}"],
                'status': 'error'
            }
    
    async def _validate_integration_workflow(self) -> Dict[str, Any]:
        """Validate end-to-end integration workflow."""
        self.logger.info("Validating integration workflow...")
        
        try:
            validation_score = 0.0
            issues = []
            
            # Step 1: Test discovery and timing analysis
            timing_report = await self.timing_analyzer.analyze_all_integration_tests()
            
            if timing_report and timing_report.total_test_count > 0:
                validation_score += 25  # Timing analysis completed
                
                # Step 2: Identify slow tests
                slow_tests = self.timing_analyzer.identify_slow_tests(
                    timing_report, threshold_seconds=0.1  # Low threshold for testing
                )
                
                if len(slow_tests) == 0:
                    # Use any passed test for workflow testing
                    all_tests = []
                    for file_result in timing_report.file_results:
                        all_tests.extend([
                            test for test in file_result.test_results 
                            if test.status == TestStatus.PASSED
                        ])
                    slow_tests = all_tests[:1]
                
                if len(slow_tests) > 0:
                    validation_score += 25  # Identified tests for profiling
                    
                    # Step 3: Profile slow tests
                    profiling_results = await self.profiler.profile_slow_tests(
                        slow_tests[:1], max_tests=1
                    )
                    
                    if len(profiling_results) > 0:
                        validation_score += 25  # Profiling completed
                        
                        # Step 4: Analyze bottlenecks
                        bottleneck_reports = self.bottleneck_analyzer.analyze_cprofile_results(profiling_results)
                        
                        if len(bottleneck_reports) > 0:
                            validation_score += 25  # Bottleneck analysis completed
                        else:
                            issues.append("Bottleneck analysis failed")
                    else:
                        issues.append("Profiling failed")
                else:
                    issues.append("No tests available for profiling")
            else:
                issues.append("Timing analysis failed or no tests found")
            
            return {
                'score': validation_score,
                'max_score': 100,
                'workflow_completed': validation_score == 100,
                'issues': issues,
                'status': 'passed' if validation_score >= 70 else 'failed'
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 100,
                'workflow_completed': False,
                'issues': [f"Integration workflow validation failed: {str(e)}"],
                'status': 'error'
            }
    
    async def _validate_error_recovery(self) -> Dict[str, Any]:
        """Validate error recovery functionality."""
        self.logger.info("Validating error recovery...")
        
        try:
            validation_score = 0.0
            issues = []
            
            # Test 1: Invalid profiling data handling
            from .models import ProfilingResult
            
            invalid_result = ProfilingResult(
                test_identifier=TestIdentifier(file_path="invalid.py"),
                profiler_used=ProfilerType.CPROFILE,
                cprofile_result=None  # Invalid data
            )
            
            try:
                reports = self.bottleneck_analyzer.analyze_cprofile_results([invalid_result])
                if isinstance(reports, list):
                    validation_score += 25  # Handled invalid data gracefully
                else:
                    issues.append("Invalid data not handled properly")
            except Exception as e:
                issues.append(f"Invalid data caused crash: {str(e)}")
            
            # Test 2: Timeout recovery in batch analysis
            short_analyzer = TestTimingAnalyzer(timeout_seconds=1)
            
            try:
                timing_report = await short_analyzer.analyze_all_integration_tests()
                if timing_report:
                    validation_score += 25  # Completed despite potential timeouts
                    
                    # Check for timeout handling
                    timeout_tests = short_analyzer.identify_timeout_tests(timing_report)
                    if len(timeout_tests) > 0:
                        validation_score += 25  # Properly identified timeouts
                    else:
                        validation_score += 15  # No timeouts occurred (also valid)
                else:
                    issues.append("Timing analysis failed with short timeout")
            except Exception as e:
                issues.append(f"Timeout handling failed: {str(e)}")
            
            # Test 3: Empty results handling
            try:
                empty_reports = self.bottleneck_analyzer.analyze_cprofile_results([])
                if isinstance(empty_reports, list) and len(empty_reports) == 0:
                    validation_score += 25  # Handled empty input correctly
                else:
                    issues.append("Empty input not handled correctly")
            except Exception as e:
                issues.append(f"Empty input caused crash: {str(e)}")
            
            return {
                'score': validation_score,
                'max_score': 100,
                'error_recovery_working': validation_score >= 70,
                'issues': issues,
                'status': 'passed' if validation_score >= 70 else 'failed'
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 100,
                'error_recovery_working': False,
                'issues': [f"Error recovery validation failed: {str(e)}"],
                'status': 'error'
            }
    
    async def _validate_performance_overhead(self) -> Dict[str, Any]:
        """Validate performance overhead of analysis tools."""
        self.logger.info("Validating performance overhead...")
        
        try:
            # Find a test to measure overhead
            discovered_tests = self.test_executor.discover_integration_tests()
            
            if not discovered_tests:
                return {
                    'score': 0,
                    'max_score': 100,
                    'issues': ['No tests available for overhead validation'],
                    'status': 'skipped'
                }
            
            test_identifier = discovered_tests[0]
            
            # Measure baseline execution time
            baseline_start = time.time()
            baseline_result = await self.test_executor.execute_single_test_method(
                test_identifier.file_path, test_identifier.method_name
            )
            baseline_end = time.time()
            baseline_time = baseline_end - baseline_start
            
            if baseline_result.status != TestStatus.PASSED:
                return {
                    'score': 0,
                    'max_score': 100,
                    'issues': ['Baseline test execution failed'],
                    'status': 'failed'
                }
            
            # Measure profiling overhead
            profiling_result = await self.profiler.profile_with_cprofile(test_identifier)
            profiling_time = profiling_result.total_profiling_time if profiling_result else 0
            
            validation_score = 0.0
            issues = []
            
            if profiling_time > 0 and baseline_time > 0:
                overhead_ratio = profiling_time / baseline_time
                
                # Profiling should take longer but not excessively
                if overhead_ratio >= 1.0:
                    validation_score += 30  # Profiling takes longer (expected)
                    
                    if overhead_ratio <= 5.0:
                        validation_score += 40  # Reasonable overhead (< 5x)
                    elif overhead_ratio <= 10.0:
                        validation_score += 20  # Acceptable overhead (< 10x)
                        issues.append(f"High profiling overhead: {overhead_ratio:.1f}x")
                    else:
                        issues.append(f"Excessive profiling overhead: {overhead_ratio:.1f}x")
                else:
                    issues.append("Profiling time less than baseline (unexpected)")
                
                # Check profiling overhead calculation
                if hasattr(profiling_result, 'profiling_overhead'):
                    if profiling_result.profiling_overhead >= 0:
                        validation_score += 30  # Overhead calculated
                    else:
                        issues.append("Negative profiling overhead")
            else:
                issues.append("Could not measure profiling overhead")
            
            return {
                'score': validation_score,
                'max_score': 100,
                'baseline_time': baseline_time,
                'profiling_time': profiling_time,
                'overhead_ratio': overhead_ratio if 'overhead_ratio' in locals() else 0,
                'issues': issues,
                'status': 'passed' if validation_score >= 70 else 'failed'
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 100,
                'baseline_time': 0,
                'profiling_time': 0,
                'issues': [f"Overhead validation failed: {str(e)}"],
                'status': 'error'
            }
    
    async def _validate_report_generation(self) -> Dict[str, Any]:
        """Validate report generation functionality."""
        self.logger.info("Validating report generation...")
        
        try:
            validation_score = 0.0
            issues = []
            
            # Test timing report generation
            timing_report = await self.timing_analyzer.analyze_all_integration_tests()
            
            if timing_report:
                validation_score += 25  # Generated timing report
                
                # Test report serialization
                try:
                    output_path = await self.timing_analyzer.save_timing_report(
                        timing_report,
                        output_path=self.temp_dir / "validation_timing_report.json"
                    )
                    
                    if output_path.exists():
                        validation_score += 25  # Saved timing report
                        
                        # Verify JSON structure
                        with open(output_path, 'r') as f:
                            report_data = json.load(f)
                        
                        required_fields = ['total_execution_time', 'analysis_timestamp', 'file_results']
                        if all(field in report_data for field in required_fields):
                            validation_score += 25  # Valid JSON structure
                        else:
                            issues.append("Invalid timing report JSON structure")
                    else:
                        issues.append("Timing report file not created")
                        
                except Exception as e:
                    issues.append(f"Timing report serialization failed: {str(e)}")
            else:
                issues.append("No timing report generated")
            
            # Test bottleneck report generation
            try:
                from .models import CProfileResult, ProfilingResult
                
                # Create minimal mock data for report testing
                test_id = TestIdentifier(file_path="report_test.py")
                cprofile_result = CProfileResult(
                    test_identifier=test_id,
                    profile_file_path="/tmp/test.prof",
                    total_time=1.0,
                    function_stats={'test_func': {'cumulative_time': 1.0, 'total_time': 1.0, 'call_count': 1, 'filename': 'test.py', 'function_name': 'test_func'}},
                    top_functions=[],
                    call_count=1
                )
                
                profiling_result = ProfilingResult(
                    test_identifier=test_id,
                    profiler_used=ProfilerType.CPROFILE,
                    cprofile_result=cprofile_result
                )
                
                bottleneck_reports = self.bottleneck_analyzer.analyze_cprofile_results([profiling_result])
                
                if len(bottleneck_reports) > 0:
                    validation_score += 25  # Generated bottleneck report
                else:
                    issues.append("No bottleneck reports generated")
                    
            except Exception as e:
                issues.append(f"Bottleneck report generation failed: {str(e)}")
            
            return {
                'score': validation_score,
                'max_score': 100,
                'timing_report_generated': timing_report is not None,
                'report_files_created': len(list(self.temp_dir.glob("*.json"))),
                'issues': issues,
                'status': 'passed' if validation_score >= 70 else 'failed'
            }
            
        except Exception as e:
            return {
                'score': 0,
                'max_score': 100,
                'timing_report_generated': False,
                'issues': [f"Report validation failed: {str(e)}"],
                'status': 'error'
            }
    
    def _calculate_overall_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        total_score = 0.0
        total_max_score = 0.0
        
        for component, result in validation_results.items():
            if isinstance(result, dict) and 'score' in result and 'max_score' in result:
                total_score += result['score']
                total_max_score += result['max_score']
        
        return (total_score / total_max_score * 100) if total_max_score > 0 else 0.0
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable validation report."""
        report_lines = [
            "Performance Analysis Tools Validation Report",
            "=" * 50,
            f"Overall Score: {validation_results['overall_score']:.1f}/100",
            f"Validation Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(validation_results['validation_timestamp']))}",
            "",
            "Component Results:",
            "-" * 20
        ]
        
        for component, result in validation_results.items():
            if isinstance(result, dict) and 'score' in result:
                status_symbol = "✓" if result['status'] == 'passed' else "✗" if result['status'] == 'failed' else "⚠"
                report_lines.append(
                    f"{status_symbol} {component.replace('_', ' ').title()}: "
                    f"{result['score']:.0f}/{result['max_score']} ({result['status']})"
                )
                
                if result.get('issues'):
                    for issue in result['issues']:
                        report_lines.append(f"    - {issue}")
        
        return "\n".join(report_lines)


@pytest.mark.asyncio
async def test_comprehensive_validation():
    """Run comprehensive validation of performance analysis tools."""
    suite = ComprehensiveValidationSuite()
    
    try:
        # Run comprehensive validation
        validation_results = await suite.run_comprehensive_validation()
        
        # Generate report
        report = suite.generate_validation_report(validation_results)
        print("\n" + report)
        
        # Assert overall validation passes
        overall_score = validation_results['overall_score']
        assert overall_score >= 60, f"Overall validation score too low: {overall_score:.1f}/100"
        
        # Check that critical components pass
        critical_components = ['test_discovery', 'timing_accuracy', 'integration_workflow']
        for component in critical_components:
            if component in validation_results:
                component_result = validation_results[component]
                assert component_result['status'] in ['passed', 'skipped'], (
                    f"Critical component {component} failed: {component_result.get('issues', [])}"
                )
        
        return validation_results
        
    finally:
        suite.cleanup()


if __name__ == "__main__":
    # Run comprehensive validation
    asyncio.run(test_comprehensive_validation())