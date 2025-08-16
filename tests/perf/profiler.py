"""
PerformanceProfiler for detailed profiling of integration tests.

This module provides the PerformanceProfiler class that applies cProfile and py-spy
profiling to slow integration tests to identify exact bottlenecks in ImplementAgent
execution and distinguish between test overhead and actual implementation performance.
"""

import asyncio
import cProfile
import pstats
import subprocess
import time
import os
import signal
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging

try:
    from .models import (
        TestIdentifier, CProfileResult, PySpyResult, ProfilingResult,
        ProfilerType, TestTimingResult
    )
    from .config import PerformanceConfig, EnvironmentConfig
    from .utils import CommandBuilder, ProcessManager, FileManager, PatternMatcher
    from .llm_profiler import LLMAPIProfiler, LLMProfilingResult
except ImportError:
    # Handle direct execution
    from models import (
        TestIdentifier, CProfileResult, PySpyResult, ProfilingResult,
        ProfilerType, TestTimingResult
    )
    from config import PerformanceConfig, EnvironmentConfig
    from utils import CommandBuilder, ProcessManager, FileManager, PatternMatcher
    from llm_profiler import LLMAPIProfiler, LLMProfilingResult


class PerformanceProfiler:
    """
    Applies detailed profiling to slow integration tests using cProfile and py-spy.
    
    This class focuses on profiling the slowest integration tests identified by
    TestTimingAnalyzer to determine whether performance issues are in the test
    infrastructure or the actual ImplementAgent implementation.
    """
    
    def __init__(self):
        """Initialize PerformanceProfiler with available profiling tools."""
        self.primary_profiler = ProfilerType.CPROFILE  # Built-in, deterministic
        self.secondary_profiler = ProfilerType.PYSPY   # For flame graphs
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize pattern matcher for component analysis
        self.pattern_matcher = PatternMatcher()
        
        # Initialize LLM API profiler
        self.llm_profiler = LLMAPIProfiler()
        
        # Check tool availability
        self._check_profiler_availability()
        
        # Ensure output directories exist
        FileManager.ensure_output_directories()
    
    def _check_profiler_availability(self):
        """Check availability of profiling tools."""
        self.pyspy_available = EnvironmentConfig.is_pyspy_available()
        
        if not self.pyspy_available:
            self.logger.warning(
                "py-spy not available. Flame graph generation will be disabled. "
                "Install py-spy for enhanced profiling: pip install py-spy"
            )
        else:
            # Check py-spy version and supported formats
            self._detect_pyspy_capabilities()
    
    def _detect_pyspy_capabilities(self):
        """Detect py-spy version and supported output formats."""
        try:
            import subprocess
            result = subprocess.run(['py-spy', '--help'], 
                                  capture_output=True, text=True, timeout=5)
            
            # Check if flamegraph format is supported
            if 'flamegraph' in result.stdout:
                self.pyspy_format = 'flamegraph'
                self.pyspy_extension = '.svg'
            elif 'svg' in result.stdout:
                self.pyspy_format = 'svg'
                self.pyspy_extension = '.svg'
            else:
                # Fallback to raw format
                self.pyspy_format = 'raw'
                self.pyspy_extension = '.txt'
                self.logger.warning(
                    "py-spy does not support flamegraph format. "
                    "Using raw format instead."
                )
            
            # Test if py-spy can actually run (check for permission issues)
            self._test_pyspy_permissions()
                
        except Exception as e:
            self.logger.warning(f"Could not detect py-spy capabilities: {e}")
            # Use safe defaults
            self.pyspy_format = 'flamegraph'
            self.pyspy_extension = '.svg'
    
    def _test_pyspy_permissions(self):
        """Test if py-spy can run with current permissions."""
        try:
            import subprocess
            import os
            
            # Try to run py-spy on current process briefly
            result = subprocess.run([
                'py-spy', 'record', 
                '-p', str(os.getpid()),
                '--duration', '1',
                '--format', 'raw'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0 and 'root' in result.stderr.lower():
                self.pyspy_needs_root = True
                self.logger.warning(
                    "py-spy requires root permissions on this system. "
                    "Flame graph generation will use fallback method. "
                    "Run with sudo for full py-spy functionality."
                )
            else:
                self.pyspy_needs_root = False
                
        except Exception as e:
            self.logger.debug(f"Could not test py-spy permissions: {e}")
            self.pyspy_needs_root = False
    
    async def profile_slow_tests(self, 
                               slow_tests: List[TestTimingResult],
                               max_tests: Optional[int] = None) -> List[ProfilingResult]:
        """
        Apply cProfile profiling to identified slow integration tests.
        
        Args:
            slow_tests: List of slow test results from timing analysis
            max_tests: Maximum number of tests to profile (default: all)
            
        Returns:
            List of ProfilingResult objects with detailed profiling data
        """
        if not slow_tests:
            self.logger.warning("No slow tests provided for profiling")
            return []
        
        # Limit number of tests if specified
        tests_to_profile = slow_tests[:max_tests] if max_tests else slow_tests
        
        self.logger.info(f"Starting profiling of {len(tests_to_profile)} slow tests")
        
        profiling_results = []
        
        for i, test_result in enumerate(tests_to_profile, 1):
            self.logger.info(
                f"Profiling test {i}/{len(tests_to_profile)}: "
                f"{test_result.test_identifier.full_name} "
                f"(baseline: {test_result.execution_time:.2f}s)"
            )
            
            try:
                # Profile with cProfile (primary method)
                profiling_result = await self.profile_with_cprofile(
                    test_result.test_identifier
                )
                
                # Add py-spy profiling if available
                if self.pyspy_available:
                    try:
                        pyspy_result = await self.profile_with_pyspy(
                            test_result.test_identifier
                        )
                        profiling_result.pyspy_result = pyspy_result
                    except Exception as e:
                        self.logger.warning(
                            f"py-spy profiling failed for {test_result.test_identifier.full_name}: {e}"
                        )
                        
                        # Create fallback flame graph if py-spy fails
                        try:
                            fallback_path = self.create_fallback_flame_graph(
                                test_result.test_identifier,
                                profiling_result.cprofile_result
                            )
                            if fallback_path:
                                self.logger.info(f"Created fallback flame graph: {fallback_path}")
                        except Exception as fallback_error:
                            self.logger.debug(f"Fallback flame graph creation failed: {fallback_error}")
                
                # Add LLM API call profiling (Task 8)
                try:
                    llm_result = await self.profile_with_llm_analysis(
                        test_result.test_identifier
                    )
                    profiling_result.llm_profiling_result = llm_result
                    
                    if llm_result.total_api_calls > 0:
                        self.logger.info(
                            f"LLM profiling: {llm_result.total_api_calls} API calls, "
                            f"{llm_result.total_llm_time:.2f}s total time"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"LLM profiling failed for {test_result.test_identifier.full_name}: {e}"
                    )
                
                profiling_results.append(profiling_result)
                
                # Log progress
                self._log_profiling_progress(profiling_result, i, len(tests_to_profile))
                
            except Exception as e:
                self.logger.error(
                    f"Profiling failed for {test_result.test_identifier.full_name}: {e}"
                )
                
                # Create error result
                error_result = ProfilingResult(
                    test_identifier=test_result.test_identifier,
                    profiler_used=ProfilerType.CPROFILE,
                    total_profiling_time=0.0
                )
                profiling_results.append(error_result)
        
        self.logger.info(f"Completed profiling of {len(profiling_results)} tests")
        return profiling_results
    
    async def profile_with_llm_analysis(self, test_identifier: TestIdentifier) -> LLMProfilingResult:
        """
        Profile LLM API calls for a specific test.
        
        Args:
            test_identifier: Test to profile for LLM API usage
            
        Returns:
            LLMProfilingResult with detailed LLM API call analysis
        """
        self.logger.debug(f"Starting LLM API profiling for {test_identifier.full_name}")
        
        try:
            # Use the LLM profiler to analyze API calls
            llm_result = await self.llm_profiler.profile_test_llm_calls(
                test_identifier, None  # We'll pass the profiling result later if needed
            )
            
            self.logger.info(
                f"LLM profiling completed: {llm_result.total_api_calls} API calls, "
                f"{llm_result.total_llm_time:.2f}s total LLM time"
            )
            
            return llm_result
            
        except Exception as e:
            self.logger.error(f"LLM profiling failed for {test_identifier.full_name}: {e}")
            # Return empty result on failure
            return LLMProfilingResult(test_identifier=test_identifier)
    
    async def profile_with_cprofile(self, test_identifier: TestIdentifier) -> ProfilingResult:
        """
        Run integration test with cProfile profiling - primary method.
        
        Args:
            test_identifier: Test to profile
            
        Returns:
            ProfilingResult with cProfile data and component analysis
        """
        self.logger.debug(f"Starting cProfile profiling for {test_identifier.full_name}")
        
        # Generate unique profile output path
        profile_filename = self._generate_profile_filename(test_identifier)
        profile_path = PerformanceConfig.get_profile_output_path(profile_filename)
        
        # Ensure unique filename
        profile_path = FileManager.get_unique_filename(profile_path)
        
        # Build cProfile command
        command = CommandBuilder.build_cprofile_command(test_identifier, profile_path)
        
        # Execute with timeout (use longer timeout for profiling)
        timeout = PerformanceConfig.DEFAULT_TIMEOUT_SECONDS * 2
        start_time = time.time()
        
        execution_result = await ProcessManager.execute_with_timeout(
            command, timeout_seconds=timeout
        )
        
        end_time = time.time()
        total_profiling_time = end_time - start_time
        
        # Calculate profiling overhead
        # Note: This is approximate since we don't have baseline without profiling
        profiling_overhead = max(0.0, total_profiling_time - execution_result.execution_time)
        
        # Analyze cProfile results
        cprofile_result = None
        component_timings = {}
        
        if profile_path.exists() and execution_result.exit_code == 0:
            try:
                cprofile_result = await self._analyze_cprofile_data(
                    test_identifier, profile_path
                )
                
                # Extract component timings from cProfile data
                component_timings = self._extract_component_timings(cprofile_result)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze cProfile data: {e}")
        else:
            self.logger.warning(
                f"cProfile output not found or test failed: {profile_path}, "
                f"exit_code: {execution_result.exit_code}"
            )
        
        return ProfilingResult(
            test_identifier=test_identifier,
            profiler_used=ProfilerType.CPROFILE,
            cprofile_result=cprofile_result,
            component_timings=component_timings,
            total_profiling_time=total_profiling_time,
            profiling_overhead=profiling_overhead
        )
    
    async def profile_with_pyspy(self, test_identifier: TestIdentifier) -> PySpyResult:
        """
        Run integration test with py-spy profiling - for flame graphs.
        
        Args:
            test_identifier: Test to profile
            
        Returns:
            PySpyResult with flame graph data
        """
        if not self.pyspy_available:
            raise RuntimeError("py-spy is not available")
        
        # Check for permission issues
        if getattr(self, 'pyspy_needs_root', False):
            raise RuntimeError(
                "py-spy requires root permissions on this system. "
                "Run with sudo for py-spy profiling or use cProfile only."
            )
        
        self.logger.debug(f"Starting py-spy profiling for {test_identifier.full_name}")
        
        # Generate flame graph output path
        flamegraph_filename = self._generate_flamegraph_filename(test_identifier)
        flamegraph_path = PerformanceConfig.get_flamegraph_output_path(flamegraph_filename)
        
        # Adjust extension based on detected format
        if hasattr(self, 'pyspy_extension'):
            flamegraph_path = flamegraph_path.with_suffix(self.pyspy_extension)
        
        # Ensure unique filename
        flamegraph_path = FileManager.get_unique_filename(flamegraph_path)
        
        # Build pytest command for the test
        pytest_command = CommandBuilder.build_pytest_command(test_identifier)
        
        # Start the test process
        start_time = time.time()
        
        test_process = await asyncio.create_subprocess_exec(
            *pytest_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Give the process a moment to start
        await asyncio.sleep(0.5)
        
        # Start py-spy profiling
        sampling_duration = PerformanceConfig.DEFAULT_TIMEOUT_SECONDS
        pyspy_format = getattr(self, 'pyspy_format', 'flamegraph')
        pyspy_command = [
            'py-spy', 'record',
            '-o', str(flamegraph_path),
            '-p', str(test_process.pid),
            '--duration', str(sampling_duration),
            '--format', pyspy_format
        ]
        
        try:
            # Run py-spy and wait for test completion
            pyspy_process = await asyncio.create_subprocess_exec(
                *pyspy_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for both processes to complete with timeout
            try:
                test_task = asyncio.create_task(test_process.communicate())
                pyspy_task = asyncio.create_task(pyspy_process.communicate())
                
                # Wait for both with a reasonable timeout
                timeout = sampling_duration + 30  # Extra buffer
                test_stdout, test_stderr = await asyncio.wait_for(test_task, timeout=timeout)
                pyspy_stdout, pyspy_stderr = await asyncio.wait_for(pyspy_task, timeout=timeout)
                
            except asyncio.TimeoutError:
                # Kill both processes on timeout
                try:
                    test_process.kill()
                    pyspy_process.kill()
                    await test_process.wait()
                    await pyspy_process.wait()
                except:
                    pass
                raise RuntimeError("py-spy profiling timed out")
            
            end_time = time.time()
            actual_duration = end_time - start_time
            
            # Check if flame graph was generated
            if flamegraph_path.exists():
                # Try to extract sample count from py-spy output
                sample_count = self._extract_sample_count(pyspy_stdout.decode())
                
                return PySpyResult(
                    test_identifier=test_identifier,
                    flamegraph_path=str(flamegraph_path),
                    sampling_duration=actual_duration,
                    sample_count=sample_count,
                    process_id=test_process.pid
                )
            else:
                # Provide more detailed error information
                error_msg = f"Flame graph not generated at {flamegraph_path}"
                if pyspy_stderr:
                    error_msg += f". py-spy error: {pyspy_stderr.decode()}"
                if test_stderr:
                    error_msg += f". Test error: {test_stderr.decode()}"
                raise RuntimeError(error_msg)
                
        except Exception as e:
            # Clean up processes
            try:
                if test_process.returncode is None:
                    test_process.kill()
                    await test_process.wait()
            except:
                pass
            
            try:
                if hasattr(pyspy_process, 'returncode') and pyspy_process.returncode is None:
                    pyspy_process.kill()
                    await pyspy_process.wait()
            except:
                pass
            
            raise RuntimeError(f"py-spy profiling failed: {e}")
    
    def generate_flame_graph_from_pyspy(self, pyspy_data: PySpyResult) -> str:
        """
        Generate SVG flame graph from py-spy data.
        
        Args:
            pyspy_data: py-spy profiling result
            
        Returns:
            Path to generated SVG flame graph
        """
        # py-spy already generates the output file directly, so just return the path
        return pyspy_data.flamegraph_path
    
    def can_use_pyspy_with_sudo(self) -> bool:
        """
        Check if py-spy can be used with sudo permissions.
        
        Returns:
            True if sudo py-spy is available and functional
        """
        if not self.pyspy_available:
            return False
        
        try:
            import subprocess
            import os
            
            # Check if we can run sudo without password prompt
            result = subprocess.run(['sudo', '-n', 'true'], 
                                  capture_output=True, timeout=5)
            
            if result.returncode == 0:
                # Test sudo py-spy briefly
                test_result = subprocess.run([
                    'sudo', 'py-spy', 'record',
                    '-p', str(os.getpid()),
                    '--duration', '1',
                    '--format', 'raw'
                ], capture_output=True, text=True, timeout=10)
                
                return test_result.returncode == 0
            
        except Exception:
            pass
        
        return False
    
    def create_fallback_flame_graph(self, test_identifier: TestIdentifier, 
                                  cprofile_result: Optional[CProfileResult] = None) -> str:
        """
        Create a fallback flame graph when py-spy is not available.
        
        This generates a simple text-based representation of the profiling data
        that can be used when py-spy flame graphs are not available.
        
        Args:
            test_identifier: Test that was profiled
            cprofile_result: Optional cProfile data to use for the fallback
            
        Returns:
            Path to generated fallback flame graph file
        """
        fallback_filename = self._generate_flamegraph_filename(test_identifier)
        fallback_path = PerformanceConfig.get_flamegraph_output_path(fallback_filename)
        fallback_path = fallback_path.with_suffix('.txt')
        
        # Ensure unique filename
        fallback_path = FileManager.get_unique_filename(fallback_path)
        
        try:
            with open(fallback_path, 'w') as f:
                f.write(f"Fallback Flame Graph for {test_identifier.full_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write("py-spy is not available. This is a text-based representation.\n")
                f.write("Install py-spy for interactive SVG flame graphs: pip install py-spy\n\n")
                
                if cprofile_result:
                    f.write("Top Functions by Cumulative Time:\n")
                    f.write("-" * 40 + "\n")
                    
                    for i, func_data in enumerate(cprofile_result.top_functions[:10], 1):
                        f.write(f"{i:2d}. {func_data['name']}\n")
                        f.write(f"    Cumulative: {func_data['cumulative_time']:.3f}s "
                               f"({func_data['percentage']:.1f}%)\n")
                        f.write(f"    Calls: {func_data['call_count']}\n\n")
                else:
                    f.write("No cProfile data available for detailed breakdown.\n")
                
                f.write(f"\nGenerated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            return str(fallback_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create fallback flame graph: {e}")
            return ""
    
    async def _analyze_cprofile_data(self, 
                                   test_identifier: TestIdentifier,
                                   profile_path: Path) -> CProfileResult:
        """
        Analyze cProfile data and extract timing information.
        
        Args:
            test_identifier: Test that was profiled
            profile_path: Path to cProfile output file
            
        Returns:
            CProfileResult with analyzed profiling data
        """
        try:
            # Load profile statistics
            stats = pstats.Stats(str(profile_path))
            
            # Get total time and call count
            total_time = 0.0
            total_calls = 0
            
            # Extract function statistics
            function_stats = {}
            top_functions = []
            
            # Sort by cumulative time to get most time-consuming functions
            stats.sort_stats(pstats.SortKey.CUMULATIVE)
            
            # Get stats data
            for func_key, (cc, nc, tt, ct, callers) in stats.stats.items():
                filename, line_num, func_name = func_key
                
                # Create function identifier
                full_func_name = f"{filename}:{line_num}({func_name})"
                
                # Store function statistics
                function_stats[full_func_name] = {
                    'cumulative_time': ct,
                    'total_time': tt,
                    'call_count': cc,
                    'filename': filename,
                    'line_number': line_num,
                    'function_name': func_name
                }
                
                total_time += tt
                total_calls += cc
            
            # Get top functions by cumulative time
            sorted_functions = sorted(
                function_stats.items(),
                key=lambda x: x[1]['cumulative_time'],
                reverse=True
            )
            
            # Take top N functions
            max_functions = PerformanceConfig.MAX_TOP_FUNCTIONS
            for func_name, func_data in sorted_functions[:max_functions]:
                top_functions.append({
                    'name': func_name,
                    'cumulative_time': func_data['cumulative_time'],
                    'total_time': func_data['total_time'],
                    'call_count': func_data['call_count'],
                    'percentage': (func_data['cumulative_time'] / total_time * 100) if total_time > 0 else 0
                })
            
            return CProfileResult(
                test_identifier=test_identifier,
                profile_file_path=str(profile_path),
                total_time=total_time,
                function_stats=function_stats,
                top_functions=top_functions,
                call_count=total_calls
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze cProfile data from {profile_path}: {e}")
            raise
    
    def _extract_component_timings(self, cprofile_result: CProfileResult) -> Dict[str, float]:
        """
        Extract timing for specific ImplementAgent components from cProfile data.
        
        Args:
            cprofile_result: Analyzed cProfile data
            
        Returns:
            Dictionary mapping component names to cumulative time spent
        """
        component_timings = {
            'TaskDecomposer': 0.0,
            'ShellExecutor': 0.0,
            'ErrorRecovery': 0.0,
            'ContextManager': 0.0,
            'ImplementAgent': 0.0,
            'LLM_API_Calls': 0.0,
            'Test_Infrastructure': 0.0,
            'Other': 0.0
        }
        
        # Analyze each function in the profile
        for func_name, func_data in cprofile_result.function_stats.items():
            cumulative_time = func_data['cumulative_time']
            function_name = func_data['function_name']
            
            # Match to ImplementAgent components
            component = self.pattern_matcher.match_component(function_name)
            if component:
                component_timings[component] += cumulative_time
            elif self.pattern_matcher.is_llm_call(function_name):
                component_timings['LLM_API_Calls'] += cumulative_time
            elif self.pattern_matcher.is_test_infrastructure(function_name):
                component_timings['Test_Infrastructure'] += cumulative_time
            else:
                component_timings['Other'] += cumulative_time
        
        return component_timings
    
    def _extract_sample_count(self, pyspy_output: str) -> int:
        """
        Extract sample count from py-spy output.
        
        Args:
            pyspy_output: stdout from py-spy command
            
        Returns:
            Number of samples collected
        """
        # Look for sample count in py-spy output
        import re
        
        # py-spy typically outputs something like "Collected X samples"
        match = re.search(r'collected\s+(\d+)\s+samples', pyspy_output, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Fallback: look for other patterns
        match = re.search(r'(\d+)\s+samples', pyspy_output, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return 0
    
    def _generate_profile_filename(self, test_identifier: TestIdentifier) -> str:
        """Generate unique filename for cProfile output."""
        # Create safe filename from test identifier
        safe_name = test_identifier.full_name.replace('/', '_').replace('::', '_')
        timestamp = int(time.time())
        return f"{safe_name}_{timestamp}"
    
    def _generate_flamegraph_filename(self, test_identifier: TestIdentifier) -> str:
        """Generate unique filename for flame graph output."""
        # Create safe filename from test identifier
        safe_name = test_identifier.full_name.replace('/', '_').replace('::', '_')
        timestamp = int(time.time())
        return f"{safe_name}_{timestamp}"
    
    def _log_profiling_progress(self, 
                              profiling_result: ProfilingResult,
                              current: int,
                              total: int):
        """Log progress information for completed profiling."""
        test_name = profiling_result.test_identifier.full_name
        
        self.logger.info(
            f"Completed profiling {current}/{total}: {test_name} "
            f"({profiling_result.total_profiling_time:.2f}s)"
        )
        
        if profiling_result.cprofile_result:
            total_time = profiling_result.cprofile_result.total_time
            call_count = profiling_result.cprofile_result.call_count
            self.logger.debug(
                f"  cProfile: {total_time:.2f}s total, {call_count} calls"
            )
        
        if profiling_result.pyspy_result:
            sample_count = profiling_result.pyspy_result.sample_count
            self.logger.debug(
                f"  py-spy: {sample_count} samples, flame graph generated"
            )
        
        # Log component timing summary
        if profiling_result.component_timings:
            implement_agent_time = sum(
                time for component, time in profiling_result.component_timings.items()
                if component in ['TaskDecomposer', 'ShellExecutor', 'ErrorRecovery', 
                               'ContextManager', 'ImplementAgent']
            )
            test_time = profiling_result.component_timings.get('Test_Infrastructure', 0.0)
            
            if implement_agent_time > 0 or test_time > 0:
                total_categorized = implement_agent_time + test_time
                impl_percentage = (implement_agent_time / total_categorized * 100) if total_categorized > 0 else 0
                
                self.logger.debug(
                    f"  Component breakdown: {impl_percentage:.1f}% ImplementAgent, "
                    f"{100-impl_percentage:.1f}% Test Infrastructure"
                )
    
    async def profile_specific_test(self, test_identifier: TestIdentifier) -> ProfilingResult:
        """
        Profile a specific test with both cProfile and py-spy if available.
        
        Args:
            test_identifier: Specific test to profile
            
        Returns:
            ProfilingResult with comprehensive profiling data
        """
        self.logger.info(f"Profiling specific test: {test_identifier.full_name}")
        
        # Profile with cProfile
        profiling_result = await self.profile_with_cprofile(test_identifier)
        
        # Add py-spy profiling if available
        if self.pyspy_available:
            try:
                pyspy_result = await self.profile_with_pyspy(test_identifier)
                profiling_result.pyspy_result = pyspy_result
            except Exception as e:
                self.logger.warning(f"py-spy profiling failed: {e}")
                
                # Create fallback flame graph if py-spy fails
                try:
                    fallback_path = self.create_fallback_flame_graph(
                        test_identifier,
                        profiling_result.cprofile_result
                    )
                    if fallback_path:
                        self.logger.info(f"Created fallback flame graph: {fallback_path}")
                except Exception as fallback_error:
                    self.logger.debug(f"Fallback flame graph creation failed: {fallback_error}")
        
        return profiling_result
    
    def get_profiling_summary(self, results: List[ProfilingResult]) -> Dict[str, Any]:
        """
        Generate summary statistics from profiling results.
        
        Args:
            results: List of profiling results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {
                'total_tests_profiled': 0,
                'successful_profiles': 0,
                'total_profiling_time': 0.0,
                'average_profiling_time': 0.0,
                'cprofile_success_rate': 0.0,
                'pyspy_success_rate': 0.0
            }
        
        successful_cprofile = sum(1 for r in results if r.cprofile_result is not None)
        successful_pyspy = sum(1 for r in results if r.pyspy_result is not None)
        total_profiling_time = sum(r.total_profiling_time for r in results)
        
        return {
            'total_tests_profiled': len(results),
            'successful_profiles': successful_cprofile,
            'total_profiling_time': total_profiling_time,
            'average_profiling_time': total_profiling_time / len(results),
            'cprofile_success_rate': (successful_cprofile / len(results)) * 100,
            'pyspy_success_rate': (successful_pyspy / len(results)) * 100 if self.pyspy_available else 0.0
        }