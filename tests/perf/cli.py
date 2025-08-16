#!/usr/bin/env python3
"""
Command-line interface for performance analysis tools.

This module provides a comprehensive CLI for running performance analysis
on integration tests, including timing-only analysis, full profiling,
selective test profiling, and report generation capabilities.
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from .timing_analyzer import TestTimingAnalyzer
    from .profiler import PerformanceProfiler
    from .bottleneck_analyzer import BottleneckAnalyzer
    from .report_generator import PerformanceReportGenerator
    from .config import PerformanceConfig, EnvironmentConfig, validate_config
    from .models import TestIdentifier, TestStatus
    from .utils import TestDiscovery
except ImportError:
    # Handle direct execution
    from timing_analyzer import TestTimingAnalyzer
    from profiler import PerformanceProfiler
    from bottleneck_analyzer import BottleneckAnalyzer
    from report_generator import PerformanceReportGenerator
    from config import PerformanceConfig, EnvironmentConfig, validate_config
    from models import TestIdentifier, TestStatus
    from utils import TestDiscovery


class PerformanceAnalysisCLI:
    """
    Command-line interface for performance analysis tools.
    
    Provides comprehensive performance analysis capabilities including:
    - Timing-only analysis of integration tests
    - Full profiling with cProfile and py-spy
    - Selective test profiling based on timing results
    - Report generation and export capabilities
    """
    
    def __init__(self):
        """Initialize CLI with components."""
        self.timing_analyzer = TestTimingAnalyzer()
        self.profiler = PerformanceProfiler()
        self.bottleneck_analyzer = BottleneckAnalyzer()
        self.report_generator = PerformanceReportGenerator()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Validate configuration
        self._validate_environment()
    
    def _setup_logging(self, verbose: bool = False) -> logging.Logger:
        """Setup logging configuration."""
        level = logging.DEBUG if verbose else logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _validate_environment(self):
        """Validate environment configuration."""
        issues = validate_config()
        if issues:
            self.logger.error("Configuration validation failed:")
            for issue in issues:
                self.logger.error(f"  - {issue}")
            sys.exit(1)
    
    async def run_timing_analysis(self, 
                                timeout: Optional[int] = None,
                                test_pattern: Optional[str] = None,
                                output_file: Optional[str] = None) -> str:
        """
        Run timing-only analysis of integration tests.
        
        Args:
            timeout: Custom timeout for individual tests (seconds)
            test_pattern: Pattern to filter tests (e.g., "test_real_*")
            output_file: Optional output file for timing report
            
        Returns:
            Path to generated timing report
        """
        self.logger.info("Starting timing-only analysis")
        
        # Configure timeout if provided
        if timeout:
            self.timing_analyzer.timeout_seconds = timeout
            self.logger.info(f"Using custom timeout: {timeout} seconds")
        
        # Run comprehensive timing analysis
        timing_report = await self.timing_analyzer.analyze_all_integration_tests()
        
        # Filter tests if pattern provided
        if test_pattern:
            timing_report = self._filter_timing_report(timing_report, test_pattern)
            self.logger.info(f"Filtered tests using pattern: {test_pattern}")
        
        # Generate summary
        summary = self.timing_analyzer.get_analysis_summary(timing_report)
        self._print_timing_summary(summary)
        
        # Identify slow tests for potential profiling
        slow_tests = self.timing_analyzer.identify_slow_tests(timing_report)
        if slow_tests:
            self.logger.info(f"\nSlow tests identified for profiling ({len(slow_tests)}):")
            for i, test in enumerate(slow_tests[:10], 1):
                self.logger.info(
                    f"  {i:2d}. {test.test_identifier.full_name}: {test.execution_time:.2f}s"
                )
        
        # Save timing report
        if output_file:
            report_path = Path(output_file)
        else:
            report_path = None
        
        saved_path = await self.timing_analyzer.save_timing_report(timing_report, report_path)
        self.logger.info(f"Timing report saved to: {saved_path}")
        
        return str(saved_path)
    
    async def run_full_profiling(self,
                               max_tests: Optional[int] = None,
                               slow_threshold: Optional[float] = None,
                               include_flamegraphs: bool = True,
                               output_dir: Optional[str] = None) -> str:
        """
        Run full profiling analysis including cProfile and py-spy.
        
        Args:
            max_tests: Maximum number of slow tests to profile
            slow_threshold: Custom threshold for identifying slow tests (seconds)
            include_flamegraphs: Whether to generate flame graphs with py-spy
            output_dir: Custom output directory for reports
            
        Returns:
            Path to generated comprehensive report
        """
        self.logger.info("Starting full profiling analysis")
        
        # First run timing analysis to identify slow tests
        self.logger.info("Phase 1: Identifying slow tests through timing analysis")
        timing_report = await self.timing_analyzer.analyze_all_integration_tests()
        
        # Identify slow tests for profiling
        if slow_threshold:
            slow_tests = self.timing_analyzer.identify_slow_tests(timing_report, slow_threshold)
        else:
            slow_tests = self.timing_analyzer.identify_slow_tests(timing_report)
        
        if not slow_tests:
            self.logger.warning("No slow tests found for profiling")
            return await self._generate_timing_only_report(timing_report, output_dir)
        
        # Limit number of tests if specified
        if max_tests and len(slow_tests) > max_tests:
            slow_tests = slow_tests[:max_tests]
            self.logger.info(f"Limited profiling to {max_tests} slowest tests")
        
        self.logger.info(f"Phase 2: Profiling {len(slow_tests)} slow tests")
        
        # Configure profiler
        if not include_flamegraphs:
            self.profiler.pyspy_available = False
            self.logger.info("Flame graph generation disabled")
        
        # Run detailed profiling
        profiling_results = await self.profiler.profile_slow_tests(slow_tests, max_tests)
        
        self.logger.info("Phase 3: Analyzing bottlenecks")
        
        # Analyze bottlenecks
        bottleneck_reports = self.bottleneck_analyzer.analyze_profiling_results_with_llm(profiling_results)
        
        self.logger.info("Phase 4: Generating comprehensive report")
        
        # Generate comprehensive report
        if output_dir:
            self.report_generator.output_dir = Path(output_dir)
        
        report_path = self.report_generator.generate_comprehensive_report(
            timing_report=timing_report,
            profiling_results=profiling_results,
            bottleneck_reports=bottleneck_reports
        )
        
        # Print summary of findings
        self._print_profiling_summary(profiling_results, bottleneck_reports)
        
        self.logger.info(f"Comprehensive report generated: {report_path}")
        return report_path
    
    async def run_selective_profiling(self,
                                    test_names: List[str],
                                    include_flamegraphs: bool = True,
                                    output_dir: Optional[str] = None) -> str:
        """
        Run profiling on specific tests selected by name.
        
        Args:
            test_names: List of specific test names to profile
            include_flamegraphs: Whether to generate flame graphs
            output_dir: Custom output directory for reports
            
        Returns:
            Path to generated report
        """
        self.logger.info(f"Starting selective profiling of {len(test_names)} tests")
        
        # Discover all tests to find matching identifiers
        all_tests = TestDiscovery.discover_integration_tests()
        
        # Find matching test identifiers
        selected_tests = []
        for test_name in test_names:
            matching_tests = [
                test for test in all_tests 
                if test_name in test.full_name or test.full_name.endswith(test_name)
            ]
            
            if matching_tests:
                selected_tests.extend(matching_tests)
                self.logger.info(f"Found {len(matching_tests)} tests matching '{test_name}'")
            else:
                self.logger.warning(f"No tests found matching '{test_name}'")
        
        if not selected_tests:
            self.logger.error("No matching tests found for selective profiling")
            sys.exit(1)
        
        # Remove duplicates
        selected_tests = list(set(selected_tests))
        self.logger.info(f"Profiling {len(selected_tests)} selected tests")
        
        # First get timing data for selected tests
        timing_results = await self.timing_analyzer.analyze_specific_tests(selected_tests)
        
        # Create minimal timing report
        from .models import TimingReport, FileTimingResult
        file_results = []
        for timing_result in timing_results:
            # Group by file
            file_path = timing_result.test_identifier.file_path
            existing_file = next((fr for fr in file_results if fr.file_path == file_path), None)
            
            if existing_file:
                existing_file.test_results.append(timing_result)
                existing_file.total_execution_time += timing_result.execution_time
            else:
                file_result = FileTimingResult(
                    file_path=file_path,
                    total_execution_time=timing_result.execution_time,
                    test_results=[timing_result],
                    file_status=timing_result.status
                )
                file_results.append(file_result)
        
        timing_report = TimingReport(
            total_execution_time=sum(result.execution_time for result in timing_results),
            file_results=file_results,
            timeout_threshold=self.timing_analyzer.timeout_seconds
        )
        
        # Configure profiler
        if not include_flamegraphs:
            self.profiler.pyspy_available = False
        
        # Run profiling on selected tests
        profiling_results = await self.profiler.profile_slow_tests(timing_results)
        
        # Analyze bottlenecks
        bottleneck_reports = self.bottleneck_analyzer.analyze_profiling_results_with_llm(profiling_results)
        
        # Generate report
        if output_dir:
            self.report_generator.output_dir = Path(output_dir)
        
        report_name = f"selective_profiling_{int(time.time())}"
        report_path = self.report_generator.generate_comprehensive_report(
            timing_report=timing_report,
            profiling_results=profiling_results,
            bottleneck_reports=bottleneck_reports,
            report_name=report_name
        )
        
        self._print_profiling_summary(profiling_results, bottleneck_reports)
        
        self.logger.info(f"Selective profiling report generated: {report_path}")
        return report_path
    
    async def generate_comparison_report(self,
                                       baseline_report: str,
                                       current_report: str,
                                       output_dir: Optional[str] = None) -> str:
        """
        Generate comparison report between two performance analysis runs.
        
        Args:
            baseline_report: Path to baseline report JSON data
            current_report: Path to current report JSON data
            output_dir: Custom output directory for comparison report
            
        Returns:
            Path to generated comparison report
        """
        self.logger.info("Generating performance comparison report")
        
        if output_dir:
            self.report_generator.output_dir = Path(output_dir)
        
        comparison_path = self.report_generator.generate_comparison_report(
            baseline_report_path=baseline_report,
            current_report_path=current_report
        )
        
        self.logger.info(f"Comparison report generated: {comparison_path}")
        return comparison_path
    
    def list_available_tests(self, pattern: Optional[str] = None):
        """
        List all available integration tests.
        
        Args:
            pattern: Optional pattern to filter test names
        """
        self.logger.info("Discovering available integration tests")
        
        all_tests = TestDiscovery.discover_integration_tests()
        
        if pattern:
            filtered_tests = [
                test for test in all_tests 
                if pattern in test.full_name
            ]
            self.logger.info(f"Found {len(filtered_tests)} tests matching pattern '{pattern}':")
            tests_to_show = filtered_tests
        else:
            self.logger.info(f"Found {len(all_tests)} integration tests:")
            tests_to_show = all_tests
        
        # Group by file for better organization
        tests_by_file = {}
        for test in tests_to_show:
            file_path = test.file_path
            if file_path not in tests_by_file:
                tests_by_file[file_path] = []
            tests_by_file[file_path].append(test)
        
        for file_path, file_tests in sorted(tests_by_file.items()):
            print(f"\n{file_path}:")
            for test in sorted(file_tests, key=lambda x: x.full_name):
                if test.method_name:
                    print(f"  - {test.full_name}")
                else:
                    print(f"  - {test.full_name} (file-level)")
    
    async def _generate_timing_only_report(self, 
                                         timing_report, 
                                         output_dir: Optional[str] = None) -> str:
        """Generate report with timing data only."""
        if output_dir:
            self.report_generator.output_dir = Path(output_dir)
        
        # Generate minimal report with just timing data
        report_name = f"timing_only_{int(time.time())}"
        report_path = self.report_generator.generate_comprehensive_report(
            timing_report=timing_report,
            profiling_results=[],
            bottleneck_reports=[],
            report_name=report_name
        )
        
        return report_path
    
    def _filter_timing_report(self, timing_report, pattern: str):
        """Filter timing report by test name pattern."""
        filtered_file_results = []
        
        for file_result in timing_report.file_results:
            filtered_test_results = [
                test_result for test_result in file_result.test_results
                if pattern in test_result.test_identifier.full_name
            ]
            
            if filtered_test_results:
                # Create new file result with filtered tests
                from .models import FileTimingResult
                filtered_file_result = FileTimingResult(
                    file_path=file_result.file_path,
                    total_execution_time=sum(tr.execution_time for tr in filtered_test_results),
                    test_results=filtered_test_results,
                    file_status=file_result.file_status
                )
                filtered_file_results.append(filtered_file_result)
        
        # Create new timing report with filtered results
        from .models import TimingReport
        return TimingReport(
            total_execution_time=sum(fr.total_execution_time for fr in filtered_file_results),
            file_results=filtered_file_results,
            timeout_threshold=timing_report.timeout_threshold
        )
    
    def _print_timing_summary(self, summary: Dict[str, Any]):
        """Print timing analysis summary."""
        print("\n" + "="*60)
        print("TIMING ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total tests analyzed: {summary['total_tests']}")
        print(f"Total execution time: {summary['total_execution_time']:.2f}s")
        print(f"Average test time: {summary['average_execution_time']:.2f}s")
        print(f"Slowest test: {summary['slowest_test_time']:.2f}s")
        print(f"Fastest test: {summary['fastest_test_time']:.2f}s")
        print(f"Tests passed: {summary['passed_count']}")
        print(f"Tests failed: {summary['failed_count']}")
        print(f"Tests timed out: {summary['timeout_count']}")
        print("="*60)
    
    def _print_profiling_summary(self, 
                               profiling_results: List,
                               bottleneck_reports: List):
        """Print profiling analysis summary."""
        print("\n" + "="*60)
        print("PROFILING ANALYSIS SUMMARY")
        print("="*60)
        print(f"Tests profiled: {len(profiling_results)}")
        
        if profiling_results:
            total_profiling_time = sum(r.total_profiling_time for r in profiling_results)
            print(f"Total profiling time: {total_profiling_time:.2f}s")
            print(f"Average profiling time: {total_profiling_time/len(profiling_results):.2f}s")
            
            cprofile_count = len([r for r in profiling_results if r.cprofile_result])
            pyspy_count = len([r for r in profiling_results if r.pyspy_result])
            llm_count = len([r for r in profiling_results if r.has_llm_data])
            
            print(f"cProfile results: {cprofile_count}")
            print(f"py-spy flame graphs: {pyspy_count}")
            print(f"LLM profiling data: {llm_count}")
        
        if bottleneck_reports:
            all_recommendations = []
            for report in bottleneck_reports:
                all_recommendations.extend(report.optimization_recommendations)
            
            high_priority = len([r for r in all_recommendations if r.expected_impact == "high"])
            medium_priority = len([r for r in all_recommendations if r.expected_impact == "medium"])
            
            print(f"Bottleneck reports: {len(bottleneck_reports)}")
            print(f"High priority recommendations: {high_priority}")
            print(f"Medium priority recommendations: {medium_priority}")
            print(f"Total recommendations: {len(all_recommendations)}")
        
        print("="*60)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Performance analysis tools for integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run timing-only analysis
  python -m tests.perf.cli timing

  # Run full profiling analysis
  python -m tests.perf.cli profile --max-tests 5

  # Profile specific tests
  python -m tests.perf.cli selective test_real_agent_manager test_real_main_controller

  # Generate comparison report
  python -m tests.perf.cli compare baseline_report.json current_report.json

  # List available tests
  python -m tests.perf.cli list --pattern "test_real_*"
        """
    )
    
    # Global options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Custom output directory for reports'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Timing analysis command
    timing_parser = subparsers.add_parser(
        'timing',
        help='Run timing-only analysis of integration tests'
    )
    timing_parser.add_argument(
        '--timeout', '-t',
        type=int,
        help='Custom timeout for individual tests (seconds)'
    )
    timing_parser.add_argument(
        '--pattern', '-p',
        type=str,
        help='Pattern to filter tests (e.g., "test_real_*")'
    )
    timing_parser.add_argument(
        '--output-file',
        type=str,
        help='Output file for timing report (JSON)'
    )
    
    # Full profiling command
    profile_parser = subparsers.add_parser(
        'profile',
        help='Run full profiling analysis with cProfile and py-spy'
    )
    profile_parser.add_argument(
        '--max-tests', '-m',
        type=int,
        default=10,
        help='Maximum number of slow tests to profile (default: 10)'
    )
    profile_parser.add_argument(
        '--slow-threshold', '-s',
        type=float,
        help='Custom threshold for identifying slow tests (seconds)'
    )
    profile_parser.add_argument(
        '--no-flamegraphs',
        action='store_true',
        help='Disable flame graph generation (skip py-spy)'
    )
    
    # Selective profiling command
    selective_parser = subparsers.add_parser(
        'selective',
        help='Profile specific tests by name'
    )
    selective_parser.add_argument(
        'test_names',
        nargs='+',
        help='Names of specific tests to profile'
    )
    selective_parser.add_argument(
        '--no-flamegraphs',
        action='store_true',
        help='Disable flame graph generation'
    )
    
    # Comparison command
    compare_parser = subparsers.add_parser(
        'compare',
        help='Generate comparison report between two analysis runs'
    )
    compare_parser.add_argument(
        'baseline_report',
        help='Path to baseline report JSON data'
    )
    compare_parser.add_argument(
        'current_report',
        help='Path to current report JSON data'
    )
    
    # List tests command
    list_parser = subparsers.add_parser(
        'list',
        help='List available integration tests'
    )
    list_parser.add_argument(
        '--pattern', '-p',
        type=str,
        help='Pattern to filter test names'
    )
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize CLI
    cli = PerformanceAnalysisCLI()
    
    # Configure verbose logging if requested
    if args.verbose:
        cli._setup_logging(verbose=True)
    
    try:
        if args.command == 'timing':
            await cli.run_timing_analysis(
                timeout=args.timeout,
                test_pattern=args.pattern,
                output_file=args.output_file
            )
        
        elif args.command == 'profile':
            await cli.run_full_profiling(
                max_tests=args.max_tests,
                slow_threshold=args.slow_threshold,
                include_flamegraphs=not args.no_flamegraphs,
                output_dir=args.output_dir
            )
        
        elif args.command == 'selective':
            await cli.run_selective_profiling(
                test_names=args.test_names,
                include_flamegraphs=not args.no_flamegraphs,
                output_dir=args.output_dir
            )
        
        elif args.command == 'compare':
            await cli.generate_comparison_report(
                baseline_report=args.baseline_report,
                current_report=args.current_report,
                output_dir=args.output_dir
            )
        
        elif args.command == 'list':
            cli.list_available_tests(pattern=args.pattern)
        
    except KeyboardInterrupt:
        cli.logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        cli.logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())