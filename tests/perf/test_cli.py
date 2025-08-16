"""
Tests for the performance analysis CLI.

This module tests the command-line interface functionality including
argument parsing, command execution, and report generation.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from argparse import Namespace

try:
    from .cli import PerformanceAnalysisCLI, create_parser
    from .models import TimingReport, TestIdentifier, TestTimingResult, TestStatus
except ImportError:
    from cli import PerformanceAnalysisCLI, create_parser
    from models import TimingReport, TestIdentifier, TestTimingResult, TestStatus


class TestPerformanceAnalysisCLI:
    """Test cases for PerformanceAnalysisCLI class."""
    
    @pytest.fixture
    def cli(self):
        """Create CLI instance for testing."""
        with patch('cli.validate_config', return_value=[]):
            return PerformanceAnalysisCLI()
    
    @pytest.fixture
    def sample_timing_report(self):
        """Create sample timing report for testing."""
        test_identifier = TestIdentifier(
            file_path="tests/integration/test_sample.py",
            class_name="TestSample",
            method_name="test_method"
        )
        
        test_result = TestTimingResult(
            test_identifier=test_identifier,
            execution_time=15.5,
            status=TestStatus.PASSED
        )
        
        from .models import FileTimingResult
        file_result = FileTimingResult(
            file_path="tests/integration/test_sample.py",
            total_execution_time=15.5,
            test_results=[test_result],
            file_status=TestStatus.PASSED
        )
        
        return TimingReport(
            total_execution_time=15.5,
            file_results=[file_result],
            timeout_threshold=60
        )
    
    @pytest.mark.asyncio
    async def test_run_timing_analysis(self, cli, sample_timing_report):
        """Test timing-only analysis functionality."""
        # Mock the timing analyzer
        cli.timing_analyzer.analyze_all_integration_tests = AsyncMock(
            return_value=sample_timing_report
        )
        cli.timing_analyzer.get_analysis_summary = Mock(return_value={
            'total_tests': 1,
            'total_execution_time': 15.5,
            'average_execution_time': 15.5,
            'slowest_test_time': 15.5,
            'fastest_test_time': 15.5,
            'passed_count': 1,
            'failed_count': 0,
            'timeout_count': 0
        })
        cli.timing_analyzer.identify_slow_tests = Mock(return_value=[])
        cli.timing_analyzer.save_timing_report = AsyncMock(
            return_value="/tmp/timing_report.json"
        )
        
        # Run timing analysis
        result_path = await cli.run_timing_analysis()
        
        # Verify calls
        cli.timing_analyzer.analyze_all_integration_tests.assert_called_once()
        cli.timing_analyzer.save_timing_report.assert_called_once()
        
        assert result_path == "/tmp/timing_report.json"
    
    @pytest.mark.asyncio
    async def test_run_timing_analysis_with_custom_timeout(self, cli, sample_timing_report):
        """Test timing analysis with custom timeout."""
        cli.timing_analyzer.analyze_all_integration_tests = AsyncMock(
            return_value=sample_timing_report
        )
        cli.timing_analyzer.get_analysis_summary = Mock(return_value={
            'total_tests': 1,
            'total_execution_time': 15.5,
            'average_execution_time': 15.5,
            'slowest_test_time': 15.5,
            'fastest_test_time': 15.5,
            'passed_count': 1,
            'failed_count': 0,
            'timeout_count': 0
        })
        cli.timing_analyzer.identify_slow_tests = Mock(return_value=[])
        cli.timing_analyzer.save_timing_report = AsyncMock(
            return_value="/tmp/timing_report.json"
        )
        
        # Run with custom timeout
        await cli.run_timing_analysis(timeout=120)
        
        # Verify timeout was set
        assert cli.timing_analyzer.timeout_seconds == 120
    
    @pytest.mark.asyncio
    async def test_run_full_profiling(self, cli, sample_timing_report):
        """Test full profiling analysis functionality."""
        # Mock timing analyzer
        cli.timing_analyzer.analyze_all_integration_tests = AsyncMock(
            return_value=sample_timing_report
        )
        
        # Create mock slow test
        slow_test = TestTimingResult(
            test_identifier=TestIdentifier(
                file_path="tests/integration/test_slow.py",
                method_name="test_slow_method"
            ),
            execution_time=25.0,
            status=TestStatus.PASSED
        )
        cli.timing_analyzer.identify_slow_tests = Mock(return_value=[slow_test])
        
        # Mock profiler
        from .models import ProfilingResult, ProfilerType
        profiling_result = ProfilingResult(
            test_identifier=slow_test.test_identifier,
            profiler_used=ProfilerType.CPROFILE,
            total_profiling_time=30.0
        )
        cli.profiler.profile_slow_tests = AsyncMock(return_value=[profiling_result])
        
        # Mock bottleneck analyzer
        from .models import BottleneckReport
        bottleneck_report = BottleneckReport(
            test_identifier=slow_test.test_identifier
        )
        cli.bottleneck_analyzer.analyze_profiling_results_with_llm = Mock(
            return_value=[bottleneck_report]
        )
        
        # Mock report generator
        cli.report_generator.generate_comprehensive_report = Mock(
            return_value="/tmp/comprehensive_report.html"
        )
        
        # Run full profiling
        result_path = await cli.run_full_profiling(max_tests=5)
        
        # Verify calls
        cli.timing_analyzer.analyze_all_integration_tests.assert_called_once()
        cli.profiler.profile_slow_tests.assert_called_once()
        cli.bottleneck_analyzer.analyze_profiling_results_with_llm.assert_called_once()
        cli.report_generator.generate_comprehensive_report.assert_called_once()
        
        assert result_path == "/tmp/comprehensive_report.html"
    
    @pytest.mark.asyncio
    async def test_run_selective_profiling(self, cli):
        """Test selective profiling functionality."""
        # Mock test discovery
        test_identifier = TestIdentifier(
            file_path="tests/integration/test_specific.py",
            method_name="test_specific_method"
        )
        
        with patch('cli.TestDiscovery.discover_integration_tests') as mock_discovery:
            mock_discovery.return_value = [test_identifier]
            
            # Mock timing analyzer
            timing_result = TestTimingResult(
                test_identifier=test_identifier,
                execution_time=20.0,
                status=TestStatus.PASSED
            )
            cli.timing_analyzer.analyze_specific_tests = AsyncMock(
                return_value=[timing_result]
            )
            
            # Mock profiler
            from .models import ProfilingResult, ProfilerType
            profiling_result = ProfilingResult(
                test_identifier=test_identifier,
                profiler_used=ProfilerType.CPROFILE,
                total_profiling_time=25.0
            )
            cli.profiler.profile_slow_tests = AsyncMock(return_value=[profiling_result])
            
            # Mock bottleneck analyzer
            from .models import BottleneckReport
            bottleneck_report = BottleneckReport(test_identifier=test_identifier)
            cli.bottleneck_analyzer.analyze_profiling_results_with_llm = Mock(
                return_value=[bottleneck_report]
            )
            
            # Mock report generator
            cli.report_generator.generate_comprehensive_report = Mock(
                return_value="/tmp/selective_report.html"
            )
            
            # Run selective profiling
            result_path = await cli.run_selective_profiling(
                test_names=["test_specific_method"]
            )
            
            # Verify calls
            cli.timing_analyzer.analyze_specific_tests.assert_called_once()
            cli.profiler.profile_slow_tests.assert_called_once()
            cli.report_generator.generate_comprehensive_report.assert_called_once()
            
            assert result_path == "/tmp/selective_report.html"
    
    @pytest.mark.asyncio
    async def test_generate_comparison_report(self, cli):
        """Test comparison report generation."""
        cli.report_generator.generate_comparison_report = Mock(
            return_value="/tmp/comparison_report.html"
        )
        
        result_path = await cli.generate_comparison_report(
            baseline_report="/tmp/baseline.json",
            current_report="/tmp/current.json"
        )
        
        cli.report_generator.generate_comparison_report.assert_called_once_with(
            baseline_report_path="/tmp/baseline.json",
            current_report_path="/tmp/current.json"
        )
        
        assert result_path == "/tmp/comparison_report.html"
    
    def test_list_available_tests(self, cli, capsys):
        """Test listing available tests."""
        # Mock test discovery
        test_identifiers = [
            TestIdentifier(
                file_path="tests/integration/test_a.py",
                method_name="test_method_1"
            ),
            TestIdentifier(
                file_path="tests/integration/test_a.py",
                method_name="test_method_2"
            ),
            TestIdentifier(
                file_path="tests/integration/test_b.py",
                method_name="test_method_3"
            )
        ]
        
        with patch('cli.TestDiscovery.discover_integration_tests') as mock_discovery:
            mock_discovery.return_value = test_identifiers
            
            cli.list_available_tests()
            
            captured = capsys.readouterr()
            assert "test_a.py" in captured.out
            assert "test_b.py" in captured.out
            assert "test_method_1" in captured.out
    
    def test_list_available_tests_with_pattern(self, cli, capsys):
        """Test listing tests with pattern filter."""
        test_identifiers = [
            TestIdentifier(
                file_path="tests/integration/test_real_agent.py",
                method_name="test_real_method"
            ),
            TestIdentifier(
                file_path="tests/integration/test_mock_agent.py",
                method_name="test_mock_method"
            )
        ]
        
        with patch('cli.TestDiscovery.discover_integration_tests') as mock_discovery:
            mock_discovery.return_value = test_identifiers
            
            cli.list_available_tests(pattern="test_real")
            
            captured = capsys.readouterr()
            assert "test_real_agent.py" in captured.out
            assert "test_mock_agent.py" not in captured.out


class TestCLIArgumentParsing:
    """Test cases for CLI argument parsing."""
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog is not None
    
    def test_timing_command_parsing(self):
        """Test timing command argument parsing."""
        parser = create_parser()
        
        # Test basic timing command
        args = parser.parse_args(['timing'])
        assert args.command == 'timing'
        assert args.timeout is None
        assert args.pattern is None
        
        # Test timing command with options
        args = parser.parse_args(['timing', '--timeout', '120', '--pattern', 'test_real_*'])
        assert args.command == 'timing'
        assert args.timeout == 120
        assert args.pattern == 'test_real_*'
    
    def test_profile_command_parsing(self):
        """Test profile command argument parsing."""
        parser = create_parser()
        
        # Test basic profile command
        args = parser.parse_args(['profile'])
        assert args.command == 'profile'
        assert args.max_tests == 10  # default
        assert args.slow_threshold is None
        assert args.no_flamegraphs is False
        
        # Test profile command with options
        args = parser.parse_args([
            'profile', '--max-tests', '5', '--slow-threshold', '15.0', '--no-flamegraphs'
        ])
        assert args.command == 'profile'
        assert args.max_tests == 5
        assert args.slow_threshold == 15.0
        assert args.no_flamegraphs is True
    
    def test_selective_command_parsing(self):
        """Test selective command argument parsing."""
        parser = create_parser()
        
        # Test selective command
        args = parser.parse_args(['selective', 'test_method_1', 'test_method_2'])
        assert args.command == 'selective'
        assert args.test_names == ['test_method_1', 'test_method_2']
        assert args.no_flamegraphs is False
        
        # Test selective command with no-flamegraphs
        args = parser.parse_args(['selective', '--no-flamegraphs', 'test_method'])
        assert args.command == 'selective'
        assert args.test_names == ['test_method']
        assert args.no_flamegraphs is True
    
    def test_compare_command_parsing(self):
        """Test compare command argument parsing."""
        parser = create_parser()
        
        args = parser.parse_args(['compare', 'baseline.json', 'current.json'])
        assert args.command == 'compare'
        assert args.baseline_report == 'baseline.json'
        assert args.current_report == 'current.json'
    
    def test_list_command_parsing(self):
        """Test list command argument parsing."""
        parser = create_parser()
        
        # Test basic list command
        args = parser.parse_args(['list'])
        assert args.command == 'list'
        assert args.pattern is None
        
        # Test list command with pattern
        args = parser.parse_args(['list', '--pattern', 'test_real_*'])
        assert args.command == 'list'
        assert args.pattern == 'test_real_*'
    
    def test_global_options_parsing(self):
        """Test global options parsing."""
        parser = create_parser()
        
        # Test verbose option
        args = parser.parse_args(['--verbose', 'timing'])
        assert args.verbose is True
        assert args.command == 'timing'
        
        # Test output-dir option
        args = parser.parse_args(['--output-dir', '/tmp/reports', 'profile'])
        assert args.output_dir == '/tmp/reports'
        assert args.command == 'profile'


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    @pytest.mark.asyncio
    async def test_cli_timing_analysis_integration(self):
        """Test CLI timing analysis with mocked components."""
        with patch('cli.validate_config', return_value=[]):
            cli = PerformanceAnalysisCLI()
            
            # Mock all dependencies
            sample_report = TimingReport(
                total_execution_time=10.0,
                file_results=[],
                timeout_threshold=60
            )
            
            cli.timing_analyzer.analyze_all_integration_tests = AsyncMock(
                return_value=sample_report
            )
            cli.timing_analyzer.get_analysis_summary = Mock(return_value={
                'total_tests': 0,
                'total_execution_time': 0.0,
                'average_execution_time': 0.0,
                'slowest_test_time': 0.0,
                'fastest_test_time': 0.0,
                'passed_count': 0,
                'failed_count': 0,
                'timeout_count': 0
            })
            cli.timing_analyzer.identify_slow_tests = Mock(return_value=[])
            cli.timing_analyzer.save_timing_report = AsyncMock(
                return_value="/tmp/test_report.json"
            )
            
            # Run timing analysis
            result = await cli.run_timing_analysis()
            
            assert result == "/tmp/test_report.json"
    
    def test_cli_error_handling(self):
        """Test CLI error handling for invalid configurations."""
        with patch('cli.validate_config', return_value=['Test error']):
            with pytest.raises(SystemExit):
                PerformanceAnalysisCLI()


if __name__ == '__main__':
    pytest.main([__file__])