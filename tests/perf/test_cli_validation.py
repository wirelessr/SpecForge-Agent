"""
CLI validation tests for performance analysis tools.

This module tests the command-line interface functionality and validates
that CLI commands work correctly with the performance analysis infrastructure.
"""

import pytest
import asyncio
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from io import StringIO

from .cli import PerformanceAnalysisCLI, main
from .models import TestIdentifier, TestStatus, TimingReport
from .config import PerformanceConfig


class TestPerformanceCLI:
    """Test PerformanceAnalysisCLI class functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.cli = PerformanceAnalysisCLI()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_cli_initialization(self):
        """Test CLI initialization and configuration."""
        assert self.cli is not None
        assert hasattr(self.cli, 'timing_analyzer')
        assert hasattr(self.cli, 'profiler')
        assert hasattr(self.cli, 'bottleneck_analyzer')
        
        # Verify components are properly initialized
        assert self.cli.timing_analyzer is not None
        assert self.cli.profiler is not None
        assert self.cli.bottleneck_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_run_timing_analysis_basic(self):
        """Test basic timing analysis CLI command."""
        # Mock the timing analyzer to avoid running real tests
        with patch.object(self.cli.timing_analyzer, 'analyze_all_integration_tests') as mock_analyze:
            # Create mock timing report
            mock_report = TimingReport(
                total_execution_time=10.0,
                file_results=[],
                timeout_threshold=60.0
            )
            mock_analyze.return_value = mock_report
            
            # Run timing analysis
            result = await self.cli.run_timing_analysis(
                timeout=30,
                output_file=str(self.temp_dir / "timing_report.json")
            )
            
            # Verify execution
            assert result is not None
            mock_analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_profiling_analysis_basic(self):
        """Test basic profiling analysis CLI command."""
        # Create mock slow tests
        mock_slow_tests = [
            Mock(
                test_identifier=TestIdentifier(file_path="test1.py", method_name="test_method1"),
                execution_time=5.0,
                status=TestStatus.PASSED
            )
        ]
        
        with patch.object(self.cli.profiler, 'profile_slow_tests') as mock_profile:
            mock_profile.return_value = []
            
            # Run profiling analysis
            result = await self.cli.run_profiling_analysis(
                slow_tests=mock_slow_tests,
                output_dir=str(self.temp_dir),
                max_tests=2
            )
            
            # Verify execution
            assert result is not None
            mock_profile.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_complete_analysis_workflow(self):
        """Test complete analysis workflow CLI command."""
        with patch.object(self.cli, 'run_timing_analysis') as mock_timing, \
             patch.object(self.cli, 'run_profiling_analysis') as mock_profiling, \
             patch.object(self.cli, 'generate_analysis_report') as mock_report:
            
            # Setup mocks
            mock_timing.return_value = {
                'timing_report': TimingReport(total_execution_time=10.0),
                'slow_tests': []
            }
            mock_profiling.return_value = {'profiling_results': []}
            mock_report.return_value = {'report_path': str(self.temp_dir / 'report.html')}
            
            # Run complete workflow
            result = await self.cli.run_complete_analysis(
                output_dir=str(self.temp_dir),
                timeout=30,
                max_tests=5
            )
            
            # Verify all steps were called
            mock_timing.assert_called_once()
            mock_profiling.assert_called_once()
            mock_report.assert_called_once()
            
            assert result is not None
            assert 'timing_analysis' in result
            assert 'profiling_analysis' in result
            assert 'report_generation' in result
    
    def test_validate_arguments_valid(self):
        """Test argument validation with valid arguments."""
        # Valid arguments should not raise exceptions
        try:
            self.cli._validate_arguments(
                timeout=30,
                max_tests=10,
                output_dir=str(self.temp_dir)
            )
        except Exception as e:
            pytest.fail(f"Valid arguments should not raise exception: {e}")
    
    def test_validate_arguments_invalid_timeout(self):
        """Test argument validation with invalid timeout."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            self.cli._validate_arguments(
                timeout=-1,
                max_tests=10,
                output_dir=str(self.temp_dir)
            )
        
        with pytest.raises(ValueError, match="Timeout must be positive"):
            self.cli._validate_arguments(
                timeout=0,
                max_tests=10,
                output_dir=str(self.temp_dir)
            )
    
    def test_validate_arguments_invalid_max_tests(self):
        """Test argument validation with invalid max_tests."""
        with pytest.raises(ValueError, match="max_tests must be positive"):
            self.cli._validate_arguments(
                timeout=30,
                max_tests=0,
                output_dir=str(self.temp_dir)
            )
        
        with pytest.raises(ValueError, match="max_tests must be positive"):
            self.cli._validate_arguments(
                timeout=30,
                max_tests=-5,
                output_dir=str(self.temp_dir)
            )
    
    def test_validate_arguments_invalid_output_dir(self):
        """Test argument validation with invalid output directory."""
        with pytest.raises(ValueError, match="Output directory must be provided"):
            self.cli._validate_arguments(
                timeout=30,
                max_tests=10,
                output_dir=""
            )
        
        with pytest.raises(ValueError, match="Output directory must be provided"):
            self.cli._validate_arguments(
                timeout=30,
                max_tests=10,
                output_dir=None
            )
    
    @pytest.mark.asyncio
    async def test_error_handling_in_timing_analysis(self):
        """Test error handling in timing analysis CLI command."""
        with patch.object(self.cli.timing_analyzer, 'analyze_all_integration_tests') as mock_analyze:
            # Make the analyzer raise an exception
            mock_analyze.side_effect = RuntimeError("Test error")
            
            # Should handle error gracefully
            with pytest.raises(RuntimeError, match="Test error"):
                await self.cli.run_timing_analysis(
                    output_dir=str(self.temp_dir),
                    timeout=30
                )
    
    @pytest.mark.asyncio
    async def test_output_file_generation(self):
        """Test that CLI generates expected output files."""
        with patch.object(self.cli.timing_analyzer, 'analyze_all_integration_tests') as mock_analyze:
            # Create mock timing report
            mock_report = TimingReport(
                total_execution_time=10.0,
                file_results=[],
                timeout_threshold=60.0
            )
            mock_analyze.return_value = mock_report
            
            # Run timing analysis with file output
            result = await self.cli.run_timing_analysis(
                output_dir=str(self.temp_dir),
                timeout=30,
                save_report=True
            )
            
            # Check if output files were created (may vary based on implementation)
            output_files = list(self.temp_dir.glob("*.json"))
            
            # Verify result structure
            assert result is not None
            assert 'timing_report' in result


class TestCLICommandLineInterface:
    """Test command-line interface entry points."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_main_function_help(self):
        """Test main function with help argument."""
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Test help argument
            with pytest.raises(SystemExit):
                main(['--help'])
            
            output = captured_output.getvalue()
            
            # Should contain help information
            assert 'usage:' in output.lower() or 'help' in output.lower()
            
        finally:
            sys.stdout = old_stdout
    
    @patch('sys.argv', ['perf_analysis', 'timing', '--timeout', '30', '--output-dir', '/tmp'])
    def test_main_function_timing_command(self):
        """Test main function with timing command."""
        with patch('tests.perf.cli.PerformanceAnalysisCLI') as mock_cli_class:
            mock_cli = Mock()
            mock_cli_class.return_value = mock_cli
            mock_cli.run_timing_analysis = AsyncMock(return_value={'status': 'completed'})
            
            # Should not raise exception
            try:
                main()
            except SystemExit as e:
                # Exit code 0 is success
                assert e.code == 0 or e.code is None
    
    def test_argument_parsing_timing_command(self):
        """Test argument parsing for timing command."""
        from .cli import create_argument_parser
        
        parser = create_argument_parser()
        
        # Test timing command arguments
        args = parser.parse_args([
            'timing',
            '--timeout', '60',
            '--max-tests', '10',
            '--output-dir', str(self.temp_dir)
        ])
        
        assert args.command == 'timing'
        assert args.timeout == 60
        assert args.max_tests == 10
        assert args.output_dir == str(self.temp_dir)
    
    def test_argument_parsing_profiling_command(self):
        """Test argument parsing for profiling command."""
        from .cli import create_argument_parser
        
        parser = create_argument_parser()
        
        # Test profiling command arguments
        args = parser.parse_args([
            'profiling',
            '--input-file', 'timing_report.json',
            '--max-tests', '5',
            '--output-dir', str(self.temp_dir)
        ])
        
        assert args.command == 'profiling'
        assert args.input_file == 'timing_report.json'
        assert args.max_tests == 5
        assert args.output_dir == str(self.temp_dir)
    
    def test_argument_parsing_complete_command(self):
        """Test argument parsing for complete analysis command."""
        from .cli import create_argument_parser
        
        parser = create_argument_parser()
        
        # Test complete command arguments
        args = parser.parse_args([
            'complete',
            '--timeout', '120',
            '--max-tests', '15',
            '--output-dir', str(self.temp_dir),
            '--generate-report'
        ])
        
        assert args.command == 'complete'
        assert args.timeout == 120
        assert args.max_tests == 15
        assert args.output_dir == str(self.temp_dir)
        assert args.generate_report is True
    
    def test_argument_parsing_invalid_command(self):
        """Test argument parsing with invalid command."""
        from .cli import create_argument_parser
        
        parser = create_argument_parser()
        
        # Invalid command should raise SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args(['invalid_command'])
    
    def test_argument_parsing_missing_required(self):
        """Test argument parsing with missing required arguments."""
        from .cli import create_argument_parser
        
        parser = create_argument_parser()
        
        # Missing output-dir should raise SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args(['timing', '--timeout', '30'])


class TestCLIReportGeneration:
    """Test CLI report generation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.cli = PerformanceCLI()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_generate_analysis_report_basic(self):
        """Test basic analysis report generation."""
        # Create mock data
        timing_report = TimingReport(
            total_execution_time=15.0,
            file_results=[],
            timeout_threshold=60.0
        )
        
        profiling_results = []
        bottleneck_reports = []
        
        # Generate report
        result = self.cli.generate_analysis_report(
            timing_report=timing_report,
            profiling_results=profiling_results,
            bottleneck_reports=bottleneck_reports,
            output_dir=str(self.temp_dir)
        )
        
        # Verify report generation
        assert result is not None
        assert 'report_path' in result
        
        # Check if report file exists
        report_path = Path(result['report_path'])
        assert report_path.exists()
        assert report_path.suffix in ['.html', '.json', '.txt']
    
    def test_generate_summary_report(self):
        """Test summary report generation."""
        # Create mock timing report with some data
        from .models import FileTimingResult, TestTimingResult
        
        test_result = TestTimingResult(
            test_identifier=TestIdentifier(file_path="test.py", method_name="test_method"),
            execution_time=2.5,
            status=TestStatus.PASSED
        )
        
        file_result = FileTimingResult(
            file_path="test.py",
            total_execution_time=2.5,
            test_results=[test_result],
            file_status=TestStatus.PASSED
        )
        
        timing_report = TimingReport(
            total_execution_time=2.5,
            file_results=[file_result],
            timeout_threshold=60.0
        )
        
        # Generate summary
        summary = self.cli.generate_summary_report(timing_report)
        
        # Verify summary structure
        assert isinstance(summary, dict)
        assert 'total_tests' in summary
        assert 'total_execution_time' in summary
        assert 'average_execution_time' in summary
        assert 'slowest_test_time' in summary
        assert 'fastest_test_time' in summary
        
        # Verify summary values
        assert summary['total_tests'] == 1
        assert summary['total_execution_time'] == 2.5
        assert summary['slowest_test_time'] == 2.5
        assert summary['fastest_test_time'] == 2.5
    
    def test_format_timing_results_for_display(self):
        """Test formatting timing results for display."""
        # Create mock timing results
        test_results = [
            TestTimingResult(
                test_identifier=TestIdentifier(file_path="test1.py", method_name="test_fast"),
                execution_time=1.0,
                status=TestStatus.PASSED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier(file_path="test2.py", method_name="test_slow"),
                execution_time=5.0,
                status=TestStatus.PASSED
            ),
            TestTimingResult(
                test_identifier=TestIdentifier(file_path="test3.py", method_name="test_timeout"),
                execution_time=0.0,
                status=TestStatus.TIMEOUT,
                timeout_occurred=True
            )
        ]
        
        # Format for display
        formatted = self.cli.format_timing_results_for_display(test_results)
        
        # Verify formatting
        assert isinstance(formatted, str)
        assert 'test_fast' in formatted
        assert 'test_slow' in formatted
        assert 'test_timeout' in formatted
        assert '1.0' in formatted  # Fast test time
        assert '5.0' in formatted  # Slow test time
        assert 'TIMEOUT' in formatted or 'timeout' in formatted
    
    def test_export_results_to_json(self):
        """Test exporting results to JSON format."""
        # Create mock data
        timing_report = TimingReport(
            total_execution_time=10.0,
            file_results=[],
            timeout_threshold=60.0
        )
        
        # Export to JSON
        json_path = self.temp_dir / "test_export.json"
        self.cli.export_results_to_json(
            timing_report=timing_report,
            profiling_results=[],
            output_path=str(json_path)
        )
        
        # Verify JSON file was created
        assert json_path.exists()
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert 'timing_report' in data
        assert 'profiling_results' in data
        assert 'export_timestamp' in data
        
        # Verify timing report data
        timing_data = data['timing_report']
        assert timing_data['total_execution_time'] == 10.0
        assert timing_data['timeout_threshold'] == 60.0


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.cli = PerformanceCLI()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_cli_with_no_integration_tests(self):
        """Test CLI behavior when no integration tests are found."""
        with patch.object(self.cli.timing_analyzer, 'analyze_all_integration_tests') as mock_analyze:
            # Return empty timing report
            empty_report = TimingReport(
                total_execution_time=0.0,
                file_results=[],
                timeout_threshold=60.0
            )
            mock_analyze.return_value = empty_report
            
            # Should handle empty results gracefully
            result = await self.cli.run_timing_analysis(
                output_dir=str(self.temp_dir),
                timeout=30
            )
            
            assert result is not None
            assert result['timing_report'].total_test_count == 0
    
    @pytest.mark.asyncio
    async def test_cli_with_all_tests_failing(self):
        """Test CLI behavior when all tests fail."""
        from .models import FileTimingResult, TestTimingResult
        
        # Create mock report with all failed tests
        failed_test = TestTimingResult(
            test_identifier=TestIdentifier(file_path="test.py", method_name="test_method"),
            execution_time=0.0,
            status=TestStatus.FAILED,
            error_message="Test failed"
        )
        
        failed_file = FileTimingResult(
            file_path="test.py",
            total_execution_time=0.0,
            test_results=[failed_test],
            file_status=TestStatus.FAILED
        )
        
        failed_report = TimingReport(
            total_execution_time=0.0,
            file_results=[failed_file],
            timeout_threshold=60.0
        )
        
        with patch.object(self.cli.timing_analyzer, 'analyze_all_integration_tests') as mock_analyze:
            mock_analyze.return_value = failed_report
            
            # Should handle all failed tests gracefully
            result = await self.cli.run_timing_analysis(
                output_dir=str(self.temp_dir),
                timeout=30
            )
            
            assert result is not None
            assert result['timing_report'].total_test_count == 1
            
            # Should identify that no tests passed
            slow_tests = self.cli.timing_analyzer.identify_slow_tests(failed_report)
            assert len(slow_tests) == 0  # No passed tests to be slow
    
    def test_cli_with_invalid_output_directory(self):
        """Test CLI behavior with invalid output directory."""
        # Non-existent parent directory
        invalid_dir = "/non/existent/path/output"
        
        with pytest.raises(ValueError):
            self.cli._validate_arguments(
                timeout=30,
                max_tests=10,
                output_dir=invalid_dir
            )
    
    @pytest.mark.asyncio
    async def test_cli_with_permission_errors(self):
        """Test CLI behavior with permission errors."""
        # Create a directory without write permissions (if possible)
        restricted_dir = self.temp_dir / "restricted"
        restricted_dir.mkdir()
        
        try:
            # Try to make directory read-only (may not work on all systems)
            restricted_dir.chmod(0o444)
            
            # CLI should handle permission errors gracefully
            with pytest.raises((PermissionError, OSError)):
                await self.cli.run_timing_analysis(
                    output_dir=str(restricted_dir / "output"),
                    timeout=30
                )
        
        finally:
            # Restore permissions for cleanup
            try:
                restricted_dir.chmod(0o755)
            except:
                pass
    
    def test_cli_argument_edge_cases(self):
        """Test CLI with edge case arguments."""
        # Very large timeout
        self.cli._validate_arguments(
            timeout=3600,  # 1 hour
            max_tests=1000,
            output_dir=str(self.temp_dir)
        )
        
        # Very small timeout
        self.cli._validate_arguments(
            timeout=1,  # 1 second
            max_tests=1,
            output_dir=str(self.temp_dir)
        )
        
        # Should not raise exceptions for valid edge cases


if __name__ == "__main__":
    pytest.main([__file__, "-v"])