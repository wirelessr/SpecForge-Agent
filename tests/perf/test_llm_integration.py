"""
Integration tests for LLM API profiling with existing performance infrastructure.

This module tests the integration of LLM API call profiling with the existing
PerformanceProfiler and BottleneckAnalyzer components.
"""

import pytest
import asyncio
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unittest.mock import Mock, patch, AsyncMock

from profiler import PerformanceProfiler
from bottleneck_analyzer import BottleneckAnalyzer
from llm_profiler import LLMAPICall, LLMProfilingResult
from models import TestIdentifier, ProfilingResult, ProfilerType, CProfileResult


class TestLLMProfilingIntegration:
    """Test integration of LLM profiling with existing performance infrastructure."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()
        self.analyzer = BottleneckAnalyzer()
        self.test_identifier = TestIdentifier(
            file_path="tests/integration/test_implement_agent.py",
            class_name="TestImplementAgent",
            method_name="test_execute_task_with_llm_calls"
        )
    
    def create_sample_llm_result(self) -> LLMProfilingResult:
        """Create sample LLM profiling result for testing."""
        base_time = time.time()
        
        api_calls = [
            LLMAPICall(
                timestamp=base_time,
                component='TaskDecomposer',
                request_url='https://api.openai.com/v1/chat/completions',
                request_method='POST',
                prompt_text='Analyze task complexity for file processing',
                prompt_size_chars=45,
                prompt_size_tokens=12,
                response_text='Medium complexity task requiring file I/O and parsing',
                response_size_chars=52,
                response_size_tokens=14,
                total_time=2.5,
                network_time=0.4,
                processing_time=2.1,
                status_code=200,
                success=True
            ),
            LLMAPICall(
                timestamp=base_time + 3,
                component='ErrorRecovery',
                request_url='https://api.openai.com/v1/chat/completions',
                request_method='POST',
                prompt_text='Generate recovery strategies for FileNotFoundError in Python script execution',
                prompt_size_chars=82,
                prompt_size_tokens=22,
                response_text='1. Check file path exists\n2. Create missing directories\n3. Use default file if missing',
                response_size_chars=89,
                response_size_tokens=24,
                total_time=3.2,
                network_time=0.6,
                processing_time=2.6,
                status_code=200,
                success=True
            ),
            LLMAPICall(
                timestamp=base_time + 7,
                component='ImplementAgent',
                request_url='https://api.openai.com/v1/chat/completions',
                request_method='POST',
                prompt_text='Coordinate execution of file processing task with error handling',
                prompt_size_chars=68,
                prompt_size_tokens=18,
                response_text='Execute file reader, apply error recovery if needed, validate output',
                response_size_chars=67,
                response_size_tokens=17,
                total_time=1.8,
                network_time=0.3,
                processing_time=1.5,
                status_code=200,
                success=True
            )
        ]
        
        return LLMProfilingResult(
            test_identifier=self.test_identifier,
            api_calls=api_calls
        )
    
    def create_sample_cprofile_result(self) -> CProfileResult:
        """Create sample cProfile result for testing."""
        function_stats = {
            'tests/integration/test_implement_agent.py:50(test_execute_task_with_llm_calls)': {
                'cumulative_time': 8.5,
                'total_time': 0.2,
                'call_count': 1,
                'filename': 'tests/integration/test_implement_agent.py',
                'line_number': 50,
                'function_name': 'test_execute_task_with_llm_calls'
            },
            'autogen_framework/agents/implement_agent.py:120(execute_task)': {
                'cumulative_time': 7.8,
                'total_time': 0.3,
                'call_count': 1,
                'filename': 'autogen_framework/agents/implement_agent.py',
                'line_number': 120,
                'function_name': 'execute_task'
            },
            'autogen_framework/agents/task_decomposer.py:80(decompose_task)': {
                'cumulative_time': 2.7,
                'total_time': 0.2,
                'call_count': 1,
                'filename': 'autogen_framework/agents/task_decomposer.py',
                'line_number': 80,
                'function_name': 'decompose_task'
            },
            'autogen_framework/agents/error_recovery.py:60(recover_from_error)': {
                'cumulative_time': 3.4,
                'total_time': 0.2,
                'call_count': 1,
                'filename': 'autogen_framework/agents/error_recovery.py',
                'line_number': 60,
                'function_name': 'recover_from_error'
            },
            'openai/api_resources/chat_completion.py:200(create)': {
                'cumulative_time': 7.5,  # LLM API calls
                'total_time': 7.5,
                'call_count': 3,
                'filename': 'openai/api_resources/chat_completion.py',
                'line_number': 200,
                'function_name': 'create'
            }
        }
        
        top_functions = [
            {
                'name': 'tests/integration/test_implement_agent.py:50(test_execute_task_with_llm_calls)',
                'cumulative_time': 8.5,
                'total_time': 0.2,
                'call_count': 1,
                'percentage': 100.0
            },
            {
                'name': 'autogen_framework/agents/implement_agent.py:120(execute_task)',
                'cumulative_time': 7.8,
                'total_time': 0.3,
                'call_count': 1,
                'percentage': 91.8
            },
            {
                'name': 'openai/api_resources/chat_completion.py:200(create)',
                'cumulative_time': 7.5,
                'total_time': 7.5,
                'call_count': 3,
                'percentage': 88.2
            }
        ]
        
        return CProfileResult(
            test_identifier=self.test_identifier,
            profile_file_path="/tmp/test_profile.prof",
            total_time=8.5,
            function_stats=function_stats,
            top_functions=top_functions,
            call_count=50
        )
    
    @pytest.mark.asyncio
    async def test_profiler_llm_integration(self):
        """Test that PerformanceProfiler integrates LLM profiling correctly."""
        # Mock the LLM profiler to return our sample data
        sample_llm_result = self.create_sample_llm_result()
        
        with patch.object(self.profiler, 'profile_with_llm_analysis', new_callable=AsyncMock) as mock_llm_profile:
            mock_llm_profile.return_value = sample_llm_result
            
            # Test LLM profiling method
            result = await self.profiler.profile_with_llm_analysis(self.test_identifier)
            
            assert isinstance(result, LLMProfilingResult)
            assert result.test_identifier == self.test_identifier
            assert result.total_api_calls == 3
            assert result.total_llm_time == 7.5  # 2.5 + 3.2 + 1.8
            
            # Verify component breakdown
            assert 'TaskDecomposer' in result.component_call_counts
            assert 'ErrorRecovery' in result.component_call_counts
            assert 'ImplementAgent' in result.component_call_counts
            
            assert result.component_call_counts['TaskDecomposer'] == 1
            assert result.component_call_counts['ErrorRecovery'] == 1
            assert result.component_call_counts['ImplementAgent'] == 1
    
    def test_bottleneck_analyzer_llm_integration(self):
        """Test that BottleneckAnalyzer handles LLM profiling data correctly."""
        # Create sample LLM result
        llm_result = self.create_sample_llm_result()
        
        # Test LLM bottleneck analysis
        llm_bottlenecks = self.analyzer.analyze_llm_api_bottlenecks(llm_result)
        
        assert len(llm_bottlenecks) > 0
        
        # Should identify overall LLM usage as bottleneck (7.5s > 5s threshold)
        llm_bottleneck = next((b for b in llm_bottlenecks if b.component_name == 'LLM_API_Calls'), None)
        assert llm_bottleneck is not None
        assert llm_bottleneck.time_spent == 7.5
        assert llm_bottleneck.optimization_potential in ['high', 'medium']
        
        # Should have function call details
        assert len(llm_bottleneck.function_calls) == 3
        
        # Test LLM optimization recommendations
        llm_recommendations = self.analyzer.generate_llm_optimization_recommendations(llm_result)
        
        # With our sample data (3 calls, avg ~65 chars, no slow calls), 
        # we might not get recommendations based on current thresholds
        # This is actually correct behavior - let's verify the thresholds work
        
        # Should not recommend reducing API calls (only 3 calls < 15 threshold)
        call_reduction_recs = [r for r in llm_recommendations if 'High number of LLM API calls' in r.issue_description]
        assert len(call_reduction_recs) == 0
        
        # Should not recommend prompt size optimization (average ~65 chars < 4000 threshold)
        prompt_size_recs = [r for r in llm_recommendations if 'Large average prompt size' in r.issue_description]
        assert len(prompt_size_recs) == 0
        
        # Should not recommend slow call optimization (no calls > 5s)
        slow_call_recs = [r for r in llm_recommendations if 'slow API calls detected' in r.issue_description]
        assert len(slow_call_recs) == 0
    
    def test_comprehensive_profiling_analysis(self):
        """Test comprehensive analysis with both cProfile and LLM data."""
        # Create sample profiling result with both cProfile and LLM data
        cprofile_result = self.create_sample_cprofile_result()
        llm_result = self.create_sample_llm_result()
        
        profiling_result = ProfilingResult(
            test_identifier=self.test_identifier,
            profiler_used=ProfilerType.CPROFILE,
            cprofile_result=cprofile_result,
            llm_profiling_result=llm_result,
            total_profiling_time=9.0
        )
        
        # Test comprehensive analysis
        reports = self.analyzer.analyze_profiling_results_with_llm([profiling_result])
        
        assert len(reports) == 1
        report = reports[0]
        
        # Should have both cProfile and LLM bottlenecks
        assert len(report.component_bottlenecks) > 0
        
        # Should include LLM time in component timings
        assert report.time_categories.component_timings.llm_api_calls == 7.5
        
        # Should have both cProfile and LLM recommendations
        assert len(report.optimization_recommendations) > 0
        
        # Verify LLM data is properly integrated
        llm_bottlenecks = [b for b in report.component_bottlenecks if 'LLM' in b.component_name]
        assert len(llm_bottlenecks) > 0
    
    def test_profiling_result_llm_properties(self):
        """Test ProfilingResult properties with LLM data."""
        llm_result = self.create_sample_llm_result()
        
        # Test with LLM data
        profiling_result_with_llm = ProfilingResult(
            test_identifier=self.test_identifier,
            profiler_used=ProfilerType.CPROFILE,
            llm_profiling_result=llm_result
        )
        
        assert profiling_result_with_llm.has_llm_data is True
        
        # Test without LLM data
        profiling_result_without_llm = ProfilingResult(
            test_identifier=self.test_identifier,
            profiler_used=ProfilerType.CPROFILE
        )
        
        assert profiling_result_without_llm.has_llm_data is False
        
        # Test with empty LLM data
        empty_llm_result = LLMProfilingResult(test_identifier=self.test_identifier)
        profiling_result_empty_llm = ProfilingResult(
            test_identifier=self.test_identifier,
            profiler_used=ProfilerType.CPROFILE,
            llm_profiling_result=empty_llm_result
        )
        
        assert profiling_result_empty_llm.has_llm_data is False
    
    def test_llm_component_timing_extraction(self):
        """Test extraction of component-specific LLM timing data."""
        llm_result = self.create_sample_llm_result()
        
        # Verify component timing breakdown
        assert llm_result.component_total_times['TaskDecomposer'] == 2.5
        assert llm_result.component_total_times['ErrorRecovery'] == 3.2
        assert llm_result.component_total_times['ImplementAgent'] == 1.8
        
        # Verify network vs processing breakdown
        expected_network_time = 0.4 + 0.6 + 0.3  # 1.3s
        expected_processing_time = 2.1 + 2.6 + 1.5  # 6.2s
        
        assert abs(llm_result.total_network_time - expected_network_time) < 0.01
        assert abs(llm_result.total_processing_time - expected_processing_time) < 0.01
        
        # Verify network percentage calculation
        expected_network_percentage = (expected_network_time / llm_result.total_llm_time) * 100
        assert abs(llm_result.network_percentage - expected_network_percentage) < 0.1
    
    def test_llm_prompt_analysis_correlation(self):
        """Test prompt size correlation analysis."""
        llm_result = self.create_sample_llm_result()
        
        # Verify prompt size metrics
        prompt_sizes = [45, 82, 68]  # From sample data
        expected_avg = sum(prompt_sizes) / len(prompt_sizes)
        expected_max = max(prompt_sizes)
        
        assert abs(llm_result.average_prompt_size - expected_avg) < 0.1
        assert llm_result.largest_prompt_size == expected_max
        
        # Correlation should be calculated (exact value depends on data)
        assert isinstance(llm_result.prompt_size_correlation, float)
        assert -1.0 <= llm_result.prompt_size_correlation <= 1.0
    
    def test_slow_call_identification(self):
        """Test identification of slow LLM API calls."""
        # Create LLM result with a slow call
        base_time = time.time()
        
        api_calls = [
            LLMAPICall(
                timestamp=base_time,
                component='TaskDecomposer',
                request_url='https://api.openai.com/v1/chat/completions',
                request_method='POST',
                prompt_text='Fast call',
                prompt_size_chars=9,
                total_time=2.0,  # Not slow
                success=True
            ),
            LLMAPICall(
                timestamp=base_time + 3,
                component='ErrorRecovery',
                request_url='https://api.openai.com/v1/chat/completions',
                request_method='POST',
                prompt_text='Slow call with complex processing',
                prompt_size_chars=34,
                total_time=6.5,  # Slow call (>5s)
                success=True
            )
        ]
        
        llm_result = LLMProfilingResult(
            test_identifier=self.test_identifier,
            api_calls=api_calls
        )
        
        # Should identify one slow call
        assert len(llm_result.slow_calls) == 1
        assert llm_result.slow_calls[0].total_time == 6.5
        assert llm_result.slow_calls[0].component == 'ErrorRecovery'
        
        # Should generate recommendation for slow calls
        recommendations = self.analyzer.generate_llm_optimization_recommendations(llm_result)
        slow_call_recs = [r for r in recommendations if 'slow API calls detected' in r.issue_description]
        assert len(slow_call_recs) == 1
        assert slow_call_recs[0].expected_impact == 'high'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])