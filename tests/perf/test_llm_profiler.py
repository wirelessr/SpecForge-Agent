"""
Tests for LLM API call profiling functionality.

This module tests the LLMAPIProfiler class and related functionality for
analyzing LLM API calls during integration test execution.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_profiler import LLMAPIProfiler, LLMAPICall, LLMProfilingResult
from models import TestIdentifier


class TestLLMAPICall:
    """Test LLMAPICall data model."""
    
    def test_llm_api_call_creation(self):
        """Test basic LLMAPICall creation and properties."""
        call = LLMAPICall(
            timestamp=time.time(),
            component='TaskDecomposer',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Test prompt',
            prompt_size_chars=11,
            response_text='Test response',
            response_size_chars=13,
            total_time=2.5,
            network_time=0.5,
            processing_time=2.0,
            status_code=200,
            success=True
        )
        
        assert call.component == 'TaskDecomposer'
        assert call.prompt_size_chars == 11
        assert call.response_size_chars == 13
        assert call.total_time == 2.5
        assert call.success is True
        assert not call.is_slow_call  # 2.5s < 5s threshold
        
        # Test calculated properties
        assert call.chars_per_second == 13 / 2.5  # response_chars / total_time
    
    def test_slow_call_detection(self):
        """Test slow call detection."""
        slow_call = LLMAPICall(
            timestamp=time.time(),
            component='ErrorRecovery',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Complex prompt',
            prompt_size_chars=14,
            total_time=6.0,  # > 5s threshold
            success=True
        )
        
        assert slow_call.is_slow_call
    
    def test_tokens_per_second_calculation(self):
        """Test token-based rate calculation."""
        call = LLMAPICall(
            timestamp=time.time(),
            component='ImplementAgent',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Test',
            prompt_size_chars=4,
            response_size_tokens=50,
            total_time=2.0,
            success=True
        )
        
        assert call.tokens_per_second == 25.0  # 50 tokens / 2.0 seconds


class TestLLMProfilingResult:
    """Test LLMProfilingResult data model and calculations."""
    
    def create_sample_calls(self) -> List[LLMAPICall]:
        """Create sample API calls for testing."""
        base_time = time.time()
        
        return [
            LLMAPICall(
                timestamp=base_time,
                component='TaskDecomposer',
                request_url='https://api.openai.com/v1/chat/completions',
                request_method='POST',
                prompt_text='Short prompt',
                prompt_size_chars=12,
                response_text='Short response',
                response_size_chars=14,
                total_time=1.5,
                network_time=0.3,
                processing_time=1.2,
                success=True
            ),
            LLMAPICall(
                timestamp=base_time + 2,
                component='TaskDecomposer',
                request_url='https://api.openai.com/v1/chat/completions',
                request_method='POST',
                prompt_text='Medium length prompt with more details',
                prompt_size_chars=40,
                response_text='Medium response with additional information',
                response_size_chars=44,
                total_time=3.0,
                network_time=0.5,
                processing_time=2.5,
                success=True
            ),
            LLMAPICall(
                timestamp=base_time + 6,
                component='ErrorRecovery',
                request_url='https://api.openai.com/v1/chat/completions',
                request_method='POST',
                prompt_text='Very long prompt with extensive context and detailed requirements that should take longer to process',
                prompt_size_chars=108,
                response_text='Comprehensive response with detailed analysis and multiple recommendations',
                response_size_chars=84,
                total_time=6.5,  # Slow call
                network_time=1.0,
                processing_time=5.5,
                success=True
            )
        ]
    
    def test_profiling_result_metrics_calculation(self):
        """Test automatic calculation of profiling metrics."""
        test_id = TestIdentifier(
            file_path="test_file.py",
            method_name="test_method"
        )
        
        calls = self.create_sample_calls()
        result = LLMProfilingResult(
            test_identifier=test_id,
            api_calls=calls
        )
        
        # Test basic metrics
        assert result.total_api_calls == 3
        assert result.total_llm_time == 11.0  # 1.5 + 3.0 + 6.5
        assert result.average_call_time == 11.0 / 3  # ~3.67
        assert result.slowest_call_time == 6.5
        assert result.fastest_call_time == 1.5
        
        # Test prompt analysis
        assert result.average_prompt_size == (12 + 40 + 108) / 3  # ~53.33
        assert result.largest_prompt_size == 108
        
        # Test network analysis
        assert result.total_network_time == 1.8  # 0.3 + 0.5 + 1.0
        assert result.total_processing_time == 9.2  # 1.2 + 2.5 + 5.5
        assert abs(result.network_percentage - (1.8 / 11.0 * 100)) < 0.1  # ~16.36%
        
        # Test component breakdown
        assert result.component_call_counts['TaskDecomposer'] == 2
        assert result.component_call_counts['ErrorRecovery'] == 1
        assert result.component_total_times['TaskDecomposer'] == 4.5  # 1.5 + 3.0
        assert result.component_total_times['ErrorRecovery'] == 6.5
        
        # Test slow calls
        assert len(result.slow_calls) == 1
        assert result.slow_calls[0].total_time == 6.5
    
    def test_correlation_calculation(self):
        """Test prompt size correlation calculation."""
        test_id = TestIdentifier(file_path="test.py")
        
        # Create calls with clear correlation between prompt size and time
        calls = [
            LLMAPICall(
                timestamp=time.time(),
                component='Test',
                request_url='https://api.test.com',
                request_method='POST',
                prompt_text='a' * 10,  # 10 chars
                prompt_size_chars=10,
                total_time=1.0,
                success=True
            ),
            LLMAPICall(
                timestamp=time.time(),
                component='Test',
                request_url='https://api.test.com',
                request_method='POST',
                prompt_text='a' * 20,  # 20 chars
                prompt_size_chars=20,
                total_time=2.0,
                success=True
            ),
            LLMAPICall(
                timestamp=time.time(),
                component='Test',
                request_url='https://api.test.com',
                request_method='POST',
                prompt_text='a' * 30,  # 30 chars
                prompt_size_chars=30,
                total_time=3.0,
                success=True
            )
        ]
        
        result = LLMProfilingResult(test_identifier=test_id, api_calls=calls)
        
        # Should have perfect positive correlation (1.0)
        assert abs(result.prompt_size_correlation - 1.0) < 0.01


class TestLLMAPIProfiler:
    """Test LLMAPIProfiler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = LLMAPIProfiler()
        self.test_identifier = TestIdentifier(
            file_path="tests/integration/test_implement_agent.py",
            class_name="TestImplementAgent",
            method_name="test_execute_task"
        )
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        assert self.profiler is not None
        assert hasattr(self.profiler, 'api_calls')
        assert hasattr(self.profiler, 'component_patterns')
        assert hasattr(self.profiler, 'llm_endpoint_patterns')
        
        # Check component patterns are loaded
        assert 'TaskDecomposer' in self.profiler.component_patterns
        assert 'ErrorRecovery' in self.profiler.component_patterns
        assert 'ImplementAgent' in self.profiler.component_patterns
        assert 'ContextManager' in self.profiler.component_patterns
    
    def test_llm_endpoint_detection(self):
        """Test LLM API endpoint detection."""
        # Test OpenAI endpoints
        assert self.profiler._is_llm_api_call('https://api.openai.com/v1/chat/completions')
        assert self.profiler._is_llm_api_call('https://api.openai.com/v1/completions')
        
        # Test Anthropic endpoints
        assert self.profiler._is_llm_api_call('https://api.anthropic.com/v1/messages')
        
        # Test non-LLM endpoints
        assert not self.profiler._is_llm_api_call('https://api.github.com/repos')
        assert not self.profiler._is_llm_api_call('https://httpbin.org/get')
    
    def test_prompt_extraction_openai_format(self):
        """Test prompt extraction from OpenAI API format."""
        request_data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'What is the capital of France?'}
            ]
        }
        
        prompt = self.profiler._extract_prompt_from_request(request_data)
        expected = 'You are a helpful assistant.\nWhat is the capital of France?'
        assert prompt == expected
    
    def test_prompt_extraction_direct_format(self):
        """Test prompt extraction from direct prompt format."""
        request_data = {
            'prompt': 'Generate a Python function to calculate fibonacci numbers',
            'max_tokens': 100
        }
        
        prompt = self.profiler._extract_prompt_from_request(request_data)
        assert prompt == 'Generate a Python function to calculate fibonacci numbers'
    
    def test_response_extraction_openai_format(self):
        """Test response extraction from OpenAI API response."""
        response_json = {
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': 'The capital of France is Paris.'
                    }
                }
            ]
        }
        
        response = self.profiler._extract_response_from_json(response_json)
        assert response == 'The capital of France is Paris.'
    
    def test_response_extraction_text_completion(self):
        """Test response extraction from text completion format."""
        response_json = {
            'choices': [
                {
                    'text': 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)'
                }
            ]
        }
        
        response = self.profiler._extract_response_from_json(response_json)
        expected = 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)'
        assert response == expected
    
    def test_component_identification(self):
        """Test component identification from stack trace."""
        # This is a simplified test - in practice, would need to mock stack trace
        component = self.profiler._identify_calling_component()
        # Should return 'Unknown' since we're not in an actual component context
        assert component == 'Unknown'
    
    def test_token_estimation(self):
        """Test token count estimation."""
        # Test empty text
        assert self.profiler._estimate_token_count('') == 0
        
        # Test short text (rough approximation: 1 token ≈ 4 characters)
        assert self.profiler._estimate_token_count('hello') == max(1, 5 // 4)  # 1 token
        assert self.profiler._estimate_token_count('hello world') == max(1, 11 // 4)  # 2 tokens
        
        # Test longer text
        long_text = 'This is a longer text that should result in more tokens'
        expected_tokens = max(1, len(long_text) // 4)
        assert self.profiler._estimate_token_count(long_text) == expected_tokens
    
    def test_timing_breakdown_calculation(self):
        """Test network vs processing time calculation."""
        # Mock response with server timing header
        mock_response = Mock()
        mock_response.headers = {'x-processing-time': '1500'}  # 1.5 seconds in ms
        
        network_time, processing_time = self.profiler._calculate_timing_breakdown(
            mock_response, 2.0  # total time
        )
        
        assert processing_time == 1.5  # From header
        assert network_time == 0.5  # Remainder
    
    def test_timing_breakdown_fallback(self):
        """Test timing breakdown when no server headers available."""
        # Mock response without timing headers
        mock_response = Mock()
        mock_response.headers = {}
        
        network_time, processing_time = self.profiler._calculate_timing_breakdown(
            mock_response, 3.0  # total time
        )
        
        # Should estimate 80% processing, 20% network
        assert abs(processing_time - 2.4) < 0.01  # 80% of 3.0
        assert abs(network_time - 0.6) < 0.01     # 20% of 3.0
    
    @pytest.mark.asyncio
    async def test_profile_test_llm_calls_basic(self):
        """Test basic LLM profiling functionality."""
        # This is a simplified test - full integration would require actual HTTP interception
        result = await self.profiler.profile_test_llm_calls(self.test_identifier, None)
        
        assert isinstance(result, LLMProfilingResult)
        assert result.test_identifier == self.test_identifier
        # Since we're not actually making API calls, should have empty results
        assert result.total_api_calls == 0
    
    def test_generate_analysis_report(self):
        """Test analysis report generation."""
        # Create sample profiling result
        calls = [
            LLMAPICall(
                timestamp=time.time(),
                component='TaskDecomposer',
                request_url='https://api.openai.com/v1/chat/completions',
                request_method='POST',
                prompt_text='Test prompt',
                prompt_size_chars=11,
                response_text='Test response',
                response_size_chars=13,
                total_time=2.5,
                network_time=0.5,
                processing_time=2.0,
                success=True
            )
        ]
        
        llm_result = LLMProfilingResult(
            test_identifier=self.test_identifier,
            api_calls=calls
        )
        
        report = self.profiler.generate_llm_analysis_report(llm_result)
        
        # Check report structure
        assert 'summary' in report
        assert 'component_breakdown' in report
        assert 'prompt_analysis' in report
        assert 'network_analysis' in report
        assert 'slow_calls' in report
        assert 'optimization_recommendations' in report
        
        # Check summary data
        assert report['summary']['total_calls'] == 1
        assert report['summary']['total_time'] == 2.5
        
        # Check component breakdown
        assert 'TaskDecomposer' in report['component_breakdown']
        assert report['component_breakdown']['TaskDecomposer']['call_count'] == 1
    
    def test_optimization_recommendations_generation(self):
        """Test optimization recommendations generation."""
        # Create result with patterns that should trigger recommendations
        calls = []
        base_time = time.time()
        
        # Create many calls to trigger frequency recommendation
        for i in range(25):
            calls.append(LLMAPICall(
                timestamp=base_time + i,
                component='TaskDecomposer',
                request_url='https://api.openai.com/v1/chat/completions',
                request_method='POST',
                prompt_text='a' * 6000,  # Large prompt
                prompt_size_chars=6000,
                total_time=1.0,
                network_time=0.8,  # High network time
                processing_time=0.2,
                success=True
            ))
        
        # Add a slow call
        calls.append(LLMAPICall(
            timestamp=base_time + 25,
            component='ErrorRecovery',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Slow prompt',
            prompt_size_chars=11,
            total_time=7.0,  # Slow call
            network_time=1.0,
            processing_time=6.0,
            success=True
        ))
        
        llm_result = LLMProfilingResult(
            test_identifier=self.test_identifier,
            api_calls=calls
        )
        
        recommendations = self.profiler._generate_llm_optimization_recommendations(llm_result)
        
        # Should have multiple recommendations
        assert len(recommendations) > 0
        
        # Check for expected recommendation types
        rec_issues = [rec['issue'] for rec in recommendations]
        
        # Should recommend reducing API calls (25 > 20 threshold)
        assert any('High number of LLM API calls' in issue for issue in rec_issues)
        
        # Should recommend optimizing prompt size (6000 > 5000 threshold)
        assert any('Large average prompt size' in issue for issue in rec_issues)
        
        # Should recommend addressing slow calls
        assert any('slow API calls detected' in issue for issue in rec_issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])