"""
LLM API Call Profiler for detailed analysis of LLM interactions.

This module provides comprehensive profiling of LLM API calls including timing,
prompt analysis, network latency separation, and correlation with ImplementAgent
component performance.
"""

import asyncio
import time
import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import aiohttp
import httpx
from urllib.parse import urlparse

try:
    from .models import TestIdentifier, ProfilingResult
    from .config import PerformanceConfig
except ImportError:
    # Handle direct execution
    from models import TestIdentifier, ProfilingResult
    from config import PerformanceConfig


@dataclass
class LLMAPICall:
    """Represents a single LLM API call with timing and content data."""
    timestamp: float
    component: str  # Which ImplementAgent component made the call
    request_url: str
    request_method: str
    prompt_text: str
    prompt_size_chars: int
    prompt_size_tokens: Optional[int] = None
    response_text: str = ""
    response_size_chars: int = 0
    response_size_tokens: Optional[int] = None
    
    # Timing measurements
    total_time: float = 0.0
    network_time: float = 0.0  # Time for network round-trip
    processing_time: float = 0.0  # Server-side processing time
    queue_time: float = 0.0  # Time waiting in queue
    
    # Status and error information
    status_code: int = 0
    success: bool = False
    error_message: Optional[str] = None
    
    # Request/response headers for additional analysis
    request_headers: Dict[str, str] = field(default_factory=dict)
    response_headers: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_slow_call(self) -> bool:
        """Check if this is considered a slow API call (>5 seconds)."""
        return self.total_time > 5.0
    
    @property
    def chars_per_second(self) -> float:
        """Calculate response generation rate in characters per second."""
        if self.total_time > 0 and self.response_size_chars > 0:
            return self.response_size_chars / self.total_time
        return 0.0
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate response generation rate in tokens per second."""
        if self.total_time > 0 and self.response_size_tokens:
            return self.response_size_tokens / self.total_time
        return 0.0


@dataclass
class LLMProfilingResult:
    """Results from LLM API call profiling analysis."""
    test_identifier: TestIdentifier
    api_calls: List[LLMAPICall] = field(default_factory=list)
    total_llm_time: float = 0.0
    total_api_calls: int = 0
    
    # Component breakdown
    component_call_counts: Dict[str, int] = field(default_factory=dict)
    component_total_times: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    average_call_time: float = 0.0
    slowest_call_time: float = 0.0
    fastest_call_time: float = float('inf')
    
    # Prompt analysis
    average_prompt_size: float = 0.0
    largest_prompt_size: int = 0
    prompt_size_correlation: float = 0.0  # Correlation between prompt size and response time
    
    # Network vs processing analysis
    total_network_time: float = 0.0
    total_processing_time: float = 0.0
    network_percentage: float = 0.0
    
    # Slow call analysis
    slow_calls: List[LLMAPICall] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.api_calls:
            self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate performance metrics from API call data."""
        if not self.api_calls:
            return
        
        self.total_api_calls = len(self.api_calls)
        self.total_llm_time = sum(call.total_time for call in self.api_calls)
        
        # Calculate timing metrics
        call_times = [call.total_time for call in self.api_calls]
        self.average_call_time = sum(call_times) / len(call_times)
        self.slowest_call_time = max(call_times)
        self.fastest_call_time = min(call_times)
        
        # Calculate prompt size metrics
        prompt_sizes = [call.prompt_size_chars for call in self.api_calls]
        self.average_prompt_size = sum(prompt_sizes) / len(prompt_sizes)
        self.largest_prompt_size = max(prompt_sizes)
        
        # Calculate prompt size correlation with response time
        self.prompt_size_correlation = self._calculate_correlation(prompt_sizes, call_times)
        
        # Calculate network vs processing time
        self.total_network_time = sum(call.network_time for call in self.api_calls)
        self.total_processing_time = sum(call.processing_time for call in self.api_calls)
        
        if self.total_llm_time > 0:
            self.network_percentage = (self.total_network_time / self.total_llm_time) * 100
        
        # Identify slow calls
        self.slow_calls = [call for call in self.api_calls if call.is_slow_call]
        
        # Calculate component breakdowns
        for call in self.api_calls:
            component = call.component
            self.component_call_counts[component] = self.component_call_counts.get(component, 0) + 1
            self.component_total_times[component] = self.component_total_times.get(component, 0.0) + call.total_time
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient between two variables."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


class LLMAPIProfiler:
    """
    Profiler for LLM API calls with detailed timing and content analysis.
    
    This class intercepts and analyzes LLM API calls during integration test
    execution to provide insights into LLM performance bottlenecks and usage patterns.
    """
    
    def __init__(self):
        """Initialize LLM API profiler."""
        self.logger = logging.getLogger(__name__)
        self.api_calls: List[LLMAPICall] = []
        self.active_calls: Dict[str, Dict[str, Any]] = {}  # Track in-progress calls
        
        # Component patterns for identifying which component made the call
        self.component_patterns = {
            'TaskDecomposer': [
                r'task.*decompos', r'decompose.*task', r'analyze.*complexity',
                r'generate.*command', r'create.*plan'
            ],
            'ErrorRecovery': [
                r'error.*recover', r'recover.*error', r'analyze.*error',
                r'generate.*strateg', r'retry.*attempt'
            ],
            'ImplementAgent': [
                r'implement.*agent', r'execute.*task', r'process.*task',
                r'coordinate.*execution'
            ],
            'ContextManager': [
                r'context.*manager', r'load.*context', r'get.*context',
                r'compress.*context'
            ]
        }
        
        # LLM API endpoint patterns
        self.llm_endpoint_patterns = [
            r'/chat/completions',
            r'/completions',
            r'/v1/chat/completions',
            r'/v1/completions',
            r'/v1/messages',  # Anthropic
            r'/anthropic/',
            r'/claude/',
            r'/openai/'
        ]
    
    async def profile_test_llm_calls(self, test_identifier: TestIdentifier,
                                   profiling_result: ProfilingResult) -> LLMProfilingResult:
        """
        Profile LLM API calls for a specific test execution.
        
        Args:
            test_identifier: Test being profiled
            profiling_result: Existing profiling result to enhance
            
        Returns:
            LLMProfilingResult with detailed LLM API analysis
        """
        self.logger.info(f"Starting LLM API profiling for {test_identifier.full_name}")
        
        # Reset state for new test
        self.api_calls.clear()
        self.active_calls.clear()
        
        # Set up API call interception
        original_aiohttp_request = None
        original_httpx_request = None
        
        try:
            # Monkey patch HTTP libraries to intercept LLM API calls
            original_aiohttp_request = self._patch_aiohttp()
            original_httpx_request = self._patch_httpx()
            
            # Re-run the test with API call interception
            # Note: This is a simplified approach - in practice, we'd need to
            # integrate this with the existing profiling infrastructure
            await self._execute_test_with_interception(test_identifier)
            
            # Analyze collected API calls
            llm_result = self._analyze_api_calls(test_identifier)
            
            self.logger.info(
                f"LLM profiling complete: {llm_result.total_api_calls} calls, "
                f"{llm_result.total_llm_time:.2f}s total time"
            )
            
            return llm_result
            
        finally:
            # Restore original HTTP methods
            if original_aiohttp_request:
                self._restore_aiohttp(original_aiohttp_request)
            if original_httpx_request:
                self._restore_httpx(original_httpx_request)
    
    def _patch_aiohttp(self) -> Optional[Any]:
        """Patch aiohttp to intercept LLM API calls."""
        try:
            import aiohttp
            
            original_request = aiohttp.ClientSession._request
            
            async def intercepted_request(session_self, method, url, **kwargs):
                """Intercept aiohttp requests to LLM APIs."""
                call_id = f"{time.time()}_{id(session_self)}"
                
                # Check if this is an LLM API call
                if self._is_llm_api_call(str(url)):
                    await self._start_api_call_tracking(call_id, method, str(url), kwargs)
                
                try:
                    # Make the actual request
                    start_time = time.time()
                    response = await original_request(session_self, method, url, **kwargs)
                    end_time = time.time()
                    
                    # Track the response if this was an LLM call
                    if call_id in self.active_calls:
                        await self._complete_api_call_tracking(call_id, response, end_time - start_time)
                    
                    return response
                    
                except Exception as e:
                    # Track the error if this was an LLM call
                    if call_id in self.active_calls:
                        await self._error_api_call_tracking(call_id, str(e))
                    raise
            
            # Apply the patch
            aiohttp.ClientSession._request = intercepted_request
            return original_request
            
        except ImportError:
            self.logger.debug("aiohttp not available for patching")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to patch aiohttp: {e}")
            return None
    
    def _patch_httpx(self) -> Optional[Any]:
        """Patch httpx to intercept LLM API calls."""
        try:
            import httpx
            
            original_request = httpx.AsyncClient.request
            
            async def intercepted_request(client_self, method, url, **kwargs):
                """Intercept httpx requests to LLM APIs."""
                call_id = f"{time.time()}_{id(client_self)}"
                
                # Check if this is an LLM API call
                if self._is_llm_api_call(str(url)):
                    await self._start_api_call_tracking(call_id, method, str(url), kwargs)
                
                try:
                    # Make the actual request
                    start_time = time.time()
                    response = await original_request(client_self, method, url, **kwargs)
                    end_time = time.time()
                    
                    # Track the response if this was an LLM call
                    if call_id in self.active_calls:
                        await self._complete_api_call_tracking(call_id, response, end_time - start_time)
                    
                    return response
                    
                except Exception as e:
                    # Track the error if this was an LLM call
                    if call_id in self.active_calls:
                        await self._error_api_call_tracking(call_id, str(e))
                    raise
            
            # Apply the patch
            httpx.AsyncClient.request = intercepted_request
            return original_request
            
        except ImportError:
            self.logger.debug("httpx not available for patching")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to patch httpx: {e}")
            return None
    
    def _restore_aiohttp(self, original_request: Any):
        """Restore original aiohttp request method."""
        try:
            import aiohttp
            aiohttp.ClientSession._request = original_request
        except Exception as e:
            self.logger.warning(f"Failed to restore aiohttp: {e}")
    
    def _restore_httpx(self, original_request: Any):
        """Restore original httpx request method."""
        try:
            import httpx
            httpx.AsyncClient.request = original_request
        except Exception as e:
            self.logger.warning(f"Failed to restore httpx: {e}")
    
    def _is_llm_api_call(self, url: str) -> bool:
        """Check if URL is an LLM API endpoint."""
        return any(re.search(pattern, url, re.IGNORECASE) 
                  for pattern in self.llm_endpoint_patterns)
    
    async def _start_api_call_tracking(self, call_id: str, method: str, 
                                     url: str, kwargs: Dict[str, Any]):
        """Start tracking an LLM API call."""
        # Extract request data
        request_data = kwargs.get('json', {}) or kwargs.get('data', {})
        
        # Extract prompt from common LLM API formats
        prompt_text = self._extract_prompt_from_request(request_data)
        
        # Identify which component made this call based on stack trace
        component = self._identify_calling_component()
        
        # Store call tracking data
        self.active_calls[call_id] = {
            'start_time': time.time(),
            'method': method,
            'url': url,
            'prompt_text': prompt_text,
            'component': component,
            'request_headers': kwargs.get('headers', {}),
            'request_data': request_data
        }
    
    async def _complete_api_call_tracking(self, call_id: str, response: Any, 
                                        total_time: float):
        """Complete tracking of an LLM API call."""
        if call_id not in self.active_calls:
            return
        
        call_data = self.active_calls.pop(call_id)
        
        # Extract response data
        try:
            if hasattr(response, 'json'):
                response_json = await response.json()
            else:
                response_json = {}
            
            response_text = self._extract_response_from_json(response_json)
            
            # Calculate timing breakdown
            network_time, processing_time = self._calculate_timing_breakdown(
                response, total_time
            )
            
            # Create API call record
            api_call = LLMAPICall(
                timestamp=call_data['start_time'],
                component=call_data['component'],
                request_url=call_data['url'],
                request_method=call_data['method'],
                prompt_text=call_data['prompt_text'],
                prompt_size_chars=len(call_data['prompt_text']),
                response_text=response_text,
                response_size_chars=len(response_text),
                total_time=total_time,
                network_time=network_time,
                processing_time=processing_time,
                status_code=getattr(response, 'status_code', 200),
                success=200 <= getattr(response, 'status_code', 200) < 300,
                request_headers=dict(call_data['request_headers']),
                response_headers=dict(getattr(response, 'headers', {}))
            )
            
            # Estimate token counts if possible
            api_call.prompt_size_tokens = self._estimate_token_count(api_call.prompt_text)
            api_call.response_size_tokens = self._estimate_token_count(api_call.response_text)
            
            self.api_calls.append(api_call)
            
        except Exception as e:
            self.logger.warning(f"Failed to process API call response: {e}")
    
    async def _error_api_call_tracking(self, call_id: str, error_message: str):
        """Handle error in API call tracking."""
        if call_id not in self.active_calls:
            return
        
        call_data = self.active_calls.pop(call_id)
        
        # Create failed API call record
        api_call = LLMAPICall(
            timestamp=call_data['start_time'],
            component=call_data['component'],
            request_url=call_data['url'],
            request_method=call_data['method'],
            prompt_text=call_data['prompt_text'],
            prompt_size_chars=len(call_data['prompt_text']),
            total_time=time.time() - call_data['start_time'],
            success=False,
            error_message=error_message,
            request_headers=dict(call_data['request_headers'])
        )
        
        self.api_calls.append(api_call)
    
    def _extract_prompt_from_request(self, request_data: Any) -> str:
        """Extract prompt text from LLM API request data."""
        if isinstance(request_data, dict):
            # OpenAI format
            if 'messages' in request_data:
                messages = request_data['messages']
                if isinstance(messages, list) and messages:
                    # Combine all message content
                    prompt_parts = []
                    for msg in messages:
                        if isinstance(msg, dict) and 'content' in msg:
                            prompt_parts.append(str(msg['content']))
                    return '\n'.join(prompt_parts)
            
            # Direct prompt format
            if 'prompt' in request_data:
                return str(request_data['prompt'])
            
            # Anthropic format
            if 'input' in request_data:
                return str(request_data['input'])
        
        # Fallback: convert entire request to string
        return str(request_data)[:1000]  # Limit to first 1000 chars
    
    def _extract_response_from_json(self, response_json: Dict[str, Any]) -> str:
        """Extract response text from LLM API response JSON."""
        # OpenAI format
        if 'choices' in response_json:
            choices = response_json['choices']
            if isinstance(choices, list) and choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    # Chat completion format
                    if 'message' in choice and 'content' in choice['message']:
                        return str(choice['message']['content'])
                    # Text completion format
                    if 'text' in choice:
                        return str(choice['text'])
        
        # Anthropic format
        if 'completion' in response_json:
            return str(response_json['completion'])
        
        # Generic content field
        if 'content' in response_json:
            return str(response_json['content'])
        
        # Fallback: convert entire response to string
        return str(response_json)[:1000]  # Limit to first 1000 chars
    
    def _calculate_timing_breakdown(self, response: Any, total_time: float) -> Tuple[float, float]:
        """
        Calculate network vs processing time breakdown.
        
        Args:
            response: HTTP response object
            total_time: Total request time
            
        Returns:
            Tuple of (network_time, processing_time)
        """
        # Try to extract server processing time from headers
        processing_time = 0.0
        
        if hasattr(response, 'headers'):
            headers = response.headers
            
            # Common server timing headers
            for header_name in ['x-processing-time', 'x-response-time', 'server-timing']:
                if header_name in headers:
                    try:
                        # Extract numeric value from header
                        header_value = headers[header_name]
                        # Look for numeric value (could be in ms or seconds)
                        import re
                        match = re.search(r'(\d+\.?\d*)', str(header_value))
                        if match:
                            value = float(match.group(1))
                            # Assume milliseconds if > 100, otherwise seconds
                            if value > 100:
                                processing_time = value / 1000.0
                            else:
                                processing_time = value
                            break
                    except (ValueError, AttributeError):
                        continue
        
        # If we couldn't extract processing time, estimate it as 80% of total
        # (assuming 20% network overhead)
        if processing_time == 0.0:
            processing_time = total_time * 0.8
        
        # Network time is the remainder
        network_time = max(0.0, total_time - processing_time)
        
        return network_time, processing_time
    
    def _identify_calling_component(self) -> str:
        """
        Identify which ImplementAgent component made the API call.
        
        Returns:
            Component name or 'Unknown' if not identifiable
        """
        import traceback
        
        # Get current stack trace
        stack = traceback.extract_stack()
        
        # Look through stack frames for component patterns
        for frame in reversed(stack):
            filename = frame.filename.lower()
            function_name = frame.name.lower()
            
            # Check each component pattern
            for component, patterns in self.component_patterns.items():
                for pattern in patterns:
                    if (re.search(pattern, function_name, re.IGNORECASE) or
                        re.search(pattern, filename, re.IGNORECASE)):
                        return component
        
        return 'Unknown'
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text.
        
        This is a rough approximation - for accurate counts, would need
        to use the actual tokenizer for the specific model.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Rough approximation: 1 token ≈ 4 characters for English text
        # This varies significantly by language and tokenizer
        return max(1, len(text) // 4)
    
    async def _execute_test_with_interception(self, test_identifier: TestIdentifier):
        """
        Execute test with LLM API call interception enabled.
        
        Note: This is a simplified implementation. In practice, this would
        need to be integrated with the existing test execution infrastructure.
        """
        # This would normally re-run the test with interception enabled
        # For now, we'll simulate some API calls for demonstration
        await asyncio.sleep(0.1)  # Simulate test execution
    
    def _analyze_api_calls(self, test_identifier: TestIdentifier) -> LLMProfilingResult:
        """Analyze collected API calls and generate profiling result."""
        return LLMProfilingResult(
            test_identifier=test_identifier,
            api_calls=self.api_calls.copy()
        )
    
    def generate_llm_analysis_report(self, llm_result: LLMProfilingResult) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report for LLM API usage.
        
        Args:
            llm_result: LLM profiling results
            
        Returns:
            Dictionary with detailed analysis and recommendations
        """
        report = {
            'summary': {
                'total_calls': llm_result.total_api_calls,
                'total_time': llm_result.total_llm_time,
                'average_call_time': llm_result.average_call_time,
                'slowest_call': llm_result.slowest_call_time,
                'fastest_call': llm_result.fastest_call_time
            },
            'component_breakdown': {},
            'prompt_analysis': {
                'average_size': llm_result.average_prompt_size,
                'largest_size': llm_result.largest_prompt_size,
                'size_correlation': llm_result.prompt_size_correlation
            },
            'network_analysis': {
                'total_network_time': llm_result.total_network_time,
                'total_processing_time': llm_result.total_processing_time,
                'network_percentage': llm_result.network_percentage
            },
            'slow_calls': [],
            'optimization_recommendations': []
        }
        
        # Component breakdown
        for component, call_count in llm_result.component_call_counts.items():
            total_time = llm_result.component_total_times.get(component, 0.0)
            avg_time = total_time / call_count if call_count > 0 else 0.0
            
            report['component_breakdown'][component] = {
                'call_count': call_count,
                'total_time': total_time,
                'average_time': avg_time,
                'percentage': (total_time / llm_result.total_llm_time * 100) if llm_result.total_llm_time > 0 else 0.0
            }
        
        # Slow calls analysis
        for call in llm_result.slow_calls:
            report['slow_calls'].append({
                'component': call.component,
                'time': call.total_time,
                'prompt_size': call.prompt_size_chars,
                'url': call.request_url,
                'error': call.error_message
            })
        
        # Generate optimization recommendations
        report['optimization_recommendations'] = self._generate_llm_optimization_recommendations(llm_result)
        
        return report
    
    def _generate_llm_optimization_recommendations(self, llm_result: LLMProfilingResult) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on LLM usage patterns."""
        recommendations = []
        
        # High frequency recommendations
        if llm_result.total_api_calls > 20:
            recommendations.append({
                'issue': f'High number of LLM API calls ({llm_result.total_api_calls})',
                'recommendation': 'Consider batching requests, implementing caching, or reducing redundant calls',
                'priority': 'high'
            })
        
        # Large prompt recommendations
        if llm_result.average_prompt_size > 5000:
            recommendations.append({
                'issue': f'Large average prompt size ({llm_result.average_prompt_size:.0f} chars)',
                'recommendation': 'Optimize prompt templates, use context compression, or implement prompt caching',
                'priority': 'medium'
            })
        
        # Network latency recommendations
        if llm_result.network_percentage > 40:
            recommendations.append({
                'issue': f'High network overhead ({llm_result.network_percentage:.1f}% of total time)',
                'recommendation': 'Consider using a closer API endpoint, connection pooling, or async batching',
                'priority': 'medium'
            })
        
        # Slow call recommendations
        if len(llm_result.slow_calls) > 0:
            recommendations.append({
                'issue': f'{len(llm_result.slow_calls)} slow API calls detected (>5s each)',
                'recommendation': 'Investigate slow calls for prompt optimization, model selection, or timeout handling',
                'priority': 'high'
            })
        
        # Component-specific recommendations
        for component, call_count in llm_result.component_call_counts.items():
            total_time = llm_result.component_total_times.get(component, 0.0)
            if call_count > 5 and total_time > 10.0:
                recommendations.append({
                    'issue': f'{component} makes frequent LLM calls ({call_count} calls, {total_time:.1f}s)',
                    'recommendation': f'Optimize {component} to reduce LLM dependency or implement component-specific caching',
                    'priority': 'medium'
                })
        
        return recommendations