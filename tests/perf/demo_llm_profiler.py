#!/usr/bin/env python3
"""
Demo script for LLM API call profiling functionality.

This script demonstrates the LLM API profiling capabilities including timing
measurement, prompt analysis, network latency separation, and optimization
recommendations.
"""

import asyncio
import time
import json
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_profiler import LLMAPIProfiler, LLMAPICall, LLMProfilingResult
from models import TestIdentifier


def create_sample_llm_calls() -> List[LLMAPICall]:
    """Create sample LLM API calls for demonstration."""
    base_time = time.time()
    
    calls = [
        # TaskDecomposer calls - analyzing task complexity
        LLMAPICall(
            timestamp=base_time,
            component='TaskDecomposer',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Analyze the complexity of this task: Create a Python function to parse CSV files',
            prompt_size_chars=78,
            prompt_size_tokens=20,
            response_text='This task has medium complexity. It requires file I/O, CSV parsing, and error handling.',
            response_size_chars=85,
            response_size_tokens=22,
            total_time=2.3,
            network_time=0.4,
            processing_time=1.9,
            status_code=200,
            success=True
        ),
        
        # TaskDecomposer calls - generating command sequence
        LLMAPICall(
            timestamp=base_time + 2.5,
            component='TaskDecomposer',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Generate shell commands to create a Python CSV parser:\n\n' + 
                       'Requirements:\n- Handle CSV files with headers\n- Support different delimiters\n- Include error handling',
            prompt_size_chars=156,
            prompt_size_tokens=42,
            response_text='1. touch csv_parser.py\n2. echo "import csv, sys" >> csv_parser.py\n3. echo "def parse_csv(file_path, delimiter=\',\'):" >> csv_parser.py',
            response_size_chars=128,
            response_size_tokens=35,
            total_time=3.1,
            network_time=0.5,
            processing_time=2.6,
            status_code=200,
            success=True
        ),
        
        # ErrorRecovery calls - analyzing error
        LLMAPICall(
            timestamp=base_time + 6.0,
            component='ErrorRecovery',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Analyze this error and suggest recovery strategies:\n\nFileNotFoundError: [Errno 2] No such file or directory: \'data.csv\'',
            prompt_size_chars=125,
            prompt_size_tokens=32,
            response_text='Error: Missing input file. Strategies: 1) Check file path, 2) Create sample file, 3) Use different input',
            response_size_chars=108,
            response_size_tokens=28,
            total_time=1.8,
            network_time=0.3,
            processing_time=1.5,
            status_code=200,
            success=True
        ),
        
        # ErrorRecovery calls - generating strategies (slow call)
        LLMAPICall(
            timestamp=base_time + 8.2,
            component='ErrorRecovery',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Generate detailed recovery strategies for Python import errors in a complex project with multiple dependencies and virtual environments. Consider various scenarios including missing packages, version conflicts, path issues, and environment activation problems.',
            prompt_size_chars=278,
            prompt_size_tokens=68,
            response_text='Comprehensive recovery strategies:\n\n1. Package Installation Issues:\n   - Check pip list for installed packages\n   - Verify virtual environment activation\n   - Use pip install --upgrade for version conflicts\n\n2. Path Resolution:\n   - Check PYTHONPATH environment variable\n   - Verify sys.path includes required directories\n   - Use absolute imports instead of relative\n\n3. Environment Problems:\n   - Recreate virtual environment if corrupted\n   - Check Python version compatibility\n   - Verify requirements.txt accuracy',
            response_size_chars=542,
            response_size_tokens=142,
            total_time=6.7,  # Slow call
            network_time=1.2,
            processing_time=5.5,
            status_code=200,
            success=True
        ),
        
        # ImplementAgent calls - coordinating execution
        LLMAPICall(
            timestamp=base_time + 15.5,
            component='ImplementAgent',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Coordinate the execution of these tasks: 1) Create CSV parser, 2) Handle file errors, 3) Test with sample data',
            prompt_size_chars=118,
            prompt_size_tokens=30,
            response_text='Execution plan: First create the parser function, then add error handling, finally create test data and run tests.',
            response_size_chars=115,
            response_size_tokens=29,
            total_time=2.1,
            network_time=0.4,
            processing_time=1.7,
            status_code=200,
            success=True
        ),
        
        # ContextManager calls - loading context (high network latency)
        LLMAPICall(
            timestamp=base_time + 18.0,
            component='ContextManager',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Summarize this project context for efficient processing:\n\nProject: CSV Parser\nFiles: csv_parser.py, test_data.csv, requirements.txt\nCurrent status: Parser created, testing in progress\nRecent errors: FileNotFoundError resolved\nNext steps: Add validation, improve error messages',
            prompt_size_chars=285,
            prompt_size_tokens=72,
            response_text='Context: CSV parser project with basic functionality complete. Error handling implemented. Focus on validation and user experience improvements.',
            response_size_chars=145,
            response_size_tokens=37,
            total_time=4.2,
            network_time=2.8,  # High network latency
            processing_time=1.4,
            status_code=200,
            success=True
        ),
        
        # Failed API call
        LLMAPICall(
            timestamp=base_time + 22.5,
            component='TaskDecomposer',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Generate advanced optimization strategies',
            prompt_size_chars=42,
            prompt_size_tokens=8,
            response_text='',
            response_size_chars=0,
            response_size_tokens=0,
            total_time=1.5,
            network_time=1.5,
            processing_time=0.0,
            status_code=429,  # Rate limited
            success=False,
            error_message='Rate limit exceeded'
        )
    ]
    
    return calls


async def demo_llm_profiling():
    """Demonstrate LLM API profiling functionality."""
    print("🔍 LLM API Call Profiling Demo")
    print("=" * 50)
    
    # Create test identifier
    test_identifier = TestIdentifier(
        file_path="tests/integration/test_implement_agent.py",
        class_name="TestImplementAgent",
        method_name="test_execute_complex_task"
    )
    
    # Create sample LLM calls
    sample_calls = create_sample_llm_calls()
    
    # Create LLM profiling result
    llm_result = LLMProfilingResult(
        test_identifier=test_identifier,
        api_calls=sample_calls
    )
    
    print(f"📊 Analysis Results for {test_identifier.full_name}")
    print("-" * 50)
    
    # Display summary metrics
    print(f"Total API Calls: {llm_result.total_api_calls}")
    print(f"Total LLM Time: {llm_result.total_llm_time:.2f}s")
    print(f"Average Call Time: {llm_result.average_call_time:.2f}s")
    print(f"Slowest Call: {llm_result.slowest_call_time:.2f}s")
    print(f"Fastest Call: {llm_result.fastest_call_time:.2f}s")
    print()
    
    # Display prompt analysis
    print("📝 Prompt Analysis:")
    print(f"  Average Prompt Size: {llm_result.average_prompt_size:.0f} characters")
    print(f"  Largest Prompt: {llm_result.largest_prompt_size} characters")
    print(f"  Size-Time Correlation: {llm_result.prompt_size_correlation:.3f}")
    print()
    
    # Display network vs processing analysis
    print("🌐 Network vs Processing Analysis:")
    print(f"  Total Network Time: {llm_result.total_network_time:.2f}s")
    print(f"  Total Processing Time: {llm_result.total_processing_time:.2f}s")
    print(f"  Network Percentage: {llm_result.network_percentage:.1f}%")
    print()
    
    # Display component breakdown
    print("🔧 Component Breakdown:")
    for component, call_count in llm_result.component_call_counts.items():
        total_time = llm_result.component_total_times.get(component, 0.0)
        avg_time = total_time / call_count if call_count > 0 else 0.0
        percentage = (total_time / llm_result.total_llm_time * 100) if llm_result.total_llm_time > 0 else 0.0
        
        print(f"  {component}:")
        print(f"    Calls: {call_count}")
        print(f"    Total Time: {total_time:.2f}s ({percentage:.1f}%)")
        print(f"    Avg Time: {avg_time:.2f}s")
    print()
    
    # Display slow calls
    if llm_result.slow_calls:
        print("🐌 Slow API Calls (>5s):")
        for i, call in enumerate(llm_result.slow_calls, 1):
            print(f"  {i}. {call.component} - {call.total_time:.2f}s")
            print(f"     Prompt: {call.prompt_text[:60]}...")
            print(f"     Network: {call.network_time:.2f}s, Processing: {call.processing_time:.2f}s")
        print()
    
    # Generate and display optimization recommendations
    profiler = LLMAPIProfiler()
    analysis_report = profiler.generate_llm_analysis_report(llm_result)
    
    print("💡 Optimization Recommendations:")
    for i, rec in enumerate(analysis_report['optimization_recommendations'], 1):
        print(f"  {i}. [{rec['priority'].upper()}] {rec['issue']}")
        print(f"     → {rec['recommendation']}")
    print()
    
    # Display detailed call analysis
    print("📋 Detailed Call Analysis:")
    for i, call in enumerate(llm_result.api_calls, 1):
        status = "✅" if call.success else "❌"
        print(f"  {i}. {status} {call.component} - {call.total_time:.2f}s")
        print(f"     Prompt: {call.prompt_size_chars} chars, Response: {call.response_size_chars} chars")
        if call.response_size_chars > 0:
            print(f"     Generation Rate: {call.chars_per_second:.1f} chars/sec")
        if call.error_message:
            print(f"     Error: {call.error_message}")
    print()
    
    # Performance insights
    print("🎯 Performance Insights:")
    
    # Identify most time-consuming component
    max_component = max(llm_result.component_total_times.items(), key=lambda x: x[1])
    print(f"  • {max_component[0]} consumes the most LLM time ({max_component[1]:.2f}s)")
    
    # Check for high network latency
    if llm_result.network_percentage > 30:
        print(f"  • High network latency detected ({llm_result.network_percentage:.1f}% of total time)")
    
    # Check for large prompts
    large_prompts = [call for call in llm_result.api_calls if call.prompt_size_chars > 200]
    if large_prompts:
        print(f"  • {len(large_prompts)} calls have large prompts (>200 chars)")
    
    # Check for failed calls
    failed_calls = [call for call in llm_result.api_calls if not call.success]
    if failed_calls:
        print(f"  • {len(failed_calls)} API calls failed")
    
    print("\n✨ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(demo_llm_profiling())