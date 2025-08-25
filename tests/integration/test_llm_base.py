"""
Base infrastructure for LLM integration tests.

This module provides the foundational classes and utilities for testing LLM interactions
and output quality validation in the AutoGen multi-agent framework.

Key Components:
- LLMIntegrationTestBase: Base test class with rate limiting and quality validation
- RateLimitHandler: Handles API rate limiting with bash sleep commands
- LLMQualityValidator: Helper class for validating LLM output quality
- Sequential test execution patterns to avoid rate limiting

Usage:
    class TestMyLLMComponent(LLMIntegrationTestBase):
        async def test_llm_functionality(self):
            result = await self.execute_with_rate_limit_handling(
                lambda: self.component.llm_operation()
            )
            quality_report = self.quality_validator.validate_llm_output(
                result, 'requirements'
            )
            assert quality_report['overall_score'] > 0.7
"""

import asyncio
import subprocess
import re
import time
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass
from pathlib import Path
import pytest
import logging

from autogen_framework.quality_metrics import QualityMetricsFramework


@dataclass
class QualityThresholds:
    """Quality thresholds for different types of LLM outputs."""
    
    requirements_generation: Dict[str, float] = None
    design_generation: Dict[str, float] = None
    task_decomposition: Dict[str, float] = None
    error_analysis: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default thresholds if not provided."""
        if self.requirements_generation is None:
            self.requirements_generation = {
                'structure_score': 0.8,
                'completeness_score': 0.85,
                'format_compliance': True,
                'overall_score': 0.75
            }
        
        if self.design_generation is None:
            self.design_generation = {
                'structure_score': 0.85,
                'completeness_score': 0.8,
                'format_compliance': True,
                'overall_score': 0.8
            }
        
        if self.task_decomposition is None:
            self.task_decomposition = {
                'structure_score': 0.9,
                'completeness_score': 0.85,
                'format_compliance': True,
                'overall_score': 0.85
            }
        
        if self.error_analysis is None:
            self.error_analysis = {
                'structure_score': 0.75,
                'completeness_score': 0.8,
                'format_compliance': False,  # Error analysis doesn't require specific format
                'overall_score': 0.75
            }


class RateLimitHandler:
    """
    Handle API rate limiting with bash sleep command execution for 429 errors.
    
    This class detects rate limit errors and uses bash sleep commands instead of
    Python sleep to handle delays, following the design specification.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rate_limit_count = 0
        self.total_delay_time = 0.0
    
    def is_rate_limit_error(self, error: Exception) -> bool:
        """
        Detect if error is due to API rate limiting (429 status).
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is a rate limit error
        """
        error_str = str(error).lower()
        rate_limit_indicators = [
            "429",
            "rate limit",
            "too many requests",
            "quota exceeded",
            "rate exceeded"
        ]
        
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    async def handle_rate_limit(self, error: Exception) -> None:
        """
        Handle rate limiting by executing bash sleep command.
        
        Args:
            error: Rate limit error that occurred
        """
        # Extract delay from error or use default
        delay_seconds = self._extract_delay_from_error(error) or 60
        
        self.rate_limit_count += 1
        self.total_delay_time += delay_seconds
        
        self.logger.info(
            f"Rate limit detected (#{self.rate_limit_count}). "
            f"Executing bash sleep for {delay_seconds} seconds..."
        )
        
        # Execute bash sleep command instead of Python sleep
        process = await asyncio.create_subprocess_exec(
            'sleep', str(delay_seconds),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            self.logger.warning(f"Sleep command failed: {stderr.decode()}")
        else:
            self.logger.info(f"Rate limit handled successfully with {delay_seconds}s delay")
    
    def _extract_delay_from_error(self, error: Exception) -> Optional[int]:
        """
        Extract retry delay from rate limit error response.
        
        Args:
            error: Rate limit error
            
        Returns:
            Delay in seconds if found, None otherwise
        """
        error_str = str(error)
        
        # Look for retry-after header or similar delay indicators
        patterns = [
            r'retry.after[:\s]+(\d+)',
            r'wait[:\s]+(\d+)',
            r'delay[:\s]+(\d+)',
            r'(\d+)\s*seconds?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiting statistics.
        
        Returns:
            Dictionary with rate limiting statistics
        """
        return {
            'rate_limit_count': self.rate_limit_count,
            'total_delay_time': self.total_delay_time,
            'average_delay': self.total_delay_time / max(1, self.rate_limit_count)
        }


class LLMQualityValidator:
    """
    Helper class for LLM integration tests using enhanced quality framework.
    
    This class provides methods to validate LLM output quality using the enhanced
    QualityMetricsFramework with LLM-specific validation capabilities.
    """
    
    def __init__(self, quality_thresholds: Optional[QualityThresholds] = None):
        """
        Initialize quality validator.
        
        Args:
            quality_thresholds: Custom quality thresholds, uses defaults if None
        """
        self.quality_metrics = QualityMetricsFramework()
        self.thresholds = quality_thresholds or QualityThresholds()
        self.logger = logging.getLogger(__name__)
    
    def validate_llm_output(self, content: str, output_type: str) -> Dict[str, Any]:
        """
        Validate LLM output using enhanced quality framework.
        
        Args:
            content: LLM-generated content to validate
            output_type: Type of output ('requirements', 'design', 'tasks', 'error_analysis')
            
        Returns:
            Dictionary containing validation results and quality scores
        """
        # Get quality assessment from enhanced framework
        assessment = self.quality_metrics.assess_llm_document_quality(content, output_type)
        
        # Get appropriate thresholds
        thresholds = self._get_thresholds_for_type(output_type)
        
        # Validate against thresholds
        validation_result = {
            'output_type': output_type,
            'assessment': assessment,
            'thresholds': thresholds,
            'passes_quality_check': True,
            'failed_criteria': [],
            'quality_summary': {}
        }
        
        # Check each threshold
        for criterion, threshold in thresholds.items():
            actual_value = assessment.get(criterion, 0.0)
            
            if isinstance(threshold, bool):
                passes = actual_value == threshold
            else:
                passes = actual_value >= threshold
            
            if not passes:
                validation_result['passes_quality_check'] = False
                validation_result['failed_criteria'].append({
                    'criterion': criterion,
                    'expected': threshold,
                    'actual': actual_value
                })
            
            validation_result['quality_summary'][criterion] = {
                'value': actual_value,
                'threshold': threshold,
                'passes': passes
            }
        
        return validation_result
    
    def validate_requirements_quality(self, content: str) -> Dict[str, Any]:
        """
        Validate requirements document quality with EARS format checking.
        
        Args:
            content: Requirements document content
            
        Returns:
            Validation results with EARS format compliance
        """
        # Use enhanced EARS validation
        ears_validation = self.quality_metrics.validate_ears_format(content)
        
        # Get general document assessment
        general_assessment = self.validate_llm_output(content, 'requirements')
        
        # Combine results
        validation_result = {
            **general_assessment,
            'ears_validation': ears_validation,
            'ears_compliant': ears_validation['compliance_score'] > 0.5
        }
        
        return validation_result
    
    def validate_design_quality(self, content: str) -> Dict[str, Any]:
        """
        Validate design document quality with Mermaid syntax checking.
        
        Args:
            content: Design document content
            
        Returns:
            Validation results with Mermaid syntax compliance
        """
        # Use enhanced Mermaid validation
        mermaid_validation = self.quality_metrics.validate_mermaid_syntax(content)
        
        # Get general document assessment
        general_assessment = self.validate_llm_output(content, 'design')
        
        # Combine results
        validation_result = {
            **general_assessment,
            'mermaid_validation': mermaid_validation,
            'mermaid_compliant': mermaid_validation['syntax_valid']
        }
        
        return validation_result
    
    def validate_tasks_quality(self, content: str) -> Dict[str, Any]:
        """
        Validate tasks document quality with structure checking.
        
        Args:
            content: Tasks document content
            
        Returns:
            Validation results with task structure compliance
        """
        # Use enhanced task structure validation
        task_assessment = self.quality_metrics.assess_task_structure(content)
        
        # Get general document assessment
        general_assessment = self.validate_llm_output(content, 'tasks')
        
        # Combine results
        validation_result = {
            **general_assessment,
            'task_structure': task_assessment,
            'structure_compliant': task_assessment['structure_score'] > 0.7
        }
        
        return validation_result
    
    def validate_revision_improvement(self, original: str, revised: str, feedback: str) -> Dict[str, Any]:
        """
        Validate that revision shows meaningful improvement.
        
        Args:
            original: Original document content
            revised: Revised document content
            feedback: Feedback that was provided
            
        Returns:
            Validation results for revision improvement
        """
        improvement_assessment = self.quality_metrics.assess_revision_improvement(
            original, revised, feedback
        )
        
        validation_result = {
            'improvement_assessment': improvement_assessment,
            'shows_improvement': improvement_assessment['quality_improved'],
            'improvement_score': improvement_assessment['improvement_score'],
            'changes_made': improvement_assessment.get('changes_summary', {}),
            'issues': improvement_assessment['issues']
        }
        
        return validation_result
    
    def _get_thresholds_for_type(self, output_type: str) -> Dict[str, float]:
        """Get quality thresholds for specific output type."""
        threshold_map = {
            'requirements': self.thresholds.requirements_generation,
            'design': self.thresholds.design_generation,
            'tasks': self.thresholds.task_decomposition,
            'error_analysis': self.thresholds.error_analysis
        }
        
        return threshold_map.get(output_type, self.thresholds.requirements_generation)


class LLMTestRetryManager:
    """
    Manage retries for flaky LLM interactions.
    
    This class handles retries for transient errors while delegating rate limit
    handling to the RateLimitHandler.
    """
    
    def __init__(self):
        self.rate_limit_handler = RateLimitHandler()
        self.logger = logging.getLogger(__name__)
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if test should be retried (not for rate limits - those are handled separately).
        
        Args:
            error: Exception that occurred
            attempt: Current attempt number (0-based)
            
        Returns:
            True if test should be retried
        """
        # Don't retry rate limits here - they're handled by RateLimitHandler
        if self.rate_limit_handler.is_rate_limit_error(error):
            return False
        
        # Retry for other transient errors
        transient_errors = ["timeout", "connection", "temporary", "network"]
        error_str = str(error).lower()
        
        is_transient = any(keyword in error_str for keyword in transient_errors)
        return attempt < 3 and is_transient
    
    async def retry_with_backoff(self, test_func: Callable, max_attempts: int = 3) -> Any:
        """
        Retry test with exponential backoff for non-rate-limit errors.
        
        Args:
            test_func: Async function to retry
            max_attempts: Maximum number of attempts
            
        Returns:
            Result of successful test execution
            
        Raises:
            Exception: If all attempts fail
        """
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return await test_func()
            except Exception as e:
                last_exception = e
                
                if self.rate_limit_handler.is_rate_limit_error(e):
                    # Let rate limit handler deal with this
                    await self.rate_limit_handler.handle_rate_limit(e)
                    return await test_func()  # Retry immediately after rate limit handling
                elif self.should_retry(e, attempt):
                    delay = 2 ** attempt  # Exponential backoff
                    self.logger.info(f"Retrying after {delay}s due to transient error: {e}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Not a retryable error or max attempts reached
                    raise
        
        # All attempts failed
        raise Exception(f"Test failed after {max_attempts} attempts. Last error: {last_exception}")


class LLMIntegrationTestBase:
    """
    Base class for all LLM integration tests.
    
    This class provides common functionality for LLM integration testing including:
    - Rate limit handling with bash sleep commands
    - Quality validation using enhanced framework
    - Sequential test execution patterns
    - Standardized test setup and teardown
    """
    
    def setup_method(self, method):
        """
        Setup method called before each test.
        
        Args:
            method: Test method being executed
        """
        # Initialize test infrastructure
        self.rate_limit_handler = RateLimitHandler()
        self.quality_validator = LLMQualityValidator()
        self.retry_manager = LLMTestRetryManager()
        self.logger = logging.getLogger(__name__)
        
        # Test execution tracking
        self.test_start_time = time.time()
        self.test_execution_stats = {
            'rate_limits_encountered': 0,
            'retries_performed': 0,
            'total_execution_time': 0.0
        }
        
        self.logger.info(f"Starting LLM integration test: {method.__name__}")
    
    def teardown_method(self, method):
        """
        Teardown method called after each test.
        
        Args:
            method: Test method that was executed
        """
        if self.test_start_time:
            execution_time = time.time() - self.test_start_time
            self.test_execution_stats['total_execution_time'] = execution_time
        
        # Log test execution statistics
        rate_limit_stats = self.rate_limit_handler.get_stats()
        self.logger.info(
            f"Completed LLM integration test: {method.__name__} "
            f"(execution_time={self.test_execution_stats['total_execution_time']:.2f}s, "
            f"rate_limits={rate_limit_stats['rate_limit_count']}, "
            f"total_delay={rate_limit_stats['total_delay_time']:.2f}s)"
        )
    
    async def execute_with_rate_limit_handling(self, llm_operation: Callable) -> Any:
        """
        Execute LLM operation with automatic rate limit handling.
        
        Args:
            llm_operation: Async callable that performs LLM operation
            
        Returns:
            Result of LLM operation
            
        Raises:
            Exception: If operation fails after rate limit handling
        """
        try:
            return await llm_operation()
        except Exception as e:
            if self.rate_limit_handler.is_rate_limit_error(e):
                self.test_execution_stats['rate_limits_encountered'] += 1
                
                # Handle rate limit and retry
                await self.rate_limit_handler.handle_rate_limit(e)
                return await llm_operation()  # Retry after rate limit handling
            else:
                # Not a rate limit error, re-raise
                raise
    
    async def execute_with_retry(self, llm_operation: Callable, max_attempts: int = 3) -> Any:
        """
        Execute LLM operation with retry logic for transient errors.
        
        Args:
            llm_operation: Async callable that performs LLM operation
            max_attempts: Maximum number of retry attempts
            
        Returns:
            Result of LLM operation
        """
        return await self.retry_manager.retry_with_backoff(llm_operation, max_attempts)
    
    def assert_quality_threshold(self, validation_result: Dict[str, Any], 
                                custom_message: str = None) -> None:
        """
        Assert that LLM output meets quality thresholds.
        
        Args:
            validation_result: Result from quality validator
            custom_message: Custom assertion message
            
        Raises:
            AssertionError: If quality thresholds are not met
        """
        if not validation_result['passes_quality_check']:
            failed_criteria = validation_result['failed_criteria']
            failure_details = []
            
            for failure in failed_criteria:
                failure_details.append(
                    f"{failure['criterion']}: expected {failure['expected']}, "
                    f"got {failure['actual']}"
                )
            
            message = custom_message or "LLM output quality below threshold"
            full_message = f"{message}. Failed criteria: {'; '.join(failure_details)}"
            
            raise AssertionError(full_message)
    
    def log_quality_assessment(self, validation_result: Dict[str, Any]) -> None:
        """
        Log quality assessment results for debugging.
        
        Args:
            validation_result: Result from quality validator
        """
        assessment = validation_result['assessment']
        
        self.logger.info(
            f"Quality Assessment - Type: {validation_result['output_type']}, "
            f"Overall Score: {assessment.get('overall_score', 0.0):.2f}, "
            f"Structure: {assessment.get('structure_score', 0.0):.2f}, "
            f"Completeness: {assessment.get('completeness_score', 0.0):.2f}, "
            f"Format Compliance: {assessment.get('format_compliance', False)}"
        )
        
        if assessment.get('issues'):
            self.logger.warning(f"Quality Issues: {'; '.join(assessment['issues'])}")
        
        if assessment.get('strengths'):
            self.logger.info(f"Quality Strengths: {'; '.join(assessment['strengths'])}")


# Sequential test execution marker
pytest_sequential = pytest.mark.sequential


def sequential_test_execution():
    """
    Decorator to ensure tests run sequentially to avoid rate limiting.
    
    Usage:
        @sequential_test_execution()
        async def test_llm_functionality(self):
            # Test implementation
    """
    def decorator(func):
        return pytest_sequential(func)
    return decorator


# Quality threshold presets for common test scenarios
QUALITY_THRESHOLDS_STRICT = QualityThresholds(
    requirements_generation={
        'structure_score': 0.9,
        'completeness_score': 0.9,
        'format_compliance': True,
        'overall_score': 0.85
    },
    design_generation={
        'structure_score': 0.9,
        'completeness_score': 0.85,
        'format_compliance': True,
        'overall_score': 0.85
    },
    task_decomposition={
        'structure_score': 0.95,
        'completeness_score': 0.9,
        'format_compliance': True,
        'overall_score': 0.9
    }
)

QUALITY_THRESHOLDS_LENIENT = QualityThresholds(
    requirements_generation={
        'structure_score': 0.6,
        'completeness_score': 0.7,
        'format_compliance': False,
        'overall_score': 0.6
    },
    design_generation={
        'structure_score': 0.7,
        'completeness_score': 0.6,
        'format_compliance': False,
        'overall_score': 0.65
    },
    task_decomposition={
        'structure_score': 0.7,
        'completeness_score': 0.7,
        'format_compliance': False,
        'overall_score': 0.7
    }
)