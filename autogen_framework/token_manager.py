"""
Token Manager for the AutoGen Multi-Agent Framework.

This module provides token usage tracking and limit management based on actual
LLM response token counts. It monitors model-specific token limits and triggers
compression when approaching configured thresholds.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .config_manager import ConfigManager


@dataclass
class TokenCheckResult:
    """Result of token limit checking."""
    current_tokens: int
    model_limit: int
    percentage_used: float
    needs_compression: bool


@dataclass
class TokenUsageStats:
    """Token usage statistics."""
    total_tokens_used: int
    requests_made: int
    compressions_performed: int
    average_tokens_per_request: float
    peak_token_usage: int


class TokenManager:
    """
    Manages token monitoring and limits based on actual LLM usage.
    
    Tracks actual token usage from LLM responses, monitors model-specific
    token limits, and triggers compression when approaching configured thresholds.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the TokenManager.
        
        Args:
            config_manager: ConfigManager instance for loading configuration.
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Load token configuration
        self.token_config = self._load_token_config()
        
        # Load model limits with defaults for unknown models
        self.model_limits = self._load_model_limits()
        
        # Track current context size and usage statistics
        self.current_context_size = 0
        self.usage_stats = {
            'total_tokens_used': 0,
            'requests_made': 0,
            'compressions_performed': 0,
            'peak_token_usage': 0,
            'usage_history': []
        }
        
        self.logger.info(f"TokenManager initialized with config: {self.token_config}")
    
    def _load_token_config(self) -> Dict[str, Any]:
        """
        Load token management configuration from ConfigManager.
        
        Returns:
            Dictionary containing token configuration parameters.
        """
        return self.config_manager.get_token_config()
    
    def _load_model_limits(self) -> Dict[str, int]:
        """
        Load model-specific token limits with safe defaults for unknown models.
        
        Returns:
            Dictionary mapping model names to their token limits.
        """
        # Common model limits - can be extended based on actual models used
        default_limits = {
            'models/gemini-2.0-flash': 1048576,  # Gemini 2.0 Flash
            'models/gemini-1.5-pro': 2097152,    # Gemini 1.5 Pro
            'models/gemini-1.5-flash': 1048576,  # Gemini 1.5 Flash
            'gpt-4': 8192,                       # GPT-4
            'gpt-4-turbo': 128000,               # GPT-4 Turbo
            'gpt-3.5-turbo': 16385,              # GPT-3.5 Turbo
            'claude-3-opus': 200000,             # Claude 3 Opus
            'claude-3-sonnet': 200000,           # Claude 3 Sonnet
            'claude-3-haiku': 200000,            # Claude 3 Haiku
        }
        
        # Use default token limit for unknown models
        default_limit = self.token_config.get('default_token_limit', 8192)
        
        # Create a defaultdict-like behavior
        class ModelLimits(dict):
            def __missing__(self, key):
                return default_limit
        
        limits = ModelLimits(default_limits)
        
        self.logger.info(f"Loaded model limits with default: {default_limit}")
        return limits
    
    # --- Estimation and extraction utilities (centralized) ---
    def estimate_tokens_from_char_count(self, char_count: int, base_overhead: int = 0) -> int:
        """
        Estimate token count from character count using a conservative heuristic.

        Args:
            char_count: Number of characters in the content to estimate.
            base_overhead: Extra tokens to account for prompt structure/headers.

        Returns:
            Estimated tokens (at least 1).
        """
        if char_count < 0:
            char_count = 0
        estimated_tokens = char_count // 4
        estimated_tokens += max(base_overhead, 0)
        return max(estimated_tokens, 1)

    def estimate_tokens_from_text(self, text: str, base_overhead: int = 0) -> int:
        """Estimate tokens directly from a text payload."""
        if text is None:
            return max(base_overhead, 1)
        return self.estimate_tokens_from_char_count(len(text), base_overhead)

    def extract_token_usage_from_response(self, response: str, prompt_overhead: int = 50) -> int:
        """
        Extract/estimate token usage from LLM response text.

        If the underlying client can't provide actual token usage, this
        serves as a single, centralized heuristic for the application.
        """
        return self.estimate_tokens_from_text(response, base_overhead=prompt_overhead)
    
    def update_token_usage(self, model: str, tokens_used: int, operation: str, is_actual: bool = True) -> None:
        """
        Update token usage from LLM response.
        
        Args:
            model: The model name that was used.
            tokens_used: Number of tokens used (actual from LLM or estimated).
            operation: Description of the operation that used tokens.
            is_actual: Whether this is actual usage from LLM (True) or estimated (False).
        """
        if tokens_used <= 0:
            self.logger.warning(f"Invalid token count: {tokens_used} for operation: {operation}")
            return
        
        # Only update context size and stats for actual LLM usage
        if is_actual:
            # Update current context size
            self.current_context_size += tokens_used
            
            # Update usage statistics
            self.usage_stats['total_tokens_used'] += tokens_used
            self.usage_stats['requests_made'] += 1
            
            # Update peak usage
            if self.current_context_size > self.usage_stats['peak_token_usage']:
                self.usage_stats['peak_token_usage'] = self.current_context_size
            
            # Calculate average tokens per request
            if self.usage_stats['requests_made'] > 0:
                self.usage_stats['average_tokens_per_request'] = (
                    self.usage_stats['total_tokens_used'] / self.usage_stats['requests_made']
                )
            
            # Record usage history entry
            usage_entry = {
                'timestamp': datetime.now().isoformat(),
                'model': model,
                'tokens_used': tokens_used,
                'operation': operation,
                'current_context_size': self.current_context_size,
                'is_actual': is_actual
            }
            self.usage_stats['usage_history'].append(usage_entry)
            
            # Log token usage
            self.log_token_usage(model, tokens_used, operation)
        else:
            # For estimated usage, just log but don't update persistent stats
            self.logger.debug(f"Estimated token usage: {tokens_used} for {operation} (not counted in stats)")
    
    def check_token_limit(self, model: str, estimated_static_tokens: Optional[int] = None) -> TokenCheckResult:
        """
        Check if current context size exceeds token limits.
        
        Uses actual token usage when available (dynamic phase), or estimated tokens
        for static content when no LLM calls have been made yet (static phase).
        
        Args:
            model: The model name to check limits for.
            estimated_static_tokens: Estimated tokens for static content (used when no actual usage yet).
            
        Returns:
            TokenCheckResult with current usage and compression recommendation.
        """
        model_limit = self.get_model_limit(model)
        
        # Determine current token count based on phase
        if self.current_context_size > 0:
            # Dynamic phase: Use actual token usage from LLM calls
            current_tokens = self.current_context_size
            phase = "dynamic"
        elif estimated_static_tokens is not None and estimated_static_tokens > 0:
            # Static phase: Use estimated tokens for static content
            current_tokens = estimated_static_tokens
            phase = "static"
        else:
            # No usage data available
            current_tokens = 0
            phase = "initial"
        
        percentage_used = current_tokens / model_limit if model_limit > 0 else 0
        
        # Check if compression is needed based on threshold
        compression_threshold = self.token_config.get('compression_threshold', 0.9)
        needs_compression = percentage_used >= compression_threshold
        
        result = TokenCheckResult(
            current_tokens=current_tokens,
            model_limit=model_limit,
            percentage_used=percentage_used,
            needs_compression=needs_compression
        )
        
        if needs_compression:
            self.logger.warning(
                f"Token limit threshold reached ({phase} phase): {current_tokens}/{model_limit} "
                f"({percentage_used:.1%}) for model {model}"
            )
        
        return result
    
    def get_model_limit(self, model: str) -> int:
        """
        Get token limit for specific model.
        
        Args:
            model: The model name to get limit for.
            
        Returns:
            Token limit for the model, or default limit if model is unknown.
        """
        limit = self.model_limits[model]  # Uses __missing__ for unknown models
        
        if self.token_config.get('verbose_logging', False):
            self.logger.debug(f"Token limit for model '{model}': {limit}")
        
        return limit
    
    def log_token_usage(self, model: str, tokens_used: int, operation: str) -> None:
        """
        Log token usage for monitoring.
        
        Args:
            model: The model name that was used.
            tokens_used: Number of tokens used.
            operation: Description of the operation.
        """
        model_limit = self.get_model_limit(model)
        percentage_used = self.current_context_size / model_limit if model_limit > 0 else 0
        
        if self.token_config.get('verbose_logging', False):
            self.logger.info(
                f"Token usage - Model: {model}, Operation: {operation}, "
                f"Tokens used: {tokens_used}, Current context: {self.current_context_size}, "
                f"Limit: {model_limit}, Usage: {percentage_used:.1%}"
            )
        else:
            self.logger.debug(
                f"Token usage: {tokens_used} tokens for {operation} "
                f"(context: {self.current_context_size}/{model_limit})"
            )
    
    def reset_context_size(self) -> None:
        """Reset context size after compression or new conversation."""
        old_size = self.current_context_size
        self.current_context_size = 0
        
        self.logger.info(f"Context size reset from {old_size} to 0")
        
        # Record the reset in usage history
        reset_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'context_reset',
            'old_context_size': old_size,
            'new_context_size': 0
        }
        self.usage_stats['usage_history'].append(reset_entry)
    
    def get_usage_statistics(self) -> TokenUsageStats:
        """
        Get comprehensive token usage statistics.
        
        Returns:
            TokenUsageStats object with current usage metrics.
        """
        return TokenUsageStats(
            total_tokens_used=self.usage_stats['total_tokens_used'],
            requests_made=self.usage_stats['requests_made'],
            compressions_performed=self.usage_stats['compressions_performed'],
            average_tokens_per_request=self.usage_stats.get('average_tokens_per_request', 0.0),
            peak_token_usage=self.usage_stats['peak_token_usage']
        )
    
    def increment_compression_count(self) -> None:
        """Increment the compression counter when compression is performed."""
        self.usage_stats['compressions_performed'] += 1
        
        compression_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'compression_performed',
            'compressions_total': self.usage_stats['compressions_performed']
        }
        self.usage_stats['usage_history'].append(compression_entry)
        
        self.logger.info(f"Compression performed. Total compressions: {self.usage_stats['compressions_performed']}")
    
    def get_detailed_usage_report(self) -> Dict[str, Any]:
        """
        Get detailed usage report including history and statistics.
        
        Returns:
            Dictionary containing comprehensive usage information.
        """
        stats = self.get_usage_statistics()
        
        return {
            'current_context_size': self.current_context_size,
            'statistics': {
                'total_tokens_used': stats.total_tokens_used,
                'requests_made': stats.requests_made,
                'compressions_performed': stats.compressions_performed,
                'average_tokens_per_request': stats.average_tokens_per_request,
                'peak_token_usage': stats.peak_token_usage
            },
            'configuration': self.token_config,
            'model_limits': dict(self.model_limits),
            'usage_history': self.usage_stats['usage_history'][-10:]  # Last 10 entries
        }