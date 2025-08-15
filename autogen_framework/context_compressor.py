"""
Context Compressor for the AutoGen Multi-Agent Framework.

This module provides a specialized context compression service that uses
LLM capabilities directly without inheriting from BaseLLMAgent to avoid
circular dependencies.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo, ModelFamily
from autogen_agentchat.messages import TextMessage

from .models import LLMConfig, CompressionResult, AgentContext
from .config_manager import ConfigManager


class ContextCompressor:
    """
    Specialized service for context compression operations.
    
    This class provides context compression functionality using LLM capabilities
    directly, without inheriting from BaseLLMAgent to avoid circular dependencies.
    """
    
    def __init__(self, llm_config: LLMConfig, token_manager=None, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the ContextCompressor.
        
        Args:
            llm_config: LLM configuration for compression operations.
            token_manager: Optional TokenManager instance for token tracking.
            config_manager: Optional ConfigManager instance for dynamic model configuration.
        """
        self.llm_config = llm_config
        self.token_manager = token_manager
        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger(f"{__name__}.ContextCompressor")
        
        # Get dynamic model configuration
        self.model_info = self.config_manager.get_model_info(llm_config.model)
        self.logger.info(f"Using model configuration: {self.model_info}")
        
        # Initialize AutoGen agent for compression
        self._autogen_agent = None
        self._initialize_autogen_agent()
        
        self.logger.info("ContextCompressor initialized")
    
    def get_max_context_size(self) -> int:
        """
        Get the maximum context size for the current model.
        
        Uses the token limit and context size ratio from configuration.
        
        Returns:
            Maximum context size in tokens.
        """
        token_limit = self.model_info["token_limit"]
        
        # Get context size ratio from framework config, default to 0.8
        framework_config = self.config_manager.get_framework_config()
        context_size_ratio = framework_config.get("context_size_ratio", 0.8)
        
        max_context_size = int(token_limit * context_size_ratio)
        
        self.logger.debug(f"Max context size: {max_context_size} (token_limit: {token_limit}, ratio: {context_size_ratio})")
        return max_context_size
    
    def _initialize_autogen_agent(self) -> bool:
        """Initialize the AutoGen agent for compression operations."""
        try:
            # Get dynamic model family from configuration
            model_family_str = self.model_info["family"]
            model_family = getattr(ModelFamily, model_family_str)
            
            # Get model capabilities from configuration
            capabilities = self.model_info["capabilities"]
            
            # Create model info using dynamic configuration
            model_info = ModelInfo(
                family=model_family,
                vision=capabilities.get("vision", False),
                function_calling=capabilities.get("function_calling", True),
                json_output=True,
                structured_output=True
            )
            
            self.logger.info(f"Using model family: {model_family_str}, capabilities: {capabilities}")
            
            # Create OpenAI client
            client = OpenAIChatCompletionClient(
                model=self.llm_config.model,
                base_url=self.llm_config.base_url,
                api_key=self.llm_config.api_key,
                model_info=model_info
            )
            
            # Create the AssistantAgent
            self._autogen_agent = AssistantAgent(
                name="context_compressor",
                model_client=client,
                system_message=self._build_compression_system_message(),
                description="Specialized service for intelligent context compression"
            )
            
            self.logger.info("AutoGen agent initialized successfully for ContextCompressor")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AutoGen agent for ContextCompressor: {e}")
            return False
    
    def _build_compression_system_message(self) -> str:
        """
        Build the system message for the compression agent.
        
        Returns:
            System message string for context compression.
        """
        return """You are a context compression specialist for an AI multi-agent framework.

Your task is to compress large context while preserving critical information including:

CRITICAL INFORMATION TO PRESERVE:
1. Current workflow state and phase
2. User requirements and specifications
3. Recent decisions and their rationales
4. Active task information and progress
5. Error states and recovery information
6. Configuration and settings
7. Key technical details and constraints

COMPRESSION GUIDELINES:
1. Summarize verbose logs and repetitive information
2. Combine similar or related information
3. Remove redundant explanations and examples
4. Preserve all specific values, names, and identifiers
5. Maintain the logical flow and structure
6. Keep all error messages and warnings
7. Preserve timestamps for critical events

COMPRESSION FORMAT:
- Use clear, concise language
- Maintain original structure where possible
- Use bullet points for lists and summaries
- Preserve code snippets and technical details
- Include compression metadata at the end

Your response should be the compressed context only, without additional commentary."""
    
    async def compress_context(self, context: Dict[str, Any], target_reduction: Optional[float] = None) -> CompressionResult:
        """
        Compress context using LLM while preserving critical information.
        
        Args:
            context: Dictionary containing context to compress.
            target_reduction: Target compression ratio. If None, uses model-specific settings.
            
        Returns:
            CompressionResult with compression details and compressed content.
        """
        # Convert context to string for compression
        original_content = self._context_to_string(context)
        original_size = len(original_content)
        
        # Get model-specific compression settings
        if target_reduction is None:
            framework_config = self.config_manager.get_framework_config()
            target_reduction = framework_config.get("compression_target_ratio", 0.5)
        
        # Get compression threshold from configuration
        framework_config = self.config_manager.get_framework_config()
        compression_threshold = framework_config.get("compression_threshold", 0.9)
        
        # Get max context size for this model
        max_context_size = self.get_max_context_size()
        
        self.logger.info(
            f"Starting context compression. Original size: {original_size} characters, "
            f"Max context size: {max_context_size} tokens, Target reduction: {target_reduction:.1%}"
        )
        
        try:
            # Create compression prompt
            compression_prompt = self._build_compression_prompt(original_content, target_reduction)
            
            # Use AutoGen agent for compression
            compressed_content = await self._generate_autogen_response(compression_prompt)
            
            # Calculate compression metrics
            compressed_size = len(compressed_content)
            compression_ratio = (original_size - compressed_size) / original_size if original_size > 0 else 0.0
            
            # Get minimum compression ratio from configuration
            min_compression_ratio = framework_config.get("min_compression_ratio", 0.3)
            
            # Check if compression meets minimum requirements
            if compression_ratio < min_compression_ratio:
                self.logger.warning(
                    f"Compression ratio {compression_ratio:.2%} below minimum {min_compression_ratio:.2%}. "
                    f"Applying additional truncation."
                )
                # Apply additional truncation
                compressed_content = self._apply_additional_truncation(
                    compressed_content, original_size, target_reduction
                )
                compressed_size = len(compressed_content)
                compression_ratio = (original_size - compressed_size) / original_size if original_size > 0 else 0.0
            
            # Create successful result
            result = CompressionResult(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                compressed_content=compressed_content,
                method_used="llm_compression_dynamic",
                success=True,
                timestamp=datetime.now().isoformat()
            )
            
            # Update token manager if available
            if self.token_manager:
                self.token_manager.increment_compression_count()
            
            self.logger.info(
                f"Context compression successful. "
                f"Size: {original_size} → {compressed_size} "
                f"({compression_ratio:.1%} reduction), Model: {self.llm_config.model}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Context compression failed: {e}")
            
            # Create failure result
            result = CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=0.0,
                compressed_content=original_content,
                method_used="compression_failed",
                success=False,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
            
            return result
    
    async def _generate_autogen_response(self, prompt: str) -> str:
        """Generate response using AutoGen agent."""
        if self._autogen_agent is None:
            raise RuntimeError("AutoGen agent not initialized")
        
        # Create message for AutoGen API
        message = TextMessage(content=prompt, source="user")
        
        # Get response from agent
        response = await self._autogen_agent.on_messages([message], cancellation_token=None)
        
        if hasattr(response, 'chat_message') and response.chat_message:
            return response.chat_message.content
        else:
            raise RuntimeError(f"Unexpected response type from AutoGen: {type(response)}")
    
    def _context_to_string(self, context: Dict[str, Any]) -> str:
        """Convert context dictionary to string for compression."""
        try:
            if isinstance(context, dict):
                parts = []
                for key, value in context.items():
                    if isinstance(value, str):
                        parts.append(f"## {key}\n{value}")
                    elif isinstance(value, (dict, list)):
                        parts.append(f"## {key}\n{json.dumps(value, indent=2)}")
                    else:
                        parts.append(f"## {key}\n{str(value)}")
                return "\n\n".join(parts)
            else:
                return str(context)
        except Exception as e:
            self.logger.warning(f"Error converting context to string: {e}")
            return str(context)
    
    def _apply_additional_truncation(self, content: str, original_size: int, target_reduction: float) -> str:
        """Apply additional truncation when compression is insufficient."""
        # Calculate target size based on model's max context size
        max_context_size = self.get_max_context_size()
        
        # Use the smaller of: target reduction or max context size
        target_size_from_reduction = int(original_size * (1 - target_reduction))
        target_size = min(target_size_from_reduction, max_context_size)
        
        if len(content) <= target_size:
            return content
        
        # Apply intelligent truncation
        lines = content.split('\n')
        truncated_lines = []
        current_size = 0
        
        # Preserve important sections (headers, errors, etc.)
        important_keywords = ['error', 'warning', 'critical', 'failed', 'success', 'completed']
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # Always include important lines
            is_important = any(keyword.lower() in line.lower() for keyword in important_keywords)
            
            if current_size + line_size <= target_size or is_important:
                truncated_lines.append(line)
                current_size += line_size
            else:
                # Add truncation notice
                if current_size + 50 <= target_size:
                    truncated_lines.append("... [Content truncated for size limits] ...")
                break
        
        self.logger.info(
            f"Applied additional truncation: {len(content)} → {current_size} characters "
            f"(target: {target_size}, max_context: {max_context_size})"
        )
        return '\n'.join(truncated_lines)
    
    def get_capabilities(self) -> List[str]:
        """
        Get a list of capabilities that this compressor provides.
        
        Returns:
            List of capability descriptions.
        """
        return [
            "Intelligent context compression using LLM",
            "Dynamic model family detection and configuration",
            "Model-specific token limit and context size calculation",
            "Configurable compression thresholds and ratios",
            "Preservation of critical workflow information",
            "Fallback truncation strategies",
            "Token usage optimization",
            "Context size reduction",
            "Batch compression operations",
            "External compression requests"
        ]
    
    async def compress_multiple_contexts(self, contexts: List[Dict[str, Any]], 
                                       target_reduction: Optional[float] = None) -> List[CompressionResult]:
        """
        Compress multiple contexts in batch.
        
        Args:
            contexts: List of context dictionaries to compress.
            target_reduction: Target compression ratio for each context. If None, uses model-specific settings.
            
        Returns:
            List of CompressionResult objects.
        """
        results = []
        
        for i, context in enumerate(contexts):
            self.logger.info(f"Compressing context {i+1}/{len(contexts)}")
            result = await self.compress_context(context, target_reduction)
            results.append(result)
            
            # Update token manager if available
            if self.token_manager and result.success:
                self.token_manager.increment_compression_count()
        
        return results
    
    def _build_compression_prompt(self, content: str, target_reduction: float) -> str:
        """Build the compression prompt for the LLM."""
        target_percentage = int(target_reduction * 100)
        
        return f"""Please compress the following context by approximately {target_percentage}% while preserving all critical information:

CRITICAL INFORMATION TO PRESERVE:
1. Current workflow state and phase
2. User requirements and specifications  
3. Recent decisions and their rationales
4. Active task information and progress
5. Error states and recovery information
6. Configuration and settings
7. Key technical details and constraints

COMPRESSION GUIDELINES:
1. Summarize verbose logs and repetitive information
2. Combine similar or related information
3. Remove redundant explanations and examples
4. Preserve all specific values, names, and identifiers
5. Maintain the logical flow and structure
6. Keep all error messages and warnings
7. Preserve timestamps for critical events

CONTENT TO COMPRESS:
{content}

COMPRESSION REQUIREMENTS:
- Target reduction: {target_percentage}%
- Preserve all critical workflow information
- Maintain technical accuracy
- Keep all specific values and identifiers
- Preserve error states and recovery information
- Maintain logical structure and flow

Provide only the compressed content without additional commentary."""