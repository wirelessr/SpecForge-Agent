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


class ContextCompressor:
    """
    Specialized service for context compression operations.
    
    This class provides context compression functionality using LLM capabilities
    directly, without inheriting from BaseLLMAgent to avoid circular dependencies.
    """
    
    def __init__(self, llm_config: LLMConfig, token_manager=None):
        """
        Initialize the ContextCompressor.
        
        Args:
            llm_config: LLM configuration for compression operations.
            token_manager: Optional TokenManager instance for token tracking.
        """
        self.llm_config = llm_config
        self.token_manager = token_manager
        self.logger = logging.getLogger(f"{__name__}.ContextCompressor")
        
        # Initialize AutoGen agent for compression
        self._autogen_agent = None
        self._initialize_autogen_agent()
        
        self.logger.info("ContextCompressor initialized")
    
    def _initialize_autogen_agent(self) -> bool:
        """Initialize the AutoGen agent for compression operations."""
        try:
            # Create model info for Gemini 2.0 Flash
            model_info = ModelInfo(
                family=ModelFamily.GEMINI_2_0_FLASH,
                vision=False,
                function_calling=True,
                json_output=True,
                structured_output=True
            )
            
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
    
    async def compress_context(self, context: Dict[str, Any], target_reduction: float = 0.5) -> CompressionResult:
        """
        Compress context using LLM while preserving critical information.
        
        Args:
            context: Dictionary containing context to compress.
            target_reduction: Target compression ratio (0.5 = 50% reduction).
            
        Returns:
            CompressionResult with compression details and compressed content.
        """
        # Convert context to string for compression
        original_content = self._context_to_string(context)
        original_size = len(original_content)
        
        self.logger.info(f"Starting context compression. Original size: {original_size} characters")
        
        try:
            # Create compression prompt
            compression_prompt = self._build_compression_prompt(original_content, target_reduction)
            
            # Use AutoGen agent for compression
            compressed_content = await self._generate_autogen_response(compression_prompt)
            
            # Calculate compression metrics
            compressed_size = len(compressed_content)
            compression_ratio = (original_size - compressed_size) / original_size if original_size > 0 else 0.0
            
            # Check if compression meets minimum requirements
            min_compression_ratio = 0.3  # Minimum 30% reduction
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
                method_used="llm_compression",
                success=True,
                timestamp=datetime.now().isoformat()
            )
            
            # Update token manager if available
            if self.token_manager:
                self.token_manager.increment_compression_count()
            
            self.logger.info(
                f"Context compression successful. "
                f"Size: {original_size} → {compressed_size} "
                f"({compression_ratio:.1%} reduction)"
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
        target_size = int(original_size * (1 - target_reduction))
        
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
        
        self.logger.info(f"Applied additional truncation: {len(content)} → {current_size} characters")
        return '\n'.join(truncated_lines)
    
    def get_capabilities(self) -> List[str]:
        """
        Get a list of capabilities that this compressor provides.
        
        Returns:
            List of capability descriptions.
        """
        return [
            "Intelligent context compression using LLM",
            "Preservation of critical workflow information",
            "Fallback truncation strategies",
            "Token usage optimization",
            "Context size reduction",
            "Batch compression operations",
            "External compression requests"
        ]
    
    async def compress_multiple_contexts(self, contexts: List[Dict[str, Any]], 
                                       target_reduction: float = 0.5) -> List[CompressionResult]:
        """
        Compress multiple contexts in batch.
        
        Args:
            contexts: List of context dictionaries to compress.
            target_reduction: Target compression ratio for each context.
            
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