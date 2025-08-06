"""
Base abstract class for all AI agents in the AutoGen multi-agent framework.

This module defines the BaseLLMAgent abstract class that provides common functionality
and interfaces for all specialized agents (Plan, Design, and Implement agents).
It handles AutoGen integration, LLM configuration, and context management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..models import LLMConfig, AgentContext, SystemInstructions, CompressionResult

# Forward declarations for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..token_manager import TokenManager
    from ..context_compressor import ContextCompressor
    from ..context_manager import ContextManager


class BaseLLMAgent(ABC):
    """
    Abstract base class for all AI agents in the framework.
    
    This class provides common functionality including:
    - AutoGen integration and configuration
    - LLM configuration management
    - Context and memory management
    - Logging and error handling
    - Common agent lifecycle methods
    
    All specialized agents (PlanAgent, DesignAgent, ImplementAgent) should inherit
    from this class and implement the required abstract methods.
    """
    
    def __init__(
        self, 
        name: str, 
        llm_config: LLMConfig, 
        system_message: str,
        description: Optional[str] = None,
        token_manager: Optional['TokenManager'] = None,
        context_compressor: Optional['ContextCompressor'] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name for this agent
            llm_config: LLM configuration for API connection
            system_message: System message/instructions for the agent
            description: Optional description of the agent's role
        """
        self.name = name
        self.llm_config = llm_config
        self.system_message = system_message
        self.description = description or f"{name} agent for the AutoGen framework"
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Context and state management
        self.context: AgentContext = {}
        self.memory_context: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_manager: Optional['ContextManager'] = None
        
        # Token management and context compression
        self.token_manager = token_manager
        self.context_compressor = context_compressor
        
        # Context compression functionality
        self.compression_history: List[CompressionResult] = []
        self.compression_stats = {
            'total_compressions': 0,
            'successful_compressions': 0,
            'failed_compressions': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'average_compression_ratio': 0.0,
            'fallback_truncations': 0
        }
        
        # AutoGen agent instance (will be initialized when needed)
        self._autogen_agent: Optional[ConversableAgent] = None
        self._is_initialized = False
        
        # Validate configuration
        if not self.llm_config.validate():
            raise ValueError(f"Invalid LLM configuration for agent {name}")
        
        self.logger.info(f"Initialized {name} agent with model {llm_config.model}")
    
    def initialize_autogen_agent(self) -> bool:
        """
        Initialize the AutoGen ConversableAgent instance.
        
        This method sets up the actual AutoGen agent with the provided configuration.
        It's called lazily when the agent is first used.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self._is_initialized and self._autogen_agent is not None:
            return True
        

        
        try:
            from autogen_core.models import ModelInfo, ModelFamily
            
            # Create model info for Gemini 2.0 Flash
            model_info = ModelInfo(
                family=ModelFamily.GEMINI_2_0_FLASH,
                vision=False,
                function_calling=True,
                json_output=True,
                structured_output=True
            )
            
            # Create OpenAI client for new AutoGen
            client = OpenAIChatCompletionClient(
                model=self.llm_config.model,
                base_url=self.llm_config.base_url,
                api_key=self.llm_config.api_key,
                model_info=model_info
            )
            
            # Create the AssistantAgent
            self._autogen_agent = AssistantAgent(
                name=self.name,
                model_client=client,
                system_message=self._build_complete_system_message(),
                description=self.description
            )
            
            self._is_initialized = True
            self.logger.info(f"AutoGen agent initialized successfully for {self.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AutoGen agent for {self.name}: {e}")
            return False
    
    def _build_complete_system_message(self) -> str:
        """
        Build the complete system message including context and memory.
        
        Returns:
            Complete system message string
        """
        message_parts = [self.system_message]
        
        # Add memory context if available
        if self.memory_context:
            message_parts.append("\n## Memory Context")
            for key, value in self.memory_context.items():
                if isinstance(value, str):
                    message_parts.append(f"\n### {key}")
                    message_parts.append(value)
                elif isinstance(value, dict):
                    # Handle nested memory structure (like global/projects)
                    message_parts.append(f"\n### {key}")
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str):
                            message_parts.append(f"\n#### {subkey}")
                            message_parts.append(subvalue)
                        elif isinstance(subvalue, dict):
                            # Handle project-specific memory
                            message_parts.append(f"\n#### {subkey}")
                            for file_key, file_content in subvalue.items():
                                message_parts.append(f"\n##### {file_key}")
                                message_parts.append(file_content)
        
        # Add current context if available
        if self.context:
            message_parts.append("\n## Current Context")
            for key, value in self.context.items():
                if isinstance(value, str):
                    message_parts.append(f"\n### {key}")
                    message_parts.append(value)
                elif isinstance(value, (dict, list)):
                    message_parts.append(f"\n### {key}")
                    message_parts.append(str(value))
        
        return "\n".join(message_parts)
    
    def update_context(self, context: AgentContext) -> None:
        """
        Update the agent's current context.
        
        Args:
            context: Dictionary containing context information
        """
        self.context.update(context)
        self.logger.debug(f"Updated context for {self.name} with {len(context)} items")
        
        # If AutoGen agent is already initialized, we need to reinitialize it with new context
        if self._is_initialized and self._autogen_agent is not None:
            try:
                # Reinitialize AutoGen agent with updated system message
                self._is_initialized = False
                self._autogen_agent = None
                self.initialize_autogen_agent()
                self.logger.debug(f"Reinitialized AutoGen agent with updated context for {self.name}")
            except Exception as e:
                self.logger.warning(f"Failed to reinitialize AutoGen agent with updated context: {e}")
    
    def update_memory_context(self, memory_context: Dict[str, Any]) -> None:
        """
        Update the agent's memory context from the memory manager.
        
        Args:
            memory_context: Memory content from MemoryManager
        """
        self.memory_context = memory_context
        self.logger.debug(f"Updated memory context for {self.name}")
        
        # Update AutoGen agent if initialized
        if self._is_initialized and self._autogen_agent is not None:
            try:
                # Reinitialize AutoGen agent with updated memory context
                self._is_initialized = False
                self._autogen_agent = None
                self.initialize_autogen_agent()
                self.logger.debug(f"Reinitialized AutoGen agent with updated memory context for {self.name}")
            except Exception as e:
                self.logger.warning(f"Failed to reinitialize AutoGen agent with memory context: {e}")
    
    def set_context_manager(self, context_manager: 'ContextManager') -> None:
        """
        Set the ContextManager for this agent.
        
        Args:
            context_manager: ContextManager instance to be used by this agent
        """
        self.context_manager = context_manager
        self.logger.info(f"ContextManager set for {self.name} agent")
    
    def add_to_conversation_history(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an entry to the conversation history.
        
        Args:
            role: Role of the speaker (user, assistant, system)
            content: Content of the message
            metadata: Optional metadata about the message
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "agent": self.name
        }
        
        if metadata:
            entry["metadata"] = metadata
        
        self.conversation_history.append(entry)
        
        # Keep history manageable (last 50 entries)
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    async def generate_response(self, prompt: str, context: Optional[AgentContext] = None) -> str:
        """
        Enhanced response generation with token management and context compression.
        
        Args:
            prompt: Input prompt for the agent
            context: Optional additional context for this specific request
            
        Returns:
            Generated response string
        """
        # Ensure AutoGen agent is initialized
        if not self.initialize_autogen_agent():
            raise RuntimeError(f"Failed to initialize AutoGen agent for {self.name}")
        
        # Update context if provided
        original_context = None
        if context:
            original_context = self.context.copy()
            self.update_context(context)
        
        try:
            # Check token limits before sending request
            if self.token_manager:
                token_check = self.token_manager.check_token_limit(self.llm_config.model)
                
                if token_check.needs_compression:
                    self.logger.info(
                        f"Token limit threshold reached for {self.name}: "
                        f"{token_check.current_tokens}/{token_check.model_limit} "
                        f"({token_check.percentage_used:.1%}). Triggering compression."
                    )
                    
                    # Attempt context compression if compressor is available
                    if self.context_compressor:
                        await self._perform_context_compression()
                    else:
                        # Fallback to truncation if no compressor available
                        self.logger.warning(f"No context compressor available for {self.name}, using truncation")
                        self._perform_fallback_truncation()
            
            # Add to conversation history
            self.add_to_conversation_history("user", prompt)
            
            # Generate response using AutoGen
            response = await self._generate_autogen_response(prompt)
            
            # Update token usage after LLM call
            if self.token_manager:
                # Extract token usage from response if available
                tokens_used = self._extract_token_usage_from_response(response)
                if tokens_used > 0:
                    self.token_manager.update_token_usage(
                        self.llm_config.model,
                        tokens_used,
                        "generate_response"
                    )
            
            # Add response to conversation history
            self.add_to_conversation_history("assistant", response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response for {self.name}: {e}")
            raise
        finally:
            # Restore original context if it was temporarily updated
            if context and original_context is not None:
                self.context = original_context
    
    async def _generate_autogen_response(self, prompt: str) -> str:
        """
        Internal method to generate response using AutoGen.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        if self._autogen_agent is None:
            raise RuntimeError("AutoGen agent not initialized")
        
        from autogen_agentchat.messages import TextMessage
        from autogen_agentchat.base import Response
        
        # Create message for new AutoGen API
        message = TextMessage(content=prompt, source="user")
        
        # Get response from agent
        response = await self._autogen_agent.on_messages([message], cancellation_token=None)
        
        if isinstance(response, Response) and response.chat_message:
            content = response.chat_message.content
            self.logger.info(f"Generated response via AutoGen for prompt: {prompt[:50]}...")
            return content
        else:
            raise RuntimeError(f"Unexpected response type from AutoGen: {type(response)}")
    
    async def _perform_context_compression(self) -> None:
        """
        Perform context compression using the available context compressor.
        """
        try:
            # Get current full context
            full_context = self._get_full_context()
            
            # Compress context using the context compressor
            compression_result = await self.context_compressor.compress_context(full_context)
            
            if compression_result.success:
                # Update system message with compressed content
                self.system_message = compression_result.compressed_content
                
                # Reset context size in token manager
                if self.token_manager:
                    self.token_manager.reset_context_size()
                    self.token_manager.increment_compression_count()
                
                # Reinitialize AutoGen agent with compressed context
                self._is_initialized = False
                self._autogen_agent = None
                self.initialize_autogen_agent()
                
                self.logger.info(
                    f"Context compression successful for {self.name}. "
                    f"Size: {compression_result.original_size} → {compression_result.compressed_size} "
                    f"({compression_result.compression_ratio:.1%} reduction)"
                )
            else:
                self.logger.warning(f"Context compression failed for {self.name}: {compression_result.error}")
                # Fallback to truncation
                self._perform_fallback_truncation()
                
        except Exception as e:
            self.logger.error(f"Error during context compression for {self.name}: {e}")
            # Fallback to truncation
            self._perform_fallback_truncation()
    
    def _perform_fallback_truncation(self) -> None:
        """
        Perform fallback truncation when compression fails or is unavailable.
        """
        try:
            # Get model limit for truncation target
            model_limit = 4000  # Conservative default
            if self.token_manager:
                model_limit = int(self.token_manager.get_model_limit(self.llm_config.model) * 0.5)
            
            # Truncate context using built-in method
            truncated_context = self.truncate_context(max_tokens=model_limit)
            
            # Update system message with truncated content
            if 'truncated_content' in truncated_context:
                self.system_message = truncated_context['truncated_content']
                
                # Reset context size in token manager
                if self.token_manager:
                    self.token_manager.reset_context_size()
                
                # Reinitialize AutoGen agent with truncated context
                self._is_initialized = False
                self._autogen_agent = None
                self.initialize_autogen_agent()
                
                self.logger.info(f"Fallback truncation applied for {self.name}")
            
        except Exception as e:
            self.logger.error(f"Error during fallback truncation for {self.name}: {e}")
    
    def _extract_token_usage_from_response(self, response: str) -> int:
        """
        Extract token usage from LLM response.
        
        Since we don't have direct access to token counts from the AutoGen response,
        we estimate based on response length. This is a rough approximation.
        
        Args:
            response: The response string from the LLM
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for most models
        # This is an approximation since we don't have access to actual token counts
        estimated_tokens = len(response) // 4
        
        # Add some tokens for the prompt processing (rough estimate)
        estimated_tokens += 50  # Base overhead for prompt processing
        
        return max(estimated_tokens, 1)  # Ensure at least 1 token is counted
    

    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current status and statistics for this agent.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            "name": self.name,
            "description": self.description,
            "initialized": self._is_initialized,
            "model": self.llm_config.model,
            "context_items": len(self.context),
            "memory_items": len(self.memory_context),
            "conversation_length": len(self.conversation_history),
            "last_activity": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
    
    def reset_agent(self) -> None:
        """
        Reset the agent to its initial state.
        
        This clears conversation history and context but preserves configuration.
        """
        self.context.clear()
        self.conversation_history.clear()
        self.logger.info(f"Reset agent {self.name}")
    
    def export_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Export the conversation history for this agent.
        
        Returns:
            List of conversation entries
        """
        return self.conversation_history.copy()
    
    @abstractmethod
    async def process_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned to this agent.
        
        This is the main method that each specialized agent must implement
        to handle their specific responsibilities.
        
        Args:
            task_input: Dictionary containing task parameters and context
            
        Returns:
            Dictionary containing task results and any relevant metadata
        """
        pass
    
    @abstractmethod
    def get_agent_capabilities(self) -> List[str]:
        """
        Get a list of capabilities that this agent provides.
        
        Returns:
            List of capability descriptions
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name}Agent(model={self.llm_config.model}, initialized={self._is_initialized})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"model='{self.llm_config.model}', "
                f"initialized={self._is_initialized}, "
                f"context_items={len(self.context)})")
    
    # Context Compression Methods
    
    async def compress_context(self, context: Optional[Dict[str, Any]] = None, 
                             target_reduction: float = 0.5) -> CompressionResult:
        """
        Compress context using LLM while preserving critical information.
        
        Args:
            context: Dictionary containing context to compress. If None, uses self.context.
            target_reduction: Target compression ratio (0.5 = 50% reduction).
            
        Returns:
            CompressionResult with compression details and compressed content.
        """
        # Use agent's own context if none provided
        if context is None:
            context = self._get_full_context()
        
        # Convert context to string for compression
        original_content = self._context_to_string(context)
        original_size = len(original_content)
        
        self.logger.info(f"Starting context compression for {self.name}. Original size: {original_size} characters")
        
        try:
            # Create compression prompt with agent-specific system message
            compression_prompt = self._build_compression_prompt(original_content, target_reduction)
            
            # Use agent's own generate_response method for compression
            compressed_content = await self.generate_response(compression_prompt)
            
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
                success=True
            )
            
            # Update statistics and history
            self._update_compression_stats(result)
            self.compression_history.append(result)
            
            self.logger.info(
                f"Context compression successful for {self.name}. "
                f"Size: {original_size} → {compressed_size} "
                f"({compression_ratio:.1%} reduction)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Context compression failed for {self.name}: {e}")
            
            # Create failure result
            result = CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=0.0,
                compressed_content=original_content,
                method_used="compression_failed",
                success=False,
                error=str(e)
            )
            
            # Update statistics
            self._update_compression_stats(result)
            self.compression_history.append(result)
            
            return result
    
    def _get_full_context(self) -> Dict[str, Any]:
        """
        Get the full context of this agent including memory and conversation history.
        
        Returns:
            Dictionary containing all agent context.
        """
        return {
            'agent_name': self.name,
            'system_message': self.system_message,
            'current_context': self.context,
            'memory_context': self.memory_context,
            'conversation_history': self.conversation_history[-10:],  # Last 10 entries
            'agent_description': self.description
        }
    
    def _context_to_string(self, context: Dict[str, Any]) -> str:
        """
        Convert context dictionary to string for compression.
        
        Args:
            context: Context dictionary to convert.
            
        Returns:
            String representation of the context.
        """
        try:
            # Handle different context structures
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
    
    def _build_compression_prompt(self, content: str, target_reduction: float) -> str:
        """
        Build the compression prompt for the LLM.
        
        Args:
            content: Content to compress.
            target_reduction: Target compression ratio.
            
        Returns:
            Compression prompt string.
        """
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
    
    def _apply_additional_truncation(self, content: str, original_size: int, 
                                   target_reduction: float) -> str:
        """
        Apply additional truncation when compression is insufficient.
        
        Args:
            content: Content to truncate further.
            original_size: Original content size.
            target_reduction: Target reduction ratio.
            
        Returns:
            Further truncated content.
        """
        target_size = int(original_size * (1 - target_reduction))
        
        if len(content) <= target_size:
            return content
        
        # Apply intelligent truncation
        lines = content.split('\n')
        truncated_lines = []
        current_size = 0
        
        # Preserve important sections (headers, errors, etc.)
        important_keywords = ['error', 'warning', 'critical', 'requirement', 'task', 'workflow']
        
        # First pass: collect all important lines
        important_lines = []
        regular_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in important_keywords):
                important_lines.append(line)
            else:
                regular_lines.append(line)
        
        # Add important lines first
        for line in important_lines:
            line_size = len(line) + 1  # +1 for newline
            if current_size + line_size <= target_size:
                truncated_lines.append(line)
                current_size += line_size
        
        # Add regular lines if space allows
        for line in regular_lines:
            line_size = len(line) + 1  # +1 for newline
            if current_size + line_size <= target_size:
                truncated_lines.append(line)
                current_size += line_size
            else:
                # Add truncation notice if space allows
                if current_size + 50 <= target_size:  # Space for truncation notice
                    truncated_lines.append("... [Content truncated for size limits] ...")
                break
        
        self.logger.info(f"Applied additional truncation for {self.name}: {len(content)} → {current_size} characters")
        return '\n'.join(truncated_lines)
    
    def truncate_context(self, context: Optional[Dict[str, Any]] = None, max_tokens: int = 4000) -> Dict[str, Any]:
        """
        Fallback truncation strategy for when compression fails.
        
        Args:
            context: Context dictionary to truncate. If None, uses self.context.
            max_tokens: Maximum number of tokens allowed.
            
        Returns:
            Truncated context dictionary.
        """
        if context is None:
            context = self._get_full_context()
            
        self.logger.info(f"Applying fallback truncation for {self.name} with max_tokens: {max_tokens}")
        
        # Convert to string and estimate tokens (rough approximation: 1 token ≈ 4 characters)
        content = self._context_to_string(context)
        estimated_tokens = len(content) // 4
        
        if estimated_tokens <= max_tokens:
            return context
        
        # Calculate target size in characters
        target_size = max_tokens * 4
        
        # Apply oldest-first truncation strategy
        truncated_content = self._apply_oldest_first_truncation(content, target_size)
        
        # Update statistics
        self.compression_stats['fallback_truncations'] += 1
        
        # Create truncation result for history
        result = CompressionResult(
            original_size=len(content),
            compressed_size=len(truncated_content),
            compression_ratio=(len(content) - len(truncated_content)) / len(content),
            compressed_content=truncated_content,
            method_used="fallback_truncation",
            success=True
        )
        
        self.compression_history.append(result)
        
        # Convert back to context structure
        return {"truncated_content": truncated_content}
    
    def _apply_oldest_first_truncation(self, content: str, target_size: int) -> str:
        """
        Apply oldest-first truncation strategy.
        
        Args:
            content: Content to truncate.
            target_size: Target size in characters.
            
        Returns:
            Truncated content.
        """
        if len(content) <= target_size:
            return content
        
        # Split content into sections
        sections = content.split('\n\n')
        
        # Keep the most recent sections that fit within target size
        truncated_sections = []
        current_size = 0
        
        # Start from the end (most recent) and work backwards
        for section in reversed(sections):
            section_size = len(section) + 2  # +2 for double newline
            
            if current_size + section_size <= target_size:
                truncated_sections.insert(0, section)
                current_size += section_size
            else:
                # Add truncation notice at the beginning
                if current_size + 50 <= target_size:
                    truncated_sections.insert(0, "... [Older content truncated] ...")
                break
        
        return '\n\n'.join(truncated_sections)
    
    def _update_compression_stats(self, result: CompressionResult) -> None:
        """
        Update compression statistics with new result.
        
        Args:
            result: CompressionResult to update statistics with.
        """
        self.compression_stats['total_compressions'] += 1
        
        if result.success:
            self.compression_stats['successful_compressions'] += 1
        else:
            self.compression_stats['failed_compressions'] += 1
        
        self.compression_stats['total_original_size'] += result.original_size
        self.compression_stats['total_compressed_size'] += result.compressed_size
        
        # Calculate average compression ratio
        if self.compression_stats['total_original_size'] > 0:
            total_reduction = (
                self.compression_stats['total_original_size'] - 
                self.compression_stats['total_compressed_size']
            )
            self.compression_stats['average_compression_ratio'] = (
                total_reduction / self.compression_stats['total_original_size']
            )
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics and history for this agent.
        
        Returns:
            Dictionary containing comprehensive compression statistics.
        """
        return {
            'agent_name': self.name,
            'statistics': self.compression_stats.copy(),
            'recent_compressions': self.compression_history[-10:],  # Last 10 compressions
            'total_history_entries': len(self.compression_history)
        }
    
    def get_detailed_compression_report(self) -> Dict[str, Any]:
        """
        Get detailed compression report including history and performance metrics.
        
        Returns:
            Dictionary containing detailed compression information.
        """
        stats = self.compression_stats.copy()
        
        # Calculate additional metrics
        success_rate = 0.0
        if stats['total_compressions'] > 0:
            success_rate = stats['successful_compressions'] / stats['total_compressions']
        
        average_original_size = 0.0
        average_compressed_size = 0.0
        if stats['successful_compressions'] > 0:
            average_original_size = stats['total_original_size'] / stats['successful_compressions']
            average_compressed_size = stats['total_compressed_size'] / stats['successful_compressions']
        
        return {
            'agent_name': self.name,
            'summary': {
                'total_compressions': stats['total_compressions'],
                'success_rate': success_rate,
                'average_compression_ratio': stats['average_compression_ratio'],
                'fallback_truncations': stats['fallback_truncations']
            },
            'size_metrics': {
                'total_original_size': stats['total_original_size'],
                'total_compressed_size': stats['total_compressed_size'],
                'average_original_size': average_original_size,
                'average_compressed_size': average_compressed_size,
                'total_size_saved': stats['total_original_size'] - stats['total_compressed_size']
            },
            'configuration': {
                'model': self.llm_config.model,
                'agent_initialized': self._is_initialized
            },
            'compression_history': self.compression_history.copy()
        }
    
    def reset_compression_stats(self) -> None:
        """Reset compression statistics and history for this agent."""
        self.compression_history.clear()
        self.compression_stats = {
            'total_compressions': 0,
            'successful_compressions': 0,
            'failed_compressions': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'average_compression_ratio': 0.0,
            'fallback_truncations': 0
        }
        
        self.logger.info(f"Compression statistics and history reset for {self.name}")
    
    async def compress_and_replace_context(self, target_reduction: float = 0.5) -> CompressionResult:
        """
        Compress the agent's current context and replace it with the compressed version.
        
        Args:
            target_reduction: Target compression ratio (0.5 = 50% reduction).
            
        Returns:
            CompressionResult with compression details.
        """
        # Compress current context
        result = await self.compress_context(target_reduction=target_reduction)
        
        if result.success:
            # Replace context with compressed version
            try:
                # Parse compressed content back to context structure if possible
                compressed_context = {
                    'compressed_content': result.compressed_content,
                    'compression_metadata': {
                        'original_size': result.original_size,
                        'compressed_size': result.compressed_size,
                        'compression_ratio': result.compression_ratio,
                        'timestamp': result.timestamp,
                        'method_used': result.method_used
                    }
                }
                
                # Update agent's context
                self.context = compressed_context
                
                # Clear old conversation history to save space
                if len(self.conversation_history) > 5:
                    self.conversation_history = self.conversation_history[-5:]
                
                self.logger.info(f"Context replaced with compressed version for {self.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to replace context with compressed version: {e}")
                result.success = False
                result.error = f"Context replacement failed: {str(e)}"
        
        return result