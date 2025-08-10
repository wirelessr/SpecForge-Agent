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

from ..models import LLMConfig, AgentContext
from ..utils.context_utils import dict_context_to_string

# Forward declarations for type hints
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..token_manager import TokenManager
    from ..context_manager import ContextManager


@dataclass
class ContextSpec:
    """Specification for context requirements."""
    context_type: str  # 'plan', 'design', 'tasks', 'implementation'
    required_params: Optional[Dict[str, Any]] = None


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
        token_manager: 'TokenManager',
        context_manager: 'ContextManager',
        description: Optional[str] = None,
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name for this agent
            llm_config: LLM configuration for API connection
            system_message: System message/instructions for the agent
            token_manager: TokenManager instance for token operations (mandatory)
            context_manager: ContextManager instance for context operations (mandatory)
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
        self.context_manager: 'ContextManager' = context_manager
        
        # Token management (mandatory)
        self.token_manager = token_manager
        
        # AutoGen agent instance (will be initialized when needed)
        self._autogen_agent: Optional[AssistantAgent] = None
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
            # Build and compress system prompt using ContextManager + TokenManager
            prepared_system = self.system_message
            estimated_static_tokens = None
            try:
                prepared = await self.context_manager.prepare_system_prompt(self._build_complete_system_message())
                prepared_system = prepared.system_prompt
                estimated_static_tokens = prepared.estimated_tokens
            except Exception as e:
                self.logger.warning(f"Failed to prepare system prompt: {e}")

            # Ensure AutoGen agent is initialized after system prompt is finalized
            # Reinitialize only if the prepared system differs
            if prepared_system != self.system_message:
                self.system_message = prepared_system
                self._is_initialized = False
                self._autogen_agent = None
                if not self.initialize_autogen_agent():
                    raise RuntimeError(f"Failed to initialize AutoGen agent for {self.name}")
            
            # Add to conversation history
            self.add_to_conversation_history("user", prompt)
            
            # Generate response using AutoGen
            response = await self._generate_autogen_response(prompt)
            
            # Update token usage after LLM call
            # Use centralized extraction heuristic
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
    
    # Deprecated: compression/truncation are handled by ContextManager now
    async def _perform_context_compression(self) -> None:
        raise RuntimeError(
            "_perform_context_compression is removed. Use ContextManager.prepare_system_prompt for compression."
        )

    def _perform_fallback_truncation(self) -> None:
        raise RuntimeError(
            "_perform_fallback_truncation is removed. Use ContextManager.prepare_system_prompt for token control."
        )
    
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
        # Delegate estimation to TokenManager if available to centralize heuristics
        return self.token_manager.extract_token_usage_from_response(response, prompt_overhead=50)
        
    

    def _estimate_static_content_tokens(self) -> int:
        """
        Estimate token count for static content (system message, context, memory).
        
        This is used in the static phase before any LLM calls have been made.
        
        Returns:
            Estimated token count for static content
        """
        total_chars = 0
        
        # Count system message
        if self.system_message:
            total_chars += len(self.system_message)
        
        # Count context content
        for key, value in self.context.items():
            if isinstance(value, str):
                total_chars += len(value)
            elif isinstance(value, (list, dict)):
                total_chars += len(str(value))
        
        # Count memory context
        for key, value in self.memory_context.items():
            if isinstance(value, str):
                total_chars += len(value)
            elif isinstance(value, (list, dict)):
                total_chars += len(str(value))
        
        # Count conversation history
        for entry in self.conversation_history:
            if 'content' in entry:
                total_chars += len(entry['content'])
        
        # Use centralized estimation in TokenManager when available
        estimated_tokens = self.token_manager.estimate_tokens_from_char_count(total_chars, base_overhead=100)
        self.logger.debug(f"Estimated static content tokens for {self.name}: {estimated_tokens}")
        return max(estimated_tokens, 1)
    

    
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
    
    async def process_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned to this agent with automatic context retrieval.
        
        This method automatically retrieves appropriate context from ContextManager
        before delegating to the concrete agent implementation.
        
        Args:
            task_input: Dictionary containing task parameters and context
            
        Returns:
            Dictionary containing task results and any relevant metadata
        """
        # Automatically retrieve context if ContextManager is available
        if self.context_manager:
            await self._retrieve_agent_context(task_input)
        
        # Delegate to concrete agent implementation
        return await self._process_task_impl(task_input)
    
    @abstractmethod
    async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Concrete implementation of task processing for each agent type.
        
        This is the method that each specialized agent must implement
        to handle their specific responsibilities.
        
        Args:
            task_input: Dictionary containing task parameters and context
            
        Returns:
            Dictionary containing task results and any relevant metadata
        """
        pass
    
    async def _retrieve_agent_context(self, task_input: Dict[str, Any]) -> None:
        """
        Automatically retrieve appropriate context based on agent type.
        
        Args:
            task_input: Dictionary containing task parameters
        """
        try:
            # Let each agent define what context it needs
            context_spec = self.get_context_requirements(task_input)
            if not context_spec:
                return
            
            # Get context based on specification
            context = await self._get_context_by_spec(context_spec, task_input)
            if context:
                # Format context for agent consumption
                formatted_context = self._format_context_for_agent(context, context_spec.context_type)
                self.update_context(formatted_context)
                self.logger.info(f"Retrieved {context_spec.context_type}Context from ContextManager")
                
        except Exception as e:
            self.logger.warning(f"Failed to retrieve context from ContextManager: {e}")
            # Continue without context - agents should handle missing context gracefully
    
    async def _get_context_by_spec(self, context_spec: 'ContextSpec', task_input: Dict[str, Any]) -> Optional[Any]:
        """
        Get context based on context specification.
        
        Args:
            context_spec: Context specification from agent
            task_input: Task input parameters
            
        Returns:
            Context object or None if retrieval fails
        """
        context_type = context_spec.context_type
        method_name = f'get_{context_type}_context'
        
        if not hasattr(self.context_manager, method_name):
            self.logger.warning(f"ContextManager does not have method: {method_name}")
            return None
        
        context_method = getattr(self.context_manager, method_name)
        
        # Call appropriate context method based on type
        if context_type in ['plan', 'design', 'tasks']:
            user_request = task_input.get('user_request')
            if not user_request:
                self.logger.warning(f"{context_type} context requires user_request")
                return None
            return await context_method(user_request)
        elif context_type == 'implementation':
            task = task_input.get('task')
            if not task:
                self.logger.warning("implementation context requires task")
                return None
            return await context_method(task)
        else:
            self.logger.warning(f"Unknown context type: {context_type}")
            return None
    
    def _format_context_for_agent(self, context: Any, context_type: str) -> Dict[str, Any]:
        """
        Format context object for agent consumption.
        
        Args:
            context: Context object from ContextManager
            context_type: Type of context (plan, design, tasks, implementation)
            
        Returns:
            Dictionary with formatted context data
        """
        formatted = {f"{context_type}_context": context}
        
        # Add commonly used context attributes
        if hasattr(context, 'project_structure') and context.project_structure:
            formatted['project_structure'] = context.project_structure
        if hasattr(context, 'memory_patterns') and context.memory_patterns:
            formatted['memory_patterns'] = context.memory_patterns
        if hasattr(context, 'compressed'):
            formatted['compressed'] = context.compressed
        
        # Add context-specific attributes
        if context_type == 'design':
            if hasattr(context, 'requirements') and context.requirements:
                formatted['requirements'] = context.requirements
        elif context_type == 'tasks':
            if hasattr(context, 'requirements') and context.requirements:
                formatted['requirements'] = context.requirements
            if hasattr(context, 'design') and context.design:
                formatted['design'] = context.design
        elif context_type == 'implementation':
            if hasattr(context, 'task') and context.task:
                formatted['task'] = context.task
            if hasattr(context, 'requirements') and context.requirements:
                formatted['requirements'] = context.requirements
            if hasattr(context, 'design') and context.design:
                formatted['design'] = context.design
            if hasattr(context, 'tasks') and context.tasks:
                formatted['tasks'] = context.tasks
            if hasattr(context, 'execution_history') and context.execution_history:
                formatted['execution_history'] = context.execution_history
            if hasattr(context, 'related_tasks') and context.related_tasks:
                formatted['related_tasks'] = context.related_tasks
        
        return formatted
    
    def get_context_requirements(self, task_input: Dict[str, Any]) -> Optional['ContextSpec']:
        """
        Define what context this agent requires.
        
        Args:
            task_input: Task input parameters
            
        Returns:
            ContextSpec defining required context, or None if no context needed
        """
        # Default implementation - agents can override this
        return None
    
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
                             target_reduction: float = 0.5) -> None:
        """
        Removed API: compression is handled by ContextManager.
        """
        raise RuntimeError(
            "compress_context is removed from BaseLLMAgent. Compression is handled by ContextManager."
        )
    
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
            if isinstance(context, dict):
                return dict_context_to_string(context)
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
    
    def truncate_context(self, context: Optional[Dict[str, Any]] = None, max_tokens: int = 4000) -> None:
        raise RuntimeError(
            "truncate_context is removed from BaseLLMAgent. Use ContextManager for token-aware preparation."
        )
    
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
    
    def _update_compression_stats(self, result: Any) -> None:
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
    
    async def compress_and_replace_context(self, target_reduction: float = 0.5) -> Dict[str, Any]:
        raise RuntimeError(
            "compress_and_replace_context is removed. Use ContextManager to manage compressed prompts/contexts."
        )