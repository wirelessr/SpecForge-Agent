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

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..models import LLMConfig, AgentContext, SystemInstructions


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
        description: Optional[str] = None
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
        Generate a response using AutoGen agent.
        
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
            # Add to conversation history
            self.add_to_conversation_history("user", prompt)
            
            # Generate response using AutoGen
            response = await self._generate_autogen_response(prompt)
            
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