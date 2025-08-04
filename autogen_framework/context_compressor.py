"""
Context Compressor for the AutoGen Multi-Agent Framework.

This module provides a specialized agent for context compression operations.
The core compression functionality is now integrated into BaseLLMAgent,
making it available to all agents in the framework.

This ContextCompressor class serves as a dedicated compression agent
for external compression requests and batch operations.
"""

import logging
from typing import Dict, Any, Optional, List

from .agents.base_agent import BaseLLMAgent
from .models import LLMConfig, CompressionResult, AgentContext


class ContextCompressor(BaseLLMAgent):
    """
    Specialized agent for context compression operations.
    
    This class extends BaseLLMAgent and serves as a dedicated compression agent
    for external compression requests and batch operations. The core compression
    functionality is inherited from BaseLLMAgent, making it available to all agents.
    """
    
    def __init__(self, llm_config: LLMConfig, token_manager=None):
        """
        Initialize the ContextCompressor.
        
        Args:
            llm_config: LLM configuration for compression operations.
            token_manager: Optional TokenManager instance for token tracking.
        """
        # Initialize base agent with compression-specific system message
        system_message = self._build_compression_system_message()
        super().__init__(
            name="context_compressor",
            llm_config=llm_config,
            system_message=system_message,
            description="Specialized agent for intelligent context compression"
        )
        
        self.token_manager = token_manager
        self.logger.info("ContextCompressor initialized")
    
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
    
    async def process_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a compression task assigned to this agent.
        
        Args:
            task_input: Dictionary containing compression parameters and context.
                       Expected keys: 'context', 'target_reduction' (optional)
            
        Returns:
            Dictionary containing compression results and metadata.
        """
        context = task_input.get('context', {})
        target_reduction = task_input.get('target_reduction', 0.5)
        
        # Use inherited compression functionality
        result = await self.compress_context(context, target_reduction)
        
        # Update token manager if available
        if self.token_manager and result.success:
            self.token_manager.increment_compression_count()
        
        return {
            'compression_result': result,
            'success': result.success,
            'original_size': result.original_size,
            'compressed_size': result.compressed_size,
            'compression_ratio': result.compression_ratio,
            'method_used': result.method_used,
            'error': result.error
        }
    
    def get_agent_capabilities(self) -> List[str]:
        """
        Get a list of capabilities that this agent provides.
        
        Returns:
            List of capability descriptions.
        """
        return [
            "Intelligent context compression using LLM",
            "Preservation of critical workflow information",
            "Fallback truncation strategies",
            "Compression statistics and reporting",
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
    
    async def compress_agent_contexts(self, agents: List[BaseLLMAgent], 
                                    target_reduction: float = 0.5) -> Dict[str, CompressionResult]:
        """
        Compress contexts from multiple agents.
        
        Args:
            agents: List of BaseLLMAgent instances to compress contexts from.
            target_reduction: Target compression ratio for each agent's context.
            
        Returns:
            Dictionary mapping agent names to their compression results.
        """
        results = {}
        
        for agent in agents:
            self.logger.info(f"Compressing context for agent: {agent.name}")
            
            # Get agent's full context
            agent_context = agent._get_full_context()
            
            # Compress the context
            result = await self.compress_context(agent_context, target_reduction)
            results[agent.name] = result
            
            # Update token manager if available
            if self.token_manager and result.success:
                self.token_manager.increment_compression_count()
        
        return results