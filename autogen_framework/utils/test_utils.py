"""
Test utilities for the AutoGen Multi-Agent Framework.

This module provides helper classes and utilities for testing,
particularly for managing dependencies and test setup.
"""

from dataclasses import dataclass
from typing import Optional

from ..config_manager import ConfigManager
from ..token_manager import TokenManager
from ..context_manager import ContextManager
from ..context_compressor import ContextCompressor
from ..memory_manager import MemoryManager
from ..models import LLMConfig


@dataclass
class ManagerDependencies:
    """
    Container for manager dependencies used in testing.
    
    This class encapsulates the creation and management of the core
    framework managers (ConfigManager, TokenManager, ContextManager)
    with proper dependency injection order.
    """
    config_manager: ConfigManager
    token_manager: TokenManager
    context_manager: ContextManager
    
    @classmethod
    def create_for_workspace(
        cls, 
        workspace_path: str, 
        memory_manager: MemoryManager, 
        llm_config: LLMConfig,
        load_env: bool = False
    ) -> 'ManagerDependencies':
        """
        Factory method to create managers with proper dependencies.
        
        Args:
            workspace_path: Path to the workspace directory
            memory_manager: MemoryManager instance to use
            llm_config: LLM configuration for the managers
            load_env: Whether to load environment variables in ConfigManager
            
        Returns:
            ManagerDependencies instance with all managers properly configured
        """
        # Create managers in correct dependency order
        config_manager = ConfigManager(load_env=load_env)
        token_manager = TokenManager(config_manager)
        context_compressor = ContextCompressor(llm_config, token_manager=token_manager)
        
        context_manager = ContextManager(
            work_dir=workspace_path,
            memory_manager=memory_manager,
            context_compressor=context_compressor,
            llm_config=llm_config,
            token_manager=token_manager,
            config_manager=config_manager,
        )
        
        return cls(
            config_manager=config_manager,
            token_manager=token_manager,
            context_manager=context_manager
        )
    
    @classmethod
    async def create_and_initialize_for_workspace(
        cls,
        workspace_path: str,
        memory_manager: MemoryManager,
        llm_config: LLMConfig,
        load_env: bool = False
    ) -> 'ManagerDependencies':
        """
        Factory method to create and initialize managers with proper dependencies.
        
        This method creates the managers and also initializes the ContextManager,
        which is required for some operations.
        
        Args:
            workspace_path: Path to the workspace directory
            memory_manager: MemoryManager instance to use
            llm_config: LLM configuration for the managers
            load_env: Whether to load environment variables in ConfigManager
            
        Returns:
            ManagerDependencies instance with all managers properly configured and initialized
        """
        managers = cls.create_for_workspace(
            workspace_path=workspace_path,
            memory_manager=memory_manager,
            llm_config=llm_config,
            load_env=load_env
        )
        
        # Initialize the context manager
        await managers.context_manager.initialize()
        
        return managers