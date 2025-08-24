"""
Dependency Container for the AutoGen Multi-Agent Framework.

This module provides a centralized dependency injection system that eliminates
the need to manually pass multiple managers to each agent. The DependencyContainer
manages all framework managers and provides them to agents on demand with
thread-safe lazy loading.
"""

import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from .models import LLMConfig


class DependencyContainer:
    """
    Central dependency injection container for all framework managers.
    
    This class manages the creation and lifecycle of all framework managers,
    providing them to agents through a clean dependency injection pattern.
    It supports both production and test configurations with simple
    lazy loading to avoid circular dependencies.
    """
    
    def __init__(self):
        """Initialize the dependency container."""
        self._managers: Dict[str, Any] = {}
        self._initialized = False
        self._work_dir: Optional[str] = None
        self._llm_config: Optional[LLMConfig] = None
        self._is_test_mode = False
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def create_production(cls, work_dir: str, llm_config: LLMConfig) -> 'DependencyContainer':
        """
        Create container with real managers for production use.
        
        Args:
            work_dir: Working directory for the framework
            llm_config: LLM configuration for agents
            
        Returns:
            DependencyContainer configured for production
        """
        container = cls()
        container._setup_production_managers(work_dir, llm_config)
        return container
    
    @classmethod
    def create_test(cls, work_dir: str, llm_config: LLMConfig) -> 'DependencyContainer':
        """
        Create container with mock managers for testing.
        
        Args:
            work_dir: Working directory for tests
            llm_config: LLM configuration for tests
            
        Returns:
            DependencyContainer configured for testing
        """
        container = cls()
        container._setup_mock_managers(work_dir, llm_config)
        return container
    
    def get_token_manager(self):
        """Get or create TokenManager instance."""
        return self._get_or_create('token_manager', self._create_token_manager)
    
    def get_context_manager(self):
        """Get or create ContextManager instance."""
        return self._get_or_create('context_manager', self._create_context_manager)
    
    def get_memory_manager(self):
        """Get or create MemoryManager instance."""
        return self._get_or_create('memory_manager', self._create_memory_manager)
    
    def get_shell_executor(self):
        """Get or create ShellExecutor instance."""
        return self._get_or_create('shell_executor', self._create_shell_executor)
    
    def get_error_recovery(self):
        """Get or create ErrorRecovery instance."""
        return self._get_or_create('error_recovery', self._create_error_recovery)
    
    def get_task_decomposer(self):
        """Get or create TaskDecomposer instance."""
        return self._get_or_create('task_decomposer', self._create_task_decomposer)
    
    def get_config_manager(self):
        """Get or create ConfigManager instance."""
        return self._get_or_create('config_manager', self._create_config_manager)
    
    def get_context_compressor(self):
        """Get or create ContextCompressor instance."""
        return self._get_or_create('context_compressor', self._create_context_compressor)
    
    def _get_or_create(self, key: str, factory: Callable) -> Any:
        """
        Simple lazy loading of managers.
        
        Args:
            key: Manager key in the container
            factory: Factory function to create the manager
            
        Returns:
            Manager instance
        """
        if key not in self._managers:
            try:
                self._managers[key] = factory()
                self.logger.debug(f"Created {key} instance")
            except Exception as e:
                self.logger.error(f"Failed to create {key}: {e}")
                raise RuntimeError(f"Failed to create {key}: {e}")
        return self._managers[key]
    
    def _setup_production_managers(self, work_dir: str, llm_config: LLMConfig):
        """
        Setup factory methods for production managers.
        
        Args:
            work_dir: Working directory for the framework
            llm_config: LLM configuration for agents
        """
        self.work_dir = work_dir
        self.llm_config = llm_config
        self._is_test_mode = False
        
        # Ensure work directory exists
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Configured production container for work_dir: {work_dir}")
    
    def _setup_mock_managers(self, work_dir: str, llm_config: LLMConfig):
        """
        Setup factory methods for mock managers.
        
        Args:
            work_dir: Working directory for tests
            llm_config: LLM configuration for tests
        """
        self.work_dir = work_dir
        self.llm_config = llm_config
        self._is_test_mode = True
        
        # Ensure work directory exists
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Configured test container for work_dir: {work_dir}")
    
    def _create_config_manager(self):
        """Create ConfigManager instance."""
        if self._is_test_mode:
            from unittest.mock import Mock
            mock_config_manager = Mock()
            mock_config_manager.get_model_info.return_value = {
                "family": "GPT_4",
                "token_limit": 8192,
                "capabilities": {
                    "vision": False,
                    "function_calling": True,
                    "streaming": False
                }
            }
            return mock_config_manager
        else:
            from .config_manager import ConfigManager
            return ConfigManager()
    
    def _create_token_manager(self):
        """Create TokenManager instance."""
        if self._is_test_mode:
            from unittest.mock import Mock
            mock_token_manager = Mock()
            mock_token_manager.get_model_limit.return_value = 8192
            mock_token_manager.get_model_context_size.return_value = 6553
            mock_token_manager.estimate_tokens_from_text.return_value = 100
            mock_token_manager.extract_token_usage_from_response.return_value = 50
            mock_token_manager.update_token_usage.return_value = None
            mock_token_manager.check_token_limit.return_value = Mock(
                current_tokens=100,
                model_limit=8192,
                percentage_used=0.01,
                needs_compression=False
            )
            return mock_token_manager
        else:
            from .token_manager import TokenManager
            config_manager = self.get_config_manager()
            return TokenManager(config_manager)
    
    def _create_memory_manager(self):
        """Create MemoryManager instance."""
        if self._is_test_mode:
            from unittest.mock import Mock
            mock_memory_manager = Mock()
            mock_memory_manager.workspace_path = self.work_dir  # Set proper workspace_path
            mock_memory_manager.load_memory.return_value = {}
            mock_memory_manager.search_memory.return_value = []
            mock_memory_manager.get_system_instructions.return_value = "Test memory instructions"
            return mock_memory_manager
        else:
            from .memory_manager import MemoryManager
            return MemoryManager(self.work_dir)
    
    def _create_context_compressor(self):
        """Create ContextCompressor instance."""
        if self._is_test_mode:
            from unittest.mock import Mock
            mock_compressor = Mock()
            mock_compressor.compress_context.return_value = Mock(
                success=True,
                compressed_content="Compressed content",
                compression_ratio=0.5,
                error=None
            )
            return mock_compressor
        else:
            from .context_compressor import ContextCompressor
            token_manager = self.get_token_manager()
            config_manager = self.get_config_manager()
            return ContextCompressor(self.llm_config, token_manager, config_manager)
    
    def _create_context_manager(self):
        """Create ContextManager instance."""
        if self._is_test_mode:
            from unittest.mock import Mock, AsyncMock
            mock_context_manager = Mock()
            mock_context_manager.initialize = AsyncMock(return_value=None)
            mock_context_manager.update_execution_history = AsyncMock(return_value=None)
            mock_context_manager.prepare_system_prompt.return_value = Mock(
                system_prompt="Test system prompt",
                estimated_tokens=100
            )
            mock_context_manager.get_plan_context.return_value = Mock(
                user_request="test request",
                project_structure=None,
                memory_patterns=[],
                compressed=False
            )
            mock_context_manager.get_design_context.return_value = Mock(
                user_request="test request",
                requirements=None,
                project_structure=None,
                memory_patterns=[],
                compressed=False
            )
            mock_context_manager.get_tasks_context.return_value = Mock(
                user_request="test request",
                requirements=None,
                design=None,
                memory_patterns=[],
                compressed=False
            )
            mock_context_manager.get_implementation_context.return_value = Mock(
                task=Mock(id="test_task", title="Test Task"),
                requirements=None,
                design=None,
                tasks=None,
                project_structure=None,
                execution_history=[],
                related_tasks=[],
                memory_patterns=[],
                compressed=False
            )
            return mock_context_manager
        else:
            from .context_manager import ContextManager
            memory_manager = self.get_memory_manager()
            context_compressor = self.get_context_compressor()
            token_manager = self.get_token_manager()
            config_manager = self.get_config_manager()
            return ContextManager(
                self.work_dir, 
                memory_manager, 
                context_compressor, 
                self.llm_config,
                token_manager,
                config_manager
            )
    
    def _create_shell_executor(self):
        """Create ShellExecutor instance."""
        if self._is_test_mode:
            from unittest.mock import Mock, AsyncMock
            mock_shell_executor = Mock()
            mock_shell_executor.execute_command = AsyncMock(return_value=Mock(
                success=True,
                command="test command",
                return_code=0,
                stdout="test output",
                stderr="",
                execution_time=1.0
            ))
            return mock_shell_executor
        else:
            from .shell_executor import ShellExecutor
            return ShellExecutor(self.work_dir)
    
    def _create_error_recovery(self):
        """Create ErrorRecovery instance."""
        if self._is_test_mode:
            from unittest.mock import Mock, AsyncMock
            mock_error_recovery = Mock()
            mock_strategy = Mock()
            mock_strategy.name = "test_strategy"
            mock_error_recovery.recover = AsyncMock(return_value=Mock(
                success=True,
                strategy_used=mock_strategy,
                error=None,
                attempted_strategies=[],
                execution_results=[],
                recovery_time=1.0
            ))
            return mock_error_recovery
        else:
            from .agents.error_recovery import ErrorRecovery
            return ErrorRecovery(
                name="ErrorRecovery",
                llm_config=self.llm_config,
                system_message="Error recovery agent for intelligent failure analysis and recovery",
                container=self
            )
    
    def _create_task_decomposer(self):
        """Create TaskDecomposer instance."""
        if self._is_test_mode:
            from unittest.mock import Mock, AsyncMock
            mock_task_decomposer = Mock()
            mock_task_decomposer.decompose_task = AsyncMock(return_value=Mock(
                task=Mock(id="test_task"),
                complexity_analysis=Mock(complexity_level="simple"),
                commands=[],
                decision_points=[],
                success_criteria=[],
                fallback_strategies=[],
                estimated_duration=5
            ))
            return mock_task_decomposer
        else:
            from .agents.task_decomposer import TaskDecomposer
            return TaskDecomposer(
                name="TaskDecomposer",
                llm_config=self.llm_config,
                system_message="Task decomposition agent for intelligent task breakdown",
                container=self
            )
    
    def is_initialized(self) -> bool:
        """Check if the container has been initialized."""
        return self._initialized
    
    def get_manager_count(self) -> int:
        """Get the number of managers currently created."""
        return len(self._managers)
    
    def get_created_managers(self) -> Dict[str, str]:
        """Get a dictionary of created managers and their types."""
        return {key: type(manager).__name__ for key, manager in self._managers.items()}
    
    def clear_managers(self) -> None:
        """Clear all created managers (useful for testing)."""
        self._managers.clear()
        self.logger.debug("Cleared all managers from container")
    
    def __repr__(self) -> str:
        """String representation of the container."""
        mode = "test" if self._is_test_mode else "production"
        return f"DependencyContainer(mode={mode}, managers={len(self._managers)}, work_dir={self.work_dir})"