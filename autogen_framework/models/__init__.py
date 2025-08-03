"""
Core data models for the AutoGen multi-agent framework.

This module defines all the data structures used throughout the framework,
including workflow states, task definitions, execution results, and configuration models.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


class WorkflowPhase(Enum):
    """Enumeration of workflow phases in the multi-agent framework."""
    PLANNING = "planning"
    DESIGN = "design"
    TASK_GENERATION = "task_generation"
    IMPLEMENTATION = "implementation"
    COMPLETED = "completed"


@dataclass
class WorkflowState:
    """
    Represents the current state of a workflow execution.
    
    This class tracks the progress through different phases of the framework,
    from initial planning through to completion.
    """
    phase: WorkflowPhase
    work_directory: str
    requirements_approved: bool = False
    design_approved: bool = False
    tasks_approved: bool = False
    current_task_index: int = 0
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def advance_phase(self) -> None:
        """Advance to the next workflow phase."""
        phase_order = [
            WorkflowPhase.PLANNING,
            WorkflowPhase.DESIGN,
            WorkflowPhase.TASK_GENERATION,
            WorkflowPhase.IMPLEMENTATION,
            WorkflowPhase.COMPLETED
        ]
        
        current_index = phase_order.index(self.phase)
        if current_index < len(phase_order) - 1:
            self.phase = phase_order[current_index + 1]
    
    def is_ready_for_next_phase(self) -> bool:
        """Check if current phase requirements are met to advance."""
        if self.phase == WorkflowPhase.PLANNING:
            return self.requirements_approved
        elif self.phase == WorkflowPhase.DESIGN:
            return self.design_approved
        elif self.phase == WorkflowPhase.TASK_GENERATION:
            return self.tasks_approved
        return False


@dataclass
class TaskDefinition:
    """
    Defines a specific task to be executed by the implement agent.
    
    Includes retry mechanism and execution tracking capabilities.
    """
    id: str
    title: str
    description: str
    steps: List[str]
    requirements_ref: List[str]
    dependencies: List[str] = field(default_factory=list)
    completed: bool = False
    execution_record: Optional[Dict[str, Any]] = None
    attempted_approaches: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    def can_retry(self) -> bool:
        """Check if task can be retried based on retry count."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment the retry counter."""
        self.retry_count += 1
    
    def add_attempt(self, approach: str, result: Dict[str, Any]) -> None:
        """Record an execution attempt."""
        attempt = {
            "approach": approach,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "retry_number": self.retry_count
        }
        self.attempted_approaches.append(attempt)
    
    def mark_completed(self, execution_record: Dict[str, Any]) -> None:
        """Mark task as completed with execution details."""
        self.completed = True
        self.execution_record = execution_record


@dataclass
class ExecutionResult:
    """
    Records the result of a shell command execution.
    
    Contains comprehensive information about command execution including
    timing, output, and approach used.
    """
    command: str
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    working_directory: str
    timestamp: str
    success: bool
    approach_used: str
    alternative_attempts: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def create_success(cls, command: str, stdout: str, execution_time: float, 
                      working_directory: str, approach_used: str) -> 'ExecutionResult':
        """Create a successful execution result."""
        return cls(
            command=command,
            return_code=0,
            stdout=stdout,
            stderr="",
            execution_time=execution_time,
            working_directory=working_directory,
            timestamp=datetime.now().isoformat(),
            success=True,
            approach_used=approach_used
        )
    
    @classmethod
    def create_failure(cls, command: str, return_code: int, stderr: str, 
                      execution_time: float, working_directory: str, 
                      approach_used: str) -> 'ExecutionResult':
        """Create a failed execution result."""
        return cls(
            command=command,
            return_code=return_code,
            stdout="",
            stderr=stderr,
            execution_time=execution_time,
            working_directory=working_directory,
            timestamp=datetime.now().isoformat(),
            success=False,
            approach_used=approach_used
        )
    
    def add_alternative_attempt(self, attempt: Dict[str, Any]) -> None:
        """Add information about an alternative approach that was tried."""
        self.alternative_attempts.append(attempt)


@dataclass
class LLMConfig:
    """
    Configuration for LLM API connection and parameters.
    
    Contains all necessary settings for connecting to the custom LLM endpoint
    and configuring model behavior. Uses Gemini-compatible parameter names.
    
    All parameters are required and must be provided explicitly - no hardcoded defaults.
    Use ConfigManager to create instances with environment-based configuration.
    """
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_output_tokens: int = 8192
    timeout: int = 60
    
    def to_autogen_config(self) -> Dict[str, Any]:
        """Convert to AutoGen-compatible configuration dictionary."""
        return {
            "config_list": [{
                "model": self.model,
                "base_url": self.base_url,
                "api_key": self.api_key,
            }],
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout": self.timeout,
        }
    
    @classmethod
    def from_config_manager(cls, config_manager) -> 'LLMConfig':
        """
        Create LLMConfig instance from ConfigManager.
        
        Args:
            config_manager: ConfigManager instance to get configuration from.
            
        Returns:
            LLMConfig instance with environment-based configuration.
            
        Raises:
            ConfigurationError: If required configuration is missing or invalid.
        """
        config_dict = config_manager.get_llm_config()
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid, False otherwise.
        """
        # Validate required string parameters
        if not self.base_url or not self.model or not self.api_key:
            return False
        
        # Validate base_url format
        if not (self.base_url.startswith('http://') or self.base_url.startswith('https://')):
            return False
        
        # Validate temperature range
        if self.temperature < 0 or self.temperature > 2:
            return False
        
        # Validate positive integers
        if self.max_output_tokens <= 0 or self.timeout <= 0:
            return False
        
        return True


# Type aliases for better code readability
AgentContext = Dict[str, Any]
MemoryContent = Dict[str, str]
SystemInstructions = str