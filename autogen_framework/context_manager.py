"""
Context Manager for the AutoGen Multi-Agent Framework.

This module provides comprehensive project context management with agent-specific
interfaces. It loads and manages project requirements, design documents, tasks,
execution history, and project structure, providing tailored context views for
different agent types.

The ContextManager integrates with MemoryManager for historical patterns and
ContextCompressor for automatic token threshold management.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from .memory_manager import MemoryManager
from .context_compressor import ContextCompressor
from .config_manager import ConfigManager
from .models import LLMConfig, TaskDefinition, ExecutionResult
from .context_utils import agent_context_to_string

# Forward declaration for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .token_manager import TokenManager


@dataclass
class ProjectStructure:
    """Represents the analyzed structure of a project."""
    root_path: str
    directories: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    key_files: Dict[str, str] = field(default_factory=dict)  # filename -> content
    file_types: Dict[str, int] = field(default_factory=dict)  # extension -> count
    total_files: int = 0
    total_directories: int = 0
    analysis_timestamp: str = ""
    
    def __post_init__(self):
        """Set analysis timestamp after initialization."""
        if not self.analysis_timestamp:
            self.analysis_timestamp = datetime.now().isoformat()


@dataclass
class RequirementsDocument:
    """Represents a parsed requirements.md document."""
    content: str
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    introduction: str = ""
    file_path: str = ""
    last_modified: str = ""
    
    def __post_init__(self):
        """Parse requirements from content after initialization."""
        if self.content and not self.requirements:
            self._parse_requirements()
    
    def _parse_requirements(self):
        """Parse structured requirements from markdown content."""
        # Simple parsing - can be enhanced later
        lines = self.content.split('\n')
        current_req = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('### Requirement'):
                if current_req:
                    self.requirements.append(current_req)
                current_req = {
                    'title': line.replace('### ', ''),
                    'user_story': '',
                    'acceptance_criteria': []
                }
            elif line.startswith('**User Story:**') and current_req:
                current_req['user_story'] = line.replace('**User Story:**', '').strip()
            elif line.startswith(('1. WHEN', '2. WHEN', '3. WHEN', '4. WHEN', '5. WHEN', '6. WHEN', '7. WHEN', '8. WHEN')) and current_req:
                current_req['acceptance_criteria'].append(line)
        
        if current_req:
            self.requirements.append(current_req)


@dataclass
class DesignDocument:
    """Represents a parsed design.md document."""
    content: str
    overview: str = ""
    architecture: str = ""
    components: List[str] = field(default_factory=list)
    file_path: str = ""
    last_modified: str = ""


@dataclass
class TasksDocument:
    """Represents a parsed tasks.md document."""
    content: str
    tasks: List[TaskDefinition] = field(default_factory=list)
    file_path: str = ""
    last_modified: str = ""
    
    def __post_init__(self):
        """Parse tasks from content after initialization."""
        if self.content and not self.tasks:
            self._parse_tasks()
    
    def _parse_tasks(self):
        """Parse task definitions from markdown content."""
        # Simple parsing - can be enhanced later
        lines = self.content.split('\n')
        current_task = None
        task_counter = 1
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('- [ ]') or line_stripped.startswith('- [x]'):
                if current_task:
                    self.tasks.append(current_task)
                
                completed = line_stripped.startswith('- [x]')
                title = line_stripped.replace('- [ ]', '').replace('- [x]', '').strip()
                
                current_task = TaskDefinition(
                    id=f"task_{task_counter}",
                    title=title,
                    description=title,
                    steps=[],
                    requirements_ref=[],
                    completed=completed
                )
                task_counter += 1
            elif line_stripped.startswith('- ') and current_task and not line_stripped.startswith('- [ ]') and not line_stripped.startswith('- [x]'):
                # This is a sub-item or detail
                current_task.steps.append(line_stripped.replace('- ', ''))
        
        if current_task:
            self.tasks.append(current_task)


@dataclass
class MemoryPattern:
    """Represents a memory pattern from MemoryManager."""
    category: str
    content: str
    relevance_score: float = 0.0
    source: str = ""


@dataclass
class PlanContext:
    """Context for PlanAgent."""
    user_request: str
    project_structure: Optional[ProjectStructure] = None
    memory_patterns: List[MemoryPattern] = field(default_factory=list)
    compressed: bool = False
    compression_ratio: Optional[float] = None


@dataclass
class DesignContext:
    """Context for DesignAgent."""
    user_request: str
    requirements: Optional[RequirementsDocument] = None
    project_structure: Optional[ProjectStructure] = None
    memory_patterns: List[MemoryPattern] = field(default_factory=list)
    compressed: bool = False
    compression_ratio: Optional[float] = None


@dataclass
class TasksContext:
    """Context for TasksAgent."""
    user_request: str
    requirements: Optional[RequirementsDocument] = None
    design: Optional[DesignDocument] = None
    memory_patterns: List[MemoryPattern] = field(default_factory=list)
    compressed: bool = False
    compression_ratio: Optional[float] = None


@dataclass
class ImplementationContext:
    """Context for ImplementAgent - focused on single task execution."""
    task: TaskDefinition
    requirements: Optional[RequirementsDocument] = None
    design: Optional[DesignDocument] = None
    tasks: Optional[TasksDocument] = None
    project_structure: Optional[ProjectStructure] = None
    execution_history: List[ExecutionResult] = field(default_factory=list)
    related_tasks: List[TaskDefinition] = field(default_factory=list)
    memory_patterns: List[MemoryPattern] = field(default_factory=list)
    compressed: bool = False
    compression_ratio: Optional[float] = None


class ContextManager:
    """
    Manages comprehensive project context with agent-specific interfaces.
    
    This class loads and manages project requirements, design documents, tasks,
    execution history, and project structure. It provides tailored context views
    for different agent types and integrates with MemoryManager and ContextCompressor.
    """
    
    def __init__(self, work_dir: str, memory_manager: MemoryManager, 
                 context_compressor: ContextCompressor, llm_config: LLMConfig,
                 token_manager: 'TokenManager', config_manager: Optional[ConfigManager] = None):
        """
        Initialize the ContextManager.
        
        Args:
            work_dir: Working directory containing project files
            memory_manager: MemoryManager instance for historical patterns
            context_compressor: ContextCompressor instance for token management
            llm_config: LLM configuration containing model information
            token_manager: TokenManager instance for actual token tracking
            config_manager: ConfigManager instance for global configuration (optional)
        """
        self.work_dir = Path(work_dir)
        self.memory_manager = memory_manager
        self.context_compressor = context_compressor
        self.llm_config = llm_config
        self.token_manager = token_manager
        self.config_manager = config_manager or ConfigManager()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load token configuration from global config
        self.token_config = self.config_manager.get_token_config()
        
        # Get model-specific token limit from TokenManager
        self.model_token_limit = self.token_manager.get_model_limit(self.llm_config.model)
        self.compression_threshold = self.token_config.get('compression_threshold', 0.9)
        
        # Initialize context data
        self.requirements: Optional[RequirementsDocument] = None
        self.design: Optional[DesignDocument] = None
        self.tasks: Optional[TasksDocument] = None
        self.execution_history: List[ExecutionResult] = []
        self.project_structure: Optional[ProjectStructure] = None
        
        # Cache timestamps for invalidation
        self._last_requirements_load: Optional[datetime] = None
        self._last_design_load: Optional[datetime] = None
        self._last_tasks_load: Optional[datetime] = None
        self._last_structure_analysis: Optional[datetime] = None
        
        self.logger.info(f"ContextManager initialized for work_dir: {work_dir}")
        self.logger.info(f"Model: {self.llm_config.model}, Token limit: {self.model_token_limit}, Compression threshold: {self.compression_threshold:.1%}")
    
    async def initialize(self) -> None:
        """Load all available project context."""
        self.logger.info("Initializing ContextManager - loading all project context")
        
        try:
            # Load project documents
            await self._load_requirements()
            await self._load_design()
            await self._load_tasks()
            
            # Analyze project structure
            await self._analyze_project_structure()
            
            # Load execution history
            await self._load_execution_history()
            
            self.logger.info("ContextManager initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during ContextManager initialization: {e}")
            raise

    @dataclass
    class PreparedPrompt:
        system_prompt: str
        estimated_tokens: int

    async def prepare_system_prompt(self, text: str) -> 'ContextManager.PreparedPrompt':
        """
        Prepare a system prompt text by checking token thresholds and compressing if needed.
        Returns compressed text when over threshold.
        """
        # Estimate tokens using centralized heuristic
        try:
            estimated = self.token_manager.estimate_tokens_from_text(text, base_overhead=100)
        except Exception:
            estimated = max(len(text) // 4 + 100, 1)

        check = self.token_manager.check_token_limit(self.llm_config.model, estimated_static_tokens=estimated)

        if not check.needs_compression:
            return ContextManager.PreparedPrompt(system_prompt=text, estimated_tokens=estimated)

        # Compute target reduction to go under threshold
        target_tokens = int(check.model_limit * self.compression_threshold)
        if check.current_tokens == 0:
            reduction = 0.2
        else:
            reduction = 1 - (target_tokens / check.current_tokens)
        reduction = max(0.1, min(0.8, reduction))

        # Attempt compression if compressor is available
        if self.context_compressor:
            try:
                payload = {"content": text}
                result = await self.context_compressor.compress_context(payload, target_reduction=reduction)
                if result.success:
                    self.token_manager.increment_compression_count()
                    return ContextManager.PreparedPrompt(
                        system_prompt=result.compressed_content,
                        estimated_tokens=target_tokens,
                    )
                else:
                    self.logger.warning(f"Compression failed: {result.error}")
            except Exception as e:
                self.logger.error(f"Compression exception: {e}")

        # No compressor or failed compression: return original
        return ContextManager.PreparedPrompt(system_prompt=text, estimated_tokens=estimated)
    
    async def _load_requirements(self) -> None:
        """Load and parse requirements.md document."""
        requirements_path = self.work_dir / "requirements.md"
        
        if requirements_path.exists():
            try:
                content = requirements_path.read_text(encoding='utf-8')
                stat = requirements_path.stat()
                
                self.requirements = RequirementsDocument(
                    content=content,
                    file_path=str(requirements_path),
                    last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
                )
                self._last_requirements_load = datetime.now()
                
                self.logger.info(f"Loaded requirements.md with {len(self.requirements.requirements)} requirements")
                
            except Exception as e:
                self.logger.error(f"Error loading requirements.md: {e}")
                self.requirements = None
        else:
            self.logger.info("requirements.md not found")
            self.requirements = None
    
    async def _load_design(self) -> None:
        """Load and parse design.md document."""
        design_path = self.work_dir / "design.md"
        
        if design_path.exists():
            try:
                content = design_path.read_text(encoding='utf-8')
                stat = design_path.stat()
                
                self.design = DesignDocument(
                    content=content,
                    file_path=str(design_path),
                    last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
                )
                self._last_design_load = datetime.now()
                
                self.logger.info("Loaded design.md successfully")
                
            except Exception as e:
                self.logger.error(f"Error loading design.md: {e}")
                self.design = None
        else:
            self.logger.info("design.md not found")
            self.design = None
    
    async def _load_tasks(self) -> None:
        """Load and parse tasks.md document."""
        tasks_path = self.work_dir / "tasks.md"
        
        if tasks_path.exists():
            try:
                content = tasks_path.read_text(encoding='utf-8')
                stat = tasks_path.stat()
                
                self.tasks = TasksDocument(
                    content=content,
                    file_path=str(tasks_path),
                    last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
                )
                self._last_tasks_load = datetime.now()
                
                self.logger.info(f"Loaded tasks.md with {len(self.tasks.tasks)} tasks")
                
            except Exception as e:
                self.logger.error(f"Error loading tasks.md: {e}")
                self.tasks = None
        else:
            self.logger.info("tasks.md not found")
            self.tasks = None
    
    async def _analyze_project_structure(self) -> None:
        """Analyze and cache project structure."""
        try:
            directories = []
            files = []
            key_files = {}
            file_types = {}
            
            # Walk through project directory
            for root, dirs, filenames in os.walk(self.work_dir):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
                
                rel_root = os.path.relpath(root, self.work_dir)
                if rel_root != '.':
                    directories.append(rel_root)
                
                for filename in filenames:
                    if not filename.startswith('.'):
                        rel_path = os.path.join(rel_root, filename) if rel_root != '.' else filename
                        files.append(rel_path)
                        
                        # Count file types
                        ext = os.path.splitext(filename)[1].lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
                        
                        # Store key files content
                        if filename in ['README.md', 'package.json', 'pyproject.toml', 'requirements.txt']:
                            try:
                                full_path = os.path.join(root, filename)
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    key_files[filename] = f.read()
                            except Exception as e:
                                self.logger.warning(f"Could not read key file {filename}: {e}")
            
            self.project_structure = ProjectStructure(
                root_path=str(self.work_dir),
                directories=directories,
                files=files,
                key_files=key_files,
                file_types=file_types,
                total_files=len(files),
                total_directories=len(directories)
            )
            
            self._last_structure_analysis = datetime.now()
            self.logger.info(f"Analyzed project structure: {len(files)} files, {len(directories)} directories")
            
        except Exception as e:
            self.logger.error(f"Error analyzing project structure: {e}")
            self.project_structure = None
    
    async def _load_execution_history(self) -> None:
        """Load execution history from previous task executions."""
        # For now, initialize empty - this will be populated as tasks are executed
        self.execution_history = []
        self.logger.info("Initialized empty execution history")
    
    async def get_plan_context(self, user_request: str) -> PlanContext:
        """
        Provides context for PlanAgent.
        
        Args:
            user_request: User's request/prompt
            
        Returns:
            PlanContext with user request, project structure, and memory patterns
            Context is automatically compressed if it exceeds token thresholds
        """
        self.logger.info("Building PlanContext")
        
        # Ensure project structure is loaded
        if not self.project_structure or self._should_refresh_structure():
            await self._analyze_project_structure()
        
        # Get memory patterns from MemoryManager
        memory_patterns = await self._get_plan_memory_patterns(user_request)
        
        # Build context
        context = PlanContext(
            user_request=user_request,
            project_structure=self.project_structure,
            memory_patterns=memory_patterns
        )
        
        # Check if compression is needed
        context = await self._compress_if_needed(context, "plan")
        
        self.logger.info(f"Built PlanContext with {len(memory_patterns)} memory patterns")
        return context
    
    async def get_design_context(self, user_request: str) -> DesignContext:
        """
        Provides context for DesignAgent.
        
        Args:
            user_request: User's request/prompt for design creation or revision
        
        Returns:
            DesignContext with user request, requirements, project structure, and memory patterns
        """
        self.logger.info("Building DesignContext")
        
        # Ensure requirements are loaded
        if not self.requirements or self._should_refresh_requirements():
            await self._load_requirements()
        
        # Ensure project structure is loaded
        if not self.project_structure or self._should_refresh_structure():
            await self._analyze_project_structure()
        
        # Get memory patterns from MemoryManager
        memory_patterns = await self._get_design_memory_patterns()
        
        # Build context
        context = DesignContext(
            user_request=user_request,
            requirements=self.requirements,
            project_structure=self.project_structure,
            memory_patterns=memory_patterns
        )
        
        # Check if compression is needed
        context = await self._compress_if_needed(context, "design")
        
        self.logger.info(f"Built DesignContext with requirements and {len(memory_patterns)} memory patterns")
        return context
    
    async def get_tasks_context(self, user_request: str) -> TasksContext:
        """
        Provides context for TasksAgent.
        
        Args:
            user_request: User's request/prompt for tasks creation or revision
        
        Returns:
            TasksContext with user request, requirements, design, and memory patterns
        """
        self.logger.info("Building TasksContext")
        
        # Ensure requirements are loaded
        if not self.requirements or self._should_refresh_requirements():
            await self._load_requirements()
        
        # Ensure design is loaded
        if not self.design or self._should_refresh_design():
            await self._load_design()
        
        # Get memory patterns from MemoryManager
        memory_patterns = await self._get_tasks_memory_patterns()
        
        # Build context
        context = TasksContext(
            user_request=user_request,
            requirements=self.requirements,
            design=self.design,
            memory_patterns=memory_patterns
        )
        
        # Check if compression is needed
        context = await self._compress_if_needed(context, "tasks")
        
        self.logger.info(f"Built TasksContext with requirements, design, and {len(memory_patterns)} memory patterns")
        return context
    
    async def get_implementation_context(self, task: TaskDefinition) -> ImplementationContext:
        """
        Provides comprehensive context for ImplementAgent.
        
        Args:
            task: Current task being implemented
            
        Returns:
            ImplementationContext with all project information, execution history, and memory patterns
            Context is automatically compressed if it exceeds token thresholds
        """
        self.logger.info(f"Building ImplementationContext for task: {task.id}")
        
        # Ensure all documents are loaded
        if not self.requirements or self._should_refresh_requirements():
            await self._load_requirements()
        
        if not self.design or self._should_refresh_design():
            await self._load_design()
        
        if not self.tasks or self._should_refresh_tasks():
            await self._load_tasks()
        
        if not self.project_structure or self._should_refresh_structure():
            await self._analyze_project_structure()
        
        # Get relevant execution history and related tasks
        relevant_history = self._get_relevant_history(task)
        related_tasks = self._find_related_tasks(task)
        
        # Get memory patterns from MemoryManager
        memory_patterns = await self._get_implementation_memory_patterns(task)
        
        # Build context
        context = ImplementationContext(
            task=task,
            requirements=self.requirements,
            design=self.design,
            tasks=self.tasks,
            project_structure=self.project_structure,
            execution_history=relevant_history,
            related_tasks=related_tasks,
            memory_patterns=memory_patterns
        )
        
        # Check if compression is needed
        context = await self._compress_if_needed(context, "implementation")
        
        self.logger.info(f"Built ImplementationContext with full project context and {len(memory_patterns)} memory patterns")
        return context
    
    async def update_execution_history(self, result: ExecutionResult) -> None:
        """Updates execution history with new results."""
        self.execution_history.append(result)
        
        # Persist to file for future sessions
        await self._persist_execution_history()
        
        self.logger.info(f"Updated execution history with result for command: {result.command}")
    
    async def refresh_project_structure(self) -> None:
        """Refreshes project structure analysis."""
        await self._analyze_project_structure()
        self.logger.info("Refreshed project structure analysis")
    
    def _get_relevant_history(self, task: TaskDefinition) -> List[ExecutionResult]:
        """Finds execution history relevant to current task."""
        # For now, return recent history - can be enhanced with relevance scoring
        return self.execution_history[-10:] if len(self.execution_history) > 10 else self.execution_history
    
    def _find_related_tasks(self, task: TaskDefinition) -> List[TaskDefinition]:
        """Finds tasks related to current task."""
        if not self.tasks:
            return []
        
        related = []
        
        # Find tasks with shared dependencies
        for other_task in self.tasks.tasks:
            if other_task.id != task.id:
                # Check for shared dependencies
                shared_deps = set(task.dependencies) & set(other_task.dependencies)
                if shared_deps:
                    related.append(other_task)
                
                # Check if current task is in other's dependencies
                if task.id in other_task.dependencies:
                    related.append(other_task)
                
                # Check if other task is in current's dependencies
                if other_task.id in task.dependencies:
                    related.append(other_task)
        
        return related
    
    async def _get_plan_memory_patterns(self, user_request: str) -> List[MemoryPattern]:
        """Get memory patterns relevant to planning."""
        try:
            # Search for planning-related patterns
            search_results = self.memory_manager.search_memory("planning requirements project")
            
            patterns = []
            for result in search_results[:5]:  # Limit to top 5 results
                pattern = MemoryPattern(
                    category=result.get("category", "unknown"),
                    content=result.get("content", ""),
                    relevance_score=result.get("relevance_score", 0.0),
                    source=result.get("file", "unknown")
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting plan memory patterns: {e}")
            return []
    
    async def _get_design_memory_patterns(self) -> List[MemoryPattern]:
        """Get memory patterns relevant to design."""
        try:
            # Search for design-related patterns
            search_results = self.memory_manager.search_memory("design architecture patterns")
            
            patterns = []
            for result in search_results[:5]:  # Limit to top 5 results
                pattern = MemoryPattern(
                    category=result.get("category", "unknown"),
                    content=result.get("content", ""),
                    relevance_score=result.get("relevance_score", 0.0),
                    source=result.get("file", "unknown")
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting design memory patterns: {e}")
            return []
    
    async def _get_tasks_memory_patterns(self) -> List[MemoryPattern]:
        """Get memory patterns relevant to task generation."""
        try:
            # Search for task-related patterns
            search_results = self.memory_manager.search_memory("tasks implementation breakdown")
            
            patterns = []
            for result in search_results[:5]:  # Limit to top 5 results
                pattern = MemoryPattern(
                    category=result.get("category", "unknown"),
                    content=result.get("content", ""),
                    relevance_score=result.get("relevance_score", 0.0),
                    source=result.get("file", "unknown")
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting tasks memory patterns: {e}")
            return []
    
    async def _get_implementation_memory_patterns(self, task: TaskDefinition) -> List[MemoryPattern]:
        """Get memory patterns relevant to implementation."""
        try:
            # Search for implementation-related patterns
            search_query = f"implementation {task.title} coding"
            search_results = self.memory_manager.search_memory(search_query)
            
            patterns = []
            for result in search_results[:5]:  # Limit to top 5 results
                pattern = MemoryPattern(
                    category=result.get("category", "unknown"),
                    content=result.get("content", ""),
                    relevance_score=result.get("relevance_score", 0.0),
                    source=result.get("file", "unknown")
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error getting implementation memory patterns: {e}")
            return []
    
    async def _compress_if_needed(self, context: Union[PlanContext, DesignContext, TasksContext, ImplementationContext], 
                                 context_type: str) -> Union[PlanContext, DesignContext, TasksContext, ImplementationContext]:
        """Compress context if it exceeds token thresholds using TokenManager."""
        try:
            # Estimate token count for static content using centralized heuristic
            context_str = self._context_to_string(context)
            try:
                estimated_tokens = self.token_manager.estimate_tokens_from_text(context_str, base_overhead=100)
            except Exception:
                # Conservative fallback
                estimated_tokens = max(len(context_str) // 4 + 100, 1)
            
            # Check if compression is needed using TokenManager with estimated static tokens
            # Don't update current_context_size here - let TokenManager handle the logic
            token_check = self.token_manager.check_token_limit(
                self.llm_config.model, 
                estimated_static_tokens=estimated_tokens
            )
            
            if token_check.needs_compression:
                self.logger.info(
                    f"Context size ({token_check.current_tokens} tokens) exceeds threshold "
                    f"({token_check.percentage_used:.1%} of {token_check.model_limit}), compressing..."
                )
                
                # Convert context to dictionary for compression
                context_dict = self._context_to_dict(context)
                
                # Calculate target reduction to get below threshold
                target_tokens = int(token_check.model_limit * self.compression_threshold)
                target_reduction = 1 - (target_tokens / token_check.current_tokens)
                target_reduction = max(0.1, min(0.8, target_reduction))  # Clamp between 10% and 80%
                
                self.logger.info(f"Target reduction: {target_reduction:.1%} (target: {target_tokens} tokens)")
                
                # Compress using ContextCompressor
                compression_result = await self.context_compressor.compress_context(context_dict, target_reduction)
                
                if compression_result.success:
                    # Update context with compressed content
                    compressed_context = self._dict_to_context(compression_result.compressed_content, context_type, context)
                    compressed_context.compressed = True
                    compressed_context.compression_ratio = compression_result.compression_ratio
                    
                    # Notify TokenManager about compression (let it handle its own state)
                    self.token_manager.increment_compression_count()
                    
                    self.logger.info(
                        f"Context compressed successfully. "
                        f"Ratio: {compression_result.compression_ratio:.2f}"
                    )
                    return compressed_context
                else:
                    self.logger.warning(f"Context compression failed: {compression_result.error}")
            else:
                self.logger.debug(
                    f"Context size ({token_check.current_tokens} tokens, {token_check.percentage_used:.1%}) "
                    f"within limits for model {self.llm_config.model}"
                )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error in context compression: {e}")
            return context
    
    def _context_to_string(self, context: Union[PlanContext, DesignContext, TasksContext, ImplementationContext]) -> str:
        """Convert context to string for token estimation (centralized utility)."""
        return agent_context_to_string(context)
    
    def _context_to_dict(self, context: Union[PlanContext, DesignContext, TasksContext, ImplementationContext]) -> Dict[str, Any]:
        """Convert context to dictionary for compression."""
        result = {}
        
        if hasattr(context, 'user_request'):
            result['user_request'] = context.user_request
        
        if hasattr(context, 'requirements') and context.requirements:
            result['requirements'] = context.requirements.content
        
        if hasattr(context, 'design') and context.design:
            result['design'] = context.design.content
        
        if hasattr(context, 'tasks') and context.tasks:
            result['tasks'] = context.tasks.content
        
        if hasattr(context, 'project_structure') and context.project_structure:
            result['project_structure'] = {
                'files': context.project_structure.files,
                'directories': context.project_structure.directories,
                'key_files': context.project_structure.key_files
            }
        
        if hasattr(context, 'memory_patterns'):
            result['memory_patterns'] = [
                {'category': p.category, 'content': p.content, 'source': p.source}
                for p in context.memory_patterns
            ]
        
        if hasattr(context, 'execution_history'):
            result['execution_history'] = [
                {'command': r.command, 'success': r.success, 'approach_used': r.approach_used}
                for r in context.execution_history
            ]
        
        return result
    
    def _dict_to_context(self, compressed_dict: Dict[str, Any], context_type: str, 
                        original_context: Union[PlanContext, DesignContext, TasksContext, ImplementationContext]) -> Union[PlanContext, DesignContext, TasksContext, ImplementationContext]:
        """Convert compressed dictionary back to context object."""
        # For now, return the original context with a flag indicating compression
        # In a full implementation, we would reconstruct the context from compressed data
        return original_context
    
    def _should_refresh_requirements(self) -> bool:
        """Check if requirements should be refreshed."""
        if not self._last_requirements_load:
            return True
        
        requirements_path = self.work_dir / "requirements.md"
        if not requirements_path.exists():
            return False
        
        file_mtime = datetime.fromtimestamp(requirements_path.stat().st_mtime)
        return file_mtime > self._last_requirements_load
    
    def _should_refresh_design(self) -> bool:
        """Check if design should be refreshed."""
        if not self._last_design_load:
            return True
        
        design_path = self.work_dir / "design.md"
        if not design_path.exists():
            return False
        
        file_mtime = datetime.fromtimestamp(design_path.stat().st_mtime)
        return file_mtime > self._last_design_load
    
    def _should_refresh_tasks(self) -> bool:
        """Check if tasks should be refreshed."""
        if not self._last_tasks_load:
            return True
        
        tasks_path = self.work_dir / "tasks.md"
        if not tasks_path.exists():
            return False
        
        file_mtime = datetime.fromtimestamp(tasks_path.stat().st_mtime)
        return file_mtime > self._last_tasks_load
    
    def _should_refresh_structure(self) -> bool:
        """Check if project structure should be refreshed."""
        if not self._last_structure_analysis:
            return True
        
        # Refresh every 5 minutes
        age = (datetime.now() - self._last_structure_analysis).total_seconds()
        return age > 300
    
    async def _persist_execution_history(self) -> None:
        """Persist execution history to file."""
        try:
            history_file = self.work_dir / "execution_history.json"
            
            # Convert ExecutionResult objects to dictionaries
            history_data = []
            for result in self.execution_history:
                history_data.append({
                    'command': result.command,
                    'return_code': result.return_code,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'execution_time': result.execution_time,
                    'working_directory': result.working_directory,
                    'timestamp': result.timestamp,
                    'success': result.success,
                    'approach_used': result.approach_used
                })
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2)
            
            self.logger.info(f"Persisted execution history with {len(history_data)} entries")
            
        except Exception as e:
            self.logger.error(f"Error persisting execution history: {e}")