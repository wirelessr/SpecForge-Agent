"""
Tasks Agent for the AutoGen multi-agent framework.

This module contains the TasksAgent class which is responsible for:
- Generating task lists (tasks.md) based on design documents
- Decomposing technical designs into implementation plans
- Creating structured task lists with specific requirements references
- Managing task generation context and formatting
"""

import os
from typing import Dict, Any, List, Optional

from .base_agent import BaseLLMAgent, ContextSpec
from ..models import LLMConfig, TaskDefinition


class TasksAgent(BaseLLMAgent):
    """
    AI agent responsible for task generation in the multi-agent framework.

    The TasksAgent handles the task generation phase of the workflow where
    technical designs are decomposed into actionable implementation plans.
    It creates structured task lists that can be executed by the ImplementAgent.

    Key capabilities:
    - Generate structured task lists from design documents
    - Reference specific requirements for each task
    - Create tasks that build incrementally on each other
    - Format tasks in markdown checkbox format
    - Maintain context access to requirements.md and design.md
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        memory_manager=None,
        token_manager=None,
        context_manager=None,
        config_manager=None,
        description: Optional[str] = None
    ):
        """
        Initialize the TasksAgent.

        Args:
            llm_config: LLM configuration for API connection
            memory_manager: Optional memory manager for context
            token_manager: TokenManager instance for token operations (mandatory)
            context_manager: ContextManager instance for context operations (mandatory)
            config_manager: ConfigManager instance for model configuration (optional)
            description: Optional description of the agent's role
        """
        super().__init__(
            name="TasksAgent",
            llm_config=llm_config,
            system_message=self._build_system_message(),
            token_manager=token_manager,
            context_manager=context_manager,
            config_manager=config_manager,
            description=description or "Task generation agent for creating implementation plans",
        )

        self.memory_manager = memory_manager
        self.current_work_directory: Optional[str] = None
        self.current_tasks: List[TaskDefinition] = []

        self.logger.info(f"TasksAgent initialized")
    def _build_system_message(self) -> str:
        """Build the system message for the TasksAgent."""
        return """You are a TasksAgent responsible for generating detailed implementation task lists.

Your role is to:
1. Analyze design documents and requirements
2. Break down technical designs into specific, actionable coding tasks
3. Create tasks that build incrementally on each other
4. Reference specific requirements for each task
5. Format tasks in markdown checkbox format

You should generate tasks that:
- Are small, testable, and specific
- Include detailed steps for implementation
- Reference specific requirements (Requirements: X.Y, Z.A)
- Are ordered by dependencies
- Focus ONLY on coding tasks that can be executed via shell commands

Always maintain the existing task generation logic, prompts, and formatting unchanged."""

    def get_context_requirements(self, task_input: Dict[str, Any]) -> Optional['ContextSpec']:
        """Define context requirements for TasksAgent."""
        if task_input.get("user_request"):
            from .base_agent import ContextSpec
            return ContextSpec(context_type="tasks")
        return None

    async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned to the TasksAgent.

        Args:
            task_input: Dictionary containing task type and parameters

        Returns:
            Dictionary containing task results and metadata
        """
        task_type = task_input.get("task_type")

        if task_type == "generate_task_list":
            return await self._handle_generate_task_list(task_input)
        elif task_type == "revision":
            return await self._handle_revision_task(task_input)
        else:
            raise ValueError(f"Unknown task type for TasksAgent: {task_type}")
    def get_agent_capabilities(self) -> List[str]:
        """
        Get a list of capabilities that this agent provides.

        Returns:
            List of capability descriptions
        """
        return [
            "Generate task lists from design documents",
            "Decompose technical designs into implementation plans",
            "Create structured task lists with requirements references",
            "Format tasks in markdown checkbox format",
            "Maintain context access to requirements.md and design.md",
            "Ensure tasks build incrementally on each other",
            "Handle task list revisions based on user feedback"
        ]

    async def generate_task_list(
        self,
        design_path: str,
        requirements_path: str,
        work_dir: str
    ) -> str:
        """
        Generate a tasks.md file based on design and requirements documents.

        Args:
            design_path: Path to the design.md file
            requirements_path: Path to the requirements.md file
            work_dir: Working directory for the project

        Returns:
            Path to the generated tasks.md file
        """
        self.current_work_directory = work_dir

        # Read design and requirements documents
        design_content = await self._read_file_content(design_path)
        requirements_content = await self._read_file_content(requirements_path)

        # Prepare context for task generation
        context = {
            "design_document": design_content,
            "requirements_document": requirements_content,
            "work_directory": work_dir,
            "task_type": "task_list_generation"
        }
        
        # Generate task list using LLM
        prompt = self._build_task_generation_prompt(design_content, requirements_content)
        
        self.logger.info(f"Generating task list for project in {work_dir}")
        
        try:
            task_list_content = await self.generate_response(prompt, context)
            
            # Save tasks.md file
            tasks_path = os.path.join(work_dir, "tasks.md")
            await self._write_file_content(tasks_path, task_list_content)
            
            # Parse and store task definitions
            self.current_tasks = self._parse_task_list(task_list_content)
            
            self.logger.info(f"Generated task list with {len(self.current_tasks)} tasks")
            return tasks_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate task list: {e}")
            raise
    
    def _build_task_generation_prompt(self, design_content: str, requirements_content: str) -> str:
        """Build prompt for task list generation."""
        return f"""Based on the following design document and requirements, generate a comprehensive tasks.md file with specific, actionable coding tasks.

DESIGN DOCUMENT:
{design_content}

REQUIREMENTS DOCUMENT:
{requirements_content}

Generate a tasks.md file that:
1. Uses markdown checkbox format with sequential numerical identifiers (- [ ] 1. Task title, - [ ] 2. Task title, etc.)
2. Each task MUST have a sequential numerical identifier (1, 2, 3, 4, 5, etc.)
3. Each task should be small, testable, and specific
4. Include detailed steps for each task (do a, do b, ...)
5. Reference specific requirements (Requirements: X.Y, Z.A)
6. Order tasks by dependencies
7. Focus ONLY on coding tasks that can be executed via shell commands

CRITICAL: Each task MUST be formatted with sequential numerical identifiers as:
- [ ] 1. Task Title
  - Step 1: Specific action
  - Step 2: Another specific action
  - Requirements: X.Y, Z.A

- [ ] 2. Next Task Title
  - Step 1: Specific action
  - Requirements: X.Y

Continue with 3, 4, 5, etc. for all tasks. Do NOT skip numbers or use any other numbering format.

Generate the complete tasks.md content now:"""
    
    def _parse_task_list(self, task_list_content: str) -> List[TaskDefinition]:
        """Parse tasks.md content into TaskDefinition objects."""
        import re
        
        tasks = []
        lines = task_list_content.split('\n')
        current_task = None
        
        for line in lines:
            line = line.strip()
            
            # Check for task checkbox
            if line.startswith('- [ ]') or line.startswith('- [x]'):
                if current_task:
                    tasks.append(current_task)
                
                title = line[5:].strip()
                
                # Extract numerical ID from task title if present (e.g., "1. Task Title" -> "1")
                numerical_id_match = re.match(r'^(\d+)\.\s*(.+)', title)
                if numerical_id_match:
                    numerical_id = numerical_id_match.group(1)
                    task_id = numerical_id  # Use numerical ID directly
                    title = numerical_id_match.group(2)  # Remove number from title for clean display
                else:
                    # Fallback to sequential ID for backward compatibility
                    task_id = f"task_{len(tasks) + 1}"
                
                current_task = TaskDefinition(
                    id=task_id,
                    title=title,
                    description=title,
                    steps=[],
                    requirements_ref=[]
                )
            
            # Check for task steps
            elif line.startswith('-') and current_task:
                step = line[1:].strip()
                if not step.startswith('Requirements:'):
                    current_task.steps.append(step)
                else:
                    # Parse requirements references
                    req_part = step.replace('Requirements:', '').strip()
                    current_task.requirements_ref = [r.strip() for r in req_part.split(',')]
        
        if current_task:
            tasks.append(current_task)
        
        return tasks
    
    async def _handle_generate_task_list(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task list generation request."""
        design_path = task_input["design_path"]
        requirements_path = task_input["requirements_path"]
        work_dir = task_input["work_dir"]
        
        tasks_path = await self.generate_task_list(design_path, requirements_path, work_dir)
        
        return {
            "success": True,
            "tasks_file": tasks_path,
            "task_count": len(self.current_tasks),
            "work_directory": work_dir
        }
    
    async def _read_file_content(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            raise FileNotFoundError(f"Could not read file: {file_path}")
    
    async def _write_file_content(self, file_path: str, content: str) -> None:
        """Write content to a file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            self.logger.error(f"Failed to write file {file_path}: {e}")
            raise IOError(f"Failed to write file: {file_path}")
    
    async def _handle_revision_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle task list revision based on user feedback.
        
        Args:
            task_input: Dictionary containing revision parameters
            
        Returns:
            Dictionary containing revision results
        """
        try:
            revision_feedback = task_input.get("revision_feedback", "")
            current_result = task_input.get("current_result", {})
            work_directory = task_input.get("work_directory", "")
            
            if not revision_feedback:
                raise ValueError("revision_feedback is required for revision tasks")
            
            if not work_directory:
                raise ValueError("work_directory is required for revision tasks")
            
            # Get current tasks.md path
            tasks_path = os.path.join(work_directory, "tasks.md")
            
            if not os.path.exists(tasks_path):
                raise ValueError(f"tasks.md not found at {tasks_path}")
            
            # Read current tasks content
            with open(tasks_path, 'r', encoding='utf-8') as f:
                current_tasks = f.read()
            
            # Apply revision using LLM
            revised_tasks = await self._apply_tasks_revision(current_tasks, revision_feedback)
            
            # Write revised tasks back to file
            with open(tasks_path, 'w', encoding='utf-8') as f:
                f.write(revised_tasks)
            
            self.logger.info(f"Tasks revised based on feedback: {revision_feedback[:100]}...")
            
            return {
                "success": True,
                "message": "Tasks revised successfully",
                "work_directory": work_directory,
                "tasks_path": tasks_path,
                "revision_applied": True
            }
            
        except Exception as e:
            self.logger.error(f"Error handling revision task: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _apply_tasks_revision(self, current_tasks: str, revision_feedback: str) -> str:
        """
        Apply revision feedback to current tasks using LLM.
        
        Args:
            current_tasks: Current tasks.md content
            revision_feedback: User's revision feedback
            
        Returns:
            Revised tasks content
        """
        revision_prompt = f"""Please revise the following tasks.md file based on the user's feedback.

Current Tasks Document:
{current_tasks}

User Feedback:
{revision_feedback}

Please provide the complete revised tasks.md content that incorporates the user's feedback while maintaining the same markdown checkbox format (- [ ] Task title). Make sure to:

1. Keep all existing tasks unless specifically asked to remove them
2. Add new tasks based on the feedback
3. Modify existing tasks if requested
4. Maintain the same structure and format
5. Keep the detailed steps for each task
6. Preserve requirement references

Provide only the revised tasks.md content, no additional explanation."""

        try:
            revised_content = await self.generate_response(revision_prompt)
            
            # Clean up the response to ensure it's properly formatted
            if revised_content.startswith("```markdown"):
                revised_content = revised_content.replace("```markdown", "").replace("```", "").strip()
            elif revised_content.startswith("```"):
                revised_content = revised_content.replace("```", "").strip()
            
            return revised_content
            
        except Exception as e:
            self.logger.error(f"Error applying tasks revision: {e}")
            # Return original content with a note about the revision attempt
            return f"{current_tasks}\n\n<!-- Revision attempted but failed: {str(e)} -->"