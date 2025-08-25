"""
TaskDecomposer Agent for the AutoGen multi-agent framework.

This module contains the TaskDecomposer class which is responsible for:
- Intelligent task breakdown and analysis using LLM capabilities
- Converting high-level tasks into executable shell command sequences
- Creating execution plans with decision points and success criteria
- Context-aware task understanding using project requirements and design
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .base_agent import BaseLLMAgent, ContextSpec
from ..models import LLMConfig, TaskDefinition


@dataclass
class ComplexityAnalysis:
    """Analysis of task complexity and requirements."""
    complexity_level: str  # "simple", "moderate", "complex", "very_complex"
    estimated_steps: int
    required_tools: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    analysis_reasoning: str = ""


@dataclass
class ShellCommand:
    """Represents a shell command with metadata."""
    command: str
    description: str
    expected_outputs: List[str] = field(default_factory=list)
    error_patterns: List[str] = field(default_factory=list)
    timeout: int = 30
    retry_on_failure: bool = True
    decision_point: bool = False
    success_indicators: List[str] = field(default_factory=list)
    failure_indicators: List[str] = field(default_factory=list)


@dataclass
class DecisionPoint:
    """Represents a decision point in task execution."""
    condition: str
    true_path: List[str]  # Command indices to execute if condition is true
    false_path: List[str]  # Command indices to execute if condition is false
    evaluation_method: str  # "output_check", "file_exists", "exit_code"
    description: str = ""


@dataclass
class ExecutionPlan:
    """Represents decomposed task execution plan."""
    task: TaskDefinition
    complexity_analysis: ComplexityAnalysis
    commands: List[ShellCommand] = field(default_factory=list)
    decision_points: List[DecisionPoint] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    fallback_strategies: List[str] = field(default_factory=list)
    estimated_duration: int = 0  # in minutes
    created_at: str = ""
    
    def __post_init__(self):
        """Set creation timestamp after initialization."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class TaskDecomposer(BaseLLMAgent):
    """
    AI agent responsible for intelligent task decomposition in the multi-agent framework.
    
    The TaskDecomposer analyzes high-level tasks and breaks them down into executable
    shell command sequences with conditional logic and decision points. It uses LLM
    capabilities for task analysis and context-aware understanding.
    
    Key capabilities:
    - Analyze task complexity using LLM capabilities
    - Decompose tasks into executable shell command sequences
    - Create execution plans with decision points and success criteria
    - Context-aware task understanding using project requirements and design
    - Generate conditional logic for command execution
    """
    
    def __init__(
        self, 
        name: str, 
        llm_config: LLMConfig, 
        system_message: str,
        container,
        description: Optional[str] = None
    ):
        """
        Initialize the TaskDecomposer.
        
        Args:
            name: Name of the agent
            llm_config: LLM configuration for API connection
            system_message: System instructions for the agent
            container: DependencyContainer instance for accessing managers
            description: Optional description of the agent's role
        """
        super().__init__(
            name=name,
            llm_config=llm_config,
            system_message=system_message,
            container=container,
            description=description or "Task decomposition agent for intelligent task breakdown"
        )
        
        # Load decomposition patterns and templates
        self.decomposition_patterns = self._load_decomposition_patterns()
        self.command_templates = self._load_command_templates()
        
        self.logger.info(f"TaskDecomposer initialized with {len(self.decomposition_patterns)} patterns")
    
    def get_context_requirements(self, task_input: Dict[str, Any]) -> Optional[ContextSpec]:
        """Define context requirements for TaskDecomposer."""
        if task_input.get("task"):
            return ContextSpec(context_type="implementation")
        return None
    
    async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task decomposition request.
        
        Args:
            task_input: Dictionary containing task and context information
            
        Returns:
            Dictionary containing decomposition results and execution plan
        """
        task_type = task_input.get("task_type", "decompose_task")
        
        if task_type == "decompose_task":
            return await self._handle_decompose_task(task_input)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def get_agent_capabilities(self) -> List[str]:
        """
        Get a list of capabilities that this agent provides.
        
        Returns:
            List of capability descriptions
        """
        return [
            "Analyze task complexity using LLM capabilities",
            "Decompose tasks into executable shell command sequences",
            "Create execution plans with decision points and success criteria",
            "Context-aware task understanding using project requirements and design",
            "Generate conditional logic for command execution",
            "Identify risk factors and dependencies in task execution",
            "Estimate task duration and resource requirements"
        ]
    
    async def decompose_task(self, task: TaskDefinition) -> ExecutionPlan:
        """
        Converts a high-level task into an executable shell command sequence
        using a single, comprehensive LLM call for efficiency.

        Args:
            task: Task definition from tasks.md

        Returns:
            ExecutionPlan with conditional shell commands
        """
        self.logger.info(f"Decomposing task with comprehensive method: {task.title}")
        try:
            plan = await self._generate_comprehensive_execution_plan(task)
            self.logger.info(f"Comprehensive task decomposition completed: {len(plan.commands)} commands")
            return plan
        except Exception as e:
            self.logger.error(f"Error in comprehensive task decomposition for {task.title}: {e}")
            # As a fallback, create a minimal plan to avoid total failure
            return ExecutionPlan(
                task=task,
                complexity_analysis=ComplexityAnalysis(
                    complexity_level="unknown",
                    estimated_steps=0,
                    analysis_reasoning=f"Decomposition failed: {e}"
                ),
                commands=[],
                success_criteria=[f"Investigate failure in task: {task.title}"],
                fallback_strategies=[f"Manual investigation required due to decomposition error: {e}"]
            )

    async def _generate_comprehensive_execution_plan(self, task: TaskDefinition) -> ExecutionPlan:
        """
        Generates a full execution plan using a single, consolidated prompt.

        Args:
            task: The task to decompose.

        Returns:
            A complete ExecutionPlan.
        """
        prompt = self._build_comprehensive_prompt(task)
        llm_response = await self.generate_response(prompt)
        return self._parse_comprehensive_response(llm_response, task)
    
    # Private helper methods for task handling
    
    async def _handle_decompose_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task decomposition request."""
        task = task_input["task"]
        
        try:
            execution_plan = await self.decompose_task(task)
            
            return {
                "success": True,
                "execution_plan": execution_plan,
                "task_id": task.id,
                "complexity_level": execution_plan.complexity_analysis.complexity_level,
                "command_count": len(execution_plan.commands),
                "decision_points": len(execution_plan.decision_points),
                "estimated_duration": execution_plan.estimated_duration
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_id": task.id
            }
    
    # Private helper methods for prompt building

    def _build_comprehensive_prompt(self, task: TaskDefinition) -> str:
        """Builds a single, comprehensive prompt for generating a full execution plan."""
        context_info = self._get_context_summary()
        prompt = f"""
You are an expert task decomposer. Analyze the following task and generate a complete execution plan in a single JSON response.

**Task Information:**
- **Title:** {task.title}
- **Description:** {task.description}
- **Steps:** {task.steps}
- **Requirements References:** {task.requirements_ref}
- **Dependencies:** {task.dependencies}

**Available Context:**
{context_info}

**Instructions:**
Generate a single JSON object that contains the entire execution plan. The JSON object should have the following structure:

1.  **`complexity_analysis` (object):**
    *   `complexity_level`: (string) "simple", "moderate", "complex", or "very_complex".
    *   `estimated_steps`: (integer) Total number of discrete steps.
    *   `required_tools`: (array of strings) Tools/technologies needed.
    *   `dependencies`: (array of strings) External dependencies or prerequisites.
    *   `risk_factors`: (array of strings) Potential challenges or failure points.
    *   `confidence_score`: (float) Your confidence in this analysis (0.0-1.0).
    *   `analysis_reasoning`: (string) Brief explanation of your assessment.

2.  **`commands` (array of objects):**
    *   `command`: (string) The actual, executable shell command.
    *   `description`: (string) What this command does.
    *   `timeout`: (integer) Max execution time in seconds.
    *   `retry_on_failure`: (boolean) Whether to retry this specific command on failure.
    *   `success_indicators`: (array of strings) Keywords in output that indicate success.
    *   `failure_indicators`: (array of strings) Keywords in output that indicate failure.

3.  **`success_criteria` (array of strings):**
    *   Specific, measurable, and verifiable criteria that confirm the entire task is complete.

4.  **`fallback_strategies` (array of strings):**
    *   Simple, alternative approaches to try if the main plan fails (e.g., "Retry after running `pip install`").

5.  **`estimated_duration` (integer):**
    *   Total estimated time to complete the task in minutes.

**Example JSON Structure:**
```json
{{
  "complexity_analysis": {{
    "complexity_level": "moderate",
    "estimated_steps": 5,
    "required_tools": ["python", "pip"],
    "dependencies": ["requests"],
    "risk_factors": ["API rate limiting"],
    "confidence_score": 0.9,
    "analysis_reasoning": "Standard Python script with external API calls."
  }},
  "commands": [
    {{
      "command": "pip install requests",
      "description": "Install the necessary requests library.",
      "timeout": 120,
      "retry_on_failure": true,
      "success_indicators": ["Successfully installed"],
      "failure_indicators": ["Could not find a version", "error"]
    }},
    {{
      "command": "python create_api_client.py",
      "description": "Run the python script to implement the feature.",
      "timeout": 300,
      "retry_on_failure": false,
      "success_indicators": ["API client created successfully"],
      "failure_indicators": ["Traceback", "Error"]
    }}
  ],
  "success_criteria": [
    "The file `api_client.py` is created.",
    "The script runs without any Python errors.",
    "A sample request using the client is successful."
  ],
  "fallback_strategies": [
    "If 'pip install' fails, try 'pip install --user requests'.",
    "If script fails, check for correct API keys in environment variables."
  ],
  "estimated_duration": 10
}}
```

Now, generate the complete JSON response for the provided task.
"""
        return prompt
    
    # Private helper methods for parsing LLM responses

    def _parse_comprehensive_response(self, response: str, task: TaskDefinition) -> ExecutionPlan:
        """Parses the comprehensive JSON response into an ExecutionPlan."""
        try:
            json_content = self._extract_json_from_response(response)
            if not json_content:
                self.logger.error("No JSON content found in LLM response for comprehensive plan.")
                raise ValueError("No JSON content found in LLM response.")

            data = json.loads(json_content)

            # Parse ComplexityAnalysis
            complexity_data = data.get("complexity_analysis", {})
            complexity = ComplexityAnalysis(
                complexity_level=complexity_data.get("complexity_level", "moderate"),
                estimated_steps=complexity_data.get("estimated_steps", 3),
                required_tools=complexity_data.get("required_tools", []),
                dependencies=complexity_data.get("dependencies", []),
                risk_factors=complexity_data.get("risk_factors", []),
                confidence_score=complexity_data.get("confidence_score", 0.7),
                analysis_reasoning=complexity_data.get("analysis_reasoning", "Parsed from comprehensive response.")
            )

            # Parse ShellCommands
            commands_data = data.get("commands", [])
            commands = [
                ShellCommand(
                    command=cmd.get("command", ""),
                    description=cmd.get("description", ""),
                    timeout=cmd.get("timeout", 60),
                    retry_on_failure=cmd.get("retry_on_failure", False),
                    success_indicators=cmd.get("success_indicators", []),
                    failure_indicators=cmd.get("failure_indicators", [])
                ) for cmd in commands_data
            ]

            # Create the final ExecutionPlan
            plan = ExecutionPlan(
                task=task,
                complexity_analysis=complexity,
                commands=commands,
                decision_points=[],  # Simplified model does not generate complex decision points
                success_criteria=data.get("success_criteria", []),
                fallback_strategies=data.get("fallback_strategies", []),
                estimated_duration=data.get("estimated_duration", 15)
            )
            return plan

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse comprehensive execution plan: {e}. Response: {response[:500]}")
            # Create a fallback plan indicating failure
            return ExecutionPlan(
                task=task,
                complexity_analysis=ComplexityAnalysis(
                    complexity_level="unknown",
                    estimated_steps=0,
                    analysis_reasoning=f"Failed to parse LLM response: {e}"
                ),
                commands=[ShellCommand(command=f"echo 'Error: Failed to parse execution plan from LLM. Please check logs.'", description="Report parsing error")],
                success_criteria=["Manual verification required."],
                fallback_strategies=[f"Parsing failed: {e}"]
            )
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON content from LLM response, handling code blocks."""
        import re

        # Pattern to find a JSON object within triple backticks
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, response, re.DOTALL)
        if match:
            return match.group(1)

        # If no code block found, try to find the first '{' and last '}'
        start_index = response.find('{')
        end_index = response.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return response[start_index:end_index+1]

        return None
    
    # Private helper methods for utilities

    def _get_context_summary(self) -> str:
        """Get summary of current context for prompts."""
        context_parts = []
        if self.context_manager:
            # This is a simplified representation. A real implementation might fetch
            # actual summaries from the context manager.
            context_parts.append("Project context (requirements, design, file structure) is available.")
            context_parts.append("Execution history for previous tasks is available.")

        if not context_parts:
            return "No additional context is available."
        return "\n".join(context_parts)
    
    def _load_decomposition_patterns(self) -> Dict[str, Any]:
        """Load decomposition patterns from configuration."""
        # For now, return basic patterns - can be enhanced with external config
        return {
            "file_creation": ["touch", "echo", "cat"],
            "directory_operations": ["mkdir", "cd", "ls"],
            "code_generation": ["python", "node", "compile"],
            "testing": ["pytest", "npm test", "make test"],
            "installation": ["pip install", "npm install", "apt-get"]
        }
    
    def _load_command_templates(self) -> Dict[str, str]:
        """Load command templates for common operations."""
        return {
            "create_file": "touch {filename}",
            "write_content": "echo '{content}' > {filename}",
            "make_executable": "chmod +x {filename}",
            "create_directory": "mkdir -p {dirname}",
            "run_tests": "python -m pytest {test_file}",
            "install_package": "pip install {package}"
        }