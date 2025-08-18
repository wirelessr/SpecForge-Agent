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
        token_manager,
        context_manager,
        config_manager=None,
        description: Optional[str] = None
    ):
        """
        Initialize the TaskDecomposer.
        
        Args:
            name: Name of the agent
            llm_config: LLM configuration for API connection
            system_message: System instructions for the agent
            token_manager: TokenManager instance for token tracking
            context_manager: ContextManager instance for context management
            config_manager: ConfigManager instance for model configuration (optional)
            description: Optional description of the agent's role
        """
        super().__init__(
            name=name,
            llm_config=llm_config,
            system_message=system_message,
            token_manager=token_manager,
            context_manager=context_manager,
            config_manager=config_manager,
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
        elif task_type == "analyze_complexity":
            return await self._handle_analyze_complexity(task_input)
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
    
    async def _handle_analyze_complexity(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle complexity analysis request."""
        task = task_input["task"]
        
        try:
            complexity = await self._analyze_complexity(task)
            
            return {
                "success": True,
                "complexity_analysis": complexity,
                "task_id": task.id
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
    
    async def generate_conditional_command_sequence(self, task: TaskDefinition, complexity: ComplexityAnalysis) -> List[ShellCommand]:
        """
        Generate shell command sequence with advanced conditional logic.
        
        This method implements requirement 1.3: conditional shell command sequence generation
        with decision points and branching logic for command execution.
        
        Args:
            task: Task definition
            complexity: Complexity analysis results
            
        Returns:
            List of ShellCommand objects with conditional logic
        """
        self.logger.info(f"Generating conditional command sequence for {task.title}")
        
        # Generate base command sequence
        base_commands = await self._generate_command_sequence(task, complexity)
        
        # Enhance commands with conditional logic
        conditional_commands = await self._add_conditional_logic(base_commands, task, complexity)
        
        # Validate command feasibility
        validated_commands = await self._validate_command_feasibility(conditional_commands, task)
        
        self.logger.info(f"Generated {len(validated_commands)} conditional commands with validation")
        return validated_commands
    
    async def _add_conditional_logic(self, commands: List[ShellCommand], task: TaskDefinition, complexity: ComplexityAnalysis) -> List[ShellCommand]:
        """
        Add conditional logic and branching to command sequence.
        
        Args:
            commands: Base command sequence
            task: Task definition
            complexity: Complexity analysis
            
        Returns:
            Enhanced commands with conditional logic
        """
        enhanced_commands = []
        
        for i, command in enumerate(commands):
            # Add pre-condition checks for critical commands
            if self._is_critical_command(command):
                # Add validation command before critical operations
                validation_cmd = self._create_validation_command(command, i)
                if validation_cmd:
                    enhanced_commands.append(validation_cmd)
            
            # Enhance command with conditional logic
            enhanced_command = await self._enhance_command_with_conditions(command, task, i)
            enhanced_commands.append(enhanced_command)
            
            # Add post-condition checks for verification
            if self._needs_verification(command):
                verification_cmd = self._create_verification_command(command, i)
                if verification_cmd:
                    enhanced_commands.append(verification_cmd)
        
        return enhanced_commands
    
    def _is_critical_command(self, command: ShellCommand) -> bool:
        """Check if a command is critical and needs pre-validation."""
        critical_patterns = [
            "rm ", "delete", "drop", "truncate",  # Destructive operations
            "sudo", "chmod", "chown",  # Permission changes
            "pip install", "npm install",  # Package installations
            "git push", "git merge"  # Version control operations
        ]
        
        return any(pattern in command.command.lower() for pattern in critical_patterns)
    
    def _needs_verification(self, command: ShellCommand) -> bool:
        """Check if a command needs post-execution verification."""
        verification_patterns = [
            "touch", "mkdir", "echo >", "cat >",  # File/directory creation
            "cp", "mv",  # File operations
            "python", "node", "compile"  # Code execution
        ]
        
        return any(pattern in command.command.lower() for pattern in verification_patterns)
    
    def _create_validation_command(self, command: ShellCommand, index: int) -> Optional[ShellCommand]:
        """Create validation command for critical operations."""
        if "rm " in command.command.lower():
            # Validate file exists before deletion
            filename = self._extract_filename_from_command(command.command)
            if filename:
                return ShellCommand(
                    command=f"test -f {filename} || echo 'Warning: File {filename} does not exist'",
                    description=f"Validate file exists before deletion",
                    timeout=5,
                    decision_point=True,
                    success_indicators=["file exists"],
                    failure_indicators=["Warning:", "does not exist"]
                )
        
        elif "pip install" in command.command.lower():
            # Check if package is already installed
            package = self._extract_package_from_pip_command(command.command)
            if package:
                return ShellCommand(
                    command=f"pip show {package} > /dev/null 2>&1 && echo 'Package {package} already installed' || echo 'Installing {package}'",
                    description=f"Check if package {package} is already installed",
                    timeout=10,
                    decision_point=True,
                    success_indicators=["already installed"],
                    failure_indicators=["Installing"]
                )
        
        return None
    
    def _create_verification_command(self, command: ShellCommand, index: int) -> Optional[ShellCommand]:
        """Create verification command for post-execution checks."""
        if "touch" in command.command.lower():
            # Verify file was created
            filename = self._extract_filename_from_command(command.command)
            if filename:
                return ShellCommand(
                    command=f"test -f {filename} && echo 'File {filename} created successfully' || echo 'Error: Failed to create {filename}'",
                    description=f"Verify file {filename} was created",
                    timeout=5,
                    success_indicators=["created successfully"],
                    failure_indicators=["Error:", "Failed to create"]
                )
        
        elif "mkdir" in command.command.lower():
            # Verify directory was created
            dirname = self._extract_dirname_from_command(command.command)
            if dirname:
                return ShellCommand(
                    command=f"test -d {dirname} && echo 'Directory {dirname} created successfully' || echo 'Error: Failed to create directory {dirname}'",
                    description=f"Verify directory {dirname} was created",
                    timeout=5,
                    success_indicators=["created successfully"],
                    failure_indicators=["Error:", "Failed to create"]
                )
        
        elif "python" in command.command.lower() and ".py" in command.command:
            # Verify Python script runs without syntax errors
            return ShellCommand(
                command=f"python -m py_compile $(echo '{command.command}' | grep -o '[^ ]*\\.py' | head -1) && echo 'Python syntax valid' || echo 'Syntax error detected'",
                description="Verify Python script has valid syntax",
                timeout=10,
                success_indicators=["syntax valid"],
                failure_indicators=["Syntax error", "SyntaxError"]
            )
        
        return None
    
    async def _enhance_command_with_conditions(self, command: ShellCommand, task: TaskDefinition, index: int) -> ShellCommand:
        """
        Enhance individual command with conditional logic.
        
        Args:
            command: Original command
            task: Task definition
            index: Command index in sequence
            
        Returns:
            Enhanced command with conditional logic
        """
        # Build conditional enhancement prompt
        enhancement_prompt = f"""
Enhance the following shell command with conditional logic and error handling.

Original Command: {command.command}
Description: {command.description}
Task Context: {task.title}
Command Index: {index}

Add conditional logic that:
1. Checks prerequisites before execution
2. Handles common error scenarios
3. Provides alternative approaches on failure
4. Includes proper error messages

Enhance the command while keeping it executable and practical.

Format your response as JSON:
{{
    "enhanced_command": "...",
    "description": "...",
    "success_indicators": [...],
    "failure_indicators": [...],
    "error_handling": "..."
}}
"""
        
        try:
            # Get LLM enhancement
            enhancement_response = await self.generate_response(enhancement_prompt)
            
            # Parse enhancement
            enhanced_data = self._parse_command_enhancement(enhancement_response)
            
            if enhanced_data:
                # Update command with enhancements
                command.command = enhanced_data.get("enhanced_command", command.command)
                command.description = enhanced_data.get("description", command.description)
                command.success_indicators = enhanced_data.get("success_indicators", command.success_indicators)
                command.failure_indicators = enhanced_data.get("failure_indicators", command.failure_indicators)
                
                # Add error handling information
                if "error_handling" in enhanced_data:
                    command.description += f" (Error handling: {enhanced_data['error_handling']})"
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance command {index}: {e}")
            # Continue with original command if enhancement fails
        
        return command
    
    def _parse_command_enhancement(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse command enhancement from LLM response."""
        try:
            if response.strip().startswith('{'):
                return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        return None
    
    async def _validate_command_feasibility(self, commands: List[ShellCommand], task: TaskDefinition) -> List[ShellCommand]:
        """
        Validate command feasibility and filter out problematic commands.
        
        This method implements requirement 1.4: command validation and feasibility checking.
        
        Args:
            commands: List of commands to validate
            task: Task definition for context
            
        Returns:
            Validated and filtered command list
        """
        validated_commands = []
        
        for i, command in enumerate(commands):
            # Perform feasibility checks
            feasibility_result = await self._check_command_feasibility(command, task, i)
            
            if feasibility_result["feasible"]:
                # Command is feasible, add to validated list
                validated_commands.append(command)
                self.logger.debug(f"Command {i} validated: {command.command}")
            else:
                # Command is not feasible, log warning and try to fix
                self.logger.warning(f"Command {i} failed feasibility check: {feasibility_result['reason']}")
                
                # Try to create alternative command
                alternative = await self._create_alternative_command(command, feasibility_result, task)
                if alternative:
                    validated_commands.append(alternative)
                    self.logger.info(f"Created alternative for command {i}: {alternative.command}")
                else:
                    self.logger.warning(f"No alternative found for command {i}, skipping")
        
        return validated_commands
    
    async def _check_command_feasibility(self, command: ShellCommand, task: TaskDefinition, index: int) -> Dict[str, Any]:
        """
        Check if a command is feasible to execute.
        
        Args:
            command: Command to check
            task: Task context
            index: Command index
            
        Returns:
            Dictionary with feasibility result and reasoning
        """
        feasibility_checks = {
            "has_command": self._check_command_exists(command.command),
            "has_valid_syntax": self._check_command_syntax(command.command),
            "has_safe_operations": self._check_command_safety(command.command),
            "has_reasonable_timeout": command.timeout > 0 and command.timeout < 3600,
            "has_description": bool(command.description.strip())
        }
        
        failed_checks = [check for check, passed in feasibility_checks.items() if not passed]
        
        if not failed_checks:
            return {
                "feasible": True,
                "reason": "All feasibility checks passed",
                "checks": feasibility_checks
            }
        else:
            return {
                "feasible": False,
                "reason": f"Failed checks: {', '.join(failed_checks)}",
                "checks": feasibility_checks,
                "failed_checks": failed_checks
            }
    
    def _check_command_exists(self, command: str) -> bool:
        """Check if the main command exists in the system."""
        # Extract the main command (first word)
        main_cmd = command.strip().split()[0] if command.strip() else ""
        
        # List of common commands that should exist
        common_commands = {
            "echo", "cat", "touch", "mkdir", "ls", "cd", "pwd", "cp", "mv", "rm",
            "grep", "find", "sort", "uniq", "head", "tail", "wc", "chmod", "chown",
            "python", "python3", "pip", "pip3", "node", "npm", "git", "curl", "wget",
            "test", "which", "whereis", "ps", "kill", "sleep", "date", "whoami"
        }
        
        # Check if it's a common command or a shell builtin
        return main_cmd in common_commands or main_cmd in ["if", "for", "while", "case", "function"]
    
    def _check_command_syntax(self, command: str) -> bool:
        """Check if command has valid shell syntax."""
        if not command.strip():
            return False
        
        # Basic syntax checks
        syntax_issues = [
            # Unmatched quotes
            command.count("'") % 2 != 0,
            command.count('"') % 2 != 0,
            # Unmatched parentheses
            command.count("(") != command.count(")"),
            # Unmatched brackets
            command.count("[") != command.count("]"),
            # Invalid characters at start
            command.strip().startswith(("&", "|", ";")),
            # Multiple consecutive operators
            any(op in command for op in ["&&&&", "||||", ";;;"])
        ]
        
        return not any(syntax_issues)
    
    def _check_command_safety(self, command: str) -> bool:
        """Check if command contains potentially dangerous operations."""
        dangerous_patterns = [
            "rm -rf /",  # Delete root
            ":(){ :|:& };:",  # Fork bomb
            "dd if=/dev/zero",  # Disk fill
            "chmod 777 /",  # Dangerous permissions
            "chown root /",  # Ownership changes
            "> /dev/sda",  # Direct disk write
            "mkfs.",  # Format filesystem
            "fdisk",  # Disk partitioning
            "shutdown", "reboot", "halt"  # System control
        ]
        
        command_lower = command.lower()
        return not any(pattern in command_lower for pattern in dangerous_patterns)
    
    async def _create_alternative_command(self, original: ShellCommand, feasibility_result: Dict[str, Any], task: TaskDefinition) -> Optional[ShellCommand]:
        """
        Create alternative command when original fails feasibility check.
        
        Args:
            original: Original command that failed
            feasibility_result: Feasibility check results
            task: Task context
            
        Returns:
            Alternative command or None if no alternative possible
        """
        failed_checks = feasibility_result.get("failed_checks", [])
        
        # Try to fix specific issues
        if "has_command" in failed_checks:
            # Try to find alternative command
            alternative_cmd = self._find_alternative_command(original.command)
            if alternative_cmd:
                return ShellCommand(
                    command=alternative_cmd,
                    description=f"Alternative to: {original.description}",
                    timeout=original.timeout,
                    retry_on_failure=original.retry_on_failure
                )
        
        if "has_valid_syntax" in failed_checks:
            # Try to fix syntax issues
            fixed_cmd = self._fix_command_syntax(original.command)
            if fixed_cmd and fixed_cmd != original.command:
                return ShellCommand(
                    command=fixed_cmd,
                    description=f"Syntax-fixed: {original.description}",
                    timeout=original.timeout,
                    retry_on_failure=original.retry_on_failure
                )
        
        if "has_safe_operations" in failed_checks:
            # Create safer version of command
            safe_cmd = self._make_command_safer(original.command)
            if safe_cmd and safe_cmd != original.command:
                return ShellCommand(
                    command=safe_cmd,
                    description=f"Safe version: {original.description}",
                    timeout=original.timeout,
                    retry_on_failure=original.retry_on_failure
                )
        
        return None
    
    def _find_alternative_command(self, command: str) -> Optional[str]:
        """Find alternative command for unsupported operations."""
        alternatives = {
            "nano": "echo 'Content' > file.txt",
            "vim": "echo 'Content' > file.txt",
            "emacs": "echo 'Content' > file.txt",
            "gedit": "echo 'Content' > file.txt",
            "code": "echo 'Content' > file.txt",
            "subl": "echo 'Content' > file.txt"
        }
        
        main_cmd = command.strip().split()[0] if command.strip() else ""
        return alternatives.get(main_cmd)
    
    def _fix_command_syntax(self, command: str) -> Optional[str]:
        """Fix basic syntax issues in command."""
        fixed = command
        
        # Fix unmatched quotes by removing them if they're unbalanced
        if fixed.count("'") % 2 != 0:
            fixed = fixed.replace("'", '"')
        
        if fixed.count('"') % 2 != 0:
            fixed = fixed.replace('"', '')
        
        # Remove leading operators
        while fixed.strip().startswith(("&", "|", ";")):
            fixed = fixed.strip()[1:].strip()
        
        # Replace multiple consecutive operators
        fixed = fixed.replace("&&&&", "&&")
        fixed = fixed.replace("||||", "||")
        fixed = fixed.replace(";;;", ";")
        
        return fixed if fixed != command else None
    
    def _make_command_safer(self, command: str) -> Optional[str]:
        """Make command safer by removing dangerous operations."""
        # For now, just return None for dangerous commands
        # In a real implementation, this could create safer alternatives
        return None
    
    # Helper methods for extracting information from commands
    
    def _extract_filename_from_command(self, command: str) -> Optional[str]:
        """Extract filename from command."""
        import re
        
        # Look for common filename patterns
        patterns = [
            r'rm\s+([^\s]+)',  # rm filename
            r'touch\s+([^\s]+)',
            r'>\s*([^\s]+)',
            r'cat\s+([^\s]+)',
            r'cp\s+\S+\s+([^\s]+)',
            r'mv\s+\S+\s+([^\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, command)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_dirname_from_command(self, command: str) -> Optional[str]:
        """Extract directory name from command."""
        import re
        
        # Look for mkdir patterns
        match = re.search(r'mkdir\s+(?:-p\s+)?([^\s]+)', command)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_package_from_pip_command(self, command: str) -> Optional[str]:
        """Extract package name from pip install command."""
        import re
        
        # Handle both pip and pip3
        match = re.search(r'pip[3]?\s+install\s+([^\s]+)', command)
        if match:
            return match.group(1)
        
        return None