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
        description: Optional[str] = None
    ):
        """
        Initialize the TaskDecomposer.
        
        Args:
            name: Name of the agent
            llm_config: LLM configuration for API connection
            system_message: System instructions for the agent
            description: Optional description of the agent's role
        """
        super().__init__(
            name=name,
            llm_config=llm_config,
            system_message=system_message,
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
        Converts a high-level task into executable shell command sequence.
        
        Args:
            task: Task definition from tasks.md
            
        Returns:
            ExecutionPlan with conditional shell commands
        """
        self.logger.info(f"Decomposing task: {task.title}")
        
        try:
            # Step 1: Analyze task complexity and requirements
            complexity = await self._analyze_complexity(task)
            
            # Step 2: Generate shell command sequence with decision points
            commands = await self._generate_command_sequence(task, complexity)
            
            # Step 3: Identify decision points and conditional logic
            decision_points = await self._identify_decision_points(commands, task)
            
            # Step 4: Define success criteria and fallback strategies
            success_criteria = await self._define_success_criteria(task)
            fallback_strategies = await self._generate_fallback_strategies(task, complexity)
            
            # Step 5: Estimate execution duration
            estimated_duration = self._estimate_duration(commands, complexity)
            
            # Create execution plan
            plan = ExecutionPlan(
                task=task,
                complexity_analysis=complexity,
                commands=commands,
                decision_points=decision_points,
                success_criteria=success_criteria,
                fallback_strategies=fallback_strategies,
                estimated_duration=estimated_duration
            )
            
            self.logger.info(f"Task decomposition completed: {len(commands)} commands, {len(decision_points)} decision points")
            return plan
            
        except Exception as e:
            self.logger.error(f"Error decomposing task {task.title}: {e}")
            raise
    
    async def _analyze_complexity(self, task: TaskDefinition) -> ComplexityAnalysis:
        """
        Analyzes task complexity based on requirements and context.
        
        Args:
            task: Task definition to analyze
            
        Returns:
            ComplexityAnalysis with detailed assessment
        """
        self.logger.debug(f"Analyzing complexity for task: {task.title}")
        
        # Build context-aware analysis prompt
        analysis_prompt = self._build_complexity_analysis_prompt(task)
        
        # Get LLM analysis
        analysis_response = await self.generate_response(analysis_prompt)
        
        # Parse the LLM response into structured analysis
        complexity = self._parse_complexity_analysis(analysis_response, task)
        
        self.logger.info(f"Complexity analysis completed: {complexity.complexity_level} ({complexity.estimated_steps} steps)")
        return complexity
    
    async def _generate_command_sequence(self, task: TaskDefinition, complexity: ComplexityAnalysis) -> List[ShellCommand]:
        """
        Generates conditional shell command sequence based on task and complexity.
        
        Args:
            task: Task definition
            complexity: Complexity analysis results
            
        Returns:
            List of ShellCommand objects with metadata
        """
        self.logger.debug(f"Generating command sequence for {complexity.complexity_level} task")
        
        # Build command generation prompt
        command_prompt = self._build_command_generation_prompt(task, complexity)
        
        # Get LLM-generated command sequence
        commands_response = await self.generate_response(command_prompt)
        
        # Parse commands from LLM response
        commands = self._parse_command_sequence(commands_response, task, complexity)
        
        self.logger.info(f"Generated {len(commands)} shell commands")
        return commands
    
    async def _identify_decision_points(self, commands: List[ShellCommand], task: TaskDefinition) -> List[DecisionPoint]:
        """
        Identifies decision points and branching logic in command sequence.
        
        Args:
            commands: List of shell commands
            task: Task definition
            
        Returns:
            List of DecisionPoint objects
        """
        decision_points = []
        
        # Identify commands marked as decision points
        for i, command in enumerate(commands):
            if command.decision_point:
                # Build decision point analysis prompt
                decision_prompt = self._build_decision_point_prompt(command, task, i)
                
                # Get LLM analysis for decision logic
                decision_response = await self.generate_response(decision_prompt)
                
                # Parse decision point from response
                decision_point = self._parse_decision_point(decision_response, i)
                if decision_point:
                    decision_points.append(decision_point)
        
        self.logger.info(f"Identified {len(decision_points)} decision points")
        return decision_points
    
    async def _define_success_criteria(self, task: TaskDefinition) -> List[str]:
        """
        Defines success criteria for task completion.
        
        Args:
            task: Task definition
            
        Returns:
            List of success criteria strings
        """
        # Build success criteria prompt
        criteria_prompt = self._build_success_criteria_prompt(task)
        
        # Get LLM-generated criteria
        criteria_response = await self.generate_response(criteria_prompt)
        
        # Parse criteria from response
        criteria = self._parse_success_criteria(criteria_response)
        
        self.logger.info(f"Defined {len(criteria)} success criteria")
        return criteria
    
    async def _generate_fallback_strategies(self, task: TaskDefinition, complexity: ComplexityAnalysis) -> List[str]:
        """
        Generates fallback strategies for error recovery.
        
        Args:
            task: Task definition
            complexity: Complexity analysis
            
        Returns:
            List of fallback strategy descriptions
        """
        # Build fallback strategies prompt
        fallback_prompt = self._build_fallback_strategies_prompt(task, complexity)
        
        # Get LLM-generated strategies
        strategies_response = await self.generate_response(fallback_prompt)
        
        # Parse strategies from response
        strategies = self._parse_fallback_strategies(strategies_response)
        
        self.logger.info(f"Generated {len(strategies)} fallback strategies")
        return strategies
    
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
    
    def _build_complexity_analysis_prompt(self, task: TaskDefinition) -> str:
        """Build prompt for task complexity analysis."""
        context_info = self._get_context_summary()
        
        prompt = f"""
Analyze the complexity of the following task and provide a detailed assessment.

Task Information:
- Title: {task.title}
- Description: {task.description}
- Steps: {task.steps}
- Requirements References: {task.requirements_ref}
- Dependencies: {task.dependencies}

{context_info}

Please analyze this task and provide:

1. Complexity Level: Choose from "simple", "moderate", "complex", "very_complex"
2. Estimated Steps: Number of discrete steps needed
3. Required Tools: List of tools/technologies needed
4. Dependencies: External dependencies or prerequisites
5. Risk Factors: Potential challenges or failure points
6. Confidence Score: Your confidence in this analysis (0.0-1.0)
7. Analysis Reasoning: Explanation of your assessment

Format your response as JSON:
{{
    "complexity_level": "...",
    "estimated_steps": ...,
    "required_tools": [...],
    "dependencies": [...],
    "risk_factors": [...],
    "confidence_score": ...,
    "analysis_reasoning": "..."
}}
"""
        return prompt
    
    def _build_command_generation_prompt(self, task: TaskDefinition, complexity: ComplexityAnalysis) -> str:
        """Build prompt for shell command sequence generation."""
        context_info = self._get_context_summary()
        
        prompt = f"""
Generate a sequence of shell commands to accomplish the following task.

Task Information:
- Title: {task.title}
- Description: {task.description}
- Steps: {task.steps}
- Complexity Level: {complexity.complexity_level}
- Estimated Steps: {complexity.estimated_steps}
- Required Tools: {complexity.required_tools}
- Risk Factors: {complexity.risk_factors}

{context_info}

Generate shell commands that:
1. Are executable and practical
2. Include proper error handling
3. Have clear success/failure indicators
4. Include decision points where branching logic is needed
5. Are optimized for the identified complexity level

For each command, provide:
- command: The actual shell command
- description: What this command does
- expected_outputs: What outputs indicate success
- error_patterns: What outputs indicate failure
- timeout: Maximum execution time in seconds
- retry_on_failure: Whether to retry on failure
- decision_point: Whether this is a decision point
- success_indicators: Specific success indicators
- failure_indicators: Specific failure indicators

Format your response as JSON array:
[
    {{
        "command": "...",
        "description": "...",
        "expected_outputs": [...],
        "error_patterns": [...],
        "timeout": ...,
        "retry_on_failure": ...,
        "decision_point": ...,
        "success_indicators": [...],
        "failure_indicators": [...]
    }},
    ...
]
"""
        return prompt
    
    def _build_decision_point_prompt(self, command: ShellCommand, task: TaskDefinition, index: int) -> str:
        """Build prompt for decision point analysis."""
        prompt = f"""
Analyze the following command as a decision point and define the branching logic.

Command: {command.command}
Description: {command.description}
Task: {task.title}
Command Index: {index}

Define the decision logic:
1. Condition: What condition determines the branch?
2. True Path: What command indices to execute if condition is true?
3. False Path: What command indices to execute if condition is false?
4. Evaluation Method: How to evaluate the condition? (output_check, file_exists, exit_code)
5. Description: Brief description of the decision logic

Format your response as JSON:
{{
    "condition": "...",
    "true_path": [...],
    "false_path": [...],
    "evaluation_method": "...",
    "description": "..."
}}
"""
        return prompt
    
    def _build_success_criteria_prompt(self, task: TaskDefinition) -> str:
        """Build prompt for success criteria definition."""
        context_info = self._get_context_summary()
        
        prompt = f"""
Define clear success criteria for the following task completion.

Task Information:
- Title: {task.title}
- Description: {task.description}
- Steps: {task.steps}
- Requirements References: {task.requirements_ref}

{context_info}

Define specific, measurable success criteria that indicate the task has been completed successfully.
Each criterion should be:
1. Specific and measurable
2. Verifiable through shell commands or file checks
3. Directly related to the task objectives

Format your response as JSON array of strings:
[
    "Criterion 1: ...",
    "Criterion 2: ...",
    ...
]
"""
        return prompt
    
    def _build_fallback_strategies_prompt(self, task: TaskDefinition, complexity: ComplexityAnalysis) -> str:
        """Build prompt for fallback strategies generation."""
        prompt = f"""
Generate fallback strategies for the following task in case of failures.

Task Information:
- Title: {task.title}
- Description: {task.description}
- Complexity Level: {complexity.complexity_level}
- Risk Factors: {complexity.risk_factors}

Consider the identified risk factors and generate alternative approaches that could be used if the primary execution plan fails.

Each strategy should:
1. Address specific failure scenarios
2. Provide alternative implementation approaches
3. Be practical and executable
4. Have lower complexity than the original approach when possible

Format your response as JSON array of strings:
[
    "Strategy 1: ...",
    "Strategy 2: ...",
    ...
]
"""
        return prompt
    
    # Private helper methods for parsing LLM responses
    
    def _parse_complexity_analysis(self, response: str, task: TaskDefinition) -> ComplexityAnalysis:
        """Parse complexity analysis from LLM response."""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                data = json.loads(response)
                return ComplexityAnalysis(
                    complexity_level=data.get("complexity_level", "moderate"),
                    estimated_steps=data.get("estimated_steps", 5),
                    required_tools=data.get("required_tools", []),
                    dependencies=data.get("dependencies", []),
                    risk_factors=data.get("risk_factors", []),
                    confidence_score=data.get("confidence_score", 0.5),
                    analysis_reasoning=data.get("analysis_reasoning", "")
                )
            else:
                # Fallback to text parsing
                return self._parse_complexity_from_text(response, task)
                
        except json.JSONDecodeError:
            # Fallback to text parsing
            return self._parse_complexity_from_text(response, task)
    
    def _parse_complexity_from_text(self, response: str, task: TaskDefinition) -> ComplexityAnalysis:
        """Parse complexity analysis from text response."""
        # Simple text-based parsing as fallback
        lines = response.lower().split('\n')
        
        complexity_level = "moderate"
        estimated_steps = len(task.steps) if task.steps else 5
        
        # Look for complexity indicators
        if any(word in response.lower() for word in ["simple", "easy", "basic"]):
            complexity_level = "simple"
            estimated_steps = max(3, estimated_steps)
        elif any(word in response.lower() for word in ["complex", "difficult", "challenging"]):
            complexity_level = "complex"
            estimated_steps = max(8, estimated_steps)
        elif any(word in response.lower() for word in ["very complex", "extremely", "highly complex"]):
            complexity_level = "very_complex"
            estimated_steps = max(12, estimated_steps)
        
        return ComplexityAnalysis(
            complexity_level=complexity_level,
            estimated_steps=estimated_steps,
            required_tools=["shell", "basic_tools"],
            dependencies=[],
            risk_factors=["execution_failure"],
            confidence_score=0.6,
            analysis_reasoning="Parsed from text response"
        )
    
    def _parse_command_sequence(self, response: str, task: TaskDefinition, complexity: ComplexityAnalysis) -> List[ShellCommand]:
        """Parse shell command sequence from LLM response."""
        try:
            # Extract JSON from response (handle code blocks)
            json_content = self._extract_json_from_response(response)
            
            if json_content:
                data = json.loads(json_content)
                commands = []
                
                for cmd_data in data:
                    command = ShellCommand(
                        command=cmd_data.get("command", ""),
                        description=cmd_data.get("description", ""),
                        expected_outputs=cmd_data.get("expected_outputs", []),
                        error_patterns=cmd_data.get("error_patterns", []),
                        timeout=cmd_data.get("timeout", 30),
                        retry_on_failure=cmd_data.get("retry_on_failure", True),
                        decision_point=cmd_data.get("decision_point", False),
                        success_indicators=cmd_data.get("success_indicators", []),
                        failure_indicators=cmd_data.get("failure_indicators", [])
                    )
                    commands.append(command)
                
                self.logger.info(f"Successfully parsed {len(commands)} commands from LLM JSON response")
                return commands
            else:
                # Fallback to text parsing
                self.logger.warning("No JSON found in response, falling back to text parsing")
                return self._parse_commands_from_text(response, task, complexity)
                
        except json.JSONDecodeError as e:
            # Fallback to text parsing
            self.logger.warning(f"JSON parsing failed: {e}, falling back to text parsing")
            return self._parse_commands_from_text(response, task, complexity)
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON content from LLM response, handling code blocks."""
        import re
        
        # Try to find JSON in code blocks first
        json_block_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
        match = re.search(json_block_pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find JSON array directly
        json_pattern = r'(\[.*?\])'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        
        # Check if response starts with [ after stripping
        stripped = response.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            return stripped
        
        return None
    
    def _parse_commands_from_text(self, response: str, task: TaskDefinition, complexity: ComplexityAnalysis) -> List[ShellCommand]:
        """Parse commands from text response."""
        commands = []
        lines = response.split('\n')
        
        current_command = None
        for line in lines:
            line = line.strip()
            
            # Look for command patterns
            if line.startswith('$') or line.startswith('> ') or 'command:' in line.lower():
                if current_command:
                    commands.append(current_command)
                
                # Extract command
                cmd = line.replace('$', '').replace('> ', '').replace('command:', '').strip()
                if cmd:
                    current_command = ShellCommand(
                        command=cmd,
                        description=f"Execute: {cmd}",
                        timeout=30,
                        retry_on_failure=True
                    )
        
        if current_command:
            commands.append(current_command)
        
        # If no commands found, create basic commands based on task
        if not commands:
            commands = self._generate_basic_commands(task, complexity)
        
        return commands
    
    def _parse_decision_point(self, response: str, index: int) -> Optional[DecisionPoint]:
        """Parse decision point from LLM response."""
        try:
            if response.strip().startswith('{'):
                data = json.loads(response)
                return DecisionPoint(
                    condition=data.get("condition", ""),
                    true_path=data.get("true_path", []),
                    false_path=data.get("false_path", []),
                    evaluation_method=data.get("evaluation_method", "exit_code"),
                    description=data.get("description", "")
                )
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _parse_success_criteria(self, response: str) -> List[str]:
        """Parse success criteria from LLM response."""
        try:
            if response.strip().startswith('['):
                return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Fallback to text parsing
        criteria = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or 'criterion' in line.lower()):
                criteria.append(line.replace('-', '').replace('*', '').strip())
        
        return criteria if criteria else ["Task execution completed without errors"]
    
    def _parse_fallback_strategies(self, response: str) -> List[str]:
        """Parse fallback strategies from LLM response."""
        try:
            if response.strip().startswith('['):
                return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Fallback to text parsing
        strategies = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or 'strategy' in line.lower()):
                strategies.append(line.replace('-', '').replace('*', '').strip())
        
        return strategies if strategies else ["Retry with simplified approach"]
    
    # Private helper methods for utilities
    
    def _get_context_summary(self) -> str:
        """Get summary of current context for prompts."""
        context_parts = []
        
        if hasattr(self, 'context') and self.context:
            if 'requirements' in self.context:
                context_parts.append("Requirements document is available")
            if 'design' in self.context:
                context_parts.append("Design document is available")
            if 'project_structure' in self.context:
                context_parts.append("Project structure is analyzed")
        
        if context_parts:
            return f"Context Available: {', '.join(context_parts)}"
        else:
            return "Context: Limited context available"
    
    def _generate_basic_commands(self, task: TaskDefinition, complexity: ComplexityAnalysis) -> List[ShellCommand]:
        """Generate basic commands when parsing fails."""
        commands = []
        
        # Create basic commands based on task description
        if "create" in task.description.lower() or "implement" in task.description.lower():
            commands.append(ShellCommand(
                command="echo 'Starting task implementation'",
                description="Initialize task execution",
                timeout=5
            ))
        
        if "test" in task.description.lower():
            commands.append(ShellCommand(
                command="echo 'Running tests'",
                description="Execute tests",
                timeout=60
            ))
        
        # Add completion command
        commands.append(ShellCommand(
            command="echo 'Task completed'",
            description="Mark task as completed",
            timeout=5
        ))
        
        return commands
    
    def _estimate_duration(self, commands: List[ShellCommand], complexity: ComplexityAnalysis) -> int:
        """Estimate execution duration in minutes."""
        base_duration = len(commands) * 2  # 2 minutes per command base
        
        # Adjust based on complexity
        complexity_multipliers = {
            "simple": 0.5,
            "moderate": 1.0,
            "complex": 1.5,
            "very_complex": 2.0
        }
        
        multiplier = complexity_multipliers.get(complexity.complexity_level, 1.0)
        return max(5, int(base_duration * multiplier))
    
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