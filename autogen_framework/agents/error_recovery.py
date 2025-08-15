"""
ErrorRecovery Agent for the AutoGen multi-agent framework.

This module contains the ErrorRecovery class which is responsible for:
- Intelligent error analysis and categorization using LLM capabilities
- Multi-strategy retry system with alternative approaches
- Recovery pattern learning and storage for future use
- Context-aware error recovery using project requirements and design
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .base_agent import BaseLLMAgent, ContextSpec
from ..models import LLMConfig


class ErrorType(Enum):
    """Enumeration of error types for classification."""
    DEPENDENCY_MISSING = "dependency_missing"
    PERMISSION_DENIED = "permission_denied"
    SYNTAX_ERROR = "syntax_error"
    ENVIRONMENT_ISSUE = "environment_issue"
    FILE_NOT_FOUND = "file_not_found"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


@dataclass
class CommandResult:
    """Result of shell command execution."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    success: bool
    error_type: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        """Set timestamp after initialization."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ErrorAnalysis:
    """Analysis of command failure."""
    error_type: ErrorType
    root_cause: str
    affected_components: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    analysis_reasoning: str = ""
    error_patterns: List[str] = field(default_factory=list)
    context_factors: List[str] = field(default_factory=list)
    severity: str = "medium"  # "low", "medium", "high", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type.value,
            "root_cause": self.root_cause,
            "affected_components": self.affected_components,
            "suggested_fixes": self.suggested_fixes,
            "confidence": self.confidence,
            "analysis_reasoning": self.analysis_reasoning,
            "error_patterns": self.error_patterns,
            "context_factors": self.context_factors,
            "severity": self.severity
        }


@dataclass
class RecoveryStrategy:
    """Alternative approach for error recovery."""
    name: str
    description: str
    commands: List[str] = field(default_factory=list)
    success_probability: float = 0.0
    resource_cost: int = 1  # 1-5 scale
    prerequisites: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    fallback_strategy: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "commands": self.commands,
            "success_probability": self.success_probability,
            "resource_cost": self.resource_cost,
            "prerequisites": self.prerequisites,
            "expected_outcome": self.expected_outcome,
            "fallback_strategy": self.fallback_strategy
        }


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    strategy_used: Optional[RecoveryStrategy] = None
    error: Optional[str] = None
    attempted_strategies: List[RecoveryStrategy] = field(default_factory=list)
    execution_results: List[CommandResult] = field(default_factory=list)
    recovery_time: float = 0.0
    lessons_learned: List[str] = field(default_factory=list)


class ErrorRecovery(BaseLLMAgent):
    """
    AI agent responsible for intelligent error recovery in the multi-agent framework.
    
    The ErrorRecovery agent analyzes command failures and generates alternative approaches
    to achieve the same objectives. It uses LLM capabilities for error analysis and
    strategy generation, while maintaining a learning system for pattern recognition.
    
    Key capabilities:
    - Categorize error types and root causes using LLM analysis
    - Generate ranked alternative strategies using LLM capabilities
    - Manage strategy switching and learning through pattern storage
    - Record successful recovery patterns for future use
    - Context-aware error recovery using project requirements and design
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
        Initialize the ErrorRecovery agent.
        
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
            description=description or "Error recovery agent for intelligent failure analysis and recovery"
        )
        
        # Load error patterns and recovery strategies
        self.error_patterns = self._load_error_patterns()
        self.recovery_patterns = self._load_recovery_patterns()
        self.strategy_history = []
        
        # Statistics tracking
        self.recovery_stats = {
            'total_recoveries_attempted': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'strategies_learned': 0,
            'most_common_errors': {},
            'most_successful_strategies': {}
        }
        
        self.logger.info(f"ErrorRecovery initialized with {len(self.error_patterns)} error patterns")
    
    def get_context_requirements(self, task_input: Dict[str, Any]) -> Optional[ContextSpec]:
        """Define context requirements for ErrorRecovery."""
        if task_input.get("failed_result") or task_input.get("task"):
            return ContextSpec(context_type="implementation")
        return None
    
    async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an error recovery request.
        
        Args:
            task_input: Dictionary containing error information and context
            
        Returns:
            Dictionary containing recovery results and strategies
        """
        task_type = task_input.get("task_type", "recover_from_error")
        
        if task_type == "recover_from_error":
            return await self._handle_error_recovery(task_input)
        elif task_type == "analyze_error":
            return await self._handle_error_analysis(task_input)
        elif task_type == "generate_strategies":
            return await self._handle_strategy_generation(task_input)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def get_agent_capabilities(self) -> List[str]:
        """
        Get a list of capabilities that this agent provides.
        
        Returns:
            List of capability descriptions
        """
        return [
            "Categorize error types and root causes using LLM error analysis",
            "Generate ranked alternative recovery strategies using LLM capabilities",
            "Manage strategy switching and learning through pattern storage",
            "Record successful recovery patterns for future use",
            "Context-aware error recovery using project requirements and design",
            "Analyze command failures with confidence scoring",
            "Learn from recovery attempts to improve future performance"
        ]
    
    async def recover(self, failed_result: CommandResult, execution_plan: Optional[Dict[str, Any]] = None) -> RecoveryResult:
        """
        Analyzes failure and generates alternative approaches.
        
        Args:
            failed_result: Result of failed command
            execution_plan: Optional current execution plan context
            
        Returns:
            RecoveryResult with alternative strategies and outcomes
        """
        self.logger.info(f"Starting error recovery for command: {failed_result.command}")
        start_time = datetime.now()
        
        try:
            self.recovery_stats['total_recoveries_attempted'] += 1
            
            # Step 1: Categorize error type and analyze root cause
            error_analysis = await self._analyze_error(failed_result)
            
            # Step 2: Generate ranked alternative strategies
            strategies = await self._generate_strategies(error_analysis, execution_plan)
            
            # Step 3: Try strategies in order of likelihood
            recovery_result = await self._execute_recovery_strategies(strategies, failed_result, error_analysis)
            
            # Step 4: Learn from the recovery attempt
            await self._learn_from_recovery(recovery_result, error_analysis)
            
            # Calculate recovery time
            recovery_result.recovery_time = (datetime.now() - start_time).total_seconds()
            
            if recovery_result.success:
                self.recovery_stats['successful_recoveries'] += 1
                self.logger.info(f"Error recovery successful using strategy: {recovery_result.strategy_used.name}")
            else:
                self.recovery_stats['failed_recoveries'] += 1
                self.logger.warning(f"Error recovery failed after trying {len(recovery_result.attempted_strategies)} strategies")
            
            return recovery_result
            
        except Exception as e:
            self.logger.error(f"Error during recovery process: {e}")
            return RecoveryResult(
                success=False,
                error=f"Recovery process failed: {str(e)}",
                recovery_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _analyze_error(self, result: CommandResult) -> ErrorAnalysis:
        """
        Categorizes error type and determines root cause using LLM analysis.
        
        Args:
            result: CommandResult with failure information
            
        Returns:
            ErrorAnalysis with detailed assessment
        """
        self.logger.debug(f"Analyzing error for command: {result.command}")
        
        # Step 1: Pattern-based classification
        pattern_analysis = self._classify_error_by_patterns(result)
        
        # Step 2: LLM-based analysis for deeper understanding
        llm_analysis = await self._analyze_error_with_llm(result, pattern_analysis)
        
        # Step 3: Combine pattern and LLM analysis
        final_analysis = self._combine_error_analyses(pattern_analysis, llm_analysis, result)
        
        # Update error statistics
        error_type_str = final_analysis.error_type.value
        if error_type_str in self.recovery_stats['most_common_errors']:
            self.recovery_stats['most_common_errors'][error_type_str] += 1
        else:
            self.recovery_stats['most_common_errors'][error_type_str] = 1
        
        self.logger.info(f"Error analysis completed: {final_analysis.error_type.value} (confidence: {final_analysis.confidence:.2f})")
        return final_analysis
    
    def _classify_error_by_patterns(self, result: CommandResult) -> ErrorAnalysis:
        """
        Classify error using predefined patterns.
        
        Args:
            result: CommandResult with failure information
            
        Returns:
            ErrorAnalysis based on pattern matching
        """
        stderr_lower = result.stderr.lower()
        stdout_lower = result.stdout.lower()
        combined_output = f"{stderr_lower} {stdout_lower}"
        
        # Check each error pattern
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_output, re.IGNORECASE):
                    return ErrorAnalysis(
                        error_type=ErrorType(error_type),
                        root_cause=f"Pattern match: {pattern}",
                        confidence=0.7,  # Pattern matching has good confidence
                        error_patterns=[pattern],
                        analysis_reasoning="Classified using predefined error patterns"
                    )
        
        # Default classification for unknown errors
        return ErrorAnalysis(
            error_type=ErrorType.UNKNOWN,
            root_cause=f"Exit code {result.exit_code}: {result.stderr[:200]}",
            confidence=0.3,  # Low confidence for unknown errors
            analysis_reasoning="No matching error patterns found"
        )
    
    async def _analyze_error_with_llm(self, result: CommandResult, pattern_analysis: ErrorAnalysis) -> ErrorAnalysis:
        """
        Analyze error using LLM for deeper understanding.
        
        Args:
            result: CommandResult with failure information
            pattern_analysis: Initial pattern-based analysis
            
        Returns:
            ErrorAnalysis enhanced with LLM insights
        """
        # Build error analysis prompt
        analysis_prompt = self._build_error_analysis_prompt(result, pattern_analysis)
        
        # Get LLM analysis
        analysis_response = await self.generate_response(analysis_prompt)
        
        # Parse the LLM response into structured analysis
        llm_analysis = self._parse_error_analysis(analysis_response, result)
        
        return llm_analysis
    
    def _combine_error_analyses(self, pattern_analysis: ErrorAnalysis, llm_analysis: ErrorAnalysis, result: CommandResult) -> ErrorAnalysis:
        """
        Combine pattern-based and LLM-based error analyses.
        
        Args:
            pattern_analysis: Analysis from pattern matching
            llm_analysis: Analysis from LLM
            result: Original command result
            
        Returns:
            Combined ErrorAnalysis with best insights from both
        """
        # Use LLM analysis as base if it has higher confidence
        if llm_analysis.confidence > pattern_analysis.confidence:
            final_analysis = llm_analysis
            # Add pattern information if available
            if pattern_analysis.error_patterns:
                final_analysis.error_patterns.extend(pattern_analysis.error_patterns)
        else:
            final_analysis = pattern_analysis
            # Enhance with LLM insights
            if llm_analysis.suggested_fixes:
                final_analysis.suggested_fixes.extend(llm_analysis.suggested_fixes)
            if llm_analysis.context_factors:
                final_analysis.context_factors.extend(llm_analysis.context_factors)
        
        # Ensure we have basic information
        if not final_analysis.root_cause:
            final_analysis.root_cause = f"Command failed with exit code {result.exit_code}"
        
        if not final_analysis.suggested_fixes:
            final_analysis.suggested_fixes = ["Retry with modified approach", "Check command syntax and parameters"]
        
        return final_analysis
    
    async def _generate_strategies(self, error_analysis: ErrorAnalysis, execution_plan: Optional[Dict[str, Any]] = None) -> List[RecoveryStrategy]:
        """
        Generates ranked alternative approaches based on error analysis.
        
        Args:
            error_analysis: Analysis of the error
            execution_plan: Optional execution plan context
            
        Returns:
            List of RecoveryStrategy objects ranked by success probability
        """
        self.logger.debug(f"Generating recovery strategies for {error_analysis.error_type.value}")
        
        # Step 1: Get template strategies for this error type
        template_strategies = self._get_template_strategies(error_analysis.error_type)
        
        # Step 2: Generate LLM-based strategies
        llm_strategies = await self._generate_strategies_with_llm(error_analysis, execution_plan)
        
        # Step 3: Combine and rank strategies
        all_strategies = template_strategies + llm_strategies
        ranked_strategies = self._rank_strategies(all_strategies, error_analysis)
        
        self.logger.info(f"Generated {len(ranked_strategies)} recovery strategies")
        return ranked_strategies
    
    def _get_template_strategies(self, error_type: ErrorType) -> List[RecoveryStrategy]:
        """
        Get template recovery strategies for specific error types.
        
        Args:
            error_type: Type of error encountered
            
        Returns:
            List of template RecoveryStrategy objects
        """
        template_strategies = {
            ErrorType.DEPENDENCY_MISSING: [
                RecoveryStrategy(
                    name="install_with_pip",
                    description="Install missing Python package using pip",
                    commands=["pip install {package_name}"],
                    success_probability=0.8,
                    resource_cost=2
                ),
                RecoveryStrategy(
                    name="install_with_system_package_manager",
                    description="Install missing system package",
                    commands=["sudo apt-get update", "sudo apt-get install -y {package_name}"],
                    success_probability=0.7,
                    resource_cost=3
                )
            ],
            ErrorType.PERMISSION_DENIED: [
                RecoveryStrategy(
                    name="use_sudo",
                    description="Retry command with sudo privileges",
                    commands=["sudo {original_command}"],
                    success_probability=0.6,
                    resource_cost=2
                ),
                RecoveryStrategy(
                    name="change_permissions",
                    description="Change file permissions and retry",
                    commands=["chmod +x {file_path}", "{original_command}"],
                    success_probability=0.7,
                    resource_cost=2
                )
            ],
            ErrorType.FILE_NOT_FOUND: [
                RecoveryStrategy(
                    name="create_missing_file",
                    description="Create missing file or directory",
                    commands=["mkdir -p {directory_path}", "touch {file_path}"],
                    success_probability=0.5,
                    resource_cost=1
                ),
                RecoveryStrategy(
                    name="use_alternative_path",
                    description="Try alternative file path",
                    commands=["{command_with_alternative_path}"],
                    success_probability=0.4,
                    resource_cost=1
                )
            ],
            ErrorType.SYNTAX_ERROR: [
                RecoveryStrategy(
                    name="fix_syntax_automatically",
                    description="Attempt automatic syntax correction",
                    commands=["{corrected_command}"],
                    success_probability=0.6,
                    resource_cost=1
                )
            ]
        }
        
        return template_strategies.get(error_type, [])
    
    async def _generate_strategies_with_llm(self, error_analysis: ErrorAnalysis, execution_plan: Optional[Dict[str, Any]] = None) -> List[RecoveryStrategy]:
        """
        Generate recovery strategies using LLM capabilities.
        
        Args:
            error_analysis: Analysis of the error
            execution_plan: Optional execution plan context
            
        Returns:
            List of LLM-generated RecoveryStrategy objects
        """
        # Build strategy generation prompt
        strategy_prompt = self._build_strategy_generation_prompt(error_analysis, execution_plan)
        
        # Get LLM-generated strategies
        strategies_response = await self.generate_response(strategy_prompt)
        
        # Parse strategies from response
        strategies = self._parse_recovery_strategies(strategies_response)
        
        return strategies
    
    def _rank_strategies(self, strategies: List[RecoveryStrategy], error_analysis: ErrorAnalysis) -> List[RecoveryStrategy]:
        """
        Rank recovery strategies by success probability and other factors.
        
        Args:
            strategies: List of recovery strategies
            error_analysis: Error analysis for context
            
        Returns:
            List of strategies ranked by likelihood of success
        """
        # Adjust success probabilities based on error analysis confidence
        for strategy in strategies:
            # Higher confidence in error analysis increases strategy success probability
            confidence_boost = error_analysis.confidence * 0.2
            strategy.success_probability = min(1.0, strategy.success_probability + confidence_boost)
            
            # Adjust based on historical success rates
            if strategy.name in self.recovery_stats.get('most_successful_strategies', {}):
                historical_success = self.recovery_stats['most_successful_strategies'][strategy.name]
                strategy.success_probability = (strategy.success_probability + historical_success) / 2
        
        # Sort by success probability (descending) and resource cost (ascending)
        return sorted(strategies, key=lambda s: (-s.success_probability, s.resource_cost))
    
    async def _execute_recovery_strategies(self, strategies: List[RecoveryStrategy], failed_result: CommandResult, error_analysis: ErrorAnalysis) -> RecoveryResult:
        """
        Execute recovery strategies in order until one succeeds.
        
        Args:
            strategies: List of ranked recovery strategies
            failed_result: Original failed command result
            error_analysis: Analysis of the original error
            
        Returns:
            RecoveryResult with outcome of recovery attempts
        """
        attempted_strategies = []
        execution_results = []
        
        for strategy in strategies:
            self.logger.info(f"Attempting recovery strategy: {strategy.name}")
            attempted_strategies.append(strategy)
            
            try:
                # Execute strategy commands (this would integrate with ShellExecutor in real implementation)
                strategy_result = await self._execute_strategy(strategy, failed_result)
                execution_results.append(strategy_result)
                
                if strategy_result.success:
                    # Strategy succeeded
                    self.logger.info(f"Recovery strategy '{strategy.name}' succeeded")
                    return RecoveryResult(
                        success=True,
                        strategy_used=strategy,
                        attempted_strategies=attempted_strategies,
                        execution_results=execution_results,
                        lessons_learned=[f"Strategy '{strategy.name}' works for {error_analysis.error_type.value} errors"]
                    )
                else:
                    self.logger.debug(f"Recovery strategy '{strategy.name}' failed: {strategy_result.stderr}")
                    
            except Exception as e:
                self.logger.warning(f"Error executing recovery strategy '{strategy.name}': {e}")
                # Create a failed result for this strategy
                strategy_result = CommandResult(
                    command=f"Strategy: {strategy.name}",
                    exit_code=1,
                    stdout="",
                    stderr=str(e),
                    execution_time=0.0,
                    success=False
                )
                execution_results.append(strategy_result)
        
        # All strategies failed
        return RecoveryResult(
            success=False,
            error=f"All {len(strategies)} recovery strategies failed",
            attempted_strategies=attempted_strategies,
            execution_results=execution_results,
            lessons_learned=[f"No effective strategy found for {error_analysis.error_type.value} errors"]
        )
    
    async def _execute_strategy(self, strategy: RecoveryStrategy, failed_result: CommandResult) -> CommandResult:
        """
        Execute a single recovery strategy.
        
        Note: This is a placeholder implementation. In the real system, this would
        integrate with the ShellExecutor to actually run the commands.
        
        Args:
            strategy: Recovery strategy to execute
            failed_result: Original failed command result
            
        Returns:
            CommandResult from strategy execution
        """
        # Placeholder implementation - in real system this would use ShellExecutor
        self.logger.debug(f"Executing strategy commands: {strategy.commands}")
        
        # Simulate strategy execution based on strategy type
        if strategy.success_probability > 0.7:
            # High probability strategies are more likely to succeed
            return CommandResult(
                command=f"Recovery: {strategy.name}",
                exit_code=0,
                stdout=f"Strategy {strategy.name} executed successfully",
                stderr="",
                execution_time=1.0,
                success=True
            )
        else:
            # Lower probability strategies may fail
            return CommandResult(
                command=f"Recovery: {strategy.name}",
                exit_code=1,
                stdout="",
                stderr=f"Strategy {strategy.name} failed to resolve the issue",
                execution_time=1.0,
                success=False
            )
    
    async def _learn_from_recovery(self, recovery_result: RecoveryResult, error_analysis: ErrorAnalysis) -> None:
        """
        Learn from recovery attempt to improve future performance.
        
        Args:
            recovery_result: Result of the recovery attempt
            error_analysis: Analysis of the original error
        """
        if recovery_result.success and recovery_result.strategy_used:
            # Record successful strategy
            strategy_name = recovery_result.strategy_used.name
            if strategy_name in self.recovery_stats['most_successful_strategies']:
                # Update success rate (simple moving average)
                current_rate = self.recovery_stats['most_successful_strategies'][strategy_name]
                self.recovery_stats['most_successful_strategies'][strategy_name] = (current_rate + 1.0) / 2
            else:
                self.recovery_stats['most_successful_strategies'][strategy_name] = 1.0
            
            # Store successful pattern for future use
            pattern = {
                'error_type': error_analysis.error_type.value,
                'strategy': recovery_result.strategy_used.to_dict(),
                'context': error_analysis.context_factors,
                'timestamp': datetime.now().isoformat()
            }
            self.recovery_patterns.append(pattern)
            self.recovery_stats['strategies_learned'] += 1
            
            self.logger.info(f"Learned new recovery pattern for {error_analysis.error_type.value} errors")
        
        # Store lessons learned
        if recovery_result.lessons_learned:
            for lesson in recovery_result.lessons_learned:
                self.logger.info(f"Lesson learned: {lesson}")
    
    # Private helper methods for task handling
    
    async def _handle_error_recovery(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error recovery request."""
        failed_result = task_input["failed_result"]
        execution_plan = task_input.get("execution_plan")
        
        try:
            recovery_result = await self.recover(failed_result, execution_plan)
            
            return {
                "success": recovery_result.success,
                "recovery_result": recovery_result,
                "strategy_used": recovery_result.strategy_used.to_dict() if recovery_result.strategy_used else None,
                "strategies_attempted": len(recovery_result.attempted_strategies),
                "recovery_time": recovery_result.recovery_time,
                "lessons_learned": recovery_result.lessons_learned
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_error_analysis(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error analysis request."""
        failed_result = task_input["failed_result"]
        
        try:
            error_analysis = await self._analyze_error(failed_result)
            
            return {
                "success": True,
                "error_analysis": error_analysis.to_dict()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_strategy_generation(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle strategy generation request."""
        error_analysis_dict = task_input["error_analysis"]
        execution_plan = task_input.get("execution_plan")
        
        try:
            # Convert dict back to ErrorAnalysis object
            error_analysis = ErrorAnalysis(
                error_type=ErrorType(error_analysis_dict["error_type"]),
                root_cause=error_analysis_dict["root_cause"],
                affected_components=error_analysis_dict.get("affected_components", []),
                suggested_fixes=error_analysis_dict.get("suggested_fixes", []),
                confidence=error_analysis_dict.get("confidence", 0.5),
                analysis_reasoning=error_analysis_dict.get("analysis_reasoning", ""),
                error_patterns=error_analysis_dict.get("error_patterns", []),
                context_factors=error_analysis_dict.get("context_factors", []),
                severity=error_analysis_dict.get("severity", "medium")
            )
            
            strategies = await self._generate_strategies(error_analysis, execution_plan)
            
            return {
                "success": True,
                "strategies": [strategy.to_dict() for strategy in strategies],
                "strategy_count": len(strategies)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }    

    # Private helper methods for prompt building
    
    def _build_error_analysis_prompt(self, result: CommandResult, pattern_analysis: ErrorAnalysis) -> str:
        """Build prompt for error analysis."""
        context_info = self._get_context_summary()
        
        prompt = f"""
Analyze the following command failure and provide detailed error analysis.

Command Information:
- Command: {result.command}
- Exit Code: {result.exit_code}
- Execution Time: {result.execution_time}s
- Timestamp: {result.timestamp}

Output Information:
- STDOUT: {result.stdout}
- STDERR: {result.stderr}

Initial Pattern Analysis:
- Error Type: {pattern_analysis.error_type.value}
- Root Cause: {pattern_analysis.root_cause}
- Confidence: {pattern_analysis.confidence}

{context_info}

Please provide a comprehensive error analysis including:

1. Error Type: Choose from dependency_missing, permission_denied, syntax_error, environment_issue, file_not_found, network_error, timeout_error, resource_exhausted, configuration_error, unknown
2. Root Cause: Detailed explanation of what caused the failure
3. Affected Components: List of system components or files affected
4. Suggested Fixes: Specific actions that could resolve the issue
5. Confidence: Your confidence in this analysis (0.0-1.0)
6. Analysis Reasoning: Explanation of your assessment
7. Error Patterns: Specific patterns in the error output
8. Context Factors: Environmental or contextual factors that may have contributed
9. Severity: Choose from low, medium, high, critical

Format your response as JSON:
{{
    "error_type": "...",
    "root_cause": "...",
    "affected_components": [...],
    "suggested_fixes": [...],
    "confidence": ...,
    "analysis_reasoning": "...",
    "error_patterns": [...],
    "context_factors": [...],
    "severity": "..."
}}
"""
        return prompt
    
    def _build_strategy_generation_prompt(self, error_analysis: ErrorAnalysis, execution_plan: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for recovery strategy generation."""
        context_info = self._get_context_summary()
        plan_info = ""
        if execution_plan:
            plan_info = f"\nExecution Plan Context: {json.dumps(execution_plan, indent=2)}"
        
        prompt = f"""
Generate recovery strategies for the following error analysis.

Error Analysis:
- Error Type: {error_analysis.error_type.value}
- Root Cause: {error_analysis.root_cause}
- Affected Components: {error_analysis.affected_components}
- Suggested Fixes: {error_analysis.suggested_fixes}
- Confidence: {error_analysis.confidence}
- Severity: {error_analysis.severity}
- Context Factors: {error_analysis.context_factors}

{context_info}{plan_info}

Generate alternative recovery strategies that:
1. Address the specific error type and root cause
2. Are practical and executable
3. Have different approaches (don't just retry the same thing)
4. Consider the context and environment
5. Are ranked by likelihood of success

For each strategy, provide:
- name: Short identifier for the strategy
- description: What this strategy does
- commands: List of shell commands to execute
- success_probability: Estimated probability of success (0.0-1.0)
- resource_cost: Resource cost on scale 1-5 (1=low, 5=high)
- prerequisites: Any prerequisites needed
- expected_outcome: What should happen if successful
- fallback_strategy: Whether this is a fallback approach

Format your response as JSON array:
[
    {{
        "name": "...",
        "description": "...",
        "commands": [...],
        "success_probability": ...,
        "resource_cost": ...,
        "prerequisites": [...],
        "expected_outcome": "...",
        "fallback_strategy": ...
    }},
    ...
]
"""
        return prompt
    
    # Private helper methods for parsing LLM responses
    
    def _parse_error_analysis(self, response: str, result: CommandResult) -> ErrorAnalysis:
        """Parse error analysis from LLM response."""
        try:
            # Extract JSON from response
            json_content = self._extract_json_from_response(response)
            
            if json_content:
                data = json.loads(json_content)
                return ErrorAnalysis(
                    error_type=ErrorType(data.get("error_type", "unknown")),
                    root_cause=data.get("root_cause", f"Command failed with exit code {result.exit_code}"),
                    affected_components=data.get("affected_components", []),
                    suggested_fixes=data.get("suggested_fixes", []),
                    confidence=data.get("confidence", 0.5),
                    analysis_reasoning=data.get("analysis_reasoning", ""),
                    error_patterns=data.get("error_patterns", []),
                    context_factors=data.get("context_factors", []),
                    severity=data.get("severity", "medium")
                )
            else:
                # Fallback to text parsing
                return self._parse_error_analysis_from_text(response, result)
                
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse error analysis JSON: {e}, falling back to text parsing")
            return self._parse_error_analysis_from_text(response, result)
    
    def _parse_error_analysis_from_text(self, response: str, result: CommandResult) -> ErrorAnalysis:
        """Parse error analysis from text response."""
        # Simple text-based parsing as fallback
        response_lower = response.lower()
        
        # Determine error type from text
        error_type = ErrorType.UNKNOWN
        if any(word in response_lower for word in ["permission", "denied", "access"]):
            error_type = ErrorType.PERMISSION_DENIED
        elif any(word in response_lower for word in ["not found", "missing", "no such"]):
            error_type = ErrorType.FILE_NOT_FOUND
        elif any(word in response_lower for word in ["syntax", "invalid", "unexpected"]):
            error_type = ErrorType.SYNTAX_ERROR
        elif any(word in response_lower for word in ["dependency", "module", "package"]):
            error_type = ErrorType.DEPENDENCY_MISSING
        
        return ErrorAnalysis(
            error_type=error_type,
            root_cause=f"Command failed with exit code {result.exit_code}",
            confidence=0.4,  # Lower confidence for text parsing
            analysis_reasoning="Parsed from text response due to JSON parsing failure"
        )
    
    def _parse_recovery_strategies(self, response: str) -> List[RecoveryStrategy]:
        """Parse recovery strategies from LLM response."""
        try:
            # Extract JSON from response
            json_content = self._extract_json_from_response(response)
            
            if json_content:
                data = json.loads(json_content)
                strategies = []
                
                for strategy_data in data:
                    strategy = RecoveryStrategy(
                        name=strategy_data.get("name", "unknown_strategy"),
                        description=strategy_data.get("description", ""),
                        commands=strategy_data.get("commands", []),
                        success_probability=strategy_data.get("success_probability", 0.5),
                        resource_cost=strategy_data.get("resource_cost", 3),
                        prerequisites=strategy_data.get("prerequisites", []),
                        expected_outcome=strategy_data.get("expected_outcome", ""),
                        fallback_strategy=strategy_data.get("fallback_strategy", False)
                    )
                    strategies.append(strategy)
                
                return strategies
            else:
                # Fallback to text parsing
                return self._parse_strategies_from_text(response)
                
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse strategies JSON: {e}, falling back to text parsing")
            return self._parse_strategies_from_text(response)
    
    def _parse_strategies_from_text(self, response: str) -> List[RecoveryStrategy]:
        """Parse strategies from text response."""
        strategies = []
        lines = response.split('\n')
        
        current_strategy = None
        for line in lines:
            line = line.strip()
            
            # Look for strategy patterns
            if any(keyword in line.lower() for keyword in ["strategy", "approach", "solution"]):
                if current_strategy:
                    strategies.append(current_strategy)
                
                # Create new strategy
                current_strategy = RecoveryStrategy(
                    name=f"strategy_{len(strategies) + 1}",
                    description=line,
                    success_probability=0.5,
                    resource_cost=3
                )
            elif current_strategy and line.startswith('-'):
                # Add command to current strategy
                command = line.replace('-', '').strip()
                if command:
                    current_strategy.commands.append(command)
        
        if current_strategy:
            strategies.append(current_strategy)
        
        # If no strategies found, create a basic retry strategy
        if not strategies:
            strategies.append(RecoveryStrategy(
                name="basic_retry",
                description="Retry the original command with slight modifications",
                commands=["# Retry original command"],
                success_probability=0.3,
                resource_cost=1
            ))
        
        return strategies
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON content from LLM response, handling code blocks."""
        import re
        
        # Try to find JSON in code blocks first
        json_block_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        match = re.search(json_block_pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find JSON object or array directly
        json_pattern = r'(\{.*?\}|\[.*?\])'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        
        # Check if response starts with { or [ after stripping
        stripped = response.strip()
        if (stripped.startswith('{') and stripped.endswith('}')) or (stripped.startswith('[') and stripped.endswith(']')):
            return stripped
        
        return None
    
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
            if 'execution_history' in self.context:
                context_parts.append("Execution history is available")
        
        if context_parts:
            return f"Context Available: {', '.join(context_parts)}"
        else:
            return "Context: Limited context available"
    
    def _load_error_patterns(self) -> Dict[str, List[str]]:
        """Load predefined error patterns for classification."""
        return {
            "dependency_missing": [
                r'command not found',
                r'No module named',
                r'package .* not found',
                r'Could not find a version',
                r'ModuleNotFoundError',
                r'ImportError',
                r'cannot import name',
                r'pip: command not found',
                r'python: command not found'
            ],
            "permission_denied": [
                r'Permission denied',
                r'Access is denied',
                r'Operation not permitted',
                r'permission denied',
                r'access denied',
                r'insufficient privileges',
                r'sudo required'
            ],
            "syntax_error": [
                r'SyntaxError',
                r'invalid syntax',
                r'unexpected token',
                r'syntax error',
                r'parse error',
                r'malformed',
                r'invalid command'
            ],
            "environment_issue": [
                r'environment variable .* not set',
                r'PATH .* not found',
                r'virtual environment not activated',
                r'PYTHONPATH',
                r'environment not found',
                r'shell not found'
            ],
            "file_not_found": [
                r'No such file or directory',
                r'file not found',
                r'cannot find',
                r'does not exist',
                r'FileNotFoundError',
                r'not found'
            ],
            "network_error": [
                r'connection refused',
                r'network unreachable',
                r'timeout',
                r'connection timed out',
                r'DNS resolution failed',
                r'unable to connect'
            ],
            "timeout_error": [
                r'timeout',
                r'timed out',
                r'operation timeout',
                r'deadline exceeded'
            ],
            "resource_exhausted": [
                r'out of memory',
                r'disk full',
                r'no space left',
                r'resource temporarily unavailable',
                r'too many open files'
            ],
            "configuration_error": [
                r'configuration error',
                r'config not found',
                r'invalid configuration',
                r'misconfigured',
                r'settings error'
            ]
        }
    
    def _load_recovery_patterns(self) -> List[Dict[str, Any]]:
        """Load historical recovery patterns."""
        # In a real implementation, this would load from persistent storage
        return []
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get current recovery statistics."""
        return self.recovery_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset recovery statistics."""
        self.recovery_stats = {
            'total_recoveries_attempted': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'strategies_learned': 0,
            'most_common_errors': {},
            'most_successful_strategies': {}
        }
        self.logger.info("Recovery statistics reset")  
  
    # Additional methods for Strategy Generation and Recovery (Task 4.2)
    
    def validate_strategy(self, strategy: RecoveryStrategy, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate a recovery strategy before execution.
        
        Args:
            strategy: Recovery strategy to validate
            context: Optional context for validation
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "recommendations": [],
            "risk_level": "low"
        }
        
        # Check if commands are valid
        for command in strategy.commands:
            if not command or command.strip() == "":
                validation_result["valid"] = False
                validation_result["issues"].append("Empty command found")
            
            # Check for potentially dangerous commands
            dangerous_patterns = ["rm -rf", "sudo rm", "format", "mkfs", "dd if="]
            for pattern in dangerous_patterns:
                if pattern in command.lower():
                    validation_result["risk_level"] = "high"
                    validation_result["recommendations"].append(f"High-risk command detected: {pattern}")
        
        # Check prerequisites
        if strategy.prerequisites:
            validation_result["recommendations"].append(f"Ensure prerequisites are met: {', '.join(strategy.prerequisites)}")
        
        # Validate success probability
        if strategy.success_probability < 0.1:
            validation_result["recommendations"].append("Very low success probability - consider alternative strategies")
        elif strategy.success_probability > 0.9:
            validation_result["recommendations"].append("Very high success probability - verify assumptions")
        
        return validation_result
    
    async def learn_from_execution_history(self, execution_history: List[Dict[str, Any]]) -> None:
        """
        Learn from execution history to improve strategy generation.
        
        Args:
            execution_history: List of execution results with outcomes
        """
        patterns_learned = 0
        
        for execution in execution_history:
            if execution.get("success") and execution.get("strategy_used"):
                # Extract successful patterns
                strategy = execution["strategy_used"]
                error_type = execution.get("error_type", "unknown")
                
                # Create learning pattern
                pattern = {
                    "error_type": error_type,
                    "strategy_name": strategy.get("name", "unknown"),
                    "commands": strategy.get("commands", []),
                    "success_rate": 1.0,  # This execution was successful
                    "context": execution.get("context", {}),
                    "learned_at": datetime.now().isoformat()
                }
                
                # Check if we already have this pattern
                existing_pattern = None
                for existing in self.recovery_patterns:
                    if (existing.get("error_type") == error_type and 
                        existing.get("strategy_name") == strategy.get("name")):
                        existing_pattern = existing
                        break
                
                if existing_pattern:
                    # Update success rate (moving average)
                    current_rate = existing_pattern.get("success_rate", 0.5)
                    existing_pattern["success_rate"] = (current_rate + 1.0) / 2
                    existing_pattern["last_updated"] = datetime.now().isoformat()
                else:
                    # Add new pattern
                    self.recovery_patterns.append(pattern)
                    patterns_learned += 1
        
        if patterns_learned > 0:
            self.recovery_stats['strategies_learned'] += patterns_learned
            self.logger.info(f"Learned {patterns_learned} new recovery patterns from execution history")
    
    def get_strategy_recommendations(self, error_type: ErrorType, context: Optional[Dict[str, Any]] = None) -> List[RecoveryStrategy]:
        """
        Get strategy recommendations based on learned patterns.
        
        Args:
            error_type: Type of error to get recommendations for
            context: Optional context for filtering recommendations
            
        Returns:
            List of recommended recovery strategies
        """
        recommendations = []
        
        # Find patterns for this error type
        relevant_patterns = [
            pattern for pattern in self.recovery_patterns
            if pattern.get("error_type") == error_type.value
        ]
        
        # Sort by success rate
        relevant_patterns.sort(key=lambda p: p.get("success_rate", 0), reverse=True)
        
        # Convert patterns to strategies
        for pattern in relevant_patterns[:5]:  # Top 5 recommendations
            strategy = RecoveryStrategy(
                name=pattern.get("strategy_name", "learned_strategy"),
                description=f"Learned strategy for {error_type.value} errors",
                commands=pattern.get("commands", []),
                success_probability=pattern.get("success_rate", 0.5),
                resource_cost=2,  # Default cost for learned strategies
                expected_outcome=f"Resolve {error_type.value} error based on historical success"
            )
            recommendations.append(strategy)
        
        return recommendations
    
    async def optimize_strategy_ranking(self, strategies: List[RecoveryStrategy], error_analysis: ErrorAnalysis) -> List[RecoveryStrategy]:
        """
        Optimize strategy ranking using advanced criteria.
        
        Args:
            strategies: List of strategies to rank
            error_analysis: Error analysis for context
            
        Returns:
            Optimally ranked list of strategies
        """
        # Calculate composite scores for each strategy
        for strategy in strategies:
            composite_score = 0.0
            
            # Base success probability (40% weight)
            composite_score += strategy.success_probability * 0.4
            
            # Resource efficiency (20% weight) - lower cost is better
            resource_efficiency = (6 - strategy.resource_cost) / 5  # Normalize to 0-1
            composite_score += resource_efficiency * 0.2
            
            # Historical success rate (25% weight)
            historical_rate = self.recovery_stats.get('most_successful_strategies', {}).get(strategy.name, 0.5)
            composite_score += historical_rate * 0.25
            
            # Context relevance (15% weight)
            context_relevance = self._calculate_context_relevance(strategy, error_analysis)
            composite_score += context_relevance * 0.15
            
            # Store composite score for sorting
            strategy.composite_score = composite_score
        
        # Sort by composite score (descending)
        optimized_strategies = sorted(strategies, key=lambda s: getattr(s, 'composite_score', 0), reverse=True)
        
        # Remove the temporary composite_score attribute
        for strategy in optimized_strategies:
            if hasattr(strategy, 'composite_score'):
                delattr(strategy, 'composite_score')
        
        return optimized_strategies
    
    def _calculate_context_relevance(self, strategy: RecoveryStrategy, error_analysis: ErrorAnalysis) -> float:
        """
        Calculate how relevant a strategy is to the current context.
        
        Args:
            strategy: Recovery strategy to evaluate
            error_analysis: Error analysis for context
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        relevance_score = 0.5  # Base relevance
        
        # Check if strategy addresses specific error patterns
        for pattern in error_analysis.error_patterns:
            if any(pattern.lower() in cmd.lower() for cmd in strategy.commands):
                relevance_score += 0.2
                break
        
        # Check if strategy addresses suggested fixes
        for fix in error_analysis.suggested_fixes:
            if any(fix.lower() in cmd.lower() for cmd in strategy.commands):
                relevance_score += 0.2
                break
        
        # Check if strategy commands are relevant to error context
        error_context_keywords = error_analysis.context_factors + [error_analysis.error_type.value]
        for keyword in error_context_keywords:
            if any(keyword.lower() in cmd.lower() for cmd in strategy.commands):
                relevance_score += 0.1
                break
        
        # Check severity alignment
        if error_analysis.severity == "critical" and strategy.resource_cost <= 2:
            relevance_score -= 0.1  # Low-cost strategies may not be sufficient for critical errors
        elif error_analysis.severity == "low" and strategy.resource_cost >= 4:
            relevance_score -= 0.1  # High-cost strategies may be overkill for low-severity errors
        
        return min(1.0, max(0.0, relevance_score))
    
    async def generate_adaptive_strategy(self, error_analysis: ErrorAnalysis, failed_strategies: List[RecoveryStrategy]) -> Optional[RecoveryStrategy]:
        """
        Generate an adaptive strategy based on what has already failed.
        
        Args:
            error_analysis: Analysis of the original error
            failed_strategies: List of strategies that have already failed
            
        Returns:
            New adaptive strategy or None if no strategy can be generated
        """
        # Build prompt for adaptive strategy generation
        failed_approaches = [f"{s.name}: {s.description}" for s in failed_strategies]
        
        adaptive_prompt = f"""
Generate a new recovery strategy that takes into account previous failures.

Error Analysis:
- Error Type: {error_analysis.error_type.value}
- Root Cause: {error_analysis.root_cause}
- Severity: {error_analysis.severity}

Failed Strategies:
{chr(10).join(f"- {approach}" for approach in failed_approaches)}

Generate a completely different approach that:
1. Avoids the patterns that have already failed
2. Takes a fundamentally different approach to the problem
3. Considers alternative tools or methods
4. May involve breaking down the problem differently

Format your response as JSON:
{{
    "name": "...",
    "description": "...",
    "commands": [...],
    "success_probability": ...,
    "resource_cost": ...,
    "expected_outcome": "...",
    "reasoning": "Why this approach is different from failed attempts"
}}
"""
        
        try:
            response = await self.generate_response(adaptive_prompt)
            json_content = self._extract_json_from_response(response)
            
            if json_content:
                data = json.loads(json_content)
                return RecoveryStrategy(
                    name=data.get("name", "adaptive_strategy"),
                    description=data.get("description", "Adaptive strategy based on failure analysis"),
                    commands=data.get("commands", []),
                    success_probability=data.get("success_probability", 0.3),
                    resource_cost=data.get("resource_cost", 3),
                    expected_outcome=data.get("expected_outcome", "Alternative approach to resolve the error"),
                    fallback_strategy=True
                )
        except Exception as e:
            self.logger.warning(f"Failed to generate adaptive strategy: {e}")
        
        return None
    
    def export_learned_patterns(self) -> Dict[str, Any]:
        """
        Export learned recovery patterns for analysis or backup.
        
        Returns:
            Dictionary containing all learned patterns and statistics
        """
        return {
            "recovery_patterns": self.recovery_patterns,
            "recovery_stats": self.recovery_stats,
            "export_timestamp": datetime.now().isoformat(),
            "total_patterns": len(self.recovery_patterns)
        }
    
    def import_learned_patterns(self, patterns_data: Dict[str, Any]) -> bool:
        """
        Import learned recovery patterns from external source.
        
        Args:
            patterns_data: Dictionary containing patterns and statistics
            
        Returns:
            True if import was successful, False otherwise
        """
        try:
            if "recovery_patterns" in patterns_data:
                # Merge with existing patterns, avoiding duplicates
                imported_count = 0
                for pattern in patterns_data["recovery_patterns"]:
                    # Check for duplicates
                    duplicate = False
                    for existing in self.recovery_patterns:
                        if (existing.get("error_type") == pattern.get("error_type") and
                            existing.get("strategy_name") == pattern.get("strategy_name")):
                            duplicate = True
                            break
                    
                    if not duplicate:
                        self.recovery_patterns.append(pattern)
                        imported_count += 1
                
                self.logger.info(f"Imported {imported_count} new recovery patterns")
            
            if "recovery_stats" in patterns_data:
                # Merge statistics
                imported_stats = patterns_data["recovery_stats"]
                for key, value in imported_stats.items():
                    if key in self.recovery_stats:
                        if isinstance(value, dict):
                            self.recovery_stats[key].update(value)
                        elif isinstance(value, (int, float)):
                            self.recovery_stats[key] += value
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import learned patterns: {e}")
            return False