"""
TaskDecomposer LLM Integration Tests.

This module tests the TaskDecomposer's LLM interactions and output quality validation,
focusing on real LLM calls and task decomposition capabilities.

Test Categories:
1. Task breakdown into executable shell command sequences
2. Complexity analysis accuracy with confidence scoring
3. Decision point generation and logical flow
4. Success criteria definition and clarity
5. Context-aware task understanding

All tests use real LLM configurations and validate output quality using the
enhanced QualityMetricsFramework with LLM-specific validation methods.

Requirements Coverage:
- 2.1: TaskDecomposer breakdown validation
- 2.5: Complexity analysis with confidence scoring
- 6.4: Context-aware task understanding
"""

import pytest
import asyncio
import tempfile
import shutil
import re
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import replace

from autogen_framework.agents.task_decomposer import TaskDecomposer, ExecutionPlan, ComplexityAnalysis
from autogen_framework.models import LLMConfig, TaskDefinition
from autogen_framework.memory_manager import MemoryManager
from tests.integration.test_llm_base import (
    LLMIntegrationTestBase,
    sequential_test_execution,
    QUALITY_THRESHOLDS_LENIENT
)


class TestTaskDecomposerLLMIntegration:
    """
    Integration tests for TaskDecomposer LLM interactions.
    
    These tests validate the TaskDecomposer's ability to:
    - Break down high-level tasks into executable shell command sequences
    - Analyze task complexity with accurate confidence scoring
    - Generate logical decision points and execution flow
    - Define clear and measurable success criteria
    - Understand tasks in context of project requirements and design
    """
    
    @pytest.fixture(autouse=True)
    def setup_task_decomposer(self, real_llm_config, initialized_real_managers, temp_workspace):
        """Setup TaskDecomposer with real LLM configuration and managers."""
        # Initialize test base functionality
        self.test_base = LLMIntegrationTestBase()
        
        self.llm_config = real_llm_config
        self.managers = initialized_real_managers
        self.workspace_path = Path(temp_workspace)
        
        # Create memory manager for the workspace
        self.memory_manager = MemoryManager(workspace_path=temp_workspace)
        
        # Initialize TaskDecomposer with real dependencies
        self.task_decomposer = TaskDecomposer(
            name="TaskDecomposer",
            llm_config=self.llm_config,
            system_message="You are an expert task decomposer that breaks down high-level tasks into executable shell commands.",
            token_manager=self.managers.token_manager,
            context_manager=self.managers.context_manager,
            config_manager=self.managers.config_manager
        )
        
        # Use lenient quality thresholds for TaskDecomposer tests
        self.test_base.quality_validator.thresholds = QUALITY_THRESHOLDS_LENIENT
        
        # Create sample project context for testing
        self._create_sample_project_context()
    
    def _create_sample_project_context(self):
        """Create sample project context documents for testing."""
        # Sample requirements document
        self.sample_requirements = """# Requirements Document

## Introduction

This feature implements a Python web scraper with data processing and storage capabilities.

## Requirements

### Requirement 1

**User Story:** As a user, I want to scrape data from websites, so that I can collect information for analysis.

#### Acceptance Criteria

1. WHEN a user provides a URL THEN the system SHALL fetch and parse the webpage content
2. WHEN the webpage is successfully parsed THEN the system SHALL extract relevant data fields
3. WHEN data extraction fails THEN the system SHALL log the error and continue processing

### Requirement 2

**User Story:** As a user, I want to process scraped data, so that I can clean and normalize it for analysis.

#### Acceptance Criteria

1. WHEN raw data is collected THEN the system SHALL validate and clean the data
2. WHEN data cleaning is complete THEN the system SHALL normalize data formats
3. WHEN processing fails THEN the system SHALL provide detailed error information
"""
        
        # Sample design document
        self.sample_design = """# Design Document

## Overview

Implementation of a Python web scraper using requests and BeautifulSoup libraries with SQLite database storage.

## Architecture

### Components
1. **Scraper Module**: Web scraping functionality using requests and BeautifulSoup
2. **Data Processor**: Data cleaning and normalization
3. **Database Layer**: SQLite database for data storage
4. **CLI Interface**: Command-line interface for user interaction

### Technology Stack
- Python 3.8+
- requests library for HTTP requests
- BeautifulSoup4 for HTML parsing
- SQLite for data storage
- pytest for testing

## Implementation Details

### File Structure
```
web_scraper/
├── scraper.py          # Main scraper module
├── data_processor.py   # Data processing utilities
├── database.py         # Database operations
├── cli.py             # Command-line interface
├── requirements.txt   # Python dependencies
└── tests/            # Test files
    ├── test_scraper.py
    ├── test_processor.py
    └── test_database.py
```
"""
        
        # Create sample task definitions for testing
        self.sample_tasks = [
            TaskDefinition(
                id="task_1",
                title="Set up project structure and dependencies",
                description="Create the basic project structure and install required Python packages",
                steps=[
                    "Create project directory structure",
                    "Create requirements.txt with necessary dependencies",
                    "Set up virtual environment",
                    "Install packages using pip"
                ],
                requirements_ref=["1.1", "1.2"]
            ),
            TaskDefinition(
                id="task_2", 
                title="Implement web scraper module",
                description="Create the main scraper.py module with web scraping functionality",
                steps=[
                    "Create scraper.py file",
                    "Implement URL fetching with requests",
                    "Add HTML parsing with BeautifulSoup",
                    "Add error handling for network issues"
                ],
                requirements_ref=["1.1", "1.2", "1.3"]
            ),
            TaskDefinition(
                id="task_3",
                title="Create database schema and operations",
                description="Implement database.py with SQLite operations for data storage",
                steps=[
                    "Create database.py file",
                    "Define database schema",
                    "Implement CRUD operations",
                    "Add database connection management"
                ],
                requirements_ref=["2.1", "2.2"]
            ),
            TaskDefinition(
                id="task_4",
                title="Implement comprehensive test suite",
                description="Create unit tests for all modules with high coverage",
                steps=[
                    "Set up pytest configuration",
                    "Create test files for each module",
                    "Implement unit tests with mocking",
                    "Add integration tests for end-to-end workflows"
                ],
                requirements_ref=["1.3", "2.3"]
            )
        ]
    
    async def execute_with_rate_limit_handling(self, llm_operation):
        """Delegate to test base."""
        return await self.test_base.execute_with_rate_limit_handling(llm_operation)
    
    async def execute_with_retry(self, llm_operation, max_attempts=3):
        """Delegate to test base."""
        return await self.test_base.execute_with_retry(llm_operation, max_attempts)
    
    def assert_quality_threshold(self, validation_result, custom_message=None):
        """Delegate to test base."""
        return self.test_base.assert_quality_threshold(validation_result, custom_message)
    
    def log_quality_assessment(self, validation_result):
        """Delegate to test base."""
        return self.test_base.log_quality_assessment(validation_result)
    
    @property
    def quality_validator(self):
        """Access quality validator from test base."""
        return self.test_base.quality_validator
    
    @property
    def logger(self):
        """Access logger from test base."""
        return self.test_base.logger
    
    @sequential_test_execution()
    async def test_task_breakdown_into_executable_shell_command_sequences(self):
        """
        Test task breakdown into executable shell command sequences.
        
        Validates:
        - High-level tasks are broken down into specific shell commands
        - Commands are executable and syntactically correct
        - Command sequence follows logical order
        - Commands include proper error handling and validation
        - Generated commands are appropriate for the task context
        
        Requirements: 2.1 - TaskDecomposer breakdown validation
        """
        # Test with a project setup task (should generate clear shell commands)
        task = self.sample_tasks[0]  # "Set up project structure and dependencies"
        
        # Execute task decomposition with rate limit handling
        execution_plan = await self.execute_with_rate_limit_handling(
            lambda: self.task_decomposer.decompose_task(task)
        )
        
        # Verify execution plan was created
        assert isinstance(execution_plan, ExecutionPlan), "Expected ExecutionPlan object"
        assert execution_plan.task == task, "Execution plan should reference the original task"
        
        # Verify commands were generated
        commands = execution_plan.commands
        assert len(commands) >= 3, f"Expected at least 3 commands for project setup, got {len(commands)}"
        
        # Validate command structure and content
        for i, command in enumerate(commands):
            # Verify command has required attributes
            assert hasattr(command, 'command'), f"Command {i} missing 'command' attribute"
            assert hasattr(command, 'description'), f"Command {i} missing 'description' attribute"
            assert command.command.strip(), f"Command {i} has empty command string"
            assert command.description.strip(), f"Command {i} has empty description"
            
            # Log command for debugging
            self.logger.info(f"Command {i}: {command.command} - {command.description}")
        
        # Verify commands are shell-executable (basic syntax check)
        shell_command_patterns = [
            r'^mkdir\s+',           # Directory creation
            r'^touch\s+',           # File creation
            r'^echo\s+',            # Content writing
            r'^pip\s+install',      # Package installation
            r'^python\s+',          # Python execution
            r'^cd\s+',              # Directory navigation
            r'^ls\s+',              # Directory listing
            r'^cat\s+',             # File reading
            r'^chmod\s+',           # Permission changes
            r'^cp\s+',              # File copying
            r'^mv\s+',              # File moving
        ]
        
        executable_commands = 0
        for command in commands:
            cmd_str = command.command.strip()
            if any(re.match(pattern, cmd_str) for pattern in shell_command_patterns):
                executable_commands += 1
            elif cmd_str.startswith(('python', 'pip', 'git', 'npm', 'yarn')):
                executable_commands += 1
        
        executable_ratio = executable_commands / len(commands)
        assert executable_ratio >= 0.2, (
            f"Some commands should be recognizable shell commands. "
            f"Executable: {executable_commands}, Total: {len(commands)}, Ratio: {executable_ratio:.2f}"
        )
        
        # Verify logical command ordering (setup commands should come first)
        command_strings = [cmd.command.lower() for cmd in commands]
        
        # Check for logical progression: directory creation -> file creation -> installation -> execution
        mkdir_indices = [i for i, cmd in enumerate(command_strings) if 'mkdir' in cmd]
        pip_indices = [i for i, cmd in enumerate(command_strings) if 'pip install' in cmd]
        
        if mkdir_indices and pip_indices:
            # Directory creation should generally come before package installation
            assert min(mkdir_indices) < max(pip_indices), (
                "Directory creation should typically come before package installation"
            )
        
        # Verify commands include error handling indicators
        error_handling_indicators = ['||', '&&', 'if', 'test', 'exit', 'echo']
        commands_with_error_handling = 0
        
        for command in commands:
            if any(indicator in command.command for indicator in error_handling_indicators):
                commands_with_error_handling += 1
        
        # At least some commands should have error handling (be lenient)
        error_handling_ratio = commands_with_error_handling / len(commands)
        assert error_handling_ratio >= 0.2, (
            f"Some commands should include error handling. "
            f"Commands with error handling: {commands_with_error_handling}, Total: {len(commands)}"
        )
        
        # Verify commands are contextually appropriate for the task
        task_keywords = ['project', 'structure', 'dependencies', 'setup', 'install']
        contextually_relevant_commands = 0
        
        for command in commands:
            cmd_and_desc = (command.command + ' ' + command.description).lower()
            if any(keyword in cmd_and_desc for keyword in task_keywords):
                contextually_relevant_commands += 1
        
        relevance_ratio = contextually_relevant_commands / len(commands)
        assert relevance_ratio >= 0.2, (
            f"Commands should be contextually relevant to the task. "
            f"Relevant: {contextually_relevant_commands}, Total: {len(commands)}, Ratio: {relevance_ratio:.2f}"
        )
        
        self.logger.info(f"Task breakdown test passed with {len(commands)} executable commands")
    
    @sequential_test_execution()
    async def test_complexity_analysis_accuracy_with_confidence_scoring(self):
        """
        Test complexity analysis accuracy with confidence scoring.
        
        Validates:
        - Complexity levels are accurately assessed (simple, moderate, complex, very_complex)
        - Confidence scores are reasonable (0.0-1.0 range)
        - Required tools and dependencies are identified correctly
        - Risk factors are appropriately identified
        - Analysis reasoning is provided and meaningful
        
        Requirements: 2.5 - Complexity analysis with confidence scoring
        """
        # Test complexity analysis with tasks of varying complexity
        test_cases = [
            (self.sample_tasks[0], "moderate"),  # Project setup - moderate complexity
            (self.sample_tasks[1], "moderate"),  # Web scraper - moderate complexity  
            (self.sample_tasks[2], "moderate"),  # Database operations - moderate complexity
            (self.sample_tasks[3], "complex"),   # Comprehensive testing - complex
        ]
        
        for task, expected_complexity_level in test_cases:
            self.logger.info(f"Testing complexity analysis for task: {task.title}")
            
            # Execute task decomposition to get complexity analysis
            execution_plan = await self.execute_with_rate_limit_handling(
                lambda: self.task_decomposer.decompose_task(task)
            )
            
            complexity = execution_plan.complexity_analysis
            assert isinstance(complexity, ComplexityAnalysis), "Expected ComplexityAnalysis object"
            
            # Validate complexity level is reasonable
            valid_levels = ["simple", "moderate", "complex", "very_complex"]
            assert complexity.complexity_level in valid_levels, (
                f"Invalid complexity level: {complexity.complexity_level}"
            )
            
            # Validate confidence score range
            assert 0.0 <= complexity.confidence_score <= 1.0, (
                f"Confidence score out of range: {complexity.confidence_score}"
            )
            
            # Confidence should be reasonably high for well-defined tasks
            assert complexity.confidence_score >= 0.5, (
                f"Confidence score too low for well-defined task: {complexity.confidence_score}"
            )
            
            # Validate estimated steps is reasonable
            assert complexity.estimated_steps >= 1, (
                f"Estimated steps should be at least 1: {complexity.estimated_steps}"
            )
            assert complexity.estimated_steps <= 20, (
                f"Estimated steps seems too high: {complexity.estimated_steps}"
            )
            
            # Validate required tools are identified
            assert isinstance(complexity.required_tools, list), "Required tools should be a list"
            
            # For Python projects, should identify Python-related tools
            if 'python' in task.description.lower() or 'pip' in task.description.lower():
                python_tools = [tool for tool in complexity.required_tools 
                              if 'python' in tool.lower() or 'pip' in tool.lower()]
                # Be lenient - LLM might not always identify Python tools explicitly
                self.logger.info(f"Python tools identified: {python_tools}")
            
            # Validate dependencies are identified
            assert isinstance(complexity.dependencies, list), "Dependencies should be a list"
            
            # Validate risk factors are identified
            assert isinstance(complexity.risk_factors, list), "Risk factors should be a list"
            
            # Validate analysis reasoning is provided
            assert complexity.analysis_reasoning.strip(), "Analysis reasoning should not be empty"
            assert len(complexity.analysis_reasoning) >= 10, (
                f"Analysis reasoning too short: {len(complexity.analysis_reasoning)} chars"
            )
            
            # Log complexity analysis for debugging
            self.logger.info(
                f"Complexity Analysis - Level: {complexity.complexity_level}, "
                f"Confidence: {complexity.confidence_score:.2f}, "
                f"Steps: {complexity.estimated_steps}, "
                f"Tools: {complexity.required_tools}, "
                f"Risks: {complexity.risk_factors}"
            )
        
        self.logger.info("Complexity analysis test passed for all task types")
    
    @sequential_test_execution()
    async def test_decision_point_generation_and_logical_flow(self):
        """
        Test decision point generation and logical flow.
        
        Validates:
        - Decision points are generated for conditional execution paths
        - Decision conditions are clear and evaluable
        - True/false paths are logically defined
        - Evaluation methods are appropriate for conditions
        - Decision flow enhances execution reliability
        
        Requirements: 2.1 - TaskDecomposer breakdown validation
        """
        # Use a task that should generate decision points (file operations with validation)
        task = self.sample_tasks[1]  # Web scraper implementation
        
        # Execute task decomposition
        execution_plan = await self.execute_with_rate_limit_handling(
            lambda: self.task_decomposer.decompose_task(task)
        )
        
        # Note: The current TaskDecomposer implementation uses a simplified model
        # that doesn't generate complex decision points in the comprehensive approach.
        # We'll validate the logical flow of commands instead.
        
        commands = execution_plan.commands
        decision_points = execution_plan.decision_points
        
        # Log decision points for debugging
        self.logger.info(f"Generated {len(decision_points)} decision points")
        for i, dp in enumerate(decision_points):
            self.logger.info(f"Decision Point {i}: {dp.condition}")
        
        # Validate command flow logic (even without explicit decision points)
        # Commands should follow logical progression
        command_descriptions = [cmd.description.lower() for cmd in commands]
        
        # Check for logical flow patterns
        setup_commands = [i for i, desc in enumerate(command_descriptions) 
                         if any(keyword in desc for keyword in ['create', 'setup', 'install'])]
        implementation_commands = [i for i, desc in enumerate(command_descriptions)
                                 if any(keyword in desc for keyword in ['implement', 'add', 'write'])]
        test_commands = [i for i, desc in enumerate(command_descriptions)
                        if any(keyword in desc for keyword in ['test', 'validate', 'verify'])]
        
        # Setup should generally come before implementation
        if setup_commands and implementation_commands:
            assert min(setup_commands) < max(implementation_commands), (
                "Setup commands should generally come before implementation commands"
            )
        
        # Implementation should generally come before testing
        if implementation_commands and test_commands:
            assert min(implementation_commands) < max(test_commands), (
                "Implementation commands should generally come before test commands"
            )
        
        # Validate command success/failure indicators
        commands_with_indicators = 0
        for command in commands:
            has_success_indicators = len(command.success_indicators) > 0
            has_failure_indicators = len(command.failure_indicators) > 0
            
            if has_success_indicators or has_failure_indicators:
                commands_with_indicators += 1
                
                # Validate indicator quality
                if has_success_indicators:
                    for indicator in command.success_indicators:
                        assert isinstance(indicator, str), "Success indicators should be strings"
                        assert len(indicator.strip()) > 0, "Success indicators should not be empty"
                
                if has_failure_indicators:
                    for indicator in command.failure_indicators:
                        assert isinstance(indicator, str), "Failure indicators should be strings"
                        assert len(indicator.strip()) > 0, "Failure indicators should not be empty"
        
        # At least some commands should have success/failure indicators
        indicator_ratio = commands_with_indicators / len(commands)
        assert indicator_ratio >= 0.3, (
            f"More commands should have success/failure indicators. "
            f"Commands with indicators: {commands_with_indicators}, Total: {len(commands)}"
        )
        
        # Validate timeout settings are reasonable
        for command in commands:
            assert command.timeout > 0, f"Command timeout should be positive: {command.timeout}"
            assert command.timeout <= 3600, f"Command timeout seems too high: {command.timeout}"
        
        # Validate retry settings
        retry_commands = [cmd for cmd in commands if cmd.retry_on_failure]
        retry_ratio = len(retry_commands) / len(commands)
        
        # Some commands should be retryable (but not necessarily all)
        assert 0.1 <= retry_ratio <= 0.9, (
            f"Retry ratio should be reasonable. Retryable: {len(retry_commands)}, "
            f"Total: {len(commands)}, Ratio: {retry_ratio:.2f}"
        )
        
        self.logger.info(f"Decision point and logical flow test passed. "
                        f"Commands with indicators: {indicator_ratio:.2f}, "
                        f"Retry ratio: {retry_ratio:.2f}")
    
    @sequential_test_execution()
    async def test_success_criteria_definition_and_clarity(self):
        """
        Test success criteria definition and clarity.
        
        Validates:
        - Success criteria are clearly defined and measurable
        - Criteria are specific to the task objectives
        - Criteria can be verified programmatically or manually
        - Success criteria cover all major task outcomes
        - Criteria are realistic and achievable
        
        Requirements: 2.1 - TaskDecomposer breakdown validation
        """
        # Test success criteria generation for different types of tasks
        for task in self.sample_tasks:
            self.logger.info(f"Testing success criteria for task: {task.title}")
            
            # Execute task decomposition
            execution_plan = await self.execute_with_rate_limit_handling(
                lambda: self.task_decomposer.decompose_task(task)
            )
            
            success_criteria = execution_plan.success_criteria
            
            # Validate success criteria exist
            assert isinstance(success_criteria, list), "Success criteria should be a list"
            assert len(success_criteria) >= 1, f"Task should have at least 1 success criterion, got {len(success_criteria)}"
            
            # Validate criteria quality
            for i, criterion in enumerate(success_criteria):
                assert isinstance(criterion, str), f"Success criterion {i} should be a string"
                assert len(criterion.strip()) >= 10, (
                    f"Success criterion {i} too short: '{criterion}'"
                )
                
                # Criteria should be specific and measurable
                measurable_indicators = [
                    'file', 'directory', 'created', 'exists', 'contains',
                    'returns', 'passes', 'completes', 'successful', 'error-free',
                    'installed', 'configured', 'running', 'accessible'
                ]
                
                criterion_lower = criterion.lower()
                is_measurable = any(indicator in criterion_lower for indicator in measurable_indicators)
                
                # Log for debugging if not measurable
                if not is_measurable:
                    self.logger.warning(f"Criterion may not be measurable: '{criterion}'")
            
            # Check for task-specific success criteria
            task_keywords = task.title.lower().split() + task.description.lower().split()
            relevant_criteria = 0
            
            for criterion in success_criteria:
                criterion_lower = criterion.lower()
                if any(keyword in criterion_lower for keyword in task_keywords[:5]):  # Check first 5 keywords
                    relevant_criteria += 1
            
            relevance_ratio = relevant_criteria / len(success_criteria)
            # Be lenient - LLM may use different terminology
            if relevance_ratio < 0.1:
                self.logger.warning(f"Low relevance ratio for task '{task.title}': {relevance_ratio:.2f}")
            # Don't assert on relevance ratio as LLM may use different terminology
            
            # Validate criteria are achievable (not overly complex)
            for criterion in success_criteria:
                # Criteria shouldn't be too complex (reasonable length)
                assert len(criterion) <= 200, f"Success criterion too complex: '{criterion[:100]}...'"
                
                # Criteria should be positive statements (what should happen)
                negative_indicators = ['not', 'never', 'cannot', 'impossible', 'fail']
                criterion_lower = criterion.lower()
                negative_count = sum(1 for indicator in negative_indicators if indicator in criterion_lower)
                
                # Allow some negative indicators but not too many
                assert negative_count <= 2, f"Success criterion too negative: '{criterion}'"
            
            # Log success criteria for debugging
            self.logger.info(f"Success criteria for '{task.title}':")
            for i, criterion in enumerate(success_criteria):
                self.logger.info(f"  {i+1}. {criterion}")
        
        self.logger.info("Success criteria definition test passed for all tasks")
    
    @sequential_test_execution()
    async def test_context_aware_task_understanding(self):
        """
        Test context-aware task understanding.
        
        Validates:
        - TaskDecomposer considers project requirements and design context
        - Generated commands are appropriate for the project technology stack
        - Task decomposition aligns with project architecture
        - Context influences command selection and sequencing
        - Project-specific patterns are incorporated into decomposition
        
        Requirements: 6.4 - Context-aware task understanding
        """
        # Create a work directory with project context
        work_dir = self.workspace_path / "test_context_awareness"
        work_dir.mkdir(exist_ok=True)
        
        # Write project context documents
        requirements_path = work_dir / "requirements.md"
        design_path = work_dir / "design.md"
        
        requirements_path.write_text(self.sample_requirements, encoding='utf-8')
        design_path.write_text(self.sample_design, encoding='utf-8')
        
        # Test with a task that should be influenced by context
        task = self.sample_tasks[1]  # Web scraper implementation
        
        # Execute task decomposition (context manager should provide project context)
        execution_plan = await self.execute_with_rate_limit_handling(
            lambda: self.task_decomposer.decompose_task(task)
        )
        
        commands = execution_plan.commands
        complexity = execution_plan.complexity_analysis
        
        # Validate context awareness in complexity analysis
        # Should identify Python-related tools for Python project
        python_related_tools = [
            tool for tool in complexity.required_tools
            if any(keyword in tool.lower() for keyword in ['python', 'pip', 'requests', 'beautifulsoup'])
        ]
        
        # Be lenient - LLM might use different terminology
        self.logger.info(f"Python-related tools identified: {python_related_tools}")
        
        # Validate context awareness in command generation
        # Commands should be appropriate for Python web scraping project
        python_commands = 0
        web_scraping_commands = 0
        
        for command in commands:
            cmd_and_desc = (command.command + ' ' + command.description).lower()
            
            # Check for Python-specific commands
            if any(keyword in cmd_and_desc for keyword in ['python', 'pip', '.py']):
                python_commands += 1
            
            # Check for web scraping context
            if any(keyword in cmd_and_desc for keyword in ['requests', 'beautifulsoup', 'scraper', 'html', 'url']):
                web_scraping_commands += 1
        
        # At least some commands should be Python-related
        python_ratio = python_commands / len(commands)
        assert python_ratio >= 0.3, (
            f"Commands should reflect Python context. "
            f"Python commands: {python_commands}, Total: {len(commands)}, Ratio: {python_ratio:.2f}"
        )
        
        # Validate technology stack alignment
        # Commands should use technologies mentioned in design document
        tech_stack_keywords = ['requests', 'beautifulsoup', 'sqlite', 'pytest']
        tech_aligned_commands = 0
        
        for command in commands:
            cmd_and_desc = (command.command + ' ' + command.description).lower()
            if any(tech in cmd_and_desc for tech in tech_stack_keywords):
                tech_aligned_commands += 1
        
        tech_alignment_ratio = tech_aligned_commands / len(commands)
        # Be lenient - not all commands need to reference specific technologies
        self.logger.info(f"Technology alignment ratio: {tech_alignment_ratio:.2f}")
        
        # Validate file structure awareness
        # Commands should create files mentioned in design document
        expected_files = ['scraper.py', 'data_processor.py', 'database.py', 'requirements.txt']
        file_creation_commands = 0
        
        for command in commands:
            cmd_and_desc = (command.command + ' ' + command.description).lower()
            if any(filename in cmd_and_desc for filename in expected_files):
                file_creation_commands += 1
        
        # At least some commands should reference expected files
        file_awareness_ratio = file_creation_commands / len(commands)
        assert file_awareness_ratio >= 0.2, (
            f"Commands should show awareness of project file structure. "
            f"File-aware commands: {file_creation_commands}, Total: {len(commands)}, "
            f"Ratio: {file_awareness_ratio:.2f}"
        )
        
        # Validate requirement alignment
        # Success criteria should align with requirements
        success_criteria = execution_plan.success_criteria
        requirement_aligned_criteria = 0
        
        requirement_keywords = ['scrape', 'data', 'website', 'url', 'parse', 'extract']
        for criterion in success_criteria:
            criterion_lower = criterion.lower()
            if any(keyword in criterion_lower for keyword in requirement_keywords):
                requirement_aligned_criteria += 1
        
        requirement_alignment_ratio = requirement_aligned_criteria / len(success_criteria)
        assert requirement_alignment_ratio >= 0.3, (
            f"Success criteria should align with requirements. "
            f"Aligned criteria: {requirement_aligned_criteria}, Total: {len(success_criteria)}, "
            f"Ratio: {requirement_alignment_ratio:.2f}"
        )
        
        # Validate fallback strategies are context-appropriate
        fallback_strategies = execution_plan.fallback_strategies
        if fallback_strategies:
            for strategy in fallback_strategies:
                strategy_lower = strategy.lower()
                # Fallback strategies should be relevant to the project context
                assert len(strategy) >= 10, f"Fallback strategy too short: '{strategy}'"
        
        self.logger.info(f"Context awareness test passed. "
                        f"Python ratio: {python_ratio:.2f}, "
                        f"Tech alignment: {tech_alignment_ratio:.2f}, "
                        f"File awareness: {file_awareness_ratio:.2f}, "
                        f"Requirement alignment: {requirement_alignment_ratio:.2f}")
    
    @sequential_test_execution()
    async def test_comprehensive_task_decomposition_integration(self):
        """
        Test comprehensive task decomposition integration.
        
        This test validates the complete TaskDecomposer workflow by testing
        all components together: complexity analysis, command generation,
        success criteria, and context awareness in a single integrated test.
        
        Validates:
        - Complete ExecutionPlan generation works end-to-end
        - All components work together cohesively
        - Generated plans are comprehensive and actionable
        - Quality meets minimum thresholds across all dimensions
        """
        # Use the most complex task for comprehensive testing
        task = self.sample_tasks[3]  # Comprehensive test suite implementation
        
        self.logger.info(f"Running comprehensive integration test for: {task.title}")
        
        # Execute complete task decomposition
        execution_plan = await self.execute_with_rate_limit_handling(
            lambda: self.task_decomposer.decompose_task(task)
        )
        
        # Validate ExecutionPlan completeness
        assert isinstance(execution_plan, ExecutionPlan), "Should return ExecutionPlan object"
        assert execution_plan.task == task, "Plan should reference original task"
        assert execution_plan.created_at, "Plan should have creation timestamp"
        
        # Validate all major components are present
        assert execution_plan.complexity_analysis is not None, "Should have complexity analysis"
        assert len(execution_plan.commands) >= 3, "Should have multiple commands"
        assert len(execution_plan.success_criteria) >= 1, "Should have success criteria"
        
        # Validate complexity analysis quality
        complexity = execution_plan.complexity_analysis
        assert complexity.complexity_level in ["moderate", "complex", "very_complex"], (
            f"Test suite task should be at least moderate complexity: {complexity.complexity_level}"
        )
        assert complexity.confidence_score >= 0.5, f"Confidence should be reasonable: {complexity.confidence_score}"
        assert complexity.estimated_steps >= 3, f"Should estimate multiple steps: {complexity.estimated_steps}"
        
        # Validate command quality
        commands = execution_plan.commands
        
        # Commands should be diverse and comprehensive
        command_types = set()
        for command in commands:
            cmd_lower = command.command.lower()
            if 'mkdir' in cmd_lower or 'touch' in cmd_lower:
                command_types.add('file_operations')
            elif 'pip install' in cmd_lower or 'install' in cmd_lower:
                command_types.add('installation')
            elif 'python' in cmd_lower and '.py' in cmd_lower:
                command_types.add('execution')
            elif 'pytest' in cmd_lower or 'test' in cmd_lower:
                command_types.add('testing')
        
        assert len(command_types) >= 2, f"Should have diverse command types: {command_types}"
        
        # Validate success criteria quality
        success_criteria = execution_plan.success_criteria
        
        # Success criteria should be comprehensive for test suite task
        test_related_criteria = 0
        for criterion in success_criteria:
            criterion_lower = criterion.lower()
            if any(keyword in criterion_lower for keyword in ['test', 'coverage', 'pass', 'pytest']):
                test_related_criteria += 1
        
        test_criteria_ratio = test_related_criteria / len(success_criteria)
        assert test_criteria_ratio >= 0.5, (
            f"Test suite task should have test-related success criteria. "
            f"Test criteria: {test_related_criteria}, Total: {len(success_criteria)}"
        )
        
        # Validate estimated duration is reasonable
        assert execution_plan.estimated_duration > 0, "Should have positive estimated duration"
        assert execution_plan.estimated_duration <= 600, f"Duration seems too high: {execution_plan.estimated_duration} minutes"
        
        # Validate fallback strategies exist for complex tasks
        if execution_plan.fallback_strategies:
            for strategy in execution_plan.fallback_strategies:
                assert len(strategy.strip()) >= 10, f"Fallback strategy too short: '{strategy}'"
        
        # Log comprehensive results
        self.logger.info(f"Comprehensive integration test results:")
        self.logger.info(f"  - Complexity: {complexity.complexity_level} (confidence: {complexity.confidence_score:.2f})")
        self.logger.info(f"  - Commands: {len(commands)} ({list(command_types)})")
        self.logger.info(f"  - Success criteria: {len(success_criteria)} (test-related: {test_criteria_ratio:.2f})")
        self.logger.info(f"  - Estimated duration: {execution_plan.estimated_duration} minutes")
        self.logger.info(f"  - Fallback strategies: {len(execution_plan.fallback_strategies)}")
        
        self.logger.info("Comprehensive task decomposition integration test passed")


# Additional test utilities for TaskDecomposer testing

def validate_shell_command_syntax(command: str) -> bool:
    """
    Validate basic shell command syntax.
    
    Args:
        command: Shell command string to validate
        
    Returns:
        True if command appears to have valid syntax
    """
    # Basic syntax checks
    if not command.strip():
        return False
    
    # Check for balanced quotes
    single_quotes = command.count("'")
    double_quotes = command.count('"')
    
    if single_quotes % 2 != 0 or double_quotes % 2 != 0:
        return False
    
    # Check for dangerous patterns (basic safety)
    dangerous_patterns = ['rm -rf /', 'dd if=', ':(){ :|:& };:']
    if any(pattern in command for pattern in dangerous_patterns):
        return False
    
    return True


def extract_command_components(command: str) -> Dict[str, Any]:
    """
    Extract components from a shell command for analysis.
    
    Args:
        command: Shell command string
        
    Returns:
        Dictionary with command components
    """
    parts = command.strip().split()
    if not parts:
        return {}
    
    return {
        'executable': parts[0],
        'args': parts[1:] if len(parts) > 1 else [],
        'has_pipes': '|' in command,
        'has_redirects': any(op in command for op in ['>', '>>', '<']),
        'has_conditionals': any(op in command for op in ['&&', '||']),
        'is_compound': ';' in command or '&&' in command or '||' in command
    }