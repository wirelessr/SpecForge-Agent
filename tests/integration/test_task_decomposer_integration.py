"""
Integration tests for TaskDecomposer agent.

Tests the TaskDecomposer's integration with real LLM services and
demonstrates successful task decomposition with actual context.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

from autogen_framework.agents.task_decomposer import TaskDecomposer, ExecutionPlan
from autogen_framework.models import TaskDefinition
from autogen_framework.context_manager import ContextManager
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.context_compressor import ContextCompressor
from autogen_framework.token_manager import TokenManager
from autogen_framework.config_manager import ConfigManager


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create sample project files
        (workspace / "requirements.md").write_text("""
# Requirements Document

## Introduction
This is a test project for task decomposition.

## Requirements

### Requirement 1: File Creation
**User Story:** As a developer, I want to create Python files, so that I can implement functionality.

#### Acceptance Criteria
1. WHEN creating a file THEN the system SHALL create a valid Python file
2. WHEN adding content THEN the system SHALL write proper Python syntax
""")
        
        (workspace / "design.md").write_text("""
# Design Document

## Overview
Simple Python module design for testing.

## Architecture
- Single module with basic functions
- Unit tests for validation
""")
        
        yield workspace


@pytest.fixture
def real_context_manager(temp_workspace, real_llm_config):
    """Create ContextManager with real configuration."""
    memory_manager = MemoryManager(workspace_path=str(temp_workspace))
    context_compressor = ContextCompressor(real_llm_config)
    config_manager = ConfigManager()
    token_manager = TokenManager(config_manager)
    
    context_manager = ContextManager(
        work_dir=str(temp_workspace),
        memory_manager=memory_manager,
        context_compressor=context_compressor,
        llm_config=real_llm_config,
        token_manager=token_manager,
        config_manager=config_manager
    )
    
    return context_manager


@pytest.fixture
def real_task_decomposer(real_llm_config, real_managers):
    """Create TaskDecomposer with real LLM configuration."""
    system_message = """
You are an expert task decomposition agent. Your role is to analyze tasks and break them down into executable shell command sequences.

Key responsibilities:
1. Analyze task complexity using available context
2. Generate practical shell commands that can be executed
3. Create decision points for conditional logic
4. Define clear success criteria
5. Provide fallback strategies for error recovery

Always provide responses in the requested JSON format when specified.
"""
    
    return TaskDecomposer(
        name="RealTaskDecomposer",
        llm_config=real_llm_config,
        system_message=system_message,
        token_manager=real_managers.token_manager,
        context_manager=real_managers.context_manager
    )


@pytest.fixture
def simple_python_task():
    """Simple Python file creation task."""
    return TaskDefinition(
        id="create_python_module",
        title="Create Python module with basic function",
        description="Create a Python module file with a simple hello world function and basic documentation",
        steps=[
            "Create Python file",
            "Add hello world function",
            "Add docstring",
            "Make file executable if needed"
        ],
        requirements_ref=["1.1"],
        dependencies=[]
    )


@pytest.fixture
def complex_testing_task():
    """Complex task involving testing."""
    return TaskDefinition(
        id="implement_with_tests",
        title="Implement module with comprehensive tests",
        description="Create a Python module with multiple functions and comprehensive unit tests",
        steps=[
            "Create main module file",
            "Implement multiple functions",
            "Create test file",
            "Write unit tests for all functions",
            "Run tests to verify functionality",
            "Generate test coverage report"
        ],
        requirements_ref=["1.1", "1.2"],
        dependencies=[]
    )


@pytest.mark.integration
class TestTaskDecomposerIntegration:
    """Integration tests for TaskDecomposer with real LLM."""
    
    @pytest.mark.asyncio
    async def test_decompose_simple_task_with_real_llm(self, real_task_decomposer, simple_python_task):
        """Test task decomposition with real LLM for simple task."""
        execution_plan = await real_task_decomposer.decompose_task(simple_python_task)
        
        # Verify execution plan structure
        assert isinstance(execution_plan, ExecutionPlan)
        assert execution_plan.task == simple_python_task
        assert execution_plan.complexity_analysis is not None
        assert len(execution_plan.commands) > 0
        assert len(execution_plan.success_criteria) > 0
        assert execution_plan.estimated_duration > 0
        
        # Verify complexity analysis
        complexity = execution_plan.complexity_analysis
        assert complexity.complexity_level in ["simple", "moderate", "complex", "very_complex"]
        assert complexity.estimated_steps > 0
        assert complexity.confidence_score >= 0.0
        assert complexity.confidence_score <= 1.0
        
        # Verify commands are practical
        commands = execution_plan.commands
        assert any("python" in cmd.command.lower() or "touch" in cmd.command.lower() or "echo" in cmd.command.lower() 
                  for cmd in commands), "Should contain file creation or Python-related commands"
        
        # Verify all commands have required fields
        for cmd in commands:
            assert cmd.command != ""
            assert cmd.description != ""
            assert isinstance(cmd.timeout, int)
            assert cmd.timeout > 0
        
        # Verify success criteria are meaningful
        assert any("file" in criterion.lower() or "function" in criterion.lower() 
                  for criterion in execution_plan.success_criteria), "Should have file or function-related criteria"
        
        print(f"✓ Simple task decomposed into {len(commands)} commands")
        print(f"✓ Complexity: {complexity.complexity_level}")
        print(f"✓ Estimated duration: {execution_plan.estimated_duration} minutes")
    
    @pytest.mark.asyncio
    async def test_decompose_complex_task_with_real_llm(self, real_task_decomposer, complex_testing_task):
        """Test task decomposition with real LLM for complex task."""
        execution_plan = await real_task_decomposer.decompose_task(complex_testing_task)
        
        # Verify execution plan structure
        assert isinstance(execution_plan, ExecutionPlan)
        assert execution_plan.task == complex_testing_task
        
        # Complex tasks should have commands and reasonable complexity
        assert len(execution_plan.commands) >= 3, "Complex task should have multiple commands"
        assert execution_plan.complexity_analysis.complexity_level in ["simple", "moderate", "complex", "very_complex"]
        assert execution_plan.estimated_duration >= 5, "Complex task should take reasonable time"
        
        # Should contain testing-related commands
        commands_text = " ".join(cmd.command.lower() for cmd in execution_plan.commands)
        assert any(test_keyword in commands_text for test_keyword in ["test", "pytest", "unittest"]), \
            "Should contain testing-related commands"
        
        # Should have comprehensive success criteria
        assert len(execution_plan.success_criteria) >= 3, "Complex task should have multiple success criteria"
        
        print(f"✓ Complex task decomposed into {len(execution_plan.commands)} commands")
        print(f"✓ Complexity: {execution_plan.complexity_analysis.complexity_level}")
        print(f"✓ Success criteria: {len(execution_plan.success_criteria)}")
    
    @pytest.mark.asyncio
    async def test_complexity_analysis_with_real_llm(self, real_task_decomposer, simple_python_task):
        """Test complexity analysis with real LLM."""
        complexity = await real_task_decomposer._analyze_complexity(simple_python_task)
        
        # Verify complexity analysis structure
        assert complexity.complexity_level in ["simple", "moderate", "complex", "very_complex"]
        assert complexity.estimated_steps > 0
        assert isinstance(complexity.required_tools, list)
        assert isinstance(complexity.dependencies, list)
        assert isinstance(complexity.risk_factors, list)
        assert 0.0 <= complexity.confidence_score <= 1.0
        assert complexity.analysis_reasoning != ""
        
        # For a simple Python task, should be reasonable
        assert complexity.estimated_steps <= 10, "Simple task shouldn't have too many steps"
        
        print(f"✓ Complexity analysis: {complexity.complexity_level}")
        print(f"✓ Estimated steps: {complexity.estimated_steps}")
        print(f"✓ Required tools: {complexity.required_tools}")
        print(f"✓ Confidence: {complexity.confidence_score:.2f}")
    
    @pytest.mark.asyncio
    async def test_command_sequence_generation_with_real_llm(self, real_task_decomposer, simple_python_task):
        """Test command sequence generation with real LLM."""
        # First get complexity analysis
        complexity = await real_task_decomposer._analyze_complexity(simple_python_task)
        
        # Then generate command sequence
        commands = await real_task_decomposer._generate_command_sequence(simple_python_task, complexity)
        
        # Verify command sequence
        assert len(commands) > 0
        assert all(isinstance(cmd.command, str) and cmd.command.strip() != "" for cmd in commands)
        assert all(isinstance(cmd.description, str) and cmd.description.strip() != "" for cmd in commands)
        
        # Commands should be executable shell commands
        command_text = " ".join(cmd.command for cmd in commands)
        assert any(keyword in command_text.lower() for keyword in ["touch", "echo", "cat", "python", "mkdir"]), \
            "Should contain executable shell commands"
        
        print(f"✓ Generated {len(commands)} shell commands")
        for i, cmd in enumerate(commands[:3]):  # Show first 3 commands
            print(f"  {i+1}. {cmd.command} - {cmd.description}")
    
    @pytest.mark.asyncio
    async def test_success_criteria_definition_with_real_llm(self, real_task_decomposer, simple_python_task):
        """Test success criteria definition with real LLM."""
        criteria = await real_task_decomposer._define_success_criteria(simple_python_task)
        
        # Verify success criteria
        assert len(criteria) > 0
        assert all(isinstance(criterion, str) and criterion.strip() != "" for criterion in criteria)
        
        # Should be relevant to the task
        criteria_text = " ".join(criteria).lower()
        assert any(keyword in criteria_text for keyword in ["file", "function", "python", "create"]), \
            "Success criteria should be relevant to Python file creation"
        
        print(f"✓ Defined {len(criteria)} success criteria")
        for i, criterion in enumerate(criteria):
            print(f"  {i+1}. {criterion}")
    
    @pytest.mark.asyncio
    async def test_fallback_strategies_generation_with_real_llm(self, real_task_decomposer, simple_python_task):
        """Test fallback strategies generation with real LLM."""
        # First get complexity analysis
        complexity = await real_task_decomposer._analyze_complexity(simple_python_task)
        
        # Then generate fallback strategies
        strategies = await real_task_decomposer._generate_fallback_strategies(simple_python_task, complexity)
        
        # Verify fallback strategies
        assert len(strategies) > 0
        assert all(isinstance(strategy, str) and strategy.strip() != "" for strategy in strategies)
        
        # Should provide alternative approaches
        strategies_text = " ".join(strategies).lower()
        assert any(keyword in strategies_text for keyword in ["alternative", "retry", "different", "fallback"]), \
            "Should provide alternative approaches"
        
        print(f"✓ Generated {len(strategies)} fallback strategies")
        for i, strategy in enumerate(strategies):
            print(f"  {i+1}. {strategy}")


@pytest.mark.integration
class TestTaskDecomposerWithContext:
    """Integration tests for TaskDecomposer with ContextManager."""
    
    @pytest.mark.asyncio
    async def test_decompose_task_with_context_manager(self, real_task_decomposer, real_context_manager, simple_python_task):
        """Test task decomposition with ContextManager integration."""
        # Set up context manager
        real_task_decomposer.set_context_manager(real_context_manager)
        await real_context_manager.initialize()
        
        # Process task with context
        task_input = {
            "task_type": "decompose_task",
            "task": simple_python_task
        }
        
        result = await real_task_decomposer.process_task(task_input)
        
        # Verify result
        assert result["success"] is True
        assert result["task_id"] == simple_python_task.id
        assert "execution_plan" in result
        assert result["command_count"] > 0
        assert result["estimated_duration"] > 0
        
        # Verify execution plan
        execution_plan = result["execution_plan"]
        assert isinstance(execution_plan, ExecutionPlan)
        assert len(execution_plan.commands) > 0
        
        print(f"✓ Task decomposed with context: {result['command_count']} commands")
        print(f"✓ Complexity: {result['complexity_level']}")
    
    @pytest.mark.asyncio
    async def test_context_aware_task_understanding(self, real_task_decomposer, real_context_manager, simple_python_task):
        """Test that TaskDecomposer uses context for better task understanding."""
        # Set up context manager
        real_task_decomposer.set_context_manager(real_context_manager)
        await real_context_manager.initialize()
        
        # Get implementation context
        context = await real_context_manager.get_implementation_context(simple_python_task)
        
        # Update agent context
        real_task_decomposer.update_context({
            "implementation_context": context,
            "requirements": context.requirements,
            "design": context.design,
            "project_structure": context.project_structure
        })
        
        # Decompose task with context
        execution_plan = await real_task_decomposer.decompose_task(simple_python_task)
        
        # Verify that context influenced the decomposition
        assert execution_plan is not None
        assert len(execution_plan.commands) > 0
        
        # The presence of context should result in more informed decisions
        complexity = execution_plan.complexity_analysis
        assert complexity.confidence_score > 0.5, "Context should increase confidence"
        
        print(f"✓ Context-aware decomposition completed")
        print(f"✓ Confidence score: {complexity.confidence_score:.2f}")
        print(f"✓ Analysis reasoning: {complexity.analysis_reasoning[:100]}...")


@pytest.mark.integration
class TestTaskDecomposerErrorHandling:
    """Integration tests for TaskDecomposer error handling."""
    
    @pytest.mark.asyncio
    async def test_decompose_task_with_invalid_task(self, real_task_decomposer):
        """Test task decomposition with invalid task."""
        invalid_task = TaskDefinition(
            id="invalid_task",
            title="",  # Empty title
            description="",  # Empty description
            steps=[],  # No steps
            requirements_ref=[],
            dependencies=[]
        )
        
        # Should still handle gracefully
        execution_plan = await real_task_decomposer.decompose_task(invalid_task)
        
        # Should return a basic plan even for invalid input
        assert isinstance(execution_plan, ExecutionPlan)
        assert execution_plan.task == invalid_task
        assert execution_plan.complexity_analysis is not None
        
        print(f"✓ Handled invalid task gracefully")
    
    @pytest.mark.asyncio
    async def test_process_task_with_unknown_type(self, real_task_decomposer, simple_python_task):
        """Test task processing with unknown task type."""
        task_input = {
            "task_type": "unknown_operation",
            "task": simple_python_task
        }
        
        with pytest.raises(ValueError, match="Unknown task type"):
            await real_task_decomposer._process_task_impl(task_input)
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_only(self, real_task_decomposer, simple_python_task):
        """Test complexity analysis as standalone operation."""
        task_input = {
            "task_type": "analyze_complexity",
            "task": simple_python_task
        }
        
        result = await real_task_decomposer._process_task_impl(task_input)
        
        assert result["success"] is True
        assert result["task_id"] == simple_python_task.id
        assert "complexity_analysis" in result
        
        complexity = result["complexity_analysis"]
        assert complexity.complexity_level in ["simple", "moderate", "complex", "very_complex"]
        
        print(f"✓ Standalone complexity analysis: {complexity.complexity_level}")


@pytest.mark.integration
class TestTaskDecomposerPerformance:
    """Integration tests for TaskDecomposer performance."""
    
    @pytest.mark.asyncio
    async def test_decompose_multiple_tasks_performance(self, real_task_decomposer):
        """Test performance when decomposing multiple tasks."""
        import time
        
        tasks = [
            TaskDefinition(
                id=f"task_{i}",
                title=f"Create Python module {i}",
                description=f"Create a Python module with function {i}",
                steps=[f"Create file {i}", f"Add function {i}"],
                requirements_ref=["1.1"],
                dependencies=[]
            )
            for i in range(3)  # Test with 3 tasks
        ]
        
        start_time = time.time()
        
        results = []
        for task in tasks:
            execution_plan = await real_task_decomposer.decompose_task(task)
            results.append(execution_plan)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all tasks were processed
        assert len(results) == 3
        assert all(isinstance(plan, ExecutionPlan) for plan in results)
        
        # Performance should be reasonable (less than 2 minutes for 3 tasks)
        assert total_time < 120, f"Processing took too long: {total_time:.2f} seconds"
        
        avg_time = total_time / len(tasks)
        print(f"✓ Processed {len(tasks)} tasks in {total_time:.2f} seconds")
        print(f"✓ Average time per task: {avg_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_complexity_analysis_consistency(self, real_task_decomposer, simple_python_task):
        """Test that complexity analysis is consistent across multiple runs."""
        results = []
        
        # Run complexity analysis multiple times
        for _ in range(3):
            complexity = await real_task_decomposer._analyze_complexity(simple_python_task)
            results.append(complexity)
        
        # Results should be consistent
        complexity_levels = [r.complexity_level for r in results]
        estimated_steps = [r.estimated_steps for r in results]
        
        # Should have some consistency (at least 2 out of 3 should match)
        most_common_level = max(set(complexity_levels), key=complexity_levels.count)
        level_count = complexity_levels.count(most_common_level)
        assert level_count >= 2, f"Inconsistent complexity levels: {complexity_levels}"
        
        # Estimated steps should be in similar range
        min_steps = min(estimated_steps)
        max_steps = max(estimated_steps)
        assert max_steps - min_steps <= 3, f"Steps vary too much: {estimated_steps}"
        
        print(f"✓ Complexity analysis consistency verified")
        print(f"✓ Levels: {complexity_levels}")
        print(f"✓ Steps: {estimated_steps}")