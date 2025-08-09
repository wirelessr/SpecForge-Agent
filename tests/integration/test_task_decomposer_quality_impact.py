"""
Quality impact tests for TaskDecomposer.

Tests the TaskDecomposer's impact on task breakdown quality by comparing
results with baseline approaches and measuring improvement metrics.
"""

import pytest
import json
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, Any, List

from autogen_framework.agents.task_decomposer import TaskDecomposer, ExecutionPlan
from autogen_framework.models import TaskDefinition


@pytest.fixture
def quality_test_tasks():
    """Representative tasks for quality testing."""
    return [
        TaskDefinition(
            id="create_python_module",
            title="Create Python module with basic functionality",
            description="Create a Python module file with functions and basic error handling",
            steps=[
                "Create Python file",
                "Add main function",
                "Add error handling",
                "Add documentation"
            ],
            requirements_ref=["1.1"],
            dependencies=[]
        ),
        TaskDefinition(
            id="implement_testing_suite",
            title="Implement comprehensive testing suite",
            description="Create unit tests with coverage reporting and CI integration",
            steps=[
                "Create test directory structure",
                "Write unit tests for all functions",
                "Set up coverage reporting",
                "Configure CI pipeline",
                "Add test documentation"
            ],
            requirements_ref=["1.2", "1.3"],
            dependencies=["create_python_module"]
        ),
        TaskDefinition(
            id="setup_database_integration",
            title="Set up database integration with ORM",
            description="Implement database models and connection management",
            steps=[
                "Choose database technology",
                "Set up database connection",
                "Create data models",
                "Implement CRUD operations",
                "Add migration system",
                "Write database tests"
            ],
            requirements_ref=["2.1", "2.2"],
            dependencies=[]
        )
    ]


@pytest.fixture
def task_decomposer_with_quality_metrics(real_llm_config):
    """TaskDecomposer configured for quality measurement."""
    system_message = """
You are an expert task decomposition agent focused on generating high-quality, executable shell command sequences.

Your primary goals:
1. Break down tasks into practical, executable shell commands
2. Ensure commands are safe and follow best practices
3. Include proper error handling and validation
4. Generate clear success criteria and fallback strategies
5. Optimize for maintainability and reliability

Always provide detailed, actionable command sequences that can be executed reliably.
"""
    
    return TaskDecomposer(
        name="QualityTaskDecomposer",
        llm_config=real_llm_config,
        system_message=system_message
    )


class QualityMetrics:
    """Quality metrics calculator for task decomposition."""
    
    @staticmethod
    def calculate_decomposition_quality(execution_plan: ExecutionPlan) -> Dict[str, float]:
        """
        Calculate quality metrics for task decomposition.
        
        Returns:
            Dictionary with quality scores (0.0-1.0 scale)
        """
        metrics = {}
        
        # 1. Command Quality (0.0-1.0)
        metrics['command_quality'] = QualityMetrics._assess_command_quality(execution_plan.commands)
        
        # 2. Complexity Appropriateness (0.0-1.0)
        metrics['complexity_appropriateness'] = QualityMetrics._assess_complexity_appropriateness(
            execution_plan.task, execution_plan.complexity_analysis
        )
        
        # 3. Success Criteria Completeness (0.0-1.0)
        metrics['success_criteria_completeness'] = QualityMetrics._assess_success_criteria(
            execution_plan.success_criteria, execution_plan.task
        )
        
        # 4. Fallback Strategy Quality (0.0-1.0)
        metrics['fallback_strategy_quality'] = QualityMetrics._assess_fallback_strategies(
            execution_plan.fallback_strategies
        )
        
        # 5. Decision Point Effectiveness (0.0-1.0)
        metrics['decision_point_effectiveness'] = QualityMetrics._assess_decision_points(
            execution_plan.decision_points, execution_plan.commands
        )
        
        # 6. Overall Executability (0.0-1.0)
        metrics['executability'] = QualityMetrics._assess_executability(execution_plan.commands)
        
        # Calculate overall quality score
        weights = {
            'command_quality': 0.25,
            'complexity_appropriateness': 0.15,
            'success_criteria_completeness': 0.20,
            'fallback_strategy_quality': 0.15,
            'decision_point_effectiveness': 0.10,
            'executability': 0.15
        }
        
        metrics['overall_quality'] = sum(
            metrics[metric] * weight for metric, weight in weights.items()
        )
        
        return metrics
    
    @staticmethod
    def _assess_command_quality(commands: List) -> float:
        """Assess the quality of generated commands."""
        if not commands:
            return 0.0
        
        quality_indicators = []
        
        for command in commands:
            cmd_quality = 0.0
            
            # Check if command has description
            if hasattr(command, 'description') and command.description.strip():
                cmd_quality += 0.2
            
            # Check if command is not empty
            if hasattr(command, 'command') and command.command.strip():
                cmd_quality += 0.2
            
            # Check for reasonable timeout
            if hasattr(command, 'timeout') and 5 <= command.timeout <= 300:
                cmd_quality += 0.2
            
            # Check for success/failure indicators
            if hasattr(command, 'success_indicators') and command.success_indicators:
                cmd_quality += 0.2
            
            if hasattr(command, 'failure_indicators') and command.failure_indicators:
                cmd_quality += 0.2
            
            quality_indicators.append(cmd_quality)
        
        return sum(quality_indicators) / len(quality_indicators)
    
    @staticmethod
    def _assess_complexity_appropriateness(task: TaskDefinition, complexity_analysis) -> float:
        """Assess if complexity analysis matches task characteristics."""
        if not complexity_analysis:
            return 0.0
        
        score = 0.0
        
        # Check if complexity level is reasonable for task steps
        step_count = len(task.steps) if task.steps else 1
        complexity_level = getattr(complexity_analysis, 'complexity_level', 'moderate')
        
        if step_count <= 3 and complexity_level in ['simple', 'moderate']:
            score += 0.3
        elif 4 <= step_count <= 6 and complexity_level in ['moderate', 'complex']:
            score += 0.3
        elif step_count > 6 and complexity_level in ['complex', 'very_complex']:
            score += 0.3
        
        # Check if estimated steps are reasonable
        estimated_steps = getattr(complexity_analysis, 'estimated_steps', 0)
        if 0.5 * step_count <= estimated_steps <= 2 * step_count:
            score += 0.3
        
        # Check confidence score
        confidence = getattr(complexity_analysis, 'confidence_score', 0)
        if confidence >= 0.6:
            score += 0.2
        
        # Check if analysis reasoning exists
        reasoning = getattr(complexity_analysis, 'analysis_reasoning', '')
        if reasoning and len(reasoning) > 10:
            score += 0.2
        
        return score
    
    @staticmethod
    def _assess_success_criteria(success_criteria: List[str], task: TaskDefinition) -> float:
        """Assess completeness and relevance of success criteria."""
        if not success_criteria:
            return 0.0
        
        score = 0.0
        
        # Check if there are enough criteria
        if len(success_criteria) >= 2:
            score += 0.3
        
        # Check if criteria are specific (not generic)
        specific_criteria = 0
        for criterion in success_criteria:
            if any(keyword in criterion.lower() for keyword in ['file', 'function', 'test', 'error', 'output']):
                specific_criteria += 1
        
        if specific_criteria >= len(success_criteria) * 0.5:
            score += 0.4
        
        # Check if criteria relate to task description
        task_keywords = task.description.lower().split()
        relevant_criteria = 0
        for criterion in success_criteria:
            if any(keyword in criterion.lower() for keyword in task_keywords[:5]):  # Check first 5 words
                relevant_criteria += 1
        
        if relevant_criteria >= len(success_criteria) * 0.3:
            score += 0.3
        
        return score
    
    @staticmethod
    def _assess_fallback_strategies(fallback_strategies: List[str]) -> float:
        """Assess quality of fallback strategies."""
        if not fallback_strategies:
            return 0.0
        
        score = 0.0
        
        # Check if there are multiple strategies
        if len(fallback_strategies) >= 2:
            score += 0.4
        
        # Check if strategies are actionable
        actionable_strategies = 0
        for strategy in fallback_strategies:
            if any(keyword in strategy.lower() for keyword in ['retry', 'alternative', 'use', 'try', 'fallback']):
                actionable_strategies += 1
        
        if actionable_strategies >= len(fallback_strategies) * 0.5:
            score += 0.6
        
        return score
    
    @staticmethod
    def _assess_decision_points(decision_points: List, commands: List) -> float:
        """Assess effectiveness of decision points."""
        if not commands:
            return 0.0
        
        # If no decision points, that's okay for simple tasks
        if not decision_points:
            return 0.7  # Neutral score
        
        score = 0.0
        
        # Check if decision points are reasonable proportion of commands
        decision_ratio = len(decision_points) / len(commands)
        if 0.1 <= decision_ratio <= 0.5:  # 10-50% of commands having decision points is reasonable
            score += 0.5
        
        # Check if decision points have proper structure
        valid_decision_points = 0
        for dp in decision_points:
            if (hasattr(dp, 'condition') and dp.condition and
                hasattr(dp, 'evaluation_method') and dp.evaluation_method):
                valid_decision_points += 1
        
        if decision_points and valid_decision_points >= len(decision_points) * 0.8:
            score += 0.5
        
        return score
    
    @staticmethod
    def _assess_executability(commands: List) -> float:
        """Assess how executable the commands are."""
        if not commands:
            return 0.0
        
        executable_commands = 0
        
        for command in commands:
            cmd_text = getattr(command, 'command', '')
            if not cmd_text:
                continue
            
            # Check for common executable patterns
            executable_patterns = [
                'echo', 'touch', 'mkdir', 'cat', 'python', 'pip', 'git',
                'ls', 'cd', 'cp', 'mv', 'chmod', 'test', 'which'
            ]
            
            if any(pattern in cmd_text.lower() for pattern in executable_patterns):
                executable_commands += 1
            
            # Avoid dangerous patterns
            dangerous_patterns = ['rm -rf /', 'format', 'fdisk', 'shutdown']
            if any(pattern in cmd_text.lower() for pattern in dangerous_patterns):
                executable_commands -= 0.5  # Penalty for dangerous commands
        
        return max(0.0, executable_commands / len(commands))


@pytest.mark.integration
class TestTaskDecomposerQualityImpact:
    """Test TaskDecomposer's quality impact on task breakdown."""
    
    @pytest.mark.asyncio
    async def test_quality_metrics_calculation(self, task_decomposer_with_quality_metrics, quality_test_tasks):
        """Test quality metrics calculation for task decomposition."""
        results = []
        
        for task in quality_test_tasks:
            # Decompose task
            execution_plan = await task_decomposer_with_quality_metrics.decompose_task(task)
            
            # Calculate quality metrics
            quality_metrics = QualityMetrics.calculate_decomposition_quality(execution_plan)
            
            result = {
                'task_id': task.id,
                'task_title': task.title,
                'complexity_level': execution_plan.complexity_analysis.complexity_level,
                'command_count': len(execution_plan.commands),
                'decision_points': len(execution_plan.decision_points),
                'success_criteria_count': len(execution_plan.success_criteria),
                'fallback_strategies_count': len(execution_plan.fallback_strategies),
                'estimated_duration': execution_plan.estimated_duration,
                'quality_metrics': quality_metrics
            }
            
            results.append(result)
            
            # Verify quality metrics are reasonable
            assert 0.0 <= quality_metrics['overall_quality'] <= 1.0
            assert quality_metrics['command_quality'] >= 0.3, f"Command quality too low for {task.id}"
            assert quality_metrics['executability'] >= 0.5, f"Executability too low for {task.id}"
            
            print(f"\n✓ Task: {task.title}")
            print(f"  Complexity: {execution_plan.complexity_analysis.complexity_level}")
            print(f"  Commands: {len(execution_plan.commands)}")
            print(f"  Overall Quality: {quality_metrics['overall_quality']:.3f}")
            print(f"  Command Quality: {quality_metrics['command_quality']:.3f}")
            print(f"  Executability: {quality_metrics['executability']:.3f}")
        
        # Calculate average quality across all tasks
        avg_quality = sum(r['quality_metrics']['overall_quality'] for r in results) / len(results)
        avg_command_quality = sum(r['quality_metrics']['command_quality'] for r in results) / len(results)
        avg_executability = sum(r['quality_metrics']['executability'] for r in results) / len(results)
        
        print(f"\n📊 Quality Summary:")
        print(f"  Average Overall Quality: {avg_quality:.3f}")
        print(f"  Average Command Quality: {avg_command_quality:.3f}")
        print(f"  Average Executability: {avg_executability:.3f}")
        
        # Quality thresholds for TaskDecomposer
        assert avg_quality >= 0.6, f"Average quality {avg_quality:.3f} below threshold"
        assert avg_command_quality >= 0.5, f"Average command quality {avg_command_quality:.3f} below threshold"
        assert avg_executability >= 0.7, f"Average executability {avg_executability:.3f} below threshold"
        
        return results
    
    @pytest.mark.asyncio
    async def test_task_breakdown_consistency(self, task_decomposer_with_quality_metrics):
        """Test consistency of task breakdown across multiple runs."""
        task = TaskDefinition(
            id="consistency_test",
            title="Create web API endpoint",
            description="Create a REST API endpoint with validation and error handling",
            steps=[
                "Define API schema",
                "Implement endpoint handler",
                "Add input validation",
                "Add error handling",
                "Write tests"
            ],
            requirements_ref=["3.1"],
            dependencies=[]
        )
        
        results = []
        
        # Run decomposition multiple times
        for run in range(3):
            execution_plan = await task_decomposer_with_quality_metrics.decompose_task(task)
            quality_metrics = QualityMetrics.calculate_decomposition_quality(execution_plan)
            
            results.append({
                'run': run + 1,
                'command_count': len(execution_plan.commands),
                'complexity_level': execution_plan.complexity_analysis.complexity_level,
                'overall_quality': quality_metrics['overall_quality'],
                'command_quality': quality_metrics['command_quality']
            })
        
        # Check consistency
        command_counts = [r['command_count'] for r in results]
        quality_scores = [r['overall_quality'] for r in results]
        
        # Command counts should be reasonably consistent (within 50% variance)
        avg_commands = sum(command_counts) / len(command_counts)
        max_variance = max(abs(count - avg_commands) for count in command_counts)
        assert max_variance <= avg_commands * 0.5, f"Command count variance too high: {max_variance}"
        
        # Quality scores should be reasonably consistent (within 0.3 variance)
        avg_quality = sum(quality_scores) / len(quality_scores)
        max_quality_variance = max(abs(score - avg_quality) for score in quality_scores)
        assert max_quality_variance <= 0.3, f"Quality variance too high: {max_quality_variance}"
        
        print(f"\n🔄 Consistency Test Results:")
        for result in results:
            print(f"  Run {result['run']}: {result['command_count']} commands, "
                  f"quality {result['overall_quality']:.3f}")
        print(f"  Average commands: {avg_commands:.1f} (variance: {max_variance:.1f})")
        print(f"  Average quality: {avg_quality:.3f} (variance: {max_quality_variance:.3f})")
    
    @pytest.mark.asyncio
    async def test_complexity_scaling(self, task_decomposer_with_quality_metrics):
        """Test that TaskDecomposer scales appropriately with task complexity."""
        tasks = [
            # Simple task
            TaskDefinition(
                id="simple_task",
                title="Create simple file",
                description="Create a text file with content",
                steps=["Create file", "Add content"],
                requirements_ref=["1.1"],
                dependencies=[]
            ),
            # Moderate task
            TaskDefinition(
                id="moderate_task",
                title="Create Python script with tests",
                description="Create a Python script with unit tests and documentation",
                steps=["Create script", "Add functions", "Write tests", "Add documentation"],
                requirements_ref=["1.1", "1.2"],
                dependencies=[]
            ),
            # Complex task
            TaskDefinition(
                id="complex_task",
                title="Build microservice with database",
                description="Build a complete microservice with database, API, tests, and deployment",
                steps=[
                    "Design database schema", "Set up database", "Create API endpoints",
                    "Add authentication", "Write comprehensive tests", "Set up CI/CD",
                    "Create documentation", "Deploy to staging"
                ],
                requirements_ref=["2.1", "2.2", "2.3"],
                dependencies=[]
            )
        ]
        
        results = []
        
        for task in tasks:
            execution_plan = await task_decomposer_with_quality_metrics.decompose_task(task)
            quality_metrics = QualityMetrics.calculate_decomposition_quality(execution_plan)
            
            results.append({
                'task_id': task.id,
                'step_count': len(task.steps),
                'command_count': len(execution_plan.commands),
                'complexity_level': execution_plan.complexity_analysis.complexity_level,
                'estimated_duration': execution_plan.estimated_duration,
                'overall_quality': quality_metrics['overall_quality']
            })
        
        # Verify scaling behavior
        simple_result = results[0]
        moderate_result = results[1]
        complex_result = results[2]
        
        # Command count should generally increase with complexity
        assert simple_result['command_count'] <= moderate_result['command_count']
        assert moderate_result['command_count'] <= complex_result['command_count'] + 2  # Allow some variance
        
        # Duration should increase with complexity
        assert simple_result['estimated_duration'] <= moderate_result['estimated_duration']
        assert moderate_result['estimated_duration'] <= complex_result['estimated_duration'] + 5  # Allow some variance
        
        # Quality should remain reasonable across all complexity levels
        for result in results:
            assert result['overall_quality'] >= 0.5, f"Quality too low for {result['task_id']}"
        
        print(f"\n📈 Complexity Scaling Results:")
        for result in results:
            print(f"  {result['task_id']}: {result['step_count']} steps → "
                  f"{result['command_count']} commands, "
                  f"{result['estimated_duration']}min, "
                  f"quality {result['overall_quality']:.3f}")
    
    @pytest.mark.asyncio
    async def test_save_quality_results(self, task_decomposer_with_quality_metrics, quality_test_tasks):
        """Save quality test results for comparison with baseline."""
        # Create results directory
        results_dir = Path("artifacts/quality-reports/taskdecomposer")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run quality tests
        test_results = []
        
        for task in quality_test_tasks:
            start_time = time.time()
            execution_plan = await task_decomposer_with_quality_metrics.decompose_task(task)
            end_time = time.time()
            
            quality_metrics = QualityMetrics.calculate_decomposition_quality(execution_plan)
            
            result = {
                'task_id': task.id,
                'task_title': task.title,
                'task_description': task.description,
                'task_steps': task.steps,
                'requirements_ref': task.requirements_ref,
                'dependencies': task.dependencies,
                'decomposition_time': end_time - start_time,
                'complexity_analysis': {
                    'complexity_level': execution_plan.complexity_analysis.complexity_level,
                    'estimated_steps': execution_plan.complexity_analysis.estimated_steps,
                    'required_tools': execution_plan.complexity_analysis.required_tools,
                    'confidence_score': execution_plan.complexity_analysis.confidence_score
                },
                'execution_plan': {
                    'command_count': len(execution_plan.commands),
                    'decision_points_count': len(execution_plan.decision_points),
                    'success_criteria_count': len(execution_plan.success_criteria),
                    'fallback_strategies_count': len(execution_plan.fallback_strategies),
                    'estimated_duration': execution_plan.estimated_duration
                },
                'commands': [
                    {
                        'command': cmd.command,
                        'description': cmd.description,
                        'timeout': cmd.timeout,
                        'decision_point': cmd.decision_point
                    }
                    for cmd in execution_plan.commands
                ],
                'quality_metrics': quality_metrics,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            test_results.append(result)
        
        # Calculate summary statistics
        summary = {
            'test_run_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tasks_tested': len(test_results),
            'average_quality_metrics': {
                'overall_quality': sum(r['quality_metrics']['overall_quality'] for r in test_results) / len(test_results),
                'command_quality': sum(r['quality_metrics']['command_quality'] for r in test_results) / len(test_results),
                'complexity_appropriateness': sum(r['quality_metrics']['complexity_appropriateness'] for r in test_results) / len(test_results),
                'success_criteria_completeness': sum(r['quality_metrics']['success_criteria_completeness'] for r in test_results) / len(test_results),
                'fallback_strategy_quality': sum(r['quality_metrics']['fallback_strategy_quality'] for r in test_results) / len(test_results),
                'decision_point_effectiveness': sum(r['quality_metrics']['decision_point_effectiveness'] for r in test_results) / len(test_results),
                'executability': sum(r['quality_metrics']['executability'] for r in test_results) / len(test_results)
            },
            'performance_metrics': {
                'average_decomposition_time': sum(r['decomposition_time'] for r in test_results) / len(test_results),
                'average_commands_per_task': sum(r['execution_plan']['command_count'] for r in test_results) / len(test_results),
                'average_estimated_duration': sum(r['execution_plan']['estimated_duration'] for r in test_results) / len(test_results)
            },
            'detailed_results': test_results
        }
        
        # Save results to file
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"taskdecomposer_quality_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Quality results saved to: {results_file}")
        print(f"📊 Summary:")
        print(f"  Tasks tested: {summary['total_tasks_tested']}")
        print(f"  Average overall quality: {summary['average_quality_metrics']['overall_quality']:.3f}")
        print(f"  Average command quality: {summary['average_quality_metrics']['command_quality']:.3f}")
        print(f"  Average executability: {summary['average_quality_metrics']['executability']:.3f}")
        print(f"  Average decomposition time: {summary['performance_metrics']['average_decomposition_time']:.2f}s")
        
        # Verify quality thresholds
        assert summary['average_quality_metrics']['overall_quality'] >= 0.6
        assert summary['average_quality_metrics']['command_quality'] >= 0.5
        assert summary['average_quality_metrics']['executability'] >= 0.7
        
        return results_file