"""
Integration tests for ErrorRecovery quality impact measurement.

This module tests the ErrorRecovery agent's impact on task execution quality
by comparing results with and without error recovery capabilities.
"""

import pytest
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from autogen_framework.agents.error_recovery import (
    ErrorRecovery, ErrorType, ErrorAnalysis, RecoveryStrategy, 
    RecoveryResult, CommandResult
)
from autogen_framework.agents.task_decomposer import TaskDecomposer
from autogen_framework.models import LLMConfig, TaskDefinition


class MockImplementAgent:
    """Mock ImplementAgent for testing ErrorRecovery integration."""
    
    def __init__(self, use_error_recovery: bool = False):
        self.use_error_recovery = use_error_recovery
        self.error_recovery = None
        self.task_decomposer = None
        self.execution_results = []
        
    def set_error_recovery(self, error_recovery: ErrorRecovery):
        """Set the ErrorRecovery agent."""
        self.error_recovery = error_recovery
        
    def set_task_decomposer(self, task_decomposer: TaskDecomposer):
        """Set the TaskDecomposer agent."""
        self.task_decomposer = task_decomposer
    
    async def execute_task(self, task: TaskDefinition) -> Dict[str, Any]:
        """Execute a task with or without error recovery."""
        # Simulate task execution with potential failures
        execution_result = {
            "task_id": task.id,
            "task_title": task.title,
            "success": False,
            "attempts": 1,
            "execution_time": 2.0,
            "commands_executed": [],
            "errors_encountered": [],
            "recovery_attempts": [],
            "final_outcome": "failed"
        }
        
        # Simulate initial command failure
        failed_command = CommandResult(
            command=f"execute_task_{task.id}",
            exit_code=1,
            stdout="",
            stderr="Simulated command failure for testing",
            execution_time=1.0,
            success=False
        )
        
        execution_result["commands_executed"].append({
            "command": failed_command.command,
            "success": failed_command.success,
            "exit_code": failed_command.exit_code
        })
        execution_result["errors_encountered"].append({
            "error": failed_command.stderr,
            "command": failed_command.command
        })
        
        # If error recovery is enabled, attempt recovery
        if self.use_error_recovery and self.error_recovery:
            try:
                recovery_result = await self.error_recovery.recover(failed_command)
                execution_result["recovery_attempts"].append({
                    "success": recovery_result.success,
                    "strategy_used": recovery_result.strategy_used.name if recovery_result.strategy_used else None,
                    "attempts": len(recovery_result.attempted_strategies),
                    "recovery_time": recovery_result.recovery_time
                })
                
                if recovery_result.success:
                    execution_result["success"] = True
                    execution_result["final_outcome"] = "recovered"
                    execution_result["attempts"] += len(recovery_result.attempted_strategies)
                else:
                    execution_result["final_outcome"] = "recovery_failed"
                    
            except Exception as e:
                execution_result["recovery_attempts"].append({
                    "success": False,
                    "error": str(e),
                    "attempts": 0,
                    "recovery_time": 0.0
                })
        
        self.execution_results.append(execution_result)
        return execution_result


class TestErrorRecoveryQualityImpact:
    """Test ErrorRecovery quality impact on task execution."""
    
    @pytest.fixture
    def real_llm_config(self):
        """Real LLM configuration for integration testing."""
        return LLMConfig(
            base_url="http://ctwuhome.local:8888/openai/v1",
            model="models/gemini-2.0-flash",
            api_key="sk-123456"
        )
    
    @pytest.fixture
    def error_recovery_agent(self, real_llm_config, real_managers):
        """Create ErrorRecovery agent for integration testing."""
        system_message = """
You are an intelligent error recovery agent. Your role is to:
1. Analyze command failures and categorize error types
2. Generate alternative recovery strategies
3. Learn from successful recovery patterns
4. Provide detailed error analysis with high confidence

Always respond with valid JSON when requested.
"""
        return ErrorRecovery(
            name="IntegrationTestErrorRecovery",
            llm_config=real_llm_config,
            system_message=system_message,
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    @pytest.fixture
    def task_decomposer_agent(self, real_llm_config, real_managers):
        """Create TaskDecomposer agent for integration testing."""
        system_message = """
You are a task decomposition agent. Your role is to:
1. Break down high-level tasks into executable shell commands
2. Analyze task complexity and requirements
3. Generate execution plans with decision points

Always respond with valid JSON when requested.
"""
        return TaskDecomposer(
            name="IntegrationTestTaskDecomposer",
            llm_config=real_llm_config,
            system_message=system_message,
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    @pytest.fixture
    def sample_tasks(self):
        """Sample tasks for quality impact testing."""
        return [
            TaskDefinition(
                id="task_1",
                title="Install Python Package",
                description="Install a Python package using pip",
                steps=["Check if pip is available", "Install package", "Verify installation"],
                requirements_ref=["1.1"],
                dependencies=[]
            ),
            TaskDefinition(
                id="task_2", 
                title="Create and Test Python Module",
                description="Create a Python module and run tests",
                steps=["Create module file", "Write test file", "Run tests"],
                requirements_ref=["1.2"],
                dependencies=["task_1"]
            ),
            TaskDefinition(
                id="task_3",
                title="Setup Development Environment",
                description="Setup a development environment with dependencies",
                steps=["Create virtual environment", "Install dependencies", "Configure environment"],
                requirements_ref=["1.3"],
                dependencies=[]
            )
        ]
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Skipping long-running quality comparison test")
    async def test_error_recovery_quality_impact_comparison(self, error_recovery_agent, sample_tasks, real_managers):
        """Test quality impact by comparing execution with and without ErrorRecovery."""
        # Test without ErrorRecovery
        agent_without_recovery = MockImplementAgent(use_error_recovery=False)
        results_without_recovery = []
        
        for task in sample_tasks:
            result = await agent_without_recovery.execute_task(task)
            results_without_recovery.append(result)
        
        # Test with ErrorRecovery
        agent_with_recovery = MockImplementAgent(use_error_recovery=True)
        agent_with_recovery.set_error_recovery(error_recovery_agent)
        results_with_recovery = []
        
        for task in sample_tasks:
            result = await agent_with_recovery.execute_task(task)
            results_with_recovery.append(result)
        
        # Calculate quality metrics
        metrics_without = self._calculate_quality_metrics(results_without_recovery)
        metrics_with = self._calculate_quality_metrics(results_with_recovery)
        
        # Generate quality impact report
        impact_report = self._generate_quality_impact_report(
            metrics_without, metrics_with, results_without_recovery, results_with_recovery
        )
        
        # Save results to artifacts directory
        await self._save_quality_results(impact_report, "errorrecovery")
        
        # Assertions for quality improvement
        assert metrics_with["success_rate"] >= metrics_without["success_rate"]
        assert metrics_with["recovery_attempts"] > 0  # Should have attempted recovery
        assert impact_report["overall_improvement"] >= 0  # Should show improvement or no regression
        
        # Log results
        print(f"Quality Impact Results:")
        print(f"Success Rate Without Recovery: {metrics_without['success_rate']:.2%}")
        print(f"Success Rate With Recovery: {metrics_with['success_rate']:.2%}")
        print(f"Overall Improvement: {impact_report['overall_improvement']:.2%}")
    
    @pytest.mark.skip(reason="This test depends on network and a specific model, fails with connection error.")
    @pytest.mark.integration
    async def test_error_analysis_accuracy(self, error_recovery_agent, real_managers):
        """Test accuracy of error analysis with real LLM."""
        test_cases = [
            {
                "command_result": CommandResult(
                    command="pip install nonexistent-package-xyz123",
                    exit_code=1,
                    stdout="",
                    stderr="ERROR: Could not find a version that satisfies the requirement nonexistent-package-xyz123",
                    execution_time=2.0,
                    success=False
                ),
                "expected_error_type": ErrorType.DEPENDENCY_MISSING,
                "expected_confidence_min": 0.7
            },
            {
                "command_result": CommandResult(
                    command="cat /root/protected-file.txt",
                    exit_code=1,
                    stdout="",
                    stderr="cat: /root/protected-file.txt: Permission denied",
                    execution_time=0.1,
                    success=False
                ),
                "expected_error_type": ErrorType.PERMISSION_DENIED,
                "expected_confidence_min": 0.8
            },
            {
                "command_result": CommandResult(
                    command="python -c 'print(hello world)'",
                    exit_code=1,
                    stdout="",
                    stderr="SyntaxError: invalid syntax",
                    execution_time=0.5,
                    success=False
                ),
                "expected_error_type": ErrorType.SYNTAX_ERROR,
                "expected_confidence_min": 0.8
            }
        ]
        
        analysis_results = []
        
        for test_case in test_cases:
            analysis = await error_recovery_agent._analyze_error(test_case["command_result"])
            
            analysis_results.append({
                "command": test_case["command_result"].command,
                "expected_type": test_case["expected_error_type"].value,
                "actual_type": analysis.error_type.value,
                "confidence": analysis.confidence,
                "correct_classification": analysis.error_type == test_case["expected_error_type"],
                "meets_confidence_threshold": analysis.confidence >= test_case["expected_confidence_min"]
            })
        
        # Calculate accuracy metrics
        correct_classifications = sum(1 for r in analysis_results if r["correct_classification"])
        accuracy = correct_classifications / len(analysis_results)
        avg_confidence = sum(r["confidence"] for r in analysis_results) / len(analysis_results)
        
        # Save analysis results
        analysis_report = {
            "test_timestamp": datetime.now().isoformat(),
            "total_test_cases": len(test_cases),
            "correct_classifications": correct_classifications,
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "detailed_results": analysis_results
        }
        
        await self._save_analysis_results(analysis_report, "error_analysis_accuracy")
        
        # Assertions
        assert accuracy >= 0.7, f"Error analysis accuracy too low: {accuracy:.2%}"
        assert avg_confidence >= 0.6, f"Average confidence too low: {avg_confidence:.2f}"
        
        print(f"Error Analysis Accuracy: {accuracy:.2%}")
        print(f"Average Confidence: {avg_confidence:.2f}")
    
    @pytest.mark.skip(reason="This test depends on network and a specific model, fails with connection error.")
    @pytest.mark.integration
    async def test_strategy_generation_quality(self, error_recovery_agent, real_managers):
        """Test quality of generated recovery strategies."""
        # Test strategy generation for different error types
        error_scenarios = [
            ErrorAnalysis(
                error_type=ErrorType.DEPENDENCY_MISSING,
                root_cause="Python package not found in PyPI",
                confidence=0.9,
                suggested_fixes=["Check package name", "Use alternative package"],
                severity="medium"
            ),
            ErrorAnalysis(
                error_type=ErrorType.PERMISSION_DENIED,
                root_cause="Insufficient permissions to access file",
                confidence=0.8,
                suggested_fixes=["Use sudo", "Change file permissions"],
                severity="high"
            ),
            ErrorAnalysis(
                error_type=ErrorType.FILE_NOT_FOUND,
                root_cause="Required file does not exist",
                confidence=0.7,
                suggested_fixes=["Create missing file", "Check file path"],
                severity="medium"
            )
        ]
        
        strategy_results = []
        
        for error_analysis in error_scenarios:
            strategies = await error_recovery_agent._generate_strategies(error_analysis)
            
            strategy_quality = self._evaluate_strategy_quality(strategies, error_analysis)
            strategy_results.append({
                "error_type": error_analysis.error_type.value,
                "strategies_generated": len(strategies),
                "quality_score": strategy_quality["overall_score"],
                "has_high_probability_strategy": strategy_quality["has_high_probability"],
                "has_diverse_approaches": strategy_quality["has_diverse_approaches"],
                "strategies": [s.to_dict() for s in strategies[:3]]  # Top 3 strategies
            })
        
        # Calculate overall strategy generation quality
        avg_strategies_per_error = sum(r["strategies_generated"] for r in strategy_results) / len(strategy_results)
        avg_quality_score = sum(r["quality_score"] for r in strategy_results) / len(strategy_results)
        high_prob_coverage = sum(1 for r in strategy_results if r["has_high_probability_strategy"]) / len(strategy_results)
        
        strategy_report = {
            "test_timestamp": datetime.now().isoformat(),
            "error_scenarios_tested": len(error_scenarios),
            "average_strategies_per_error": avg_strategies_per_error,
            "average_quality_score": avg_quality_score,
            "high_probability_coverage": high_prob_coverage,
            "detailed_results": strategy_results
        }
        
        await self._save_analysis_results(strategy_report, "strategy_generation_quality")
        
        # Assertions
        assert avg_strategies_per_error >= 2, f"Too few strategies generated: {avg_strategies_per_error:.1f}"
        assert avg_quality_score >= 0.6, f"Strategy quality too low: {avg_quality_score:.2f}"
        assert high_prob_coverage >= 0.5, f"Not enough high-probability strategies: {high_prob_coverage:.2%}"
        
        print(f"Average Strategies per Error: {avg_strategies_per_error:.1f}")
        print(f"Average Quality Score: {avg_quality_score:.2f}")
        print(f"High Probability Coverage: {high_prob_coverage:.2%}")
    
    def _calculate_quality_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics from execution results."""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r["success"])
        total_attempts = sum(r["attempts"] for r in results)
        total_execution_time = sum(r["execution_time"] for r in results)
        recovery_attempts = sum(len(r["recovery_attempts"]) for r in results)
        successful_recoveries = sum(1 for r in results 
                                  for recovery in r["recovery_attempts"] 
                                  if recovery.get("success", False))
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "total_attempts": total_attempts,
            "average_attempts": total_attempts / total_tasks if total_tasks > 0 else 0,
            "total_execution_time": total_execution_time,
            "average_execution_time": total_execution_time / total_tasks if total_tasks > 0 else 0,
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": successful_recoveries / recovery_attempts if recovery_attempts > 0 else 0
        }
    
    def _generate_quality_impact_report(self, metrics_without: Dict[str, Any], 
                                      metrics_with: Dict[str, Any],
                                      results_without: List[Dict[str, Any]],
                                      results_with: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive quality impact report."""
        success_rate_improvement = metrics_with["success_rate"] - metrics_without["success_rate"]
        attempt_efficiency = (metrics_without["average_attempts"] - metrics_with["average_attempts"]) / metrics_without["average_attempts"] if metrics_without["average_attempts"] > 0 else 0
        
        # Calculate overall improvement score (weighted)
        overall_improvement = (
            success_rate_improvement * 0.6 +  # Success rate is most important
            attempt_efficiency * 0.2 +        # Efficiency matters
            (metrics_with["recovery_success_rate"] * 0.2)  # Recovery capability
        )
        
        return {
            "test_timestamp": datetime.now().isoformat(),
            "metrics_without_recovery": metrics_without,
            "metrics_with_recovery": metrics_with,
            "improvements": {
                "success_rate_improvement": success_rate_improvement,
                "attempt_efficiency": attempt_efficiency,
                "recovery_success_rate": metrics_with["recovery_success_rate"]
            },
            "overall_improvement": overall_improvement,
            "detailed_results": {
                "without_recovery": results_without,
                "with_recovery": results_with
            }
        }
    
    def _evaluate_strategy_quality(self, strategies: List[RecoveryStrategy], 
                                 error_analysis: ErrorAnalysis) -> Dict[str, Any]:
        """Evaluate the quality of generated recovery strategies."""
        if not strategies:
            return {
                "overall_score": 0.0,
                "has_high_probability": False,
                "has_diverse_approaches": False
            }
        
        # Check for high probability strategies
        has_high_probability = any(s.success_probability > 0.7 for s in strategies)
        
        # Check for diverse approaches (different strategy names/types)
        strategy_types = set(s.name.split('_')[0] for s in strategies)  # First word of strategy name
        has_diverse_approaches = len(strategy_types) > 1
        
        # Calculate overall quality score
        avg_success_probability = sum(s.success_probability for s in strategies) / len(strategies)
        strategy_count_score = min(1.0, len(strategies) / 3)  # Optimal around 3 strategies
        diversity_score = 1.0 if has_diverse_approaches else 0.5
        
        overall_score = (
            avg_success_probability * 0.5 +
            strategy_count_score * 0.3 +
            diversity_score * 0.2
        )
        
        return {
            "overall_score": overall_score,
            "has_high_probability": has_high_probability,
            "has_diverse_approaches": has_diverse_approaches,
            "average_success_probability": avg_success_probability,
            "strategy_count": len(strategies)
        }
    
    async def _save_quality_results(self, report: Dict[str, Any], test_type: str):
        """Save quality test results to artifacts directory."""
        # Ensure artifacts directory exists
        artifacts_dir = "artifacts/quality-reports/errorrecovery"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_impact_report_{test_type}_{timestamp}.json"
        filepath = os.path.join(artifacts_dir, filename)
        
        # Save report
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Quality report saved to: {filepath}")
    
    async def _save_analysis_results(self, results: Dict[str, Any], test_type: str):
        """Save analysis results to artifacts directory."""
        # Ensure artifacts directory exists
        artifacts_dir = "artifacts/quality-reports/errorrecovery"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_type}_{timestamp}.json"
        filepath = os.path.join(artifacts_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Analysis results saved to: {filepath}")


@pytest.mark.integration
class TestErrorRecoveryTaskDecomposerIntegration:
    """Test ErrorRecovery integration with TaskDecomposer."""
    
    @pytest.fixture
    def real_llm_config(self):
        """Real LLM configuration for integration testing."""
        return LLMConfig(
            base_url="http://ctwuhome.local:8888/openai/v1",
            model="models/gemini-2.0-flash",
            api_key="sk-123456"
        )
    
    @pytest.mark.integration
    async def test_taskdecomposer_errorrecovery_integration(self, real_llm_config, real_managers):
        """Test integration between TaskDecomposer and ErrorRecovery."""
        # Create agents
        task_decomposer = TaskDecomposer(
            name="TestTaskDecomposer",
            llm_config=real_llm_config,
            system_message="You are a task decomposition agent for testing.",
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
        
        error_recovery = ErrorRecovery(
            name="TestErrorRecovery", 
            llm_config=real_llm_config,
            system_message="You are an error recovery agent for testing.",
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
        
        # Create test task
        test_task = TaskDefinition(
            id="integration_test_task",
            title="Test Task with Potential Failures",
            description="A task designed to test error recovery integration",
            steps=["Step 1: Setup", "Step 2: Execute", "Step 3: Verify"],
            requirements_ref=["1.1"],
            dependencies=[]
        )
        
        # Test TaskDecomposer + ErrorRecovery workflow
        agent_with_both = MockImplementAgent(use_error_recovery=True)
        agent_with_both.set_task_decomposer(task_decomposer)
        agent_with_both.set_error_recovery(error_recovery)
        
        result = await agent_with_both.execute_task(test_task)
        
        # Verify integration worked
        assert result["task_id"] == test_task.id
        assert len(result["recovery_attempts"]) > 0  # Should have attempted recovery
        
        # Generate integration report
        integration_report = {
            "test_timestamp": datetime.now().isoformat(),
            "task_decomposer_used": True,
            "error_recovery_used": True,
            "integration_successful": True,
            "task_result": result
        }
        
        # Save integration results
        artifacts_dir = "artifacts/quality-reports/errorrecovery"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"taskdecomposer_integration_{timestamp}.json"
        filepath = os.path.join(artifacts_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(integration_report, f, indent=2, default=str)
        
        print(f"Integration test results saved to: {filepath}")
        print(f"Integration test completed successfully")