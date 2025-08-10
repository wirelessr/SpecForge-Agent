"""
Integration tests for ImplementAgent with TaskDecomposer integration.

This module tests the enhanced ImplementAgent that uses TaskDecomposer for
intelligent task breakdown and execution, comparing quality metrics before
and after integration.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime

from autogen_framework.agents.implement_agent import ImplementAgent
from autogen_framework.agents.task_decomposer import TaskDecomposer
from autogen_framework.models import LLMConfig, TaskDefinition
from autogen_framework.shell_executor import ShellExecutor


class TestImplementAgentTaskDecomposerIntegration:
    """Test ImplementAgent integration with TaskDecomposer."""
    
    # real_llm_config fixture is provided by tests/integration/conftest.py
    # It loads configuration from .env.integration file for secure testing
    
    @pytest.fixture
    def shell_executor(self):
        """Create ShellExecutor for testing."""
        return ShellExecutor()
    
    @pytest.fixture
    def task_decomposer(self, real_llm_config, real_managers):
        """Create TaskDecomposer for testing."""
        return TaskDecomposer(
            name="test_decomposer",
            llm_config=real_llm_config,
            system_message="You are a task decomposition expert.",
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    @pytest.fixture
    def enhanced_implement_agent(self, real_llm_config, shell_executor, task_decomposer, real_managers):
        """Create enhanced ImplementAgent with TaskDecomposer."""
        return ImplementAgent(
            name="enhanced_implement_agent",
            llm_config=real_llm_config,
            system_message="You are an enhanced implementation agent with intelligent task decomposition.",
            shell_executor=shell_executor,
            task_decomposer=task_decomposer,
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    @pytest.fixture
    def original_implement_agent(self, real_llm_config, shell_executor, real_managers):
        """Create original ImplementAgent without TaskDecomposer for comparison."""
        return ImplementAgent(
            name="original_implement_agent",
            llm_config=real_llm_config,
            system_message="You are an implementation agent.",
            shell_executor=shell_executor,
            task_decomposer=None,  # Use default simple decomposer
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    @pytest.fixture
    def test_tasks(self):
        """Create test tasks for comparison."""
        return [
            TaskDefinition(
                id="test_1",
                title="Create Python Module with Basic Functionality",
                description="Create a Python module named 'calculator.py' with basic arithmetic functions (add, subtract, multiply, divide) and proper error handling.",
                requirements_ref=["1.1", "1.2"],
                steps=[
                    "Create calculator.py file",
                    "Implement arithmetic functions",
                    "Add error handling",
                    "Create basic tests"
                ]
            ),
            TaskDefinition(
                id="test_2", 
                title="Set up Project Structure",
                description="Create a standard Python project structure with src/, tests/, and docs/ directories, and initialize with __init__.py files.",
                requirements_ref=["2.1"],
                steps=[
                    "Create directory structure",
                    "Add __init__.py files",
                    "Create README.md",
                    "Set up basic configuration"
                ]
            )
        ]
    
    @pytest.mark.integration
    async def test_enhanced_vs_original_quality_comparison(
        self, 
        enhanced_implement_agent, 
        original_implement_agent, 
        test_tasks
    ):
        """
        Compare quality metrics between enhanced and original ImplementAgent.
        
        This test executes the same tasks with both agents and compares:
        - Success rates
        - Execution time
        - Command quality
        - Overall execution quality
        """
        results = {
            "enhanced": [],
            "original": [],
            "comparison": {}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            
            # Test enhanced agent with TaskDecomposer
            print("\n=== Testing Enhanced ImplementAgent with TaskDecomposer ===")
            for task in test_tasks:
                print(f"\nExecuting task: {task.title}")
                
                # Create separate work directory for this task
                task_work_dir = work_dir / f"enhanced_{task.id}"
                task_work_dir.mkdir(exist_ok=True)
                
                start_time = asyncio.get_event_loop().time()
                result = await enhanced_implement_agent.execute_task(task, str(task_work_dir))
                execution_time = asyncio.get_event_loop().time() - start_time
                
                result["actual_execution_time"] = execution_time
                result["agent_type"] = "enhanced"
                results["enhanced"].append(result)
                
                print(f"Enhanced result: Success={result['success']}, Time={execution_time:.2f}s")
                if result.get("quality_metrics"):
                    print(f"Quality metrics: {result['quality_metrics']}")
            
            # Test original agent for comparison
            print("\n=== Testing Original ImplementAgent (Baseline) ===")
            for task in test_tasks:
                print(f"\nExecuting task: {task.title}")
                
                # Reset task state for fair comparison
                task.completed = False
                task.retry_count = 0
                task.attempts = []
                
                # Create separate work directory for this task
                task_work_dir = work_dir / f"original_{task.id}"
                task_work_dir.mkdir(exist_ok=True)
                
                start_time = asyncio.get_event_loop().time()
                result = await original_implement_agent.execute_task(task, str(task_work_dir))
                execution_time = asyncio.get_event_loop().time() - start_time
                
                result["actual_execution_time"] = execution_time
                result["agent_type"] = "original"
                results["original"].append(result)
                
                print(f"Original result: Success={result['success']}, Time={execution_time:.2f}s")
        
        # Calculate comparison metrics
        results["comparison"] = self._calculate_comparison_metrics(
            results["enhanced"], 
            results["original"]
        )
        
        # Save results for analysis
        await self._save_comparison_results(results)
        
        # Print comparison summary
        self._print_comparison_summary(results["comparison"])
        
        # Assertions to verify improvement
        comparison = results["comparison"]
        
        # Enhanced agent should have better or equal success rate
        assert comparison["success_rate_improvement"] >= 0, \
            f"Enhanced agent success rate should not be worse: {comparison['success_rate_improvement']}"
        
        # Enhanced agent should have quality metrics (original might not)
        enhanced_with_quality = sum(1 for r in results["enhanced"] if r.get("quality_metrics"))
        assert enhanced_with_quality > 0, "Enhanced agent should provide quality metrics"
        
        # If both succeed, enhanced should have better quality metrics
        if comparison["enhanced_success_rate"] > 0 and comparison["original_success_rate"] > 0:
            enhanced_avg_quality = comparison.get("enhanced_avg_quality_score", 0)
            assert enhanced_avg_quality > 0.5, \
                f"Enhanced agent should have reasonable quality score: {enhanced_avg_quality}"
        
        print(f"\n✅ Integration test completed successfully!")
        print(f"Enhanced agent shows {comparison['success_rate_improvement']:.1%} success rate improvement")
        
        return results
    
    def _calculate_comparison_metrics(self, enhanced_results: list, original_results: list) -> dict:
        """Calculate comparison metrics between enhanced and original results."""
        comparison = {
            "enhanced_success_rate": 0.0,
            "original_success_rate": 0.0,
            "success_rate_improvement": 0.0,
            "enhanced_avg_execution_time": 0.0,
            "original_avg_execution_time": 0.0,
            "execution_time_improvement": 0.0,
            "enhanced_avg_quality_score": 0.0,
            "quality_metrics_available": False
        }
        
        # Calculate success rates
        if enhanced_results:
            comparison["enhanced_success_rate"] = sum(1 for r in enhanced_results if r["success"]) / len(enhanced_results)
            comparison["enhanced_avg_execution_time"] = sum(r["actual_execution_time"] for r in enhanced_results) / len(enhanced_results)
        
        if original_results:
            comparison["original_success_rate"] = sum(1 for r in original_results if r["success"]) / len(original_results)
            comparison["original_avg_execution_time"] = sum(r["actual_execution_time"] for r in original_results) / len(original_results)
        
        # Calculate improvements
        comparison["success_rate_improvement"] = comparison["enhanced_success_rate"] - comparison["original_success_rate"]
        
        if comparison["original_avg_execution_time"] > 0:
            comparison["execution_time_improvement"] = (
                comparison["original_avg_execution_time"] - comparison["enhanced_avg_execution_time"]
            ) / comparison["original_avg_execution_time"]
        
        # Calculate quality metrics for enhanced agent
        enhanced_quality_scores = []
        for result in enhanced_results:
            if result.get("quality_metrics") and result["quality_metrics"].get("overall_score"):
                enhanced_quality_scores.append(result["quality_metrics"]["overall_score"])
        
        if enhanced_quality_scores:
            comparison["enhanced_avg_quality_score"] = sum(enhanced_quality_scores) / len(enhanced_quality_scores)
            comparison["quality_metrics_available"] = True
        
        return comparison
    
    async def _save_comparison_results(self, results: dict):
        """Save comparison results to artifacts directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure artifacts directory exists
        artifacts_dir = Path("artifacts/quality-reports/taskdecomposer-integration")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = artifacts_dir / f"integration_comparison_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        summary_file = artifacts_dir / f"integration_summary_{timestamp}.md"
        summary_content = self._generate_summary_report(results)
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        print(f"\n📊 Results saved to:")
        print(f"  - Detailed: {results_file}")
        print(f"  - Summary: {summary_file}")
    
    def _generate_summary_report(self, results: dict) -> str:
        """Generate markdown summary report."""
        comparison = results["comparison"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# ImplementAgent TaskDecomposer Integration Test Results

## Test Summary

**Test Date**: {timestamp}  
**Tasks Tested**: {len(results["enhanced"])} tasks  
**Test Type**: Quality comparison between enhanced and original ImplementAgent  

## Results Overview

### Success Rates
- **Enhanced Agent**: {comparison["enhanced_success_rate"]:.1%}
- **Original Agent**: {comparison["original_success_rate"]:.1%}
- **Improvement**: {comparison["success_rate_improvement"]:+.1%}

### Execution Time
- **Enhanced Agent**: {comparison["enhanced_avg_execution_time"]:.2f}s average
- **Original Agent**: {comparison["original_avg_execution_time"]:.2f}s average
- **Time Improvement**: {comparison["execution_time_improvement"]:+.1%}

### Quality Metrics
- **Quality Metrics Available**: {comparison["quality_metrics_available"]}
- **Enhanced Agent Quality Score**: {comparison["enhanced_avg_quality_score"]:.2f}

## Task-by-Task Results

### Enhanced Agent Results
"""
        
        for i, result in enumerate(results["enhanced"], 1):
            report += f"""
#### Task {i}: {result["task_title"]}
- **Success**: {result["success"]}
- **Execution Time**: {result["actual_execution_time"]:.2f}s
- **Approaches Attempted**: {len(result["attempts"])}
"""
            if result.get("quality_metrics"):
                qm = result["quality_metrics"]
                report += f"""- **Quality Score**: {qm.get("overall_score", 0):.2f}
- **Command Success Rate**: {qm.get("command_success_rate", 0):.2f}
- **Execution Efficiency**: {qm.get("execution_efficiency", 0):.2f}
"""
        
        report += f"""
### Original Agent Results
"""
        
        for i, result in enumerate(results["original"], 1):
            report += f"""
#### Task {i}: {result["task_title"]}
- **Success**: {result["success"]}
- **Execution Time**: {result["actual_execution_time"]:.2f}s
- **Approaches Attempted**: {len(result["attempts"])}
"""
        
        report += f"""
## Conclusions

### Key Improvements
1. **TaskDecomposer Integration**: Enhanced agent uses intelligent task breakdown
2. **Quality Metrics**: Enhanced agent provides detailed quality analysis
3. **Success Rate**: {comparison["success_rate_improvement"]:+.1%} improvement in task completion
4. **Execution Efficiency**: {comparison["execution_time_improvement"]:+.1%} time improvement

### Recommendations
1. **Deploy Enhanced Agent**: Results support using TaskDecomposer integration
2. **Monitor Quality Metrics**: Continue tracking quality improvements
3. **Expand Testing**: Test with more complex tasks and scenarios

---
**Test Framework**: ImplementAgent TaskDecomposer Integration Tests  
**Quality Threshold**: {'PASSED' if comparison['success_rate_improvement'] >= 0 else 'NEEDS_IMPROVEMENT'}
"""
        
        return report
    
    def _print_comparison_summary(self, comparison: dict):
        """Print comparison summary to console."""
        print(f"\n{'='*60}")
        print(f"TASKDECOMPOSER INTEGRATION COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Enhanced Success Rate:    {comparison['enhanced_success_rate']:.1%}")
        print(f"Original Success Rate:    {comparison['original_success_rate']:.1%}")
        print(f"Success Rate Improvement: {comparison['success_rate_improvement']:+.1%}")
        print(f"")
        print(f"Enhanced Avg Time:        {comparison['enhanced_avg_execution_time']:.2f}s")
        print(f"Original Avg Time:        {comparison['original_avg_execution_time']:.2f}s")
        print(f"Time Improvement:         {comparison['execution_time_improvement']:+.1%}")
        print(f"")
        if comparison['quality_metrics_available']:
            print(f"Enhanced Quality Score:   {comparison['enhanced_avg_quality_score']:.2f}")
        else:
            print(f"Enhanced Quality Score:   Not available")
        print(f"{'='*60}")


@pytest.mark.integration
class TestTaskDecomposerExecutionFlow:
    """Test the specific TaskDecomposer execution flow in ImplementAgent."""
    
    # real_llm_config fixture is provided by tests/integration/conftest.py
    # It loads configuration from .env.integration file for secure testing
    
    @pytest.fixture
    def enhanced_agent(self, real_llm_config, real_managers):
        """Create enhanced ImplementAgent for testing."""
        shell_executor = ShellExecutor()
        return ImplementAgent(
            name="test_enhanced_agent",
            llm_config=real_llm_config,
            system_message="You are an enhanced implementation agent.",
            shell_executor=shell_executor,
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    async def test_task_decomposer_execution_flow(self, enhanced_agent):
        """Test the complete TaskDecomposer execution flow."""
        task = TaskDefinition(
            id="flow_test",
            title="Create Simple Python Script",
            description="Create a simple Python script that prints 'Hello, TaskDecomposer!' and saves the output to a file.",
            requirements_ref=["1.1"],
            steps=["Create Python script", "Execute script", "Verify output"]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await enhanced_agent.execute_task(task, temp_dir)
            
            # Verify TaskDecomposer was used
            assert result.get("decomposition_plan") is not None, "TaskDecomposer should provide decomposition plan"
            
            # Verify quality metrics are available
            assert result.get("quality_metrics") is not None, "Quality metrics should be available"
            
            # Verify execution details
            decomposition_plan = result["decomposition_plan"]
            assert decomposition_plan["commands_count"] > 0, "Should have generated commands"
            assert decomposition_plan["decomposition_time"] > 0, "Should have decomposition time"
            
            # Verify quality metrics structure
            quality_metrics = result["quality_metrics"]
            expected_metrics = ["overall_score", "command_success_rate", "execution_efficiency", "plan_accuracy"]
            for metric in expected_metrics:
                assert metric in quality_metrics, f"Quality metrics should include {metric}"
            
            print(f"\n✅ TaskDecomposer execution flow test completed!")
            print(f"Decomposition plan: {decomposition_plan['commands_count']} commands")
            print(f"Quality score: {quality_metrics['overall_score']:.2f}")
            
            return result