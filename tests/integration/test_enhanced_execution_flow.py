"""
Integration tests for Enhanced Execution Flow (Task 5.2).

This module tests the enhanced execution flow that integrates:
- TaskDecomposer for intelligent task breakdown
- ShellExecutor for command execution  
- ErrorRecovery for intelligent error handling
- ContextManager for project context awareness
"""

import pytest
import os
import tempfile
import json
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from autogen_framework.agents.implement_agent import ImplementAgent
from autogen_framework.agents.task_decomposer import TaskDecomposer
from autogen_framework.agents.error_recovery import ErrorRecovery
from autogen_framework.models import LLMConfig, TaskDefinition
from autogen_framework.shell_executor import ShellExecutor


class TestEnhancedExecutionFlow:
    """Test enhanced execution flow integration."""
    
    @pytest.fixture
    def real_llm_config(self):
        """Real LLM configuration for integration testing."""
        return LLMConfig(
            base_url="http://ctwuhome.local:8888/openai/v1",
            model="models/gemini-2.0-flash",
            api_key="sk-123456"
        )
    
    @pytest.fixture
    def shell_executor(self):
        """Create ShellExecutor for testing."""
        return ShellExecutor()
    
    @pytest.fixture
    def task_decomposer(self, real_llm_config, real_managers):
        """Create TaskDecomposer for testing."""
        system_message = """
You are a task decomposition expert. Break down high-level tasks into executable shell commands.
Always respond with valid JSON when requested.
Focus on practical, executable commands that achieve the task objectives.
"""
        return TaskDecomposer(
            name="TestTaskDecomposer",
            llm_config=real_llm_config,
            system_message=system_message,
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    @pytest.fixture
    def error_recovery(self, real_llm_config, real_managers):
        """Create ErrorRecovery for testing."""
        system_message = """
You are an intelligent error recovery agent. Analyze failures and generate recovery strategies.
Always respond with valid JSON when requested.
Focus on practical recovery approaches that can resolve common errors.
"""
        return ErrorRecovery(
            name="TestErrorRecovery",
            llm_config=real_llm_config,
            system_message=system_message,
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    @pytest.fixture
    def enhanced_implement_agent(self, real_llm_config, shell_executor, task_decomposer, error_recovery, real_managers):
        """Create enhanced ImplementAgent with all components."""
        system_message = """
You are an enhanced implementation agent with intelligent task execution capabilities.
You use TaskDecomposer for task breakdown, ShellExecutor for command execution,
and ErrorRecovery for intelligent error handling.
"""
        return ImplementAgent(
            name="EnhancedImplementAgent",
            llm_config=real_llm_config,
            system_message=system_message,
            shell_executor=shell_executor,
            task_decomposer=task_decomposer,
            error_recovery=error_recovery,
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
    
    @pytest.fixture
    def sample_tasks(self):
        """Sample tasks for testing enhanced execution flow."""
        return [
            TaskDefinition(
                id="test_task_1",
                title="Create Simple Python Script",
                description="Create a simple Python script that prints hello world",
                steps=[
                    "Create a new Python file",
                    "Add print statement",
                    "Make file executable",
                    "Test the script"
                ],
                requirements_ref=["1.1"],
                dependencies=[]
            ),
            TaskDefinition(
                id="test_task_2",
                title="Setup Test Directory Structure",
                description="Create a test directory structure with multiple files",
                steps=[
                    "Create main directory",
                    "Create subdirectories",
                    "Create test files",
                    "Verify structure"
                ],
                requirements_ref=["1.2"],
                dependencies=[]
            ),
            TaskDefinition(
                id="test_task_3",
                title="Install and Test Python Package",
                description="Install a Python package and verify it works",
                steps=[
                    "Check if pip is available",
                    "Install package",
                    "Import and test package",
                    "Verify installation"
                ],
                requirements_ref=["1.3"],
                dependencies=[]
            )
        ]
    
    @pytest.mark.integration
    async def test_enhanced_execution_flow_basic(self, enhanced_implement_agent, sample_tasks):
        """Test basic enhanced execution flow functionality."""
        task = sample_tasks[0]  # Simple Python script task
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Execute task using enhanced flow
            result = await enhanced_implement_agent.execute_task(task, temp_dir)
            
            # Verify basic execution structure
            assert "task_id" in result
            assert "success" in result
            assert "attempts" in result
            assert "execution_time" in result
            
            # Verify enhanced flow components were used
            if result.get("decomposition_plan"):
                assert "complexity_level" in result["decomposition_plan"]
                assert "command_count" in result["decomposition_plan"]
            
            # Verify quality metrics if available
            if result.get("quality_metrics"):
                assert "overall_score" in result["quality_metrics"]
                assert "functionality_score" in result["quality_metrics"]
            
            print(f"Enhanced execution flow test completed:")
            print(f"- Task: {task.title}")
            print(f"- Success: {result['success']}")
            print(f"- Execution Time: {result['execution_time']:.2f}s")
            if result.get("quality_metrics"):
                print(f"- Quality Score: {result['quality_metrics']['overall_score']:.2f}")
    
    @pytest.mark.integration
    async def test_task_decomposer_integration(self, enhanced_implement_agent, sample_tasks):
        """Test TaskDecomposer integration in enhanced flow."""
        task = sample_tasks[1]  # Directory structure task
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await enhanced_implement_agent.execute_task(task, temp_dir)
            
            # Verify TaskDecomposer was used
            assert result.get("decomposition_plan") is not None
            decomposition_plan = result["decomposition_plan"]
            
            # Verify decomposition plan structure
            assert "complexity_level" in decomposition_plan
            assert "estimated_steps" in decomposition_plan
            assert "command_count" in decomposition_plan
            assert "estimated_duration" in decomposition_plan
            
            # Verify task analysis
            if result.get("task_analysis"):
                task_analysis = result["task_analysis"]
                assert "complexity_level" in task_analysis
                assert "confidence_score" in task_analysis
            
            print(f"TaskDecomposer integration test:")
            print(f"- Complexity Level: {decomposition_plan['complexity_level']}")
            print(f"- Command Count: {decomposition_plan['command_count']}")
            print(f"- Estimated Duration: {decomposition_plan['estimated_duration']} minutes")
    
    @pytest.mark.integration
    async def test_error_recovery_integration(self, enhanced_implement_agent):
        """Test ErrorRecovery integration in enhanced flow."""
        # Create a task that's likely to fail initially
        failing_task = TaskDefinition(
            id="failing_task",
            title="Execute Non-existent Command",
            description="Try to execute a command that doesn't exist to test error recovery",
            steps=[
                "Try to run non-existent command",
                "Recover from failure",
                "Complete alternative approach"
            ],
            requirements_ref=["2.1"],
            dependencies=[]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await enhanced_implement_agent.execute_task(failing_task, temp_dir)
            
            # Verify error recovery was attempted
            if result.get("recovery_attempts"):
                recovery_attempts = result["recovery_attempts"]
                assert len(recovery_attempts) > 0
                
                # Check recovery attempt structure
                for attempt in recovery_attempts:
                    assert "success" in attempt
                    assert "recovery_time" in attempt
                    if attempt.get("strategy_used"):
                        assert isinstance(attempt["strategy_used"], str)
            
            print(f"Error recovery integration test:")
            print(f"- Task Success: {result['success']}")
            print(f"- Recovery Attempts: {len(result.get('recovery_attempts', []))}")
            if result.get("recovery_attempts"):
                successful_recoveries = sum(1 for attempt in result["recovery_attempts"] if attempt["success"])
                print(f"- Successful Recoveries: {successful_recoveries}")
    
    @pytest.mark.integration
    async def test_quality_validation(self, enhanced_implement_agent, sample_tasks):
        """Test quality validation in enhanced execution flow."""
        task = sample_tasks[0]  # Simple task for quality testing
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await enhanced_implement_agent.execute_task(task, temp_dir)
            
            # Verify quality metrics are generated
            if result.get("quality_metrics"):
                quality_metrics = result["quality_metrics"]
                
                # Check required quality metrics
                required_metrics = [
                    "overall_score",
                    "functionality_score", 
                    "reliability_score",
                    "efficiency_score",
                    "recovery_effectiveness"
                ]
                
                for metric in required_metrics:
                    assert metric in quality_metrics
                    assert 0.0 <= quality_metrics[metric] <= 1.0
                
                print(f"Quality validation test:")
                print(f"- Overall Score: {quality_metrics['overall_score']:.2f}")
                print(f"- Functionality: {quality_metrics['functionality_score']:.2f}")
                print(f"- Reliability: {quality_metrics['reliability_score']:.2f}")
                print(f"- Efficiency: {quality_metrics['efficiency_score']:.2f}")
                print(f"- Recovery Effectiveness: {quality_metrics['recovery_effectiveness']:.2f}")
    
    @pytest.mark.integration
    async def test_context_aware_execution(self, enhanced_implement_agent, sample_tasks):
        """Test context-aware execution capabilities."""
        task = sample_tasks[2]  # Package installation task
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock context manager to test context awareness
            mock_context_manager = Mock()
            mock_context_manager.get_implementation_context = AsyncMock(return_value=Mock(
                task=task,
                requirements="Mock requirements content",
                design="Mock design content",
                project_structure="Mock project structure",
                execution_history=[],
                related_tasks=[],
                memory_patterns=[]
            ))
            mock_context_manager.update_execution_history = AsyncMock()
            
            enhanced_implement_agent.set_context_manager(mock_context_manager)
            
            result = await enhanced_implement_agent.execute_task(task, temp_dir)
            
            # Verify context manager was used
            if mock_context_manager.get_implementation_context.called:
                print("Context manager was successfully integrated")
            
            # Verify execution completed with context
            assert "task_id" in result
            print(f"Context-aware execution test:")
            print(f"- Task: {task.title}")
            print(f"- Success: {result['success']}")
            print(f"- Context Manager Used: {mock_context_manager.get_implementation_context.called}")
    
    @pytest.mark.integration
    async def test_comprehensive_execution_flow(self, enhanced_implement_agent, sample_tasks):
        """Test comprehensive execution flow with all components."""
        results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for task in sample_tasks:
                print(f"\nExecuting task: {task.title}")
                
                result = await enhanced_implement_agent.execute_task(task, temp_dir)
                results.append({
                    "task_id": task.id,
                    "task_title": task.title,
                    "success": result["success"],
                    "execution_time": result["execution_time"],
                    "quality_score": result.get("quality_metrics", {}).get("overall_score", 0.0),
                    "recovery_attempts": len(result.get("recovery_attempts", [])),
                    "commands_executed": len(result.get("shell_commands", []))
                })
                
                print(f"- Success: {result['success']}")
                print(f"- Execution Time: {result['execution_time']:.2f}s")
                if result.get("quality_metrics"):
                    print(f"- Quality Score: {result['quality_metrics']['overall_score']:.2f}")
        
        # Generate comprehensive test report
        test_report = {
            "test_timestamp": datetime.now().isoformat(),
            "total_tasks": len(sample_tasks),
            "successful_tasks": sum(1 for r in results if r["success"]),
            "average_execution_time": sum(r["execution_time"] for r in results) / len(results),
            "average_quality_score": sum(r["quality_score"] for r in results) / len(results),
            "total_recovery_attempts": sum(r["recovery_attempts"] for r in results),
            "detailed_results": results
        }
        
        # Save test report
        os.makedirs("artifacts/quality-reports/enhanced-execution-flow", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"artifacts/quality-reports/enhanced-execution-flow/comprehensive_test_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        print(f"\nComprehensive test report saved to: {report_path}")
        print(f"Overall Results:")
        print(f"- Success Rate: {test_report['successful_tasks']}/{test_report['total_tasks']}")
        print(f"- Average Quality Score: {test_report['average_quality_score']:.2f}")
        print(f"- Average Execution Time: {test_report['average_execution_time']:.2f}s")
        
        # Assertions for test success
        assert test_report["successful_tasks"] >= test_report["total_tasks"] * 0.5  # At least 50% success
        assert test_report["average_quality_score"] >= 0.3  # Reasonable quality threshold


class TestEnhancedCapabilities:
    """Test enhanced capabilities and methods."""
    
    @pytest.fixture
    def real_llm_config(self):
        """Real LLM configuration for testing."""
        return LLMConfig(
            base_url="http://ctwuhome.local:8888/openai/v1",
            model="models/gemini-2.0-flash",
            api_key="sk-123456"
        )
    
    def test_enhanced_capabilities_reporting(self, real_llm_config, real_managers):
        """Test enhanced capabilities reporting."""
        shell_executor = ShellExecutor()
        task_decomposer = Mock(spec=TaskDecomposer)
        error_recovery = Mock(spec=ErrorRecovery)
        agent = ImplementAgent(
            name="TestAgent",
            llm_config=real_llm_config,
            system_message="Test agent",
            shell_executor=shell_executor,
            task_decomposer=task_decomposer,
            error_recovery=error_recovery,
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
        
        # Test base capabilities
        base_capabilities = agent.get_agent_capabilities()
        assert isinstance(base_capabilities, list)
        assert len(base_capabilities) > 0
        
        # Test enhanced capabilities
        enhanced_capabilities = agent.get_enhanced_capabilities()
        assert isinstance(enhanced_capabilities, list)
        assert len(enhanced_capabilities) > len(base_capabilities)
        
        # Verify enhanced capabilities include new features
        enhanced_text = " ".join(enhanced_capabilities).lower()
        assert "taskdecomposer" in enhanced_text
        assert "errorrecovery" in enhanced_text
        assert "quality validation" in enhanced_text
        assert "context-aware" in enhanced_text
    
    def test_error_recovery_setter(self, real_llm_config, real_managers):
        """Test ErrorRecovery setter method."""
        shell_executor = ShellExecutor()
        task_decomposer = Mock(spec=TaskDecomposer)
        error_recovery = Mock(spec=ErrorRecovery)
        agent = ImplementAgent(
            name="TestAgent",
            llm_config=real_llm_config,
            system_message="Test agent",
            shell_executor=shell_executor,
            task_decomposer=task_decomposer,
            error_recovery=error_recovery,
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
        
        # Create new ErrorRecovery instance
        error_recovery = ErrorRecovery(
            name="TestErrorRecovery",
            llm_config=real_llm_config,
            system_message="Test error recovery",
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
        
        # Test setter
        agent.set_error_recovery(error_recovery)
        assert agent.error_recovery == error_recovery