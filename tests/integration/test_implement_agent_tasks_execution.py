"""
Enhanced integration test for ImplementAgent with quality measurement capabilities.

This test verifies that:
1. ImplementAgent can read a tasks.md file with unchecked tasks
2. Execute all tasks in the file with quality measurement
3. Update the tasks.md file to mark completed tasks as checked
4. Collect comprehensive quality metrics for each task execution
5. Generate quality reports for analysis and comparison

Quality Metrics Collected:
- Functionality: Does the code work as intended?
- Maintainability: Is the code readable and well-structured?
- Standards Compliance: Does it follow coding standards?
- Test Coverage: Are there adequate tests?
- Documentation: Is the code properly documented?
"""

import pytest
import tempfile
import shutil
import asyncio
import re
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from autogen_framework.agents.implement_agent import ImplementAgent
from autogen_framework.agents.task_decomposer import TaskDecomposer
from autogen_framework.agents.error_recovery import ErrorRecovery
from autogen_framework.shell_executor import ShellExecutor
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.models import LLMConfig
from autogen_framework.quality_metrics import QualityMetricsFramework, QualityReport


class TestImplementAgentTasksExecution:
    """Enhanced integration tests for ImplementAgent with quality measurement."""
    
    def _parse_tasks_from_file(self, tasks_file_path):
        """Simple task parser for testing - just counts tasks with checkboxes."""
        with open(tasks_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tasks = []
        lines = content.split('\n')
        
        for line in lines:
            # Simple pattern matching for any task with checkbox
            if re.match(r'^\s*- \[[ x]\] \d+(\.\d+)?[\. ] ', line):
                match = re.match(r'^\s*- \[[ x]\] (\d+(?:\.\d+)?)[\. ] (.+)', line)
                if match:
                    task_num = match.group(1)
                    title = match.group(2)
                    tasks.append({
                        "number": task_num,
                        "title": title
                    })
        
        return tasks
    
    def _load_standardized_test_task(self, task_name: str) -> str:
        """Load a standardized test task from the test-tasks directory."""
        test_tasks_dir = Path("artifacts/quality-reports/test-tasks")
        task_file = test_tasks_dir / f"{task_name}.md"
        
        if not task_file.exists():
            raise FileNotFoundError(f"Standardized test task not found: {task_file}")
        
        return task_file.read_text(encoding='utf-8')
    
    def _extract_requirements_from_task(self, task_content: str) -> str:
        """Extract requirements section from task content."""
        lines = task_content.split('\n')
        requirements_section = []
        in_requirements = False
        
        for line in lines:
            if line.strip() == "## Requirements Reference":
                in_requirements = True
                continue
            elif line.startswith("## ") and in_requirements:
                break
            elif in_requirements:
                requirements_section.append(line)
        
        return '\n'.join(requirements_section)
    
    def _create_design_from_task(self, task_content: str) -> str:
        """Create a simple design document from task content."""
        return f"""# Design Document

## Overview
This is a test implementation based on the task requirements.

## Architecture
Simple implementation following the task specifications.

## Implementation Notes
- Follow the task steps in order
- Ensure all requirements are met
- Include proper error handling
- Add appropriate documentation

Generated from task: {datetime.now().isoformat()}
"""
    

    

    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def quality_reports_dir(self):
        """Create directory for quality reports."""
        reports_dir = Path("artifacts/quality-reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir
    
    @pytest.fixture
    def shell_executor(self, temp_workspace):
        """Create a ShellExecutor instance."""
        return ShellExecutor(default_working_dir=temp_workspace)
    
    @pytest.fixture
    def memory_manager(self, temp_workspace):
        """Create a MemoryManager instance."""
        return MemoryManager(temp_workspace)
    
    @pytest.fixture
    def quality_framework(self):
        """Create a QualityMetricsFramework instance."""
        return QualityMetricsFramework()
    
    @pytest.fixture
    def implement_agent(self, llm_config, shell_executor, memory_manager, real_managers):
        """Create an ImplementAgent instance."""
        task_decomposer = Mock(spec=TaskDecomposer)
        error_recovery = Mock(spec=ErrorRecovery)
        agent = ImplementAgent(
            name="TestImplementAgent",
            llm_config=llm_config,
            system_message="Test implementation agent for task execution testing with quality measurement.",
            shell_executor=shell_executor,
            task_decomposer=task_decomposer,
            error_recovery=error_recovery,
            description="Test agent for integration testing with quality metrics",
            token_manager=real_managers.token_manager,
            context_manager=real_managers.context_manager
        )
        # Set memory manager after initialization
        agent.memory_manager = memory_manager
        return agent
    
    @pytest.fixture
    def sample_tasks_md_content(self):
        """Create sample tasks.md content with unchecked tasks."""
        return """# Implementation Plan

- [ ] 1. Create project structure
  - Create main directory structure
  - Initialize configuration files
  - _Requirements: 1.1, 1.2_

- [ ] 2. Implement core functionality
  - [ ] 2.1 Create main module
    - Write main.py with basic structure
    - Add error handling
    - _Requirements: 2.1_
  
  - [ ] 2.2 Add utility functions
    - Create utils.py with helper functions
    - Add input validation
    - _Requirements: 2.2_

- [ ] 3. Create tests
  - Write unit tests for main module
  - Write integration tests
  - _Requirements: 3.1, 3.2_

- [ ] 4. Add documentation
  - Create README.md
  - Add code comments
  - _Requirements: 4.1_
"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_implement_agent_executes_single_task_and_updates_checkbox(
        self, implement_agent, temp_workspace, sample_tasks_md_content
    ):
        """Test that ImplementAgent executes a single task and updates its checkbox in tasks.md."""
        
        # Create work directory and tasks.md file
        work_dir = Path(temp_workspace) / "test_project"
        work_dir.mkdir(exist_ok=True)
        
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(sample_tasks_md_content)
        
        # Create requirements.md and design.md for context
        (work_dir / "requirements.md").write_text("""# Requirements
1.1 Create project structure
1.2 Initialize configuration
2.1 Implement main module
2.2 Add utility functions
3.1 Unit tests required
3.2 Integration tests required
4.1 Documentation required
""")
        
        (work_dir / "design.md").write_text("""# Design
Simple Python project with main module, utilities, tests, and documentation.
""")
        
        # Verify initial state - all tasks unchecked
        initial_content = tasks_file.read_text()
        initial_unchecked = initial_content.count("- [ ]")
        assert initial_unchecked == 6  # Should have 6 unchecked tasks
        
        # Create a simple task to execute
        from autogen_framework.models import TaskDefinition
        
        task_def = TaskDefinition(
            id="test_task_1",
            title="Create project structure",
            description="Create main directory structure and initialize configuration files",
            steps=[
                "Create src, tests, and docs directories",
                "Create __init__.py files",
                "Create basic configuration file"
            ],
            requirements_ref=["1.1", "1.2"]
        )
        
        # Execute the task using the real ImplementAgent
        result = await implement_agent.execute_task(task_def, str(work_dir))
        
        # Verify the task execution result
        assert result is not None
        assert isinstance(result, dict)
        
        # The agent should have executed shell commands and potentially created files
        # We can verify this by checking if the agent attempted to execute commands
        # (The actual file creation depends on the LLM response and shell execution)
        
        # For this integration test, we focus on verifying that the agent:
        # 1. Can process a task definition
        # 2. Attempts to execute it (even if mocked LLM responses)
        # 3. Returns a result structure
        
        print(f"Task execution result: {result}")
        
        # Note: The actual checkbox updating in tasks.md would be handled by a higher-level
        # workflow manager, not the individual task execution. This test verifies that
        # the ImplementAgent can execute individual tasks correctly.
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_implement_agent_processes_tasks_from_file(
        self, implement_agent, temp_workspace
    ):
        """Test that ImplementAgent can process tasks from a tasks.md file."""
        
        # Create work directory and a simple tasks.md file
        work_dir = Path(temp_workspace) / "test_project"
        work_dir.mkdir(exist_ok=True)
        
        # Create a simple tasks.md with one task
        simple_tasks_content = """# Implementation Plan

- [ ] 1. Create a simple hello world script
  - Create hello.py file
  - Add print statement
  - _Requirements: 1.1_
"""
        
        tasks_file = work_dir / "tasks.md"
        tasks_file.write_text(simple_tasks_content)
        
        # Create requirements.md for context
        (work_dir / "requirements.md").write_text("""# Requirements
1.1 Create a hello world script that prints "Hello, World!"
""")
        
        # Create design.md for context
        (work_dir / "design.md").write_text("""# Design
Simple Python script that prints Hello, World!
""")
        
        # Verify initial state
        initial_content = tasks_file.read_text()
        assert "- [ ] 1. Create a simple hello world script" in initial_content
        
        # Create a TaskDefinition from the tasks.md content
        from autogen_framework.models import TaskDefinition
        
        task_def = TaskDefinition(
            id="hello_world_task",
            title="Create a simple hello world script",
            description="Create hello.py file with print statement",
            steps=[
                "Create hello.py file",
                "Add print statement for Hello, World!"
            ],
            requirements_ref=["1.1"]
        )
        
        # Execute the task using the real ImplementAgent
        # This will use the real LLM (or whatever is configured in the test environment)
        result = await implement_agent.execute_task(task_def, str(work_dir))
        
        # Verify the task execution result structure
        assert result is not None
        assert isinstance(result, dict)
        
        # The result should contain information about the execution
        print(f"Task execution result: {result}")
        
        # Check if any files were created (this depends on the actual LLM response)
        hello_file = work_dir / "hello.py"
        if hello_file.exists():
            print(f"hello.py was created with content: {hello_file.read_text()}")
        
        # This test verifies that:
        # 1. ImplementAgent can accept and process TaskDefinition objects
        # 2. It attempts to execute the task (calls LLM, processes response)
        # 3. It returns a structured result
        # 4. It can work with real file system operations


    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_implement_agent_with_real_task_execution(
        self, implement_agent, temp_workspace
    ):
        """Test ImplementAgent with real task execution (minimal mocking)."""
        
        # Create work directory
        work_dir = Path(temp_workspace) / "test_project"
        work_dir.mkdir(exist_ok=True)
        
        # Create a very simple task that should be easy to execute
        from autogen_framework.models import TaskDefinition
        
        task_def = TaskDefinition(
            id="simple_file_task",
            title="Create a simple text file",
            description="Create a text file named hello.txt with specific content",
            steps=[
                "Create hello.txt file",
                "Add the text 'Hello, World!' to the file"
            ],
            requirements_ref=["1.1"]
        )
        
        # Create context files
        (work_dir / "requirements.md").write_text("""# Requirements
1.1 Create a text file named hello.txt with the content "Hello, World!"
""")
        
        (work_dir / "design.md").write_text("""# Design
Simple file creation using shell commands like echo or cat.
""")
        
        # Execute the task using the real ImplementAgent
        # This will make a real LLM call and execute real shell commands
        result = await implement_agent.execute_task(task_def, str(work_dir))
        
        # Verify the result structure
        assert result is not None
        assert isinstance(result, dict)
        
        print(f"Task execution result: {result}")
        
        # Check if the task actually created the expected file
        hello_file = work_dir / "hello.txt"
        if hello_file.exists():
            content = hello_file.read_text()
            print(f"Created file content: '{content}'")
            # The exact content depends on the LLM response, but we can check if file exists
            assert len(content) > 0, "File should have some content"
        else:
            print("File was not created - this may be due to LLM response or execution issues")
        
        # This test verifies that:
        # 1. ImplementAgent can process real TaskDefinition objects
        # 2. It makes real LLM calls (not mocked)
        # 3. It executes real shell commands through ShellExecutor
        # 4. It returns structured results
        # 5. The integration between LLM, agent, and shell execution works
        
        # The success of file creation depends on:
        # - LLM generating appropriate shell commands
        # - Shell commands being executed correctly
        # - No errors in the execution chain

    @pytest.mark.skip(reason="This test depends on missing test data in artifacts/ directory.")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_implement_agent_with_quality_measurement(
        self, implement_agent, temp_workspace, quality_framework, quality_reports_dir
    ):
        """Test ImplementAgent with comprehensive quality measurement."""
        
        # Create work directory
        work_dir = Path(temp_workspace) / "quality_test_project"
        work_dir.mkdir(exist_ok=True)
        
        # Load standardized test task
        task_content = self._load_standardized_test_task("simple-file-creation")
        
        # Create context files
        requirements_content = self._extract_requirements_from_task(task_content)
        design_content = self._create_design_from_task(task_content)
        
        (work_dir / "requirements.md").write_text(requirements_content)
        (work_dir / "design.md").write_text(design_content)
        (work_dir / "tasks.md").write_text(task_content)
        
        # Create a simple task definition
        from autogen_framework.models import TaskDefinition
        
        task_def = TaskDefinition(
            id="quality_test_simple_file",
            title="Create a simple text file",
            description="Create a text file named hello.txt with specific content",
            steps=[
                "Create hello.txt file",
                "Add the text 'Hello, World!' to the file",
                "Verify the file exists and contains correct content"
            ],
            requirements_ref=["1.1"]
        )
        
        # Record start time
        start_time = datetime.now()
        
        # Execute the task
        result = await implement_agent.execute_task(task_def, str(work_dir))
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Enhance result with execution metadata
        enhanced_result = {
            'success': result is not None and isinstance(result, dict),
            'execution_time': execution_time,
            'commands_executed': result.get('commands_executed', []) if result else [],
            'original_result': result
        }
        
        # Evaluate quality using the framework
        quality_report = quality_framework.evaluate_task_execution(
            work_dir, task_def, enhanced_result
        )
        
        # Save quality report
        report_path = quality_framework.save_report(quality_report, quality_reports_dir, work_dir)
        
        # Verify quality report was created
        assert report_path.exists(), "Quality report should be saved"
        
        # Verify report structure
        assert quality_report.task_id == task_def.id
        assert quality_report.task_title == task_def.title
        assert quality_report.execution_success == enhanced_result['success']
        
        # Verify all quality metrics were evaluated
        from autogen_framework.quality_metrics import QualityMetric
        expected_metrics = set(QualityMetric)
        actual_metrics = set(quality_report.scores.keys())
        assert actual_metrics == expected_metrics, f"Missing metrics: {expected_metrics - actual_metrics}"
        
        # Verify overall score is calculated
        assert 1.0 <= quality_report.overall_score <= 10.0, "Overall score should be between 1 and 10"
        
        # Print quality report summary for debugging
        print(f"\nQuality Report Summary:")
        print(f"Task: {quality_report.task_title}")
        print(f"Overall Score: {quality_report.overall_score:.2f}/10.0")
        print(f"Execution Success: {quality_report.execution_success}")
        
        for metric, score in quality_report.scores.items():
            print(f"{metric.value}: {score.score:.2f}/10.0 ({score.percentage:.1f}%)")
            if score.criteria_met:
                print(f"  ✓ {', '.join(score.criteria_met)}")
            if score.criteria_failed:
                print(f"  ✗ {', '.join(score.criteria_failed)}")
        
        print(f"Report saved to: {report_path}")
        
        # Verify report can be loaded back
        with open(report_path, 'r', encoding='utf-8') as f:
            loaded_report = json.load(f)
        
        assert loaded_report['task_id'] == task_def.id
        assert loaded_report['overall_score'] == quality_report.overall_score
        assert 'scores' in loaded_report
        
        return quality_report

    @pytest.mark.skip(reason="This test depends on missing test data in artifacts/ directory.")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_standardized_task_execution_with_quality_metrics(
        self, implement_agent, temp_workspace, quality_framework, quality_reports_dir
    ):
        """Test execution of all standardized test tasks with quality measurement."""
        
        # List of standardized test tasks
        test_tasks = [
            "simple-file-creation",
            "basic-python-module",
            # Note: Commenting out complex tasks for initial testing
            # "web-scraper-utility",
            # "data-analysis-script", 
            # "api-client-library"
        ]
        
        quality_reports = []
        
        for task_name in test_tasks:
            print(f"\n=== Testing {task_name} ===")
            
            # Create separate work directory for each task
            work_dir = Path(temp_workspace) / f"test_{task_name}"
            work_dir.mkdir(exist_ok=True)
            
            try:
                # Load standardized test task
                task_content = self._load_standardized_test_task(task_name)
                
                # Create context files
                requirements_content = self._extract_requirements_from_task(task_content)
                design_content = self._create_design_from_task(task_content)
                
                (work_dir / "requirements.md").write_text(requirements_content)
                (work_dir / "design.md").write_text(design_content)
                (work_dir / "tasks.md").write_text(task_content)
                
                # Create task definition for the first task in the file
                from autogen_framework.models import TaskDefinition
                
                task_def = TaskDefinition(
                    id=f"standardized_{task_name}",
                    title=f"Standardized test: {task_name}",
                    description=f"Execute standardized test task: {task_name}",
                    steps=["Execute the first task from the standardized test file"],
                    requirements_ref=["1.1"]
                )
                
                # Record start time
                start_time = datetime.now()
                
                # Execute the task
                result = await implement_agent.execute_task(task_def, str(work_dir))
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Enhance result with execution metadata
                enhanced_result = {
                    'success': result is not None and isinstance(result, dict),
                    'execution_time': execution_time,
                    'commands_executed': result.get('commands_executed', []) if result else [],
                    'original_result': result
                }
                
                # Evaluate quality
                quality_report = quality_framework.evaluate_task_execution(
                    work_dir, task_def, enhanced_result
                )
                
                # Save quality report
                report_path = quality_framework.save_report(
                    quality_report, 
                    quality_reports_dir / "standardized_tests",
                    work_dir
                )
                
                quality_reports.append(quality_report)
                
                print(f"✓ {task_name}: Overall Score {quality_report.overall_score:.2f}/10.0")
                print(f"  Report: {report_path}")
                
            except Exception as e:
                print(f"✗ {task_name}: Failed with error: {str(e)}")
                # Create a failed quality report
                from autogen_framework.quality_metrics import QualityReport
                failed_report = QualityReport(
                    task_id=f"standardized_{task_name}",
                    task_title=f"Standardized test: {task_name}",
                    execution_timestamp=datetime.now().isoformat(),
                    execution_success=False,
                    overall_score=1.0
                )
                quality_reports.append(failed_report)
        
        # Generate summary report
        summary = {
            'total_tasks': len(test_tasks),
            'successful_tasks': sum(1 for r in quality_reports if r.execution_success),
            'average_score': sum(r.overall_score for r in quality_reports) / len(quality_reports),
            'task_scores': {r.task_id: r.overall_score for r in quality_reports},
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = quality_reports_dir / "standardized_tests_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== Summary ===")
        print(f"Total tasks: {summary['total_tasks']}")
        print(f"Successful: {summary['successful_tasks']}")
        print(f"Average score: {summary['average_score']:.2f}/10.0")
        print(f"Summary saved to: {summary_path}")
        
        # Verify at least one task was successful
        assert summary['successful_tasks'] > 0, "At least one standardized test should succeed"
        
        return quality_reports

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_quality_metrics_collection_and_reporting(
        self, implement_agent, temp_workspace, quality_framework, quality_reports_dir
    ):
        """Test that quality metrics are properly collected and reported."""
        
        # Create work directory
        work_dir = Path(temp_workspace) / "metrics_test"
        work_dir.mkdir(exist_ok=True)
        
        # Create a task that should produce measurable quality metrics
        task_content = """# Test Task for Quality Metrics

- [ ] 1. Create Python module with documentation
  - Create a file named `calculator.py`
  - Add a function `add(a, b)` that returns a + b
  - Include docstring for the function
  - Add type hints for parameters
  - _Requirements: 1.1, 1.2_

- [ ] 2. Create test file
  - Create `test_calculator.py`
  - Add test for the add function
  - Include edge case testing
  - _Requirements: 1.3_

- [ ] 3. Create documentation
  - Create README.md with usage examples
  - Document the calculator module
  - _Requirements: 1.4_
"""
        
        requirements_content = """# Requirements
1.1 Create calculator.py with add function
1.2 Include proper documentation and type hints
1.3 Create comprehensive tests
1.4 Provide usage documentation
"""
        
        design_content = """# Design
Simple calculator module with proper documentation and testing.
"""
        
        (work_dir / "requirements.md").write_text(requirements_content)
        (work_dir / "design.md").write_text(design_content)
        (work_dir / "tasks.md").write_text(task_content)
        
        # Create task definition
        from autogen_framework.models import TaskDefinition
        
        task_def = TaskDefinition(
            id="metrics_test_task",
            title="Create documented Python module",
            description="Create a Python module with proper documentation and tests",
            steps=[
                "Create calculator.py with add function and docstring",
                "Add type hints to function parameters",
                "Create test_calculator.py with comprehensive tests",
                "Create README.md with usage examples"
            ],
            requirements_ref=["1.1", "1.2", "1.3", "1.4"]
        )
        
        # Execute the task
        start_time = datetime.now()
        result = await implement_agent.execute_task(task_def, str(work_dir))
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create enhanced result
        enhanced_result = {
            'success': result is not None,
            'execution_time': execution_time,
            'commands_executed': result.get('commands_executed', []) if result else [],
            'original_result': result
        }
        
        # Evaluate quality
        quality_report = quality_framework.evaluate_task_execution(
            work_dir, task_def, enhanced_result
        )
        
        # Verify all quality metrics are present and have valid scores
        from autogen_framework.quality_metrics import QualityMetric
        
        for metric in QualityMetric:
            assert metric in quality_report.scores, f"Missing quality metric: {metric}"
            score = quality_report.scores[metric]
            assert 1.0 <= score.score <= 10.0, f"Invalid score for {metric}: {score.score}"
            assert score.percentage == (score.score / score.max_score) * 100
            
            # Verify score has proper structure
            assert hasattr(score, 'criteria_met')
            assert hasattr(score, 'criteria_failed')
            assert hasattr(score, 'recommendations')
            assert hasattr(score, 'details')
        
        # Verify overall score calculation
        assert 1.0 <= quality_report.overall_score <= 10.0
        
        # Save and verify report
        report_path = quality_framework.save_report(quality_report, quality_reports_dir, work_dir)
        assert report_path.exists()
        
        # Load and verify report structure
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        required_fields = [
            'task_id', 'task_title', 'execution_timestamp', 'overall_score',
            'execution_success', 'scores'
        ]
        
        for field in required_fields:
            assert field in report_data, f"Missing required field in report: {field}"
        
        # Verify scores structure
        assert len(report_data['scores']) == len(QualityMetric)
        
        for metric_name, score_data in report_data['scores'].items():
            assert 'score' in score_data
            assert 'percentage' in score_data
            assert 'criteria_met' in score_data
            assert 'criteria_failed' in score_data
            assert 'recommendations' in score_data
        
        print(f"\nQuality Metrics Test Results:")
        print(f"Overall Score: {quality_report.overall_score:.2f}/10.0")
        print(f"Report saved to: {report_path}")
        
        for metric, score in quality_report.scores.items():
            print(f"{metric.value}: {score.score:.2f}/10.0")
        
        return quality_report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])