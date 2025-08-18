import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from autogen_framework.agents.task_decomposer import TaskDecomposer, ExecutionPlan, ComplexityAnalysis, ShellCommand
from autogen_framework.models import TaskDefinition, LLMConfig

# Sample data for testing
SAMPLE_TASK = TaskDefinition(
    id="test_task_123",
    title="Create a Python script",
    description="Write a simple Python script that prints 'Hello, World!'",
    steps=["Create file", "Write content", "Run script"],
    requirements_ref=["REQ-001"],
    dependencies=["python3"]
)

MOCK_LLM_RESPONSE = {
    "complexity_analysis": {
        "complexity_level": "simple",
        "estimated_steps": 3,
        "required_tools": ["python"],
        "dependencies": [],
        "risk_factors": ["File system permissions"],
        "confidence_score": 0.95,
        "analysis_reasoning": "A very straightforward task involving basic file I/O and execution."
    },
    "commands": [
        {
            "command": "touch hello_world.py",
            "description": "Create the Python script file.",
            "timeout": 10,
            "retry_on_failure": False,
            "success_indicators": [],
            "failure_indicators": ["No such file or directory"]
        },
        {
            "command": "echo 'print(\"Hello, World!\")' > hello_world.py",
            "description": "Write the Python code to the file.",
            "timeout": 10,
            "retry_on_failure": True,
            "success_indicators": [],
            "failure_indicators": ["Permission denied"]
        },
        {
            "command": "python hello_world.py",
            "description": "Execute the Python script.",
            "timeout": 15,
            "retry_on_failure": False,
            "success_indicators": ["Hello, World!"],
            "failure_indicators": ["Traceback", "Error"]
        }
    ],
    "success_criteria": [
        "The file 'hello_world.py' is created.",
        "The script 'hello_world.py' contains the correct print statement.",
        "Executing the script prints 'Hello, World!' to standard output."
    ],
    "fallback_strategies": [
        "Ensure Python 3 is installed and in the system's PATH.",
        "Check write permissions for the current directory."
    ],
    "estimated_duration": 2
}

@pytest.fixture
def mock_dependencies():
    """Provides a fixture for mocked dependencies."""
    mock_llm_config = LLMConfig(model="mock-model", api_key="mock-key", base_url="http://localhost:1234")
    mock_token_manager = MagicMock()
    mock_context_manager = MagicMock()
    mock_config_manager = MagicMock()
    return mock_llm_config, mock_token_manager, mock_context_manager, mock_config_manager

@pytest.fixture
def task_decomposer(mock_dependencies):
    """Provides a TaskDecomposer instance with mocked dependencies."""
    llm_config, token_manager, context_manager, config_manager = mock_dependencies
    return TaskDecomposer(
        name="TestTaskDecomposer",
        llm_config=llm_config,
        system_message="You are a test agent.",
        token_manager=token_manager,
        context_manager=context_manager,
        config_manager=config_manager
    )

@pytest.mark.asyncio
class TestTaskDecompositionFlow:
    """Tests the primary task decomposition flow."""

    async def test_decompose_task_successful_parsing(self, task_decomposer):
        """
        Tests the happy path where the LLM returns a valid, complete JSON response.
        """
        # Mock the LLM response
        mock_response_str = f"```json\n{json.dumps(MOCK_LLM_RESPONSE)}\n```"
        task_decomposer.generate_response = AsyncMock(return_value=mock_response_str)

        # Call the method to be tested
        execution_plan = await task_decomposer.decompose_task(SAMPLE_TASK)

        # Assertions
        assert isinstance(execution_plan, ExecutionPlan)
        assert execution_plan.task == SAMPLE_TASK

        # Assert ComplexityAnalysis
        complexity = execution_plan.complexity_analysis
        expected_complexity = MOCK_LLM_RESPONSE["complexity_analysis"]
        assert isinstance(complexity, ComplexityAnalysis)
        assert complexity.complexity_level == expected_complexity["complexity_level"]
        assert complexity.estimated_steps == expected_complexity["estimated_steps"]
        assert complexity.confidence_score == expected_complexity["confidence_score"]
        assert complexity.analysis_reasoning == expected_complexity["analysis_reasoning"]

        # Assert ShellCommands
        assert len(execution_plan.commands) == len(MOCK_LLM_RESPONSE["commands"])
        first_cmd = execution_plan.commands[0]
        expected_first_cmd = MOCK_LLM_RESPONSE["commands"][0]
        assert isinstance(first_cmd, ShellCommand)
        assert first_cmd.command == expected_first_cmd["command"]
        assert first_cmd.description == expected_first_cmd["description"]
        assert first_cmd.timeout == expected_first_cmd["timeout"]

        # Assert other plan details
        assert execution_plan.success_criteria == MOCK_LLM_RESPONSE["success_criteria"]
        assert execution_plan.fallback_strategies == MOCK_LLM_RESPONSE["fallback_strategies"]
        assert execution_plan.estimated_duration == MOCK_LLM_RESPONSE["estimated_duration"]

        # Verify that generate_response was called
        task_decomposer.generate_response.assert_called_once()

    async def test_decompose_task_invalid_json_fallback(self, task_decomposer):
        """
        Tests the fallback mechanism when the LLM response is not valid JSON.
        """
        # Mock a malformed JSON response
        mock_response_str = "This is not JSON, just some text."
        task_decomposer.generate_response = AsyncMock(return_value=mock_response_str)

        # Patch logger to spy on it
        with patch.object(task_decomposer.logger, 'error') as mock_logger:
            execution_plan = await task_decomposer.decompose_task(SAMPLE_TASK)

            # Assert that a fallback plan is created
            assert isinstance(execution_plan, ExecutionPlan)
            assert execution_plan.complexity_analysis.complexity_level == "unknown"
            assert "Failed to parse" in execution_plan.complexity_analysis.analysis_reasoning
            assert len(execution_plan.commands) == 1
            assert "Error: Failed to parse execution plan" in execution_plan.commands[0].command
            assert execution_plan.success_criteria == ["Manual verification required."]
            
            # Assert that an error was logged
            assert mock_logger.call_count == 2
            # Check that both expected messages were logged
            assert "No JSON content found" in mock_logger.call_args_list[0][0][0]
            assert "Failed to parse comprehensive execution plan" in mock_logger.call_args_list[1][0][0]

    async def test_decompose_task_empty_json_fallback(self, task_decomposer):
        """
        Tests the fallback mechanism when the LLM returns an empty JSON object.
        """
        # Mock an empty JSON response
        mock_response_str = "```json\n{}\n```"
        task_decomposer.generate_response = AsyncMock(return_value=mock_response_str)

        execution_plan = await task_decomposer.decompose_task(SAMPLE_TASK)

        # Assert that the plan is created with default values
        assert isinstance(execution_plan, ExecutionPlan)
        assert execution_plan.complexity_analysis.complexity_level == "moderate" # default value
        assert len(execution_plan.commands) == 0
        assert len(execution_plan.success_criteria) == 0
        assert execution_plan.estimated_duration == 15 # default value

    async def test_decompose_task_handles_llm_exception(self, task_decomposer):
        """
        Tests the fallback mechanism when the `generate_response` call raises an exception.
        """
        # Mock `generate_response` to raise an exception
        llm_error = Exception("LLM API is down")
        task_decomposer.generate_response = AsyncMock(side_effect=llm_error)

        with patch.object(task_decomposer.logger, 'error') as mock_logger:
            execution_plan = await task_decomposer.decompose_task(SAMPLE_TASK)

            # Assert that a minimal fallback plan is created
            assert isinstance(execution_plan, ExecutionPlan)
            assert execution_plan.complexity_analysis.complexity_level == "unknown"
            assert "Decomposition failed" in execution_plan.complexity_analysis.analysis_reasoning
            assert len(execution_plan.commands) == 0
            assert f"Investigate failure in task: {SAMPLE_TASK.title}" in execution_plan.success_criteria
            
            # Assert that the exception was logged
            mock_logger.assert_called_once()
            assert f"Error in comprehensive task decomposition for {SAMPLE_TASK.title}" in mock_logger.call_args[0][0]

class TestTaskDecomposerHelpers:
    """Tests the helper methods of the TaskDecomposer."""

    def test_get_context_summary(self, task_decomposer):
        """
        Tests the context summary generation.
        """
        # Mock the context manager
        task_decomposer.context_manager = MagicMock()
        
        summary = task_decomposer._get_context_summary()
        
        assert "Project context" in summary
        assert "Execution history" in summary

    def test_get_context_summary_no_context_manager(self, task_decomposer):
        """
        Tests context summary generation when no context manager is present.
        """
        task_decomposer.context_manager = None
        summary = task_decomposer._get_context_summary()
        assert summary == "No additional context is available."

    @pytest.mark.parametrize("response, expected", [
        ("```json\n{\"key\": \"value\"}\n```", "{\"key\": \"value\"}"),
        ("Some text before\n```json\n{\"key\": \"value\"}\n```\nand after", "{\"key\": \"value\"}"),
        ("No code block, but has JSON {\"key\": \"value\"}", "{\"key\": \"value\"}"),
        ("Invalid response", None),
        ("{}", "{}"),
    ])
    def test_extract_json_from_response(self, task_decomposer, response, expected):
        """
        Tests the JSON extraction logic from various LLM response formats.
        """
        extracted = task_decomposer._extract_json_from_response(response)
        assert extracted == expected

    async def test_process_task_impl_routes_correctly(self, task_decomposer):
        """
        Tests that _process_task_impl routes to the correct handler.
        """
        task_input = {"task_type": "decompose_task", "task": SAMPLE_TASK}
        
        # Mock the handler to verify it gets called
        with patch.object(task_decomposer, '_handle_decompose_task', new_callable=AsyncMock) as mock_handler:
            mock_handler.return_value = {"status": "handled"}
            result = await task_decomposer._process_task_impl(task_input)
            
            mock_handler.assert_called_once_with(task_input)
            assert result == {"status": "handled"}

    async def test_process_task_impl_raises_for_unknown_type(self, task_decomposer):
        """
        Tests that _process_task_impl raises a ValueError for an unknown task type.
        """
        task_input = {"task_type": "unknown_task", "task": SAMPLE_TASK}
        
        with pytest.raises(ValueError, match="Unknown task type: unknown_task"):
            await task_decomposer._process_task_impl(task_input)
