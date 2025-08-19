import pytest
from pathlib import Path
import asyncio

from autogen_framework.main_controller import MainController
from autogen_framework.models import TaskDefinition

@pytest.mark.integration
class TestAgentCapabilities:
    @pytest.fixture
    def main_controller(self, temp_workspace, real_llm_config):
        """Fixture to create and initialize a MainController."""
        controller = MainController(workspace_path=temp_workspace)
        assert controller.initialize_framework(llm_config=real_llm_config)
        return controller

    @pytest.mark.asyncio
    async def test_error_recovery_handles_bad_command(self, main_controller, temp_workspace):
        """Test that the system can handle and log an error from a bad command."""
        task_def = TaskDefinition(
            id="error-test",
            title="Test error recovery",
            description="A task designed to fail to test error recovery.",
            steps=[
                "Create a file named `test.txt` with the content 'hello'.",
                "Execute the command `fake_command --and-exit`.",
                "Create another file `test2.txt` (should not be reached)."
            ],
            requirements_ref=[]
        )

        task_input = {
            "task_type": "execute_task",
            "task": task_def,
            "work_dir": temp_workspace
        }

        # Run the implement agent, which should encounter an error but handle it
        result = await main_controller.agent_manager.implement_agent.process_task(task_input=task_input)

        # The overall task should be marked as failed
        assert not result.get("success")

        # Verify that the first step was attempted
        first_file = Path(temp_workspace) / "test.txt"
        # The exact outcome depends on the decomposer, but we can check if it was created.
        # It's possible the decomposer creates it.
        # A more robust test would check the execution log.
        # For now, we check if the error was handled.

        # Verify that the third step was NOT reached
        second_file = Path(temp_workspace) / "test2.txt"
        assert not second_file.exists()

        # Verify that an error was logged and recovery was attempted
        assert "attempts" in result
        assert len(result["attempts"]) > 0
        last_attempt = result["attempts"][-1]
        assert not last_attempt["success"]
        # Make the error check more general to be robust
        assert "fake_command" in last_attempt.get("error", "").lower()

        await asyncio.sleep(20)

    @pytest.mark.asyncio
    async def test_context_is_passed_between_agents(self, main_controller, temp_workspace):
        """
        Test that context from previous steps (files) is used by subsequent agents,
        even if the direct prompt is contradictory.
        """
        # Setup: Create a work directory with requirements and design files for a "blog engine"
        work_dir = Path(temp_workspace) / "context-test-workspace"
        work_dir.mkdir()
        requirements_path = work_dir / "requirements.md"
        requirements_path.write_text("# Requirements\nCreate a Python-based blog engine with posts and comments.")
        design_path = work_dir / "design.md"
        design_path.write_text("# Design\nUse FastAPI and SQLAlchemy. Create models for Post and Comment.")

        # Now, run the TasksAgent with a completely different user_request
        task_input = {
            "user_request": "Create a simple calculator app",
            "requirements_path": str(requirements_path),
            "design_path": str(design_path),
            "work_dir": str(work_dir),
            "task_type": "generate_task_list"
        }

        result = await main_controller.agent_manager.tasks_agent.process_task(task_input=task_input)

        assert result.get("success"), f"TasksAgent failed: {result.get('error')}"
        tasks_path_str = result.get("tasks_path")
        assert tasks_path_str

        tasks_path = Path(tasks_path_str)
        assert tasks_path.exists()

        # Verify that the content is about the BLOG ENGINE, not the calculator.
        # This proves that the context from the files was prioritized over the prompt.
        content = tasks_path.read_text().lower()
        assert "blog" in content
        assert "post" in content
        assert "fastapi" in content
        assert "calculator" not in content

        await asyncio.sleep(20)
