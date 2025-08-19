import pytest
from pathlib import Path
import asyncio

from autogen_framework.main_controller import MainController

@pytest.mark.integration
class TestDocumentGeneration:
    @pytest.fixture
    def main_controller(self, temp_workspace, real_llm_config):
        """Fixture to create and initialize a MainController."""
        controller = MainController(workspace_path=temp_workspace)
        assert controller.initialize_framework(llm_config=real_llm_config)
        return controller

    @pytest.mark.asyncio
    async def test_plan_agent_generates_requirements(self, main_controller, temp_workspace):
        """Test that PlanAgent can generate requirements.md."""
        task_input = {"user_request": "Create a tic-tac-toe game in python"}

        result = await main_controller.agent_manager.plan_agent.process_task(task_input=task_input)

        assert result.get("success"), f"PlanAgent failed: {result.get('error')}"
        requirements_path_str = result.get("requirements_path")
        assert requirements_path_str is not None

        requirements_path = Path(requirements_path_str)
        assert requirements_path.exists()

        content = requirements_path.read_text()
        assert len(content) > 50
        assert "tic-tac-toe" in content.lower()
        assert "python" in content.lower()

        await asyncio.sleep(20)

    @pytest.mark.asyncio
    async def test_design_agent_generates_design(self, main_controller, temp_workspace):
        """Test that DesignAgent can generate design.md."""
        # Setup: Create a work directory and a requirements file, similar to what PlanAgent would do.
        work_dir = Path(temp_workspace) / "test-design-workspace"
        work_dir.mkdir()
        requirements_path = work_dir / "requirements.md"
        requirements_path.write_text("The user wants to create a tic-tac-toe game in Python.")

        task_input = {
            "user_request": "Create a tic-tac-toe game in python",
            "requirements_path": str(requirements_path),
            "work_directory": str(work_dir)
        }

        result = await main_controller.agent_manager.design_agent.process_task(task_input=task_input)

        assert result.get("success"), f"DesignAgent failed: {result.get('error')}"
        design_path_str = result.get("design_path")
        assert design_path_str is not None

        design_path = Path(design_path_str)
        assert design_path.exists()

        content = design_path.read_text()
        assert len(content) > 50
        assert "class" in content.lower() or "function" in content.lower()

        await asyncio.sleep(20)

    @pytest.mark.asyncio
    async def test_tasks_agent_generates_tasks(self, main_controller, temp_workspace):
        """Test that TasksAgent can generate tasks.md."""
        # Setup: Create a work directory with requirements and design files.
        work_dir = Path(temp_workspace) / "test-tasks-workspace"
        work_dir.mkdir()
        requirements_path = work_dir / "requirements.md"
        requirements_path.write_text("The user wants to create a tic-tac-toe game in Python.")
        design_path = work_dir / "design.md"
        design_path.write_text("We will create a `Game` class with methods for drawing the board, checking for wins, and handling player input.")

        task_input = {
            "user_request": "Create a tic-tac-toe game in python",
            "requirements_path": str(requirements_path),
            "design_path": str(design_path),
            "work_dir": str(work_dir),
            "task_type": "generate_task_list"
        }

        result = await main_controller.agent_manager.tasks_agent.process_task(task_input=task_input)

        assert result.get("success"), f"TasksAgent failed: {result.get('error')}"
        tasks_path_str = result.get("tasks_path")
        assert tasks_path_str is not None

        tasks_path = Path(tasks_path_str)
        assert tasks_path.exists()

        content = tasks_path.read_text()
        assert len(content) > 50
        # Check for list items instead of the word "task" to make the test more robust
        assert "- [" in content
        assert "1." in content

        await asyncio.sleep(20)

    @pytest.mark.asyncio
    async def test_revise_command_modifies_document(self, main_controller, temp_workspace):
        """Test that the revise command can modify an existing document."""
        initial_task = {"user_request": "Create a snake game in javascript"}

        plan_result = await main_controller.agent_manager.plan_agent.process_task(task_input=initial_task)
        assert plan_result.get("success"), f"Initial plan generation failed: {plan_result.get('error')}"
        requirements_path_str = plan_result.get("requirements_path")
        assert requirements_path_str
        requirements_path = Path(requirements_path_str)

        initial_content = requirements_path.read_text()
        assert "snake" in initial_content.lower()
        assert "javascript" in initial_content.lower()

        await asyncio.sleep(20) # Add delay before the next LLM call

        revision_task = {
            "task_type": "revision",
            "revision_feedback": "Actually, let's make it in Python instead of Javascript.",
            "current_result": plan_result,
            "work_directory": str(requirements_path.parent)
        }

        revision_result = await main_controller.agent_manager.plan_agent.process_task(task_input=revision_task)
        assert revision_result.get("success"), f"Revision failed: {revision_result.get('error')}"

        revised_content = requirements_path.read_text()
        assert revised_content != initial_content
        assert "python" in revised_content.lower()
        assert "javascript" not in revised_content.lower()
        assert "snake" in revised_content.lower()

        await asyncio.sleep(20)
