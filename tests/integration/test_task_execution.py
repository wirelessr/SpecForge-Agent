import pytest
from pathlib import Path
import asyncio

from autogen_framework.main_controller import MainController
from autogen_framework.models import TaskDefinition

@pytest.mark.integration
class TestTaskExecution:
    @pytest.fixture
    def main_controller(self, temp_workspace, real_llm_config):
        """Fixture to create and initialize a MainController."""
        controller = MainController(workspace_path=temp_workspace)
        assert controller.initialize_framework(llm_config=real_llm_config)
        return controller


    @pytest.mark.asyncio
    async def test_implement_agent_creates_and_executes_code(self, main_controller, temp_workspace):
        """Test that ImplementAgent can create a file with code and that the code is valid."""
        task_def = TaskDefinition(
            id="1",
            title="Create calculator function",
            description="Create a python function `add(a, b)` that returns the sum of two numbers. Save it in a file named `calculator.py`.",
            steps=[
                "Create a new file named `calculator.py`.",
                "Inside `calculator.py`, write a Python function `add(a, b)` that takes two arguments and returns their sum."
            ],
            requirements_ref=[]
        )

        task_input = {
            "task_type": "execute_task",
            "task": task_def,
            "work_dir": temp_workspace
        }

        # Run the implement agent
        result = await main_controller.agent_manager.implement_agent.process_task(task_input=task_input)

        assert result.get("success"), f"ImplementAgent failed: {result.get('error', 'No error details')}"

        # Verify that the file was created
        code_path = Path(temp_workspace) / "calculator.py"
        assert code_path.exists(), "The file 'calculator.py' was not created."

        # Verify the content of the file
        code_content = code_path.read_text()
        assert "def add(a, b):" in code_content, "The 'add' function is not defined correctly."
        assert "return a + b" in code_content or "return a+b" in code_content, "The function body is incorrect."

        # Verify that the generated code is executable and correct
        try:
            namespace = {}
            exec(code_content, namespace)
            add_func = namespace.get('add')
            assert add_func is not None, "The 'add' function could not be found after execution."
            assert add_func(5, 3) == 8
            assert add_func(-1, 1) == 0
        except Exception as e:
            pytest.fail(f"The generated Python code failed to execute or was incorrect: {e}\nCode:\n{code_content}")

