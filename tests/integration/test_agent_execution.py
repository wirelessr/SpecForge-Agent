import pytest
import asyncio
from pathlib import Path
import subprocess

from autogen_framework.agents.implement_agent import ImplementAgent
from autogen_framework.agents.task_decomposer import TaskDecomposer
from autogen_framework.agents.error_recovery import ErrorRecovery
from autogen_framework.shell_executor import ShellExecutor
from autogen_framework.models import TaskDefinition

@pytest.fixture
def shell_executor(temp_workspace):
    """Create a ShellExecutor instance for the temp workspace."""
    return ShellExecutor(default_working_dir=temp_workspace)

@pytest.fixture
def implement_agent(real_llm_config, shell_executor, real_managers):
    """Create an ImplementAgent instance with real dependencies."""
    task_decomposer = TaskDecomposer(
        name="TestTaskDecomposer",
        llm_config=real_llm_config,
        system_message="You are a task decomposer.",
        token_manager=real_managers.token_manager,
        context_manager=real_managers.context_manager,
    )
    error_recovery = ErrorRecovery(
        name="TestErrorRecovery",
        llm_config=real_llm_config,
        system_message="You are an error recovery agent.",
        token_manager=real_managers.token_manager,
        context_manager=real_managers.context_manager,
    )

    return ImplementAgent(
        name="TestImplementAgent",
        llm_config=real_llm_config,
        system_message="You are a helpful AI assistant that writes Python code to complete tasks.",
        shell_executor=shell_executor,
        task_decomposer=task_decomposer,
        error_recovery=error_recovery,
        description="Test agent for integration testing",
        token_manager=real_managers.token_manager,
        context_manager=real_managers.context_manager
    )

@pytest.mark.integration
@pytest.mark.asyncio
async def test_implement_agent_executes_simple_task(implement_agent, temp_workspace):
    """
    Test that ImplementAgent can execute a simple task to create and run a Python script.
    """
    await asyncio.sleep(120)
    work_dir = Path(temp_workspace)
    task_def = TaskDefinition(
        id="create_hello_py",
        title="Create a Python hello world script",
        description="Create a Python script named 'hello.py' that prints 'Hello from workflow'.",
        steps=["Create a file named hello.py", "Add python code to print 'Hello from workflow'"],
        requirements_ref=[]
    )
    (work_dir / "requirements.md").write_text("# Requirements\n- A python script that prints 'Hello from workflow'")
    (work_dir / "design.md").write_text("# Design\n- Use the `print()` function in Python.")
    result = await implement_agent.execute_task(task_def, str(work_dir))
    assert result.get("success") is True, f"Agent task execution failed: {result}"
    output_file = work_dir / "hello.py"
    assert output_file.exists(), "The file 'hello.py' should have been created."
    content = output_file.read_text()
    assert "Hello from workflow" in content
    process_result = subprocess.run(
        ["python", str(output_file)], capture_output=True, text=True, cwd=work_dir
    )
    assert process_result.returncode == 0, f"Script failed to run: {process_result.stderr}"
    assert "Hello from workflow" in process_result.stdout

@pytest.mark.integration
@pytest.mark.asyncio
async def test_implement_agent_revises_code(implement_agent, temp_workspace):
    """
    Test that ImplementAgent can revise an existing file based on a new task.
    """
    await asyncio.sleep(120)
    work_dir = Path(temp_workspace)
    initial_code = "def add(a, b):\n    return a + b\n"
    script_file = work_dir / "calculator.py"
    script_file.write_text(initial_code)
    task_def = TaskDefinition(
        id="revise_calculator_py",
        title="Revise calculator script",
        description="Append a new 'subtract' function to the end of the 'calculator.py' script.",
        steps=["Append the new function to calculator.py"],
        requirements_ref=[]
    )
    result = await implement_agent.execute_task(task_def, str(work_dir))
    assert result.get("success") is True, f"Agent revision task failed: {result}"
    content = script_file.read_text()
    assert "def add(a, b):" in content and "def subtract(a, b):" in content
    process_result = subprocess.run(
        ["python", "-c", "import calculator"], capture_output=True, text=True, cwd=work_dir
    )
    assert process_result.returncode == 0

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_handles_task_decomposition(implement_agent, temp_workspace):
    """
    Test that the ImplementAgent can handle a complex task that requires decomposition.
    """
    await asyncio.sleep(120)
    work_dir = Path(temp_workspace)
    task_def = TaskDefinition(
        id="prime_number_test_suite",
        title="Create and test a prime number function",
        description=(
            "Create a Python utility file 'math_utils.py' with a function `is_prime(n)` "
            "and a test file 'test_primes.py' that uses pytest to test it."
        ),
        steps=[],
        requirements_ref=[]
    )
    result = await implement_agent.execute_task(task_def, str(work_dir))
    assert result.get("success") is True, f"Agent decomposition task failed: {result}"
    test_file = work_dir / "test_primes.py"
    assert (work_dir / "math_utils.py").exists()
    assert test_file.exists()
    process_result = subprocess.run(
        ["python", "-m", "pytest", str(test_file)],
        capture_output=True, text=True, cwd=work_dir
    )
    assert process_result.returncode == 0, f"Pytest execution failed: {process_result.stderr}"

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_recovers_from_error(implement_agent, temp_workspace):
    """
    Test that the ImplementAgent can use the ErrorRecovery agent to fix a failing task.
    """
    await asyncio.sleep(120)
    work_dir = Path(temp_workspace)
    buggy_code = "def buggy_function():\n    print('This has a syntax error'"
    script_file = work_dir / "buggy_script.py"
    script_file.write_text(buggy_code)
    task_def = TaskDefinition(id="error_task", title="Run a script that will fail", description=f"Run the script {script_file.name}", steps=[], requirements_ref=[])

    result = await implement_agent.execute_task(task_def, str(work_dir))

    assert result.get("success") is True, "Agent should ultimately succeed after recovery."
    fixed_content = script_file.read_text()
    assert "print('This has a syntax error')" in fixed_content
    process_result = subprocess.run(
        ["python", str(script_file)],
        capture_output=True, text=True, cwd=work_dir
    )
    assert process_result.returncode == 0, f"Fixed script still fails to run: {process_result.stderr}"
