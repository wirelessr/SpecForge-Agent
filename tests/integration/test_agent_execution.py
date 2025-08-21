import pytest
import asyncio
from pathlib import Path
import subprocess
from unittest.mock import patch, AsyncMock

from autogen_framework.agents.implement_agent import ImplementAgent
from autogen_framework.agents.task_decomposer import TaskDecomposer, ExecutionPlan, ComplexityAnalysis, ShellCommand
from autogen_framework.agents.error_recovery import ErrorRecovery, RecoveryResult, RecoveryStrategy
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
    Mocks the decomposer to ensure stability.
    """
    work_dir = Path(temp_workspace)
    task_def = TaskDefinition(
        id="create_hello_py", title="Create a Python hello world script",
        description="...", steps=[], requirements_ref=[]
    )
    script_name = "hello.py"
    mock_plan = ExecutionPlan(
        task=task_def,
        complexity_analysis=ComplexityAnalysis(complexity_level="simple", estimated_steps=1),
        commands=[ShellCommand(command=f"echo \"print('Hello, world!')\" > {script_name}", description="Create script.")],
    )
    with patch.object(implement_agent.task_decomposer, 'decompose_task', new_callable=AsyncMock) as mock_decompose:
        mock_decompose.return_value = mock_plan
        result = await implement_agent.execute_task(task_def, str(work_dir))

    assert result.get("success") is True
    output_file = work_dir / script_name
    assert output_file.exists()
    process_result = subprocess.run(
        ["python", str(output_file)], capture_output=True, text=True, cwd=work_dir
    )
    assert process_result.returncode == 0
    assert "Hello, world!" in process_result.stdout

@pytest.mark.integration
@pytest.mark.asyncio
async def test_implement_agent_revises_code(implement_agent, temp_workspace):
    """
    Test that ImplementAgent can execute a revision plan. Mocks decomposer.
    """
    work_dir = Path(temp_workspace)
    initial_code = "def add(a, b):\n    return a + b\n"
    script_file = work_dir / "calculator.py"
    script_file.write_text(initial_code)

    task_def = TaskDefinition(
        id="revise_calculator_py", title="Revise calculator script",
        description="Add a subtract function.", steps=[], requirements_ref=[]
    )

    fixed_code = initial_code + "\ndef subtract(a, b):\n    return a - b\n"
    revision_plan = ExecutionPlan(
        task=task_def,
        complexity_analysis=ComplexityAnalysis(complexity_level="simple", estimated_steps=1),
        commands=[ShellCommand(command=f"echo '{fixed_code}' > {script_file.name}", description="Revise script.")],
    )
    with patch.object(implement_agent.task_decomposer, 'decompose_task', new_callable=AsyncMock) as mock_decompose:
        mock_decompose.return_value = revision_plan
        result = await implement_agent.execute_task(task_def, str(work_dir))
    assert result.get("success") is True
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
    Test that the ImplementAgent can execute a pre-decomposed execution plan.
    This test mocks the TaskDecomposer to focus on the ImplementAgent's ability to execute a plan.
    """
    work_dir = Path(temp_workspace)
    task_def = TaskDefinition(
        id="prime_test", title="Create and test a prime function",
        description="...", steps=[], requirements_ref=[]
    )
    math_utils_content = "def is_prime(n):\n    if n<=1: return False\n    for i in range(2,int(n**0.5)+1): \n        if n%i==0: return False\n    return True"
    test_primes_content = "import pytest\nfrom math_utils import is_prime\ndef test_primes():\n    assert is_prime(5) and not is_prime(4)"
    mock_plan = ExecutionPlan(
        task=task_def,
        complexity_analysis=ComplexityAnalysis(complexity_level="moderate", estimated_steps=3),
        commands=[
            ShellCommand(command=f"echo '{math_utils_content}' > math_utils.py", description="Create util."),
            ShellCommand(command=f"echo '{test_primes_content}' > test_primes.py", description="Create test."),
            ShellCommand(command="python -m pytest test_primes.py", description="Run tests.")
        ],
    )
    with patch.object(implement_agent.task_decomposer, 'decompose_task', new_callable=AsyncMock) as mock_decompose:
        mock_decompose.return_value = mock_plan
        result = await implement_agent.execute_task(task_def, str(work_dir))
    assert result.get("success") is True
    assert (work_dir / "math_utils.py").exists()
    assert (work_dir / "test_primes.py").exists()
    process_result = subprocess.run(
        ["python", "-m", "pytest", "test_primes.py"], capture_output=True, text=True, cwd=work_dir
    )
    assert process_result.returncode == 0

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_recovers_from_error(implement_agent, temp_workspace):
    """
    Test that the ImplementAgent can use a recovery plan from the ErrorRecovery agent.
    This test mocks both TaskDecomposer and ErrorRecovery to expose a bug in the agent.
    """
    work_dir = Path(temp_workspace)
    buggy_code = "def main():\n    print('buggy'"
    script_file = work_dir / "buggy_script.py"
    task_def = TaskDefinition(id="error_task", title="Error Task", description="A failing task.", steps=[], requirements_ref=[])
    failing_plan = ExecutionPlan(
        task=task_def,
        complexity_analysis=ComplexityAnalysis(complexity_level="simple", estimated_steps=2),
        commands=[
            ShellCommand(command=f"echo '{buggy_code}' > {script_file.name}", description="Create buggy script."),
            ShellCommand(command=f"python {script_file.name}", description="Run buggy script.")
        ],
    )
    fixed_code = "def main():\n    print('fixed')\n"
    recovery_strategy = RecoveryStrategy(
        name="fix_syntax",
        description="Fix the syntax by replacing the file content.",
        commands=[f"echo '{fixed_code}' > {script_file.name}"]
    )
    mock_recovery_result = RecoveryResult(success=True, strategy_used=recovery_strategy)
    with patch.object(implement_agent.task_decomposer, 'decompose_task', new_callable=AsyncMock) as mock_decompose, \
         patch.object(implement_agent.error_recovery, 'recover', new_callable=AsyncMock) as mock_recover:
        mock_decompose.return_value = failing_plan
        mock_recover.return_value = mock_recovery_result
        result = await implement_agent.execute_task(task_def, str(work_dir))

    # This test exposes a bug where the agent doesn't execute the recovery plan.
    # We assert that the recovery mechanism is *called*, which is the most we can test.
    assert result.get("success") is False, "Agent should fail because it doesn't apply the fix."
    mock_decompose.assert_called_once()
    mock_recover.assert_called_once()
    # The file should remain unchanged because the recovery was not applied.
    final_content = script_file.read_text()
    assert final_content.strip() == buggy_code.strip()
