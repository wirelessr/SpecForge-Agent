import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock

from autogen_framework.agents.plan_agent import PlanAgent
from autogen_framework.agents.design_agent import DesignAgent
from autogen_framework.agents.tasks_agent import TasksAgent

@pytest.mark.integration
@pytest.mark.asyncio
async def test_plan_agent_generates_requirements(real_managers, real_llm_config, real_memory_manager, temp_workspace):
    """
    Test that PlanAgent can generate a requirements.md file from a user prompt.
    """
    plan_agent = PlanAgent(
        llm_config=real_llm_config, memory_manager=real_memory_manager,
        token_manager=real_managers.token_manager, context_manager=real_managers.context_manager,
    )
    user_prompt = "Create a python script that acts as a countdown timer."
    task_input = {"user_request": user_prompt}
    result = await plan_agent._process_task_impl(task_input)
    assert result["success"] is True, f"Task failed. Result: {result}"
    work_dir = Path(result["work_directory"])
    requirements_file = work_dir / "requirements.md"
    assert requirements_file.exists()
    content = requirements_file.read_text()
    assert len(content) > 50
    assert "# Requirements Document" in content

@pytest.mark.integration
@pytest.mark.asyncio
async def test_design_agent_generates_design(real_managers, real_llm_config, temp_workspace):
    """
    Test that DesignAgent can generate a design.md file from a requirements.md file.
    """
    design_agent = DesignAgent(
        llm_config=real_llm_config,
        token_manager=real_managers.token_manager, context_manager=real_managers.context_manager,
    )
    work_dir = Path(temp_workspace) / "design-test-project"
    work_dir.mkdir()
    requirements_path = work_dir / "requirements.md"
    requirements_path.write_text("# Requirements\nAs a user, I want a script that counts down from 10 to 0.")
    task_input = {"requirements_path": str(requirements_path), "work_directory": str(work_dir)}
    result = await design_agent._process_task_impl(task_input)
    assert result["success"] is True, f"Task failed. Result: {result}"
    design_file = work_dir / "design.md"
    assert design_file.exists()
    content = design_file.read_text()
    assert len(content) > 50
    assert "# Design Document" in content
    assert "mermaid" in content.lower()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_tasks_agent_generates_tasks(real_managers, real_llm_config, temp_workspace):
    """
    Test that TasksAgent can generate a tasks.md file from a design.md file.
    """
    tasks_agent = TasksAgent(
        llm_config=real_llm_config,
        token_manager=real_managers.token_manager, context_manager=real_managers.context_manager,
    )
    work_dir = Path(temp_workspace) / "tasks-test-project"
    work_dir.mkdir()
    (work_dir / "requirements.md").write_text("...")
    (work_dir / "design.md").write_text("...")
    task_input = {
        "task_type": "generate_task_list", "requirements_path": str(work_dir / "requirements.md"),
        "design_path": str(work_dir / "design.md"), "work_dir": str(work_dir)
    }
    result = await tasks_agent._process_task_impl(task_input)
    assert result["success"] is True, f"Task failed. Result: {result}"
    tasks_file = work_dir / "tasks.md"
    assert tasks_file.exists()
    content = tasks_file.read_text()
    assert len(content) > 50
    assert "- [ ]" in content
