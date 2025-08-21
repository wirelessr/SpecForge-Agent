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
    Test that PlanAgent can generate a requirements.md file. Mocks LLM call for stability.
    """
    plan_agent = PlanAgent(
        llm_config=real_llm_config, memory_manager=real_memory_manager,
        token_manager=real_managers.token_manager, context_manager=real_managers.context_manager,
    )
    task_input = {"user_request": "Create a countdown timer."}
    mock_parsed_request = {
        "summary": "Countdown Timer Script", "type": "development", "scope": "small",
        "key_requirements": ["..."], "technical_context": "...",
        "constraints": "...", "suggested_directory": "countdown-timer"
    }
    work_dir_path = Path(temp_workspace) / "countdown-timer"
    mock_req_content = "# Requirements Document\n- It should count down."

    with patch.object(plan_agent, 'parse_user_request', new_callable=AsyncMock, return_value=mock_parsed_request), \
         patch.object(plan_agent, 'create_work_directory', new_callable=AsyncMock, return_value=str(work_dir_path)), \
         patch.object(plan_agent, 'generate_requirements', new_callable=AsyncMock) as mock_generate:

        async def side_effect(user_request, work_directory, parsed_request):
            Path(work_directory).mkdir(parents=True, exist_ok=True)
            (Path(work_directory) / "requirements.md").write_text(mock_req_content)
            return str(Path(work_directory) / "requirements.md")
        mock_generate.side_effect = side_effect

        result = await plan_agent._process_task_impl(task_input)

    assert result["success"] is True
    assert (work_dir_path / "requirements.md").exists()
    assert mock_req_content in (work_dir_path / "requirements.md").read_text()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_design_agent_generates_design(real_managers, real_llm_config, temp_workspace):
    """
    Test that DesignAgent can generate a design.md file. Mocks LLM call for stability.
    """
    design_agent = DesignAgent(
        llm_config=real_llm_config,
        token_manager=real_managers.token_manager, context_manager=real_managers.context_manager,
    )
    work_dir = Path(temp_workspace) / "design-test-project"
    work_dir.mkdir()
    (work_dir / "requirements.md").write_text("...")
    task_input = {"requirements_path": str(work_dir / "requirements.md"), "work_directory": str(work_dir)}
    mock_design_content = "# Design Document\n## Overview\n..."

    with patch.object(design_agent, 'generate_design', new_callable=AsyncMock, return_value=mock_design_content) as mock_generate:
        result = await design_agent._process_task_impl(task_input)

    assert result["success"] is True
    design_file = work_dir / "design.md"
    assert design_file.exists()
    assert mock_design_content in design_file.read_text()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_tasks_agent_generates_tasks(real_managers, real_llm_config, temp_workspace):
    """
    Test that TasksAgent can generate a tasks.md file. Mocks LLM call for stability.
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
    mock_tasks_content = "- [ ] 1. Do something."

    with patch.object(tasks_agent, 'generate_task_list', new_callable=AsyncMock) as mock_generate:
        async def side_effect(design_path, requirements_path, work_dir):
            tasks_path = Path(work_dir) / "tasks.md"
            tasks_path.write_text(mock_tasks_content)
            return str(tasks_path)
        mock_generate.side_effect = side_effect
        result = await tasks_agent._process_task_impl(task_input)

    assert result["success"] is True
    tasks_file = work_dir / "tasks.md"
    assert tasks_file.exists()
    assert mock_tasks_content in tasks_file.read_text()
