#!/usr/bin/env python3
"""
Test script to verify TasksAgent works correctly with container-based initialization.
"""

import tempfile
import os
from autogen_framework.models import LLMConfig
from autogen_framework.agents.tasks_agent import TasksAgent
from autogen_framework.dependency_container import DependencyContainer

def test_tasks_agent_container_integration():
    """Test TasksAgent with container-based initialization."""
    
    # Create test LLM config
    llm_config = LLMConfig(
        base_url='http://test.local:8888/openai/v1',
        model='test-model',
        api_key='test-key'
    )
    
    # Create container
    container = DependencyContainer.create_test('test_workspace', llm_config)
    
    # Test 1: Create TasksAgent with container only
    print("Test 1: Creating TasksAgent with container only...")
    agent1 = TasksAgent(container=container)
    assert agent1.name == "TasksAgent"
    assert agent1.llm_config.model == "test-model"
    print("✓ Container-only initialization works")
    
    # Test 2: Create TasksAgent with container and custom parameters
    print("\nTest 2: Creating TasksAgent with container and custom parameters...")
    custom_llm_config = LLMConfig(
        base_url='http://custom.local:8888/openai/v1',
        model='custom-model',
        api_key='custom-key'
    )
    agent2 = TasksAgent(
        container=container,
        name="CustomTasksAgent",
        llm_config=custom_llm_config,
        description="Custom tasks agent"
    )
    assert agent2.name == "CustomTasksAgent"
    assert agent2.llm_config.model == "custom-model"
    assert "Custom tasks agent" in agent2.description
    print("✓ Custom parameter initialization works")
    
    # Test 3: Verify manager access through container
    print("\nTest 3: Verifying manager access through container...")
    managers = [
        ('token_manager', agent1.token_manager),
        ('context_manager', agent1.context_manager),
        ('memory_manager', agent1.memory_manager),
        ('shell_executor', agent1.shell_executor),
        ('error_recovery', agent1.error_recovery),
        ('task_decomposer', agent1.task_decomposer),
        ('config_manager', agent1.config_manager),
        ('context_compressor', agent1.context_compressor),
    ]
    
    for manager_name, manager in managers:
        assert manager is not None, f"{manager_name} should not be None"
        print(f"✓ {manager_name} accessible")
    
    # Test 4: Verify capabilities
    print("\nTest 4: Verifying agent capabilities...")
    capabilities = agent1.get_agent_capabilities()
    expected_capabilities = [
        "Generate task lists from design documents",
        "Decompose technical designs into implementation plans",
        "Create structured task lists with requirements references",
        "Format tasks in markdown checkbox format",
        "Maintain context access to requirements.md and design.md",
        "Ensure tasks build incrementally on each other",
        "Handle task list revisions based on user feedback"
    ]
    
    assert len(capabilities) == len(expected_capabilities)
    for expected in expected_capabilities:
        assert expected in capabilities, f"Missing capability: {expected}"
    print(f"✓ All {len(capabilities)} capabilities present")
    
    # Test 5: Test task parsing functionality
    print("\nTest 5: Testing task parsing functionality...")
    test_task_content = """# Implementation Tasks

- [ ] 1. Create base structure
  - Create directory
  - Initialize files
  - Requirements: 1.1, 2.3

- [ ] 2. Implement functionality
  - Write code
  - Add tests
  - Requirements: 2.1"""
    
    tasks = agent1._parse_task_list(test_task_content)
    assert len(tasks) == 2
    assert tasks[0].id == "1"
    assert tasks[0].title == "Create base structure"
    assert tasks[0].requirements_ref == ["1.1", "2.3"]
    assert tasks[1].id == "2"
    assert tasks[1].title == "Implement functionality"
    assert tasks[1].requirements_ref == ["2.1"]
    print("✓ Task parsing works correctly")
    
    print("\n🎉 All tests passed! TasksAgent container integration is working correctly.")

if __name__ == "__main__":
    test_tasks_agent_container_integration()