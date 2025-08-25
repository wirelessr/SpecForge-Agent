#!/usr/bin/env python3
"""
Example code showing before/after initialization patterns for the AutoGen Framework.

This file demonstrates the migration from complex manual dependency injection
to the simplified container-based approach.
"""

import asyncio
from pathlib import Path
from autogen_framework.config_manager import LLMConfig

# =============================================================================
# BEFORE: Complex Manual Initialization (Legacy Pattern)
# =============================================================================

def legacy_initialization_example():
    """
    Example of the old complex initialization pattern.
    
    This approach required manual creation and wiring of all dependencies,
    making it error-prone and difficult to test.
    """
    print("=== LEGACY PATTERN (BEFORE) ===")
    
    # Manual dependency creation - complex and error-prone
    from autogen_framework.token_manager import TokenManager
    from autogen_framework.context_manager import ContextManager
    from autogen_framework.memory_manager import MemoryManager
    from autogen_framework.shell_executor import ShellExecutor
    from autogen_framework.agents.error_recovery import ErrorRecovery
    from autogen_framework.agents.task_decomposer import TaskDecomposer
    from autogen_framework.agents.plan_agent import PlanAgent
    from autogen_framework.agents.design_agent import DesignAgent
    from autogen_framework.agents.tasks_agent import TasksAgent
    from autogen_framework.agents.implement_agent import ImplementAgent
    
    # Configuration
    work_dir = "/tmp/example_project"
    llm_config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="gpt-4",
        api_key="sk-example"
    )
    
    # Step 1: Create all managers manually (order matters!)
    print("Creating managers manually...")
    token_manager = TokenManager(llm_config)
    memory_manager = MemoryManager(work_dir)
    
    # Context manager needs memory manager and token manager
    context_manager = ContextManager(
        work_dir=work_dir,
        memory_manager=memory_manager,
        token_manager=token_manager
    )
    
    # Error recovery needs multiple dependencies
    error_recovery = ErrorRecovery(
        llm_config=llm_config,
        token_manager=token_manager,
        context_manager=context_manager
    )
    
    # Shell executor needs error recovery
    shell_executor = ShellExecutor(error_recovery=error_recovery)
    
    # Task decomposer needs context manager
    task_decomposer = TaskDecomposer(
        llm_config=llm_config,
        context_manager=context_manager
    )
    
    # Step 2: Create agents with all dependencies (verbose and error-prone)
    print("Creating agents with manual dependency injection...")
    
    plan_agent = PlanAgent(
        name="PlanAgent",
        llm_config=llm_config,
        system_message="Generate project requirements",
        token_manager=token_manager,
        context_manager=context_manager,
        memory_manager=memory_manager,
        shell_executor=shell_executor
    )
    
    design_agent = DesignAgent(
        name="DesignAgent",
        llm_config=llm_config,
        system_message="Create technical design",
        token_manager=token_manager,
        context_manager=context_manager,
        memory_manager=memory_manager,
        shell_executor=shell_executor
    )
    
    tasks_agent = TasksAgent(
        name="TasksAgent",
        llm_config=llm_config,
        system_message="Generate implementation tasks",
        token_manager=token_manager,
        context_manager=context_manager,
        memory_manager=memory_manager,
        shell_executor=shell_executor
    )
    
    implement_agent = ImplementAgent(
        name="ImplementAgent",
        llm_config=llm_config,
        system_message="Execute implementation tasks",
        token_manager=token_manager,
        context_manager=context_manager,
        memory_manager=memory_manager,
        shell_executor=shell_executor,
        error_recovery=error_recovery,
        task_decomposer=task_decomposer
    )
    
    print(f"✓ Created {plan_agent.name}")
    print(f"✓ Created {design_agent.name}")
    print(f"✓ Created {tasks_agent.name}")
    print(f"✓ Created {implement_agent.name}")
    print("❌ Complex initialization with many parameters and potential for errors")
    print()


# =============================================================================
# AFTER: Simple Container-Based Initialization (New Pattern)
# =============================================================================

def modern_initialization_example():
    """
    Example of the new simplified container-based initialization pattern.
    
    This approach uses dependency injection to eliminate manual wiring
    and dramatically simplify agent creation.
    """
    print("=== MODERN PATTERN (AFTER) ===")
    
    from autogen_framework.dependency_container import DependencyContainer
    from autogen_framework.agents.plan_agent import PlanAgent
    from autogen_framework.agents.design_agent import DesignAgent
    from autogen_framework.agents.tasks_agent import TasksAgent
    from autogen_framework.agents.implement_agent import ImplementAgent
    
    # Configuration
    work_dir = "/tmp/example_project"
    llm_config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="gpt-4",
        api_key="sk-example"
    )
    
    # Step 1: Create dependency container (handles all complexity internally)
    print("Creating dependency container...")
    container = DependencyContainer.create_production(work_dir, llm_config)
    
    # Step 2: Create agents with simple, clean initialization
    print("Creating agents with container-based initialization...")
    
    plan_agent = PlanAgent(
        name="PlanAgent",
        llm_config=llm_config,
        system_message="Generate project requirements",
        container=container
    )
    
    design_agent = DesignAgent(
        name="DesignAgent",
        llm_config=llm_config,
        system_message="Create technical design",
        container=container
    )
    
    tasks_agent = TasksAgent(
        name="TasksAgent",
        llm_config=llm_config,
        system_message="Generate implementation tasks",
        container=container
    )
    
    implement_agent = ImplementAgent(
        name="ImplementAgent",
        llm_config=llm_config,
        system_message="Execute implementation tasks",
        container=container
    )
    
    print(f"✓ Created {plan_agent.name}")
    print(f"✓ Created {design_agent.name}")
    print(f"✓ Created {tasks_agent.name}")
    print(f"✓ Created {implement_agent.name}")
    print("✅ Simple initialization with clean, maintainable code")
    print()


# =============================================================================
# Testing Patterns: Before and After
# =============================================================================

def legacy_test_setup_example():
    """Example of complex test setup with manual mocking."""
    print("=== LEGACY TEST SETUP (BEFORE) ===")
    
    # Complex manual mock setup
    from unittest.mock import Mock, MagicMock
    from autogen_framework.agents.plan_agent import PlanAgent
    
    # Manual creation of all mock dependencies
    mock_token_manager = Mock()
    mock_context_manager = Mock()
    mock_memory_manager = Mock()
    mock_shell_executor = Mock()
    
    # Configure mock behavior
    mock_context_manager.get_plan_context = MagicMock(return_value="mock context")
    mock_memory_manager.save_requirements = MagicMock()
    
    # Create agent with all mocked dependencies
    llm_config = LLMConfig(base_url="http://test", model="test", api_key="test")
    
    agent = PlanAgent(
        name="TestAgent",
        llm_config=llm_config,
        system_message="Test message",
        token_manager=mock_token_manager,
        context_manager=mock_context_manager,
        memory_manager=mock_memory_manager,
        shell_executor=mock_shell_executor
    )
    
    print("❌ Complex test setup with manual mock configuration")
    print("❌ Easy to forget dependencies or misconfigure mocks")
    print()


def modern_test_setup_example():
    """Example of simple test setup with container fixtures."""
    print("=== MODERN TEST SETUP (AFTER) ===")
    
    # Simple fixture-based setup (this would be in a pytest test)
    print("# In your test file:")
    print("""
def test_plan_agent_requirements(simple_plan_agent):
    '''Test requirements generation with automatic mocking.'''
    agent = simple_plan_agent
    
    # All dependencies are automatically mocked through the container
    result = agent.generate_requirements("Create a web API", "/tmp/test")
    
    # Verify interactions with mocked managers
    agent.context_manager.get_plan_context.assert_called_once()
    agent.memory_manager.save_requirements.assert_called_once()
""")
    
    print("✅ Simple test setup using pytest fixtures")
    print("✅ Automatic mock configuration through container")
    print("✅ Consistent behavior across all tests")
    print()


# =============================================================================
# Integration Test Patterns: Before and After
# =============================================================================

async def legacy_integration_test_example():
    """Example of complex integration test setup."""
    print("=== LEGACY INTEGRATION TEST (BEFORE) ===")
    
    # Manual setup of real dependencies for integration testing
    from autogen_framework.token_manager import TokenManager
    from autogen_framework.context_manager import ContextManager
    from autogen_framework.memory_manager import MemoryManager
    from autogen_framework.agents.plan_agent import PlanAgent
    
    work_dir = "/tmp/integration_test"
    llm_config = LLMConfig(
        base_url="http://localhost:8000/v1",
        model="gpt-4",
        api_key="sk-real-key"
    )
    
    # Manual creation of real dependencies
    token_manager = TokenManager(llm_config)
    memory_manager = MemoryManager(work_dir)
    context_manager = ContextManager(work_dir, memory_manager, token_manager)
    
    # Create agent for integration testing
    agent = PlanAgent(
        name="IntegrationTestAgent",
        llm_config=llm_config,
        system_message="Generate requirements",
        token_manager=token_manager,
        context_manager=context_manager,
        memory_manager=memory_manager,
        shell_executor=None  # Not needed for this test
    )
    
    print("❌ Manual setup of real dependencies")
    print("❌ Risk of inconsistent configuration between tests")
    print()


async def modern_integration_test_example():
    """Example of simple integration test with container."""
    print("=== MODERN INTEGRATION TEST (AFTER) ===")
    
    print("# In your integration test file:")
    print("""
@pytest.mark.integration
async def test_plan_agent_real_llm(real_plan_agent):
    '''Test requirements generation with real LLM interaction.'''
    agent = real_plan_agent
    
    # Test with real LLM - all dependencies configured automatically
    result = await agent.generate_requirements(
        "Create a simple web API with user authentication",
        "/tmp/integration_test"
    )
    
    # Validate real output quality
    assert "# Requirements Document" in result
    assert "User Story:" in result
    assert "Acceptance Criteria" in result
""")
    
    print("✅ Simple integration test using real_plan_agent fixture")
    print("✅ Automatic configuration of real dependencies")
    print("✅ Consistent real LLM testing across all integration tests")
    print()


# =============================================================================
# Performance and Maintenance Benefits
# =============================================================================

def benefits_comparison():
    """Compare the benefits of the new approach."""
    print("=== BENEFITS COMPARISON ===")
    
    print("📊 Code Complexity:")
    print("  Legacy:  ~50 lines to create 4 agents with dependencies")
    print("  Modern:  ~15 lines to create 4 agents with dependencies")
    print("  Improvement: 70% reduction in initialization code")
    print()
    
    print("🧪 Test Setup:")
    print("  Legacy:  Manual mock creation and configuration for each test")
    print("  Modern:  Single fixture provides consistent mocked environment")
    print("  Improvement: 80% reduction in test setup code")
    print()
    
    print("🔧 Maintainability:")
    print("  Legacy:  Changes to dependencies require updates in multiple places")
    print("  Modern:  Changes to dependencies handled centrally in container")
    print("  Improvement: Single point of change for dependency management")
    print()
    
    print("🚀 Performance:")
    print("  Legacy:  Multiple instances of managers created unnecessarily")
    print("  Modern:  Shared singleton instances with lazy loading")
    print("  Improvement: Reduced memory usage and faster initialization")
    print()
    
    print("🛡️ Error Prevention:")
    print("  Legacy:  Easy to forget dependencies or pass them in wrong order")
    print("  Modern:  Container ensures all dependencies are available")
    print("  Improvement: Compile-time safety and runtime validation")
    print()


# =============================================================================
# Migration Guide Example
# =============================================================================

def migration_example():
    """Show step-by-step migration from legacy to modern pattern."""
    print("=== MIGRATION EXAMPLE ===")
    
    print("Step 1: Replace manual manager creation with container")
    print("  Before: token_manager = TokenManager(config)")
    print("  After:  container = DependencyContainer.create_production(work_dir, config)")
    print()
    
    print("Step 2: Update agent constructors")
    print("  Before: PlanAgent(name, config, message, token_manager, context_manager, ...)")
    print("  After:  PlanAgent(name, config, message, container)")
    print()
    
    print("Step 3: Update tests to use fixtures")
    print("  Before: Manual mock setup in each test")
    print("  After:  Use simple_plan_agent or real_plan_agent fixtures")
    print()
    
    print("Step 4: Verify all tests pass")
    print("  Run: pytest tests/unit/ -x --tb=short -q")
    print("  Run: pytest tests/integration/ -x --tb=short -q")
    print()


# =============================================================================
# Main Example Runner
# =============================================================================

def main():
    """Run all examples to demonstrate the patterns."""
    print("AutoGen Framework Initialization Patterns")
    print("=" * 50)
    print()
    
    # Show the before and after patterns
    legacy_initialization_example()
    modern_initialization_example()
    
    # Show testing patterns
    legacy_test_setup_example()
    modern_test_setup_example()
    
    # Show integration testing
    asyncio.run(legacy_integration_test_example())
    asyncio.run(modern_integration_test_example())
    
    # Show benefits and migration
    benefits_comparison()
    migration_example()
    
    print("🎉 Migration to container-based initialization complete!")
    print("📖 See docs/dependency-injection-guide.md for detailed documentation")


if __name__ == "__main__":
    main()