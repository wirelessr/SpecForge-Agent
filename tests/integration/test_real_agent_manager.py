"""
Integration tests for AgentManager with real components.

This module contains tests that use actual MemoryManager, ShellExecutor, and other
real components to verify the integration between the framework components.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from autogen_framework.agent_manager import AgentManager
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.shell_executor import ShellExecutor


class TestAgentManagerRealIntegration:
    """Integration tests for AgentManager with real components."""
    
    @pytest.fixture
    def agent_manager(self, temp_workspace):
        """Create an AgentManager instance for integration testing."""
        return AgentManager(temp_workspace)
    
    @pytest.mark.integration
    def test_agent_manager_with_real_memory_manager(self, agent_manager, temp_workspace, real_llm_config):
        """Test AgentManager with real MemoryManager integration."""
        # Verify memory manager integration
        assert isinstance(agent_manager.memory_manager, MemoryManager)
        assert agent_manager.memory_manager.workspace_path == Path(temp_workspace)
        
        # Test memory loading
        memory_content = agent_manager.memory_manager.load_memory()
        assert isinstance(memory_content, dict)
        
        # Test that memory can be saved and loaded
        agent_manager.memory_manager.save_memory(
            "test_integration",
            "# Test Integration\n\nThis is a test for integration.",
            "global"
        )
        
        # Reload memory and verify
        updated_memory = agent_manager.memory_manager.load_memory()
        assert "global" in updated_memory
        global_memory = updated_memory["global"]
        assert any("test_integration" in key for key in global_memory.keys())
    
    @pytest.mark.integration
    def test_agent_manager_with_real_shell_executor(self, agent_manager, temp_workspace):
        """Test AgentManager with real ShellExecutor integration."""
        # Verify shell executor integration
        assert isinstance(agent_manager.shell_executor, ShellExecutor)
        assert agent_manager.shell_executor.default_working_dir == temp_workspace
        
        # Test that shell executor can execute real commands
        import asyncio
        
        async def test_shell_execution():
            result = await agent_manager.shell_executor.execute_command("echo 'integration test'")
            assert result.success is True
            assert "integration test" in result.stdout
            return result
        
        # Run the async test
        result = asyncio.run(test_shell_execution())
        assert result.success is True
    
    @pytest.mark.integration
    def test_agent_manager_coordination_logging(self, agent_manager, temp_workspace, real_llm_config):
        """Test that coordination events are properly logged with real components."""
        # Mock only the agent initialization to avoid AutoGen dependencies
        with patch('autogen_framework.agents.base_agent.BaseLLMAgent.initialize_autogen_agent') as mock_init:
            mock_init.return_value = True
            
            # Setup agents with real components - this now creates and injects managers
            result = agent_manager.setup_agents(real_llm_config)
            assert result is True
            
            # Verify coordination log contains initialization event
            assert len(agent_manager.coordination_log) > 0
            init_event = agent_manager.coordination_log[0]
            assert init_event["event_type"] == "agent_initialization"
            assert "agents_initialized" in init_event["details"]
            assert len(init_event["details"]["agents_initialized"]) == 4
            
            # Verify that real LLM config was used
            assert init_event["details"]["llm_config"]["model"] == real_llm_config.model
            assert init_event["details"]["llm_config"]["base_url"] == real_llm_config.base_url
            
            # Verify that managers were created and injected
            assert agent_manager.plan_agent is not None
            assert agent_manager.design_agent is not None
            assert agent_manager.tasks_agent is not None
            assert agent_manager.implement_agent is not None
            
            # Verify agents have manager dependencies
            assert hasattr(agent_manager.plan_agent, 'token_manager')
            assert hasattr(agent_manager.plan_agent, 'context_manager')
            assert hasattr(agent_manager.design_agent, 'token_manager')
            assert hasattr(agent_manager.design_agent, 'context_manager')
            assert hasattr(agent_manager.tasks_agent, 'token_manager')
            assert hasattr(agent_manager.tasks_agent, 'context_manager')
            assert hasattr(agent_manager.implement_agent, 'token_manager')
            assert hasattr(agent_manager.implement_agent, 'context_manager')
    
    @pytest.mark.integration
    def test_real_memory_and_shell_integration(self, agent_manager, temp_workspace, real_llm_config):
        """Test integration between real MemoryManager and ShellExecutor."""
        # Add some memory content
        agent_manager.memory_manager.save_memory(
            "shell_commands",
            "# Shell Commands\n\nUseful commands:\n- ls -la\n- pwd\n- echo 'test'",
            "global"
        )
        
        # Load memory
        memory_content = agent_manager.memory_manager.load_memory()
        assert "global" in memory_content
        
        # Test shell execution with commands from memory
        import asyncio
        
        async def test_memory_shell_integration():
            # Execute a command that was mentioned in memory
            result = await agent_manager.shell_executor.execute_command("pwd")
            assert result.success is True
            assert temp_workspace in result.stdout
            
            # Execute another command
            result2 = await agent_manager.shell_executor.execute_command("echo 'memory integration test'")
            assert result2.success is True
            assert "memory integration test" in result2.stdout
            
            return result, result2
        
        results = asyncio.run(test_memory_shell_integration())
        assert all(r.success for r in results)
        
        # Verify execution history
        history = agent_manager.shell_executor.execution_history
        assert len(history) >= 2
        assert any("pwd" in cmd.command for cmd in history)
        assert any("echo" in cmd.command for cmd in history)
    
    @pytest.mark.integration
    def test_agent_manager_export_coordination_report(self, agent_manager, temp_workspace, real_llm_config):
        """Test coordination report export with real components."""
        # Mock agent initialization
        with patch('autogen_framework.agents.base_agent.BaseLLMAgent.initialize_autogen_agent') as mock_init:
            mock_init.return_value = True
            
            # Setup agents and create some coordination events
            agent_manager.setup_agents(real_llm_config)
            
            # Add some test interactions
            agent_manager._record_coordination_event(
                "test_event",
                {"test_data": "integration_test"}
            )
            
            # Export report
            output_path = Path(temp_workspace) / "coordination_report.json"
            result = agent_manager.export_coordination_report(str(output_path))
            
            assert result is True
            assert output_path.exists()
            
            # Verify report content
            import json
            with open(output_path, 'r') as f:
                report = json.load(f)
            
            assert "timestamp" in report
            assert "workspace_path" in report
            assert "coordination_events" in report
            assert "agent_interactions" in report
            assert "agent_status" in report
            
            # Verify workspace path is correct
            assert report["workspace_path"] == temp_workspace
            
            # Verify coordination events include our test event
            events = report["coordination_events"]
            assert len(events) >= 2  # At least initialization + test event
            assert any(event["event_type"] == "test_event" for event in events)
    
    @pytest.mark.integration
    def test_agent_capabilities_with_real_components(self, agent_manager, temp_workspace, real_llm_config):
        """Test agent capabilities reporting with real components."""
        # Mock agent initialization
        with patch('autogen_framework.agents.base_agent.BaseLLMAgent.initialize_autogen_agent') as mock_init:
            mock_init.return_value = True
            
            # Setup agents
            agent_manager.setup_agents(real_llm_config)
            
            # Get capabilities
            capabilities = agent_manager.get_agent_capabilities()
            
            assert isinstance(capabilities, dict)
            assert len(capabilities) == 4  # plan, design, tasks, implement
            assert "plan" in capabilities
            assert "design" in capabilities
            assert "tasks" in capabilities
            assert "implement" in capabilities
            
            # Each agent should have capabilities
            for agent_name, agent_capabilities in capabilities.items():
                assert isinstance(agent_capabilities, list)
                assert len(agent_capabilities) > 0
    
    @pytest.mark.integration
    def test_coordination_statistics_with_real_components(self, agent_manager, temp_workspace, real_llm_config):
        """Test coordination statistics with real components."""
        # Mock agent initialization
        with patch('autogen_framework.agents.base_agent.BaseLLMAgent.initialize_autogen_agent') as mock_init:
            mock_init.return_value = True
            
            # Setup agents
            agent_manager.setup_agents(real_llm_config)
            
            # Add some test interactions
            agent_manager._record_agent_interaction("test_interaction_1", "plan", "test_start", {"test": True})
            agent_manager._record_agent_interaction("test_interaction_2", "design", "test_complete", {"test": True})
            
            # Get statistics
            stats = agent_manager.get_coordination_statistics()
            
            assert isinstance(stats, dict)
            assert "total_interactions" in stats
            assert "total_coordination_events" in stats
            assert "agent_interaction_counts" in stats
            assert "event_type_counts" in stats
            assert "agents_initialized" in stats
            
            # Verify statistics are accurate
            assert stats["total_interactions"] >= 2
            assert stats["agents_initialized"] == 4
            assert "plan" in stats["agent_interaction_counts"]
            assert "design" in stats["agent_interaction_counts"]
    
    @pytest.mark.integration
    def test_memory_context_update_with_real_components(self, agent_manager, temp_workspace, real_llm_config):
        """Test memory context updates with real components."""
        # Add memory content
        agent_manager.memory_manager.save_memory(
            "project_context",
            "# Project Context\n\nThis is an integration test project.",
            "global"
        )
        
        # Mock agent initialization
        with patch('autogen_framework.agents.base_agent.BaseLLMAgent.initialize_autogen_agent') as mock_init:
            mock_init.return_value = True
            
            # Setup agents
            agent_manager.setup_agents(real_llm_config)
            
            # Load memory and update agent memory
            memory_context = agent_manager.memory_manager.load_memory()
            agent_manager.update_agent_memory(memory_context)
            
            # Verify agents received memory context
            for agent_name, agent in agent_manager.agents.items():
                # Check that update_memory_context was called
                # (This would be verified by checking agent's memory_context attribute)
                assert hasattr(agent, 'memory_context')
    
    @pytest.mark.integration
    def test_manager_creation_and_injection(self, agent_manager, temp_workspace, real_llm_config):
        """Test that managers are properly created and injected into agents."""
        # Mock agent initialization
        with patch('autogen_framework.agents.base_agent.BaseLLMAgent.initialize_autogen_agent') as mock_init:
            mock_init.return_value = True
            
            # Initialize agents - this should create and inject managers
            result = agent_manager.setup_agents(real_llm_config)
            assert result is True
            
            # Verify all agents were created
            assert agent_manager.plan_agent is not None
            assert agent_manager.design_agent is not None
            assert agent_manager.tasks_agent is not None
            assert agent_manager.implement_agent is not None
            
            # Verify each agent has the required manager dependencies
            agents = [
                agent_manager.plan_agent,
                agent_manager.design_agent,
                agent_manager.tasks_agent,
                agent_manager.implement_agent
            ]
            
            for agent in agents:
                # Check that agent has manager dependencies
                assert hasattr(agent, 'token_manager'), f"Agent {agent.name} missing token_manager"
                assert hasattr(agent, 'context_manager'), f"Agent {agent.name} missing context_manager"
                
                # Verify manager types
                from autogen_framework.token_manager import TokenManager
                from autogen_framework.context_manager import ContextManager
                
                assert isinstance(agent.token_manager, TokenManager), f"Agent {agent.name} has wrong token_manager type"
                assert isinstance(agent.context_manager, ContextManager), f"Agent {agent.name} has wrong context_manager type"
                
                # Verify managers are properly configured
                assert agent.token_manager.config_manager is not None
                assert str(agent.context_manager.work_dir) == temp_workspace
                assert agent.context_manager.llm_config == real_llm_config
    
    @pytest.mark.integration
    def test_context_management_integration_scenarios(self, agent_manager, temp_workspace, real_llm_config):
        """Test context management integration scenarios with real managers."""
        # Mock agent initialization
        with patch('autogen_framework.agents.base_agent.BaseLLMAgent.initialize_autogen_agent') as mock_init:
            mock_init.return_value = True
            
            # Initialize agents
            result = agent_manager.setup_agents(real_llm_config)
            assert result is True
            
            # Test that context managers can be used for context preparation
            import asyncio
            
            async def test_context_preparation():
                # Get a context manager from one of the agents
                context_manager = agent_manager.plan_agent.context_manager
                
                # Initialize the context manager
                await context_manager.initialize()
                
                # Test context preparation
                test_prompt = "Test system prompt for integration testing"
                prepared = await context_manager.prepare_system_prompt(test_prompt)
                
                assert hasattr(prepared, 'system_prompt')
                assert hasattr(prepared, 'estimated_tokens')
                assert isinstance(prepared.system_prompt, str)
                assert isinstance(prepared.estimated_tokens, int)
                assert len(prepared.system_prompt) > 0
                assert prepared.estimated_tokens > 0
                
                return prepared
            
            # Run the async test
            prepared = asyncio.run(test_context_preparation())
            assert prepared is not None
    
    @pytest.mark.integration
    def test_token_management_integration_scenarios(self, agent_manager, temp_workspace, real_llm_config):
        """Test token management integration scenarios with real managers."""
        # Mock agent initialization
        with patch('autogen_framework.agents.base_agent.BaseLLMAgent.initialize_autogen_agent') as mock_init:
            mock_init.return_value = True
            
            # Initialize agents
            result = agent_manager.setup_agents(real_llm_config)
            assert result is True
            
            # Test that token managers can be used for token operations
            token_manager = agent_manager.plan_agent.token_manager
            
            # Test token estimation
            test_text = "This is a test text for token estimation in integration testing."
            estimated_tokens = token_manager.estimate_tokens_from_text(test_text)
            assert isinstance(estimated_tokens, int)
            assert estimated_tokens > 0
            
            # Test character count estimation
            char_count = len(test_text)
            char_based_tokens = token_manager.estimate_tokens_from_char_count(char_count)
            assert isinstance(char_based_tokens, int)
            assert char_based_tokens > 0
            
            # Test token limit checking
            limit_check = token_manager.check_token_limit(real_llm_config.model, estimated_static_tokens=estimated_tokens)
            assert hasattr(limit_check, 'needs_compression')
            assert hasattr(limit_check, 'current_tokens')
            assert hasattr(limit_check, 'model_limit')
            assert isinstance(limit_check.needs_compression, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])