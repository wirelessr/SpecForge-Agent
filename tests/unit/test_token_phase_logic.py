"""
Unit tests for token phase logic (static vs dynamic phase).

Tests the enhanced token management that distinguishes between:
- Static phase: Uses estimated tokens when no LLM calls have been made
- Dynamic phase: Uses actual token usage from LLM responses
"""

import pytest
from unittest.mock import Mock, patch

from autogen_framework.config_manager import ConfigManager
from autogen_framework.token_manager import TokenManager, TokenCheckResult
from autogen_framework.agents.base_agent import BaseLLMAgent
from autogen_framework.models import LLMConfig


class TestTokenPhaseLogic:
    """Test token phase logic functionality."""
    
    @pytest.fixture
    def token_manager(self, test_llm_config):
        """Create a TokenManager instance for testing."""
        config_manager = Mock(spec=ConfigManager)
        config_manager.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.9,
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': False
        }
        return TokenManager(config_manager)
    
    @pytest.fixture
    def test_agent(self, test_llm_config, token_manager, mock_context_manager):
        """Create a test agent for testing."""
        class TestAgent(BaseLLMAgent):
            async def _process_task_impl(self, task_input):
                return {"result": "test"}
            
            def get_agent_capabilities(self):
                return ["test_capability"]
        
        agent = TestAgent(
            name="test_agent",
            llm_config=test_llm_config,
            system_message="Test agent",
            token_manager=token_manager,
            context_manager=mock_context_manager,
            description="Test agent for phase logic"
        )
        return agent
    
    def test_static_phase_detection(self, token_manager):
        """Test that static phase is correctly detected when no LLM calls made."""
        # Initially, no requests have been made
        assert token_manager.usage_stats['requests_made'] == 0
        assert token_manager.current_context_size == 0
        
        # Check token limit with estimated tokens (static phase)
        estimated_tokens = 5000
        result = token_manager.check_token_limit("test-model", estimated_tokens)
        
        # Should use estimated tokens in static phase
        assert result.current_tokens == estimated_tokens
        assert isinstance(result, TokenCheckResult)
    
    def test_dynamic_phase_detection(self, token_manager):
        """Test that dynamic phase is correctly detected after LLM calls."""
        # Simulate LLM usage
        token_manager.update_token_usage("test-model", 1000, "test_call", is_actual=True)
        
        # Now we're in dynamic phase
        assert token_manager.usage_stats['requests_made'] == 1
        assert token_manager.current_context_size == 1000
        
        # Check token limit - should use actual tokens even if estimated provided
        estimated_tokens = 5000
        result = token_manager.check_token_limit("test-model", estimated_tokens)
        
        # Should use actual tokens in dynamic phase, ignoring estimated
        assert result.current_tokens == 1000
        assert result.current_tokens != estimated_tokens
    
    def test_phase_transition(self, token_manager):
        """Test transition from static to dynamic phase."""
        # Start in static phase
        estimated_tokens = 3000
        result_static = token_manager.check_token_limit("test-model", estimated_tokens)
        assert result_static.current_tokens == estimated_tokens
        
        # Make first LLM call - transition to dynamic phase
        token_manager.update_token_usage("test-model", 500, "first_call", is_actual=True)
        
        # Now in dynamic phase
        result_dynamic = token_manager.check_token_limit("test-model", estimated_tokens)
        assert result_dynamic.current_tokens == 500  # Uses actual, not estimated
    
    def test_static_phase_compression_threshold(self, token_manager):
        """Test compression threshold in static phase."""
        # Set low threshold for testing
        token_manager.token_config['compression_threshold'] = 0.1
        
        # Get model limit
        model_limit = token_manager.get_model_limit("test-model")
        
        # Estimated tokens above threshold
        high_estimated = int(model_limit * 0.15)  # 15% > 10% threshold
        result = token_manager.check_token_limit("test-model", high_estimated)
        
        assert result.needs_compression is True
        assert result.percentage_used > 0.1
    
    def test_dynamic_phase_compression_threshold(self, token_manager):
        """Test compression threshold in dynamic phase."""
        # Build up actual usage
        model_limit = token_manager.get_model_limit("test-model")
        high_usage = int(model_limit * 0.95)  # 95% > 90% default threshold
        
        token_manager.update_token_usage("test-model", high_usage, "big_call", is_actual=True)
        
        result = token_manager.check_token_limit("test-model")
        
        assert result.needs_compression is True
        assert result.percentage_used > 0.9
        assert result.current_tokens == high_usage
    
    def test_estimated_tokens_ignored_in_dynamic_phase(self, token_manager):
        """Test that estimated tokens are ignored when actual usage exists."""
        # Set up dynamic phase with actual usage
        token_manager.update_token_usage("test-model", 2000, "actual_call", is_actual=True)
        
        # Provide very high estimated tokens
        very_high_estimated = 1000000
        result = token_manager.check_token_limit("test-model", very_high_estimated)
        
        # Should use actual tokens (2000), not estimated (1000000)
        assert result.current_tokens == 2000
        assert result.current_tokens != very_high_estimated
    
    def test_no_estimated_tokens_in_static_phase(self, token_manager):
        """Test behavior when no estimated tokens provided in static phase."""
        # No LLM calls made, no estimated tokens provided
        result = token_manager.check_token_limit("test-model")
        
        # Should use 0 tokens
        assert result.current_tokens == 0
        assert result.needs_compression is False
    
    def test_update_token_usage_actual_vs_estimated(self, token_manager):
        """Test the difference between actual and estimated token updates."""
        # Update with actual usage
        token_manager.update_token_usage("test-model", 1000, "actual", is_actual=True)
        
        # Check stats are updated
        assert token_manager.usage_stats['requests_made'] == 1
        assert token_manager.usage_stats['total_tokens_used'] == 1000
        assert token_manager.current_context_size == 1000
        
        # Update with estimated usage (should not affect stats)
        token_manager.update_token_usage("test-model", 500, "estimated", is_actual=False)
        
        # Stats should remain unchanged
        assert token_manager.usage_stats['requests_made'] == 1
        assert token_manager.usage_stats['total_tokens_used'] == 1000
        assert token_manager.current_context_size == 1000
    
    def test_context_manager_integration(self, test_agent, token_manager):
        """Test that agent can be configured with ContextManager."""
        from unittest.mock import Mock
        
        # Create mock ContextManager
        mock_context_manager = Mock()
        mock_context_manager.token_manager = token_manager
        
        # Set ContextManager on agent
        test_agent.set_context_manager(mock_context_manager)
        
        # Verify ContextManager integration
        assert test_agent.context_manager is not None
        assert test_agent.context_manager == mock_context_manager
    
    def test_unified_token_management_architecture(self, test_agent, token_manager):
        """Test that the unified token management architecture works correctly."""
        # Add some context to make estimation meaningful
        test_agent.update_context({"test": "data" * 100})
        
        # Test static phase estimation
        estimated_tokens = test_agent._estimate_static_content_tokens()
        assert estimated_tokens > 0
        
        # Test token limit checking with estimated tokens
        token_check = token_manager.check_token_limit("test-model", estimated_tokens)
        assert token_check.current_tokens == estimated_tokens
        
        # Simulate actual LLM usage
        token_manager.update_token_usage("test-model", 500, "test_call", is_actual=True)
        
        # Test dynamic phase - should use actual tokens
        token_check_dynamic = token_manager.check_token_limit("test-model", estimated_tokens)
        assert token_check_dynamic.current_tokens == 500  # Uses actual, not estimated
        assert token_check_dynamic.current_tokens != estimated_tokens
    
    def test_estimate_static_content_tokens(self, test_agent):
        """Test estimation of static content tokens."""
        # Add some context
        test_agent.update_context({
            "key1": "value1" * 100,
            "key2": "value2" * 200
        })
        
        # Add memory context
        test_agent.memory_context = {
            "memory1": "memory content" * 50
        }
        
        # Add conversation history
        test_agent.add_to_conversation_history("user", "test message" * 30)
        test_agent.add_to_conversation_history("assistant", "test response" * 40)
        
        # Estimate tokens
        estimated = test_agent._estimate_static_content_tokens()
        
        # Should be a reasonable estimate (> 0)
        assert estimated > 0
        assert isinstance(estimated, int)
    
    def test_estimate_static_content_empty(self, test_agent):
        """Test estimation with empty content."""
        # Clear all content
        test_agent.context.clear()
        test_agent.memory_context.clear()
        test_agent.conversation_history.clear()
        test_agent.system_message = ""
        
        estimated = test_agent._estimate_static_content_tokens()
        
        # Should still return at least 1 (base overhead)
        assert estimated >= 1


class TestTokenPhaseIntegration:
    """Integration tests for token phase logic."""
    
    @pytest.fixture
    def integration_setup(self, test_llm_config):
        """Set up integration test environment."""
        from unittest.mock import Mock
        config_manager = Mock(spec=ConfigManager)
        config_manager.get_token_config.return_value = {
            'default_token_limit': 8192,
            'compression_threshold': 0.8,  # 80% threshold
            'compression_enabled': True,
            'compression_target_ratio': 0.5,
            'verbose_logging': False
        }
        
        token_manager = TokenManager(config_manager)
        
        class TestAgent(BaseLLMAgent):
            async def _process_task_impl(self, task_input):
                return {"result": "test"}
            
            def get_agent_capabilities(self):
                return ["test_capability"]
        
        from unittest.mock import Mock
        mock_context_manager = Mock()
        
        agent = TestAgent(
            name="integration_agent",
            llm_config=test_llm_config,
            system_message="Integration test agent",
            token_manager=token_manager,
            context_manager=mock_context_manager,
            description="Agent for integration testing"
        )
        
        return {
            'token_manager': token_manager,
            'agent': agent,
            'llm_config': test_llm_config
        }
    
    def test_full_static_to_dynamic_flow(self, integration_setup):
        """Test complete flow from static to dynamic phase."""
        token_manager = integration_setup['token_manager']
        agent = integration_setup['agent']
        
        # Phase 1: Static phase with large context
        large_context = {"data": "x" * 10000}  # Large context
        agent.update_context(large_context)
        
        # Estimate static tokens
        estimated = agent._estimate_static_content_tokens()
        assert estimated > 1000  # Should be substantial
        
        # Check in static phase
        static_result = token_manager.check_token_limit("test-model", estimated)
        assert static_result.current_tokens == estimated
        
        # Phase 2: Transition to dynamic phase
        token_manager.update_token_usage("test-model", 500, "first_call", is_actual=True)
        
        # Check in dynamic phase
        dynamic_result = token_manager.check_token_limit("test-model", estimated)
        assert dynamic_result.current_tokens == 500  # Uses actual, not estimated
        assert dynamic_result.current_tokens != estimated
        
        # Phase 3: Build up dynamic usage
        for i in range(5):
            token_manager.update_token_usage("test-model", 1000, f"call_{i}", is_actual=True)
        
        final_result = token_manager.check_token_limit("test-model", estimated)
        assert final_result.current_tokens == 5500  # 500 + 5*1000
        assert final_result.current_tokens != estimated
    
    def test_compression_trigger_in_both_phases(self, integration_setup):
        """Test that compression can be triggered in both phases."""
        token_manager = integration_setup['token_manager']
        agent = integration_setup['agent']
        
        # Set very low threshold for testing
        token_manager.token_config['compression_threshold'] = 0.01  # 1%
        
        # Test static phase compression trigger
        huge_context = {"data": "x" * 100000}  # Very large context
        agent.update_context(huge_context)
        
        estimated = agent._estimate_static_content_tokens()
        static_result = token_manager.check_token_limit("test-model", estimated)
        
        # Should trigger compression in static phase
        assert static_result.needs_compression is True
        
        # Reset threshold to normal
        token_manager.token_config['compression_threshold'] = 0.8
        
        # Transition to dynamic phase
        token_manager.update_token_usage("test-model", 100, "small_call", is_actual=True)
        
        # Build up to trigger compression in dynamic phase
        model_limit = token_manager.get_model_limit("test-model")
        large_usage = int(model_limit * 0.85)  # 85% > 80% threshold
        
        token_manager.update_token_usage("test-model", large_usage, "large_call", is_actual=True)
        
        dynamic_result = token_manager.check_token_limit("test-model")
        
        # Should trigger compression in dynamic phase
        assert dynamic_result.needs_compression is True
        assert dynamic_result.current_tokens > model_limit * 0.8