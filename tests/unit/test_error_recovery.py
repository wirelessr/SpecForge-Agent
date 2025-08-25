"""
Unit tests for ErrorRecovery agent.

This module tests the ErrorRecovery agent's capabilities including:
- Error analysis and classification
- Recovery strategy generation
- Strategy ranking and optimization
- Learning from recovery attempts
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from autogen_framework.agents.error_recovery import (
    ErrorRecovery, ErrorType, ErrorAnalysis, RecoveryStrategy, 
    RecoveryResult, CommandResult
)
from autogen_framework.models import LLMConfig


class TestErrorRecovery:
    """Test cases for ErrorRecovery agent."""
    
    @pytest.fixture
    def test_llm_config(self):
        """Test LLM configuration."""
        return LLMConfig(
            base_url="http://test.local:8888/openai/v1",
            model="test-model",
            api_key="test-key"
        )
    
    @pytest.fixture
    def error_recovery_agent(self, test_llm_config, mock_dependency_container):
        """Create ErrorRecovery agent for testing with required manager dependencies."""
        system_message = "You are an error recovery agent for testing."
        return ErrorRecovery(
            name="TestErrorRecovery",
            llm_config=test_llm_config,
            system_message=system_message,
            container=mock_dependency_container
        )
    
    @pytest.fixture
    def sample_command_result(self):
        """Sample failed command result."""
        return CommandResult(
            command="pip install nonexistent-package",
            exit_code=1,
            stdout="",
            stderr="ERROR: Could not find a version that satisfies the requirement nonexistent-package",
            execution_time=2.5,
            success=False
        )
    
    @pytest.fixture
    def sample_error_analysis(self):
        """Sample error analysis."""
        return ErrorAnalysis(
            error_type=ErrorType.DEPENDENCY_MISSING,
            root_cause="Package not found in PyPI",
            affected_components=["pip", "python_environment"],
            suggested_fixes=["Check package name", "Use alternative package"],
            confidence=0.8,
            analysis_reasoning="Clear package not found error",
            error_patterns=["Could not find a version"],
            context_factors=["pip", "python"],
            severity="medium"
        )
    
    def test_initialization(self, error_recovery_agent):
        """Test ErrorRecovery agent initialization."""
        assert error_recovery_agent.name == "TestErrorRecovery"
        assert error_recovery_agent.description == "Error recovery agent for intelligent failure analysis and recovery"
        assert len(error_recovery_agent.error_patterns) > 0
        assert error_recovery_agent.recovery_stats['total_recoveries_attempted'] == 0
    
    def test_get_agent_capabilities(self, error_recovery_agent):
        """Test agent capabilities reporting."""
        capabilities = error_recovery_agent.get_agent_capabilities()
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        assert any("error analysis" in cap.lower() for cap in capabilities)
        assert any("recovery strategies" in cap.lower() for cap in capabilities)
    
    def test_get_context_requirements(self, error_recovery_agent):
        """Test context requirements specification."""
        # Test with failed result
        task_input = {"failed_result": Mock()}
        context_spec = error_recovery_agent.get_context_requirements(task_input)
        assert context_spec is not None
        assert context_spec.context_type == "implementation"
        
        # Test with task
        task_input = {"task": Mock()}
        context_spec = error_recovery_agent.get_context_requirements(task_input)
        assert context_spec is not None
        assert context_spec.context_type == "implementation"
        
        # Test without relevant input
        task_input = {}
        context_spec = error_recovery_agent.get_context_requirements(task_input)
        assert context_spec is None
    
    def test_classify_error_by_patterns(self, error_recovery_agent, sample_command_result):
        """Test pattern-based error classification."""
        analysis = error_recovery_agent._classify_error_by_patterns(sample_command_result)
        
        assert isinstance(analysis, ErrorAnalysis)
        assert analysis.error_type == ErrorType.DEPENDENCY_MISSING
        assert analysis.confidence > 0.5
        assert len(analysis.error_patterns) > 0
    
    def test_get_template_strategies(self, error_recovery_agent):
        """Test template strategy retrieval."""
        # Test dependency missing strategies
        strategies = error_recovery_agent._get_template_strategies(ErrorType.DEPENDENCY_MISSING)
        assert len(strategies) > 0
        assert any("pip install" in str(s.commands) for s in strategies)
        
        # Test permission denied strategies
        strategies = error_recovery_agent._get_template_strategies(ErrorType.PERMISSION_DENIED)
        assert len(strategies) > 0
        assert any("sudo" in str(s.commands) for s in strategies)
        
        # Test unknown error type
        strategies = error_recovery_agent._get_template_strategies(ErrorType.UNKNOWN)
        assert len(strategies) == 0
    
    def test_rank_strategies(self, error_recovery_agent, sample_error_analysis):
        """Test strategy ranking."""
        strategies = [
            RecoveryStrategy(
                name="high_prob_strategy",
                description="High probability strategy",
                success_probability=0.9,
                resource_cost=2
            ),
            RecoveryStrategy(
                name="low_prob_strategy", 
                description="Low probability strategy",
                success_probability=0.3,
                resource_cost=1
            ),
            RecoveryStrategy(
                name="medium_prob_strategy",
                description="Medium probability strategy", 
                success_probability=0.6,
                resource_cost=3
            )
        ]
        
        ranked = error_recovery_agent._rank_strategies(strategies, sample_error_analysis)
        
        # Should be ranked by success probability (descending)
        assert ranked[0].name == "high_prob_strategy"
        assert ranked[1].name == "medium_prob_strategy"
        assert ranked[2].name == "low_prob_strategy"
    
    def test_validate_strategy(self, error_recovery_agent):
        """Test strategy validation."""
        # Valid strategy
        valid_strategy = RecoveryStrategy(
            name="safe_strategy",
            description="Safe recovery strategy",
            commands=["pip install --user package"],
            success_probability=0.7,
            resource_cost=2
        )
        
        result = error_recovery_agent.validate_strategy(valid_strategy)
        assert result["valid"] is True
        assert result["risk_level"] == "low"
        
        # Dangerous strategy
        dangerous_strategy = RecoveryStrategy(
            name="dangerous_strategy",
            description="Dangerous recovery strategy",
            commands=["sudo rm -rf /tmp/*"],
            success_probability=0.8,
            resource_cost=1
        )
        
        result = error_recovery_agent.validate_strategy(dangerous_strategy)
        assert result["risk_level"] == "high"
        assert len(result["recommendations"]) > 0
    
    def test_calculate_context_relevance(self, error_recovery_agent, sample_error_analysis):
        """Test context relevance calculation."""
        # Relevant strategy
        relevant_strategy = RecoveryStrategy(
            name="relevant_strategy",
            description="Strategy that matches error patterns",
            commands=["pip install alternative-package"],  # Matches pip context
            success_probability=0.6,
            resource_cost=2
        )
        
        relevance = error_recovery_agent._calculate_context_relevance(relevant_strategy, sample_error_analysis)
        assert relevance > 0.5
        
        # Irrelevant strategy
        irrelevant_strategy = RecoveryStrategy(
            name="irrelevant_strategy",
            description="Strategy unrelated to error",
            commands=["systemctl restart nginx"],  # Unrelated to pip error
            success_probability=0.6,
            resource_cost=2
        )
        
        relevance = error_recovery_agent._calculate_context_relevance(irrelevant_strategy, sample_error_analysis)
        assert relevance <= 0.7  # Should be lower than relevant strategy
    
    def test_get_strategy_recommendations(self, error_recovery_agent):
        """Test strategy recommendations based on learned patterns."""
        # Add some learned patterns
        error_recovery_agent.recovery_patterns = [
            {
                "error_type": "dependency_missing",
                "strategy_name": "learned_pip_install",
                "commands": ["pip install --upgrade pip", "pip install {package}"],
                "success_rate": 0.9
            },
            {
                "error_type": "dependency_missing", 
                "strategy_name": "learned_conda_install",
                "commands": ["conda install {package}"],
                "success_rate": 0.7
            }
        ]
        
        recommendations = error_recovery_agent.get_strategy_recommendations(ErrorType.DEPENDENCY_MISSING)
        
        assert len(recommendations) == 2
        assert recommendations[0].name == "learned_pip_install"  # Higher success rate should be first
        assert recommendations[1].name == "learned_conda_install"
    
    def test_export_import_learned_patterns(self, error_recovery_agent):
        """Test exporting and importing learned patterns."""
        # Add some patterns
        error_recovery_agent.recovery_patterns = [
            {
                "error_type": "dependency_missing",
                "strategy_name": "test_strategy",
                "commands": ["test command"],
                "success_rate": 0.8
            }
        ]
        error_recovery_agent.recovery_stats['strategies_learned'] = 1
        
        # Export patterns
        exported = error_recovery_agent.export_learned_patterns()
        assert "recovery_patterns" in exported
        assert "recovery_stats" in exported
        assert exported["total_patterns"] == 1
        
        # Create new agent and import patterns
        new_agent = ErrorRecovery(
            name="NewAgent",
            llm_config=error_recovery_agent.llm_config,
            system_message="Test",
            container=error_recovery_agent.container
        )
        
        success = new_agent.import_learned_patterns(exported)
        assert success is True
        assert len(new_agent.recovery_patterns) == 1
        assert new_agent.recovery_stats['strategies_learned'] == 1
    
    def test_get_recovery_statistics(self, error_recovery_agent):
        """Test recovery statistics retrieval."""
        stats = error_recovery_agent.get_recovery_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_recoveries_attempted' in stats
        assert 'successful_recoveries' in stats
        assert 'failed_recoveries' in stats
        assert 'strategies_learned' in stats
        assert 'most_common_errors' in stats
        assert 'most_successful_strategies' in stats
    
    def test_reset_statistics(self, error_recovery_agent):
        """Test statistics reset."""
        # Modify some statistics
        error_recovery_agent.recovery_stats['total_recoveries_attempted'] = 10
        error_recovery_agent.recovery_stats['successful_recoveries'] = 7
        
        # Reset
        error_recovery_agent.reset_statistics()
        
        # Verify reset
        assert error_recovery_agent.recovery_stats['total_recoveries_attempted'] == 0
        assert error_recovery_agent.recovery_stats['successful_recoveries'] == 0
    
    @pytest.mark.asyncio
    async def test_process_task_impl_error_analysis(self, error_recovery_agent, sample_command_result):
        """Test error analysis task processing."""
        with patch.object(error_recovery_agent, '_analyze_error', new_callable=AsyncMock) as mock_analyze:
            mock_analysis = ErrorAnalysis(
                error_type=ErrorType.DEPENDENCY_MISSING,
                root_cause="Test error",
                confidence=0.8
            )
            mock_analyze.return_value = mock_analysis
            
            task_input = {
                "task_type": "analyze_error",
                "failed_result": sample_command_result
            }
            
            result = await error_recovery_agent._process_task_impl(task_input)
            
            assert result["success"] is True
            assert "error_analysis" in result
            mock_analyze.assert_called_once_with(sample_command_result)
    
    @pytest.mark.asyncio
    async def test_process_task_impl_strategy_generation(self, error_recovery_agent, sample_error_analysis):
        """Test strategy generation task processing."""
        with patch.object(error_recovery_agent, '_generate_strategies', new_callable=AsyncMock) as mock_generate:
            mock_strategies = [
                RecoveryStrategy(name="test_strategy", description="Test strategy")
            ]
            mock_generate.return_value = mock_strategies
            
            task_input = {
                "task_type": "generate_strategies",
                "error_analysis": sample_error_analysis.to_dict()
            }
            
            result = await error_recovery_agent._process_task_impl(task_input)
            
            assert result["success"] is True
            assert "strategies" in result
            assert result["strategy_count"] == 1
    
    @pytest.mark.asyncio
    async def test_process_task_impl_unknown_task_type(self, error_recovery_agent):
        """Test handling of unknown task type."""
        task_input = {"task_type": "unknown_task"}
        
        with pytest.raises(ValueError, match="Unknown task type"):
            await error_recovery_agent._process_task_impl(task_input)
    
    def test_extract_json_from_response(self, error_recovery_agent):
        """Test JSON extraction from LLM responses."""
        # Test JSON in code block
        response_with_block = '''
        Here's the analysis:
        ```json
        {"error_type": "dependency_missing", "confidence": 0.8}
        ```
        '''
        
        json_content = error_recovery_agent._extract_json_from_response(response_with_block)
        assert json_content is not None
        assert "error_type" in json_content
        
        # Test direct JSON
        response_direct = '{"error_type": "syntax_error", "confidence": 0.9}'
        json_content = error_recovery_agent._extract_json_from_response(response_direct)
        assert json_content is not None
        assert "syntax_error" in json_content
        
        # Test no JSON
        response_no_json = "This is just text without any JSON content."
        json_content = error_recovery_agent._extract_json_from_response(response_no_json)
        assert json_content is None
    
    def test_parse_error_analysis_from_text(self, error_recovery_agent, sample_command_result):
        """Test parsing error analysis from text response."""
        text_response = """
        This appears to be a permission denied error.
        The command failed because access was denied to the file.
        """
        
        analysis = error_recovery_agent._parse_error_analysis_from_text(text_response, sample_command_result)
        
        assert isinstance(analysis, ErrorAnalysis)
        assert analysis.error_type == ErrorType.PERMISSION_DENIED
        assert analysis.confidence > 0
    
    def test_parse_strategies_from_text(self, error_recovery_agent):
        """Test parsing strategies from text response."""
        text_response = """
        Strategy 1: Try installing with pip
        - pip install package
        - pip install --user package
        
        Strategy 2: Use conda instead
        - conda install package
        """
        
        strategies = error_recovery_agent._parse_strategies_from_text(text_response)
        
        assert len(strategies) >= 1
        assert any("pip install" in str(s.commands) for s in strategies)


class TestErrorAnalysis:
    """Test cases for ErrorAnalysis data class."""
    
    def test_to_dict(self):
        """Test ErrorAnalysis to_dict conversion."""
        analysis = ErrorAnalysis(
            error_type=ErrorType.DEPENDENCY_MISSING,
            root_cause="Package not found",
            affected_components=["pip"],
            suggested_fixes=["Check package name"],
            confidence=0.8
        )
        
        result = analysis.to_dict()
        
        assert isinstance(result, dict)
        assert result["error_type"] == "dependency_missing"
        assert result["root_cause"] == "Package not found"
        assert result["confidence"] == 0.8


class TestRecoveryStrategy:
    """Test cases for RecoveryStrategy data class."""
    
    def test_to_dict(self):
        """Test RecoveryStrategy to_dict conversion."""
        strategy = RecoveryStrategy(
            name="test_strategy",
            description="Test recovery strategy",
            commands=["pip install package"],
            success_probability=0.7,
            resource_cost=2
        )
        
        result = strategy.to_dict()
        
        assert isinstance(result, dict)
        assert result["name"] == "test_strategy"
        assert result["commands"] == ["pip install package"]
        assert result["success_probability"] == 0.7


class TestCommandResult:
    """Test cases for CommandResult data class."""
    
    def test_initialization_with_timestamp(self):
        """Test CommandResult initialization sets timestamp."""
        result = CommandResult(
            command="test command",
            exit_code=0,
            stdout="success",
            stderr="",
            execution_time=1.0,
            success=True
        )
        
        assert result.timestamp != ""
        assert isinstance(result.timestamp, str)
        
        # Test that timestamp is a valid ISO format
        datetime.fromisoformat(result.timestamp.replace('Z', '+00:00'))