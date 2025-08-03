"""
Unit tests for core data models.

Tests all data classes and their methods to ensure proper functionality
and data integrity.
"""

import pytest
from datetime import datetime
from autogen_framework.models import (
    WorkflowPhase,
    WorkflowState,
    TaskDefinition,
    ExecutionResult,
    LLMConfig
)


class TestWorkflowPhase:
    """Test cases for WorkflowPhase enumeration."""
    
    def test_workflow_phase_values(self):
        """Test that all workflow phases have correct string values."""
        assert WorkflowPhase.PLANNING.value == "planning"
        assert WorkflowPhase.DESIGN.value == "design"
        assert WorkflowPhase.TASK_GENERATION.value == "task_generation"
        assert WorkflowPhase.IMPLEMENTATION.value == "implementation"
        assert WorkflowPhase.COMPLETED.value == "completed"


class TestWorkflowState:
    """Test cases for WorkflowState data class."""
    
    def test_workflow_state_initialization(self):
        """Test WorkflowState initialization with required parameters."""
        state = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/dir"
        )
        
        assert state.phase == WorkflowPhase.PLANNING
        assert state.work_directory == "/test/dir"
        assert state.requirements_approved is False
        assert state.design_approved is False
        assert state.tasks_approved is False
        assert state.current_task_index == 0
        assert state.execution_log == []
    
    def test_advance_phase(self):
        """Test phase advancement through workflow."""
        state = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/dir"
        )
        
        # Test advancing through all phases
        state.advance_phase()
        assert state.phase == WorkflowPhase.DESIGN
        
        state.advance_phase()
        assert state.phase == WorkflowPhase.TASK_GENERATION
        
        state.advance_phase()
        assert state.phase == WorkflowPhase.IMPLEMENTATION
        
        state.advance_phase()
        assert state.phase == WorkflowPhase.COMPLETED
        
        # Test that advancing from completed doesn't change phase
        state.advance_phase()
        assert state.phase == WorkflowPhase.COMPLETED
    
    def test_is_ready_for_next_phase(self):
        """Test readiness check for phase advancement."""
        state = WorkflowState(
            phase=WorkflowPhase.PLANNING,
            work_directory="/test/dir"
        )
        
        # Planning phase - needs requirements approval
        assert not state.is_ready_for_next_phase()
        state.requirements_approved = True
        assert state.is_ready_for_next_phase()
        
        # Design phase - needs design approval
        state.phase = WorkflowPhase.DESIGN
        state.design_approved = False
        assert not state.is_ready_for_next_phase()
        state.design_approved = True
        assert state.is_ready_for_next_phase()
        
        # Task generation phase - needs tasks approval
        state.phase = WorkflowPhase.TASK_GENERATION
        state.tasks_approved = False
        assert not state.is_ready_for_next_phase()
        state.tasks_approved = True
        assert state.is_ready_for_next_phase()
        
        # Implementation and completed phases
        state.phase = WorkflowPhase.IMPLEMENTATION
        assert not state.is_ready_for_next_phase()
        
        state.phase = WorkflowPhase.COMPLETED
        assert not state.is_ready_for_next_phase()


class TestTaskDefinition:
    """Test cases for TaskDefinition data class."""
    
    def test_task_definition_initialization(self):
        """Test TaskDefinition initialization."""
        task = TaskDefinition(
            id="task-1",
            title="Test Task",
            description="A test task",
            steps=["step 1", "step 2"],
            requirements_ref=["req-1", "req-2"]
        )
        
        assert task.id == "task-1"
        assert task.title == "Test Task"
        assert task.description == "A test task"
        assert task.steps == ["step 1", "step 2"]
        assert task.requirements_ref == ["req-1", "req-2"]
        assert task.dependencies == []
        assert task.completed is False
        assert task.execution_record is None
        assert task.attempted_approaches == []
        assert task.retry_count == 0
        assert task.max_retries == 3
    
    def test_can_retry(self):
        """Test retry capability check."""
        task = TaskDefinition(
            id="task-1",
            title="Test Task",
            description="A test task",
            steps=["step 1"],
            requirements_ref=["req-1"]
        )
        
        # Should be able to retry initially
        assert task.can_retry()
        
        # Test retry limit
        task.retry_count = 2
        assert task.can_retry()
        
        task.retry_count = 3
        assert not task.can_retry()
        
        task.retry_count = 4
        assert not task.can_retry()
    
    def test_increment_retry(self):
        """Test retry counter increment."""
        task = TaskDefinition(
            id="task-1",
            title="Test Task",
            description="A test task",
            steps=["step 1"],
            requirements_ref=["req-1"]
        )
        
        assert task.retry_count == 0
        task.increment_retry()
        assert task.retry_count == 1
        task.increment_retry()
        assert task.retry_count == 2
    
    def test_add_attempt(self):
        """Test adding execution attempts."""
        task = TaskDefinition(
            id="task-1",
            title="Test Task",
            description="A test task",
            steps=["step 1"],
            requirements_ref=["req-1"]
        )
        
        result = {"success": False, "error": "test error"}
        task.add_attempt("approach-1", result)
        
        assert len(task.attempted_approaches) == 1
        attempt = task.attempted_approaches[0]
        assert attempt["approach"] == "approach-1"
        assert attempt["result"] == result
        assert attempt["retry_number"] == 0
        assert "timestamp" in attempt
    
    def test_mark_completed(self):
        """Test marking task as completed."""
        task = TaskDefinition(
            id="task-1",
            title="Test Task",
            description="A test task",
            steps=["step 1"],
            requirements_ref=["req-1"]
        )
        
        execution_record = {"files_created": ["test.py"], "commands_run": ["python test.py"]}
        task.mark_completed(execution_record)
        
        assert task.completed is True
        assert task.execution_record == execution_record


class TestExecutionResult:
    """Test cases for ExecutionResult data class."""
    
    def test_execution_result_initialization(self):
        """Test ExecutionResult initialization."""
        result = ExecutionResult(
            command="ls -la",
            return_code=0,
            stdout="file1.txt\nfile2.txt",
            stderr="",
            execution_time=0.5,
            working_directory="/test/dir",
            timestamp="2024-01-01T12:00:00",
            success=True,
            approach_used="direct"
        )
        
        assert result.command == "ls -la"
        assert result.return_code == 0
        assert result.stdout == "file1.txt\nfile2.txt"
        assert result.stderr == ""
        assert result.execution_time == 0.5
        assert result.working_directory == "/test/dir"
        assert result.timestamp == "2024-01-01T12:00:00"
        assert result.success is True
        assert result.approach_used == "direct"
        assert result.alternative_attempts == []
    
    def test_create_success(self):
        """Test creating successful execution result."""
        result = ExecutionResult.create_success(
            command="echo hello",
            stdout="hello",
            execution_time=0.1,
            working_directory="/test",
            approach_used="direct"
        )
        
        assert result.command == "echo hello"
        assert result.return_code == 0
        assert result.stdout == "hello"
        assert result.stderr == ""
        assert result.execution_time == 0.1
        assert result.working_directory == "/test"
        assert result.success is True
        assert result.approach_used == "direct"
        assert isinstance(result.timestamp, str)
    
    def test_create_failure(self):
        """Test creating failed execution result."""
        result = ExecutionResult.create_failure(
            command="invalid-command",
            return_code=127,
            stderr="command not found",
            execution_time=0.05,
            working_directory="/test",
            approach_used="direct"
        )
        
        assert result.command == "invalid-command"
        assert result.return_code == 127
        assert result.stdout == ""
        assert result.stderr == "command not found"
        assert result.execution_time == 0.05
        assert result.working_directory == "/test"
        assert result.success is False
        assert result.approach_used == "direct"
        assert isinstance(result.timestamp, str)
    
    def test_add_alternative_attempt(self):
        """Test adding alternative attempt information."""
        result = ExecutionResult.create_success(
            command="test",
            stdout="output",
            execution_time=0.1,
            working_directory="/test",
            approach_used="direct"
        )
        
        alternative = {"approach": "patch", "result": "failed", "reason": "file not found"}
        result.add_alternative_attempt(alternative)
        
        assert len(result.alternative_attempts) == 1
        assert result.alternative_attempts[0] == alternative


class TestLLMConfig:
    """Test cases for LLMConfig data class."""
    
    def test_test_llm_config_explicit_values(self):
        """Test LLMConfig with explicit values (no defaults)."""
        test_base_url = "http://test.example.com:8888/openai/v1"
        test_model = "models/test-model"
        test_api_key = "test-api-key-123"
        
        config = LLMConfig(
            base_url=test_base_url,
            model=test_model,
            api_key=test_api_key
        )
        
        # Verify configuration structure and patterns
        assert config.base_url == test_base_url
        assert config.base_url.startswith("http://")
        assert "/openai/v1" in config.base_url
        
        assert config.model == test_model
        assert config.model.startswith("models/")
        
        assert config.api_key == test_api_key
        assert len(config.api_key) > 10  # API keys should be reasonably long
        assert config.api_key.startswith("test-")  # Test API key pattern
        
        # Verify default values
        assert config.temperature == 0.7  # Optional parameter with default
        assert config.max_output_tokens == 8192  # Optional parameter with default
        assert config.timeout == 60  # Optional parameter with default
    
    def test_test_llm_config_custom_values(self):
        """Test LLMConfig with custom values."""
        config = LLMConfig(
            base_url="http://custom.api.com",
            model="custom-model",
            api_key="custom-key",
            temperature=0.5,
            max_output_tokens=2000,
            timeout=30
        )
        
        assert config.base_url == "http://custom.api.com"
        assert config.model == "custom-model"
        assert config.api_key == "custom-key"
        assert config.temperature == 0.5
        assert config.max_output_tokens == 2000
        assert config.timeout == 30
    
    def test_to_autogen_config(self):
        """Test conversion to AutoGen configuration format."""
        config = LLMConfig(
            base_url="http://test.com",
            model="test-model",
            api_key="test-key",
            temperature=0.8,
            max_output_tokens=3000,
            timeout=45
        )
        
        autogen_config = config.to_autogen_config()
        
        expected = {
            "config_list": [{
                "model": "test-model",
                "base_url": "http://test.com",
                "api_key": "test-key",
            }],
            "temperature": 0.8,
            "max_output_tokens": 3000,
            "timeout": 45,
        }
        
        assert autogen_config == expected
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = LLMConfig(
            base_url="http://test.com",
            model="test-model",
            api_key="test-key"
        )
        assert config.validate() is True
        
        config = LLMConfig(
            base_url="http://test.com",
            model="test-model",
            api_key="test-key",
            temperature=1.0,
            max_output_tokens=1000,
            timeout=30
        )
        assert config.validate() is True
    
    def test_validate_invalid_config(self):
        """Test validation of invalid configurations."""
        # Empty base_url
        config = LLMConfig(base_url="", model="test-model", api_key="test-key")
        assert config.validate() is False
        
        # Empty model
        config = LLMConfig(base_url="http://test.com", model="", api_key="test-key")
        assert config.validate() is False
        
        # Empty api_key
        config = LLMConfig(base_url="http://test.com", model="test-model", api_key="")
        assert config.validate() is False
        
        # Invalid base_url format
        config = LLMConfig(base_url="invalid-url", model="test-model", api_key="test-key")
        assert config.validate() is False
        
        # Invalid temperature
        config = LLMConfig(base_url="http://test.com", model="test-model", api_key="test-key", temperature=-0.1)
        assert config.validate() is False
        
        config = LLMConfig(base_url="http://test.com", model="test-model", api_key="test-key", temperature=2.1)
        assert config.validate() is False
        
        # Invalid max_output_tokens
        config = LLMConfig(base_url="http://test.com", model="test-model", api_key="test-key", max_output_tokens=0)
        assert config.validate() is False
        
        config = LLMConfig(base_url="http://test.com", model="test-model", api_key="test-key", max_output_tokens=-100)
        assert config.validate() is False
        
        # Invalid timeout
        config = LLMConfig(base_url="http://test.com", model="test-model", api_key="test-key", timeout=0)
        assert config.validate() is False
        
        config = LLMConfig(base_url="http://test.com", model="test-model", api_key="test-key", timeout=-10)
        assert config.validate() is False
    
    def test_from_config_manager(self):
        """Test creating LLMConfig from ConfigManager."""
        from autogen_framework.config_manager import ConfigManager
        import os
        
        # Set up test environment variables
        test_env = {
            'LLM_BASE_URL': 'http://test.local:8888/openai/v1',
            'LLM_MODEL': 'test-model',
            'LLM_API_KEY': 'test-api-key',
            'LLM_TEMPERATURE': '0.8',
            'LLM_MAX_OUTPUT_TOKENS': '4096',
            'LLM_TIMEOUT_SECONDS': '45'
        }
        
        # Temporarily set environment variables
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            config_manager = ConfigManager(load_env=False)  # Don't load .env file
            config = LLMConfig.from_config_manager(config_manager)
            
            assert config.base_url == 'http://test.local:8888/openai/v1'
            assert config.model == 'test-model'
            assert config.api_key == 'test-api-key'
            assert config.temperature == 0.8
            assert config.max_output_tokens == 4096
            assert config.timeout == 45
            assert config.validate() is True
            
        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value


if __name__ == "__main__":
    pytest.main([__file__])