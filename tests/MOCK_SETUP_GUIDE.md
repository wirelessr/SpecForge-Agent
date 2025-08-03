# Mock Setup Guide for Complex Test Scenarios

This guide provides detailed explanations and examples for setting up complex mock scenarios in the AutoGen Multi-Agent Framework tests.

## Table of Contents

1. [Complex Mock Patterns](#complex-mock-patterns)
2. [Multi-Component Mocking](#multi-component-mocking)
3. [Async Mock Configurations](#async-mock-configurations)
4. [State-Dependent Mocking](#state-dependent-mocking)
5. [Mock Verification Patterns](#mock-verification-patterns)
6. [Common Complex Scenarios](#common-complex-scenarios)

## Complex Mock Patterns

### Pattern 1: Sequential Mock Responses with side_effect

Use `side_effect` when a method needs to return different values on successive calls:

```python
@pytest.mark.asyncio
async def test_multi_phase_workflow(self, component):
    """Test workflow that requires different responses for each phase."""
    
    # Mock method that returns different results for each workflow phase
    component.coordinate_agents = AsyncMock()
    component.coordinate_agents.side_effect = [
        {"success": True, "phase": "requirements", "path": "/work/requirements.md"},
        {"success": True, "phase": "design", "path": "/work/design.md"},
        {"success": True, "phase": "tasks", "path": "/work/tasks.md"}
    ]
    
    # Execute workflow phases
    req_result = await component.coordinate_agents("requirements_generation", {})
    design_result = await component.coordinate_agents("design_generation", {})
    tasks_result = await component.coordinate_agents("task_execution", {})
    
    # Verify each phase received correct response
    assert req_result["phase"] == "requirements"
    assert design_result["phase"] == "design"
    assert tasks_result["phase"] == "tasks"
    
    # Verify method was called correct number of times
    assert component.coordinate_agents.call_count == 3
```

### Pattern 2: Parameter-Dependent Mock Responses

Use a custom function with `side_effect` when responses depend on input parameters:

```python
@pytest.mark.asyncio
async def test_parameter_dependent_responses(self, agent):
    """Test agent responses that vary based on input parameters."""
    
    def mock_coordinate_workflow(task_type, context):
        """
        Mock function that returns different responses based on task_type.
        
        This simulates the actual coordinate_agents behavior where different
        task types trigger different agent workflows and return structures.
        """
        if task_type == "requirements_generation":
            return {
                "success": True,
                "work_directory": context.get("workspace", "/default"),
                "requirements_path": f"{context.get('workspace', '/default')}/requirements.md"
            }
        elif task_type == "design_generation":
            return {
                "success": True,
                "design_path": f"{context.get('workspace', '/default')}/design.md"
            }
        elif task_type == "task_execution":
            return {
                "success": True,
                "tasks_file": f"{context.get('workspace', '/default')}/tasks.md"
            }
        else:
            return {"success": False, "error": f"Unknown task type: {task_type}"}
    
    # Apply the mock function
    agent.coordinate_agents = AsyncMock(side_effect=mock_coordinate_workflow)
    
    # Test different parameter combinations
    req_result = await agent.coordinate_agents("requirements_generation", {"workspace": "/test"})
    assert req_result["requirements_path"] == "/test/requirements.md"
    
    design_result = await agent.coordinate_agents("design_generation", {"workspace": "/test"})
    assert design_result["design_path"] == "/test/design.md"
    
    # Test error case
    error_result = await agent.coordinate_agents("invalid_type", {})
    assert error_result["success"] is False
    assert "Unknown task type" in error_result["error"]
```

### Pattern 3: Stateful Mock Objects

Create mocks that maintain state across calls:

```python
class StatefulMockAgent:
    """Mock agent that maintains state across method calls."""
    
    def __init__(self):
        self.workflow_state = "idle"
        self.generated_files = []
        self.call_history = []
    
    async def coordinate_agents(self, task_type, context):
        """Mock coordination that updates internal state."""
        self.call_history.append({"task_type": task_type, "context": context})
        
        if task_type == "requirements_generation":
            self.workflow_state = "requirements_pending"
            file_path = f"{context['workspace']}/requirements.md"
            self.generated_files.append(file_path)
            return {"success": True, "requirements_path": file_path}
            
        elif task_type == "design_generation":
            if self.workflow_state != "requirements_approved":
                return {"success": False, "error": "Requirements not approved"}
            
            self.workflow_state = "design_pending"
            file_path = f"{context['workspace']}/design.md"
            self.generated_files.append(file_path)
            return {"success": True, "design_path": file_path}
        
        return {"success": False, "error": "Invalid workflow state"}
    
    def approve_phase(self, phase):
        """Mock approval that updates workflow state."""
        if phase == "requirements" and self.workflow_state == "requirements_pending":
            self.workflow_state = "requirements_approved"
            return True
        elif phase == "design" and self.workflow_state == "design_pending":
            self.workflow_state = "design_approved"
            return True
        return False

@pytest.mark.asyncio
async def test_stateful_workflow(self, controller):
    """Test workflow with stateful mock that tracks progression."""
    
    # Create and apply stateful mock
    mock_agent = StatefulMockAgent()
    controller.agent_manager = mock_agent
    
    # Test requirements phase
    req_result = await mock_agent.coordinate_agents("requirements_generation", {"workspace": "/test"})
    assert req_result["success"] is True
    assert mock_agent.workflow_state == "requirements_pending"
    
    # Test design phase before approval (should fail)
    design_result = await mock_agent.coordinate_agents("design_generation", {"workspace": "/test"})
    assert design_result["success"] is False
    assert "Requirements not approved" in design_result["error"]
    
    # Approve requirements and retry design
    assert mock_agent.approve_phase("requirements") is True
    design_result = await mock_agent.coordinate_agents("design_generation", {"workspace": "/test"})
    assert design_result["success"] is True
    assert mock_agent.workflow_state == "design_pending"
    
    # Verify state tracking
    assert len(mock_agent.generated_files) == 2
    assert "/test/requirements.md" in mock_agent.generated_files
    assert "/test/design.md" in mock_agent.generated_files
```

## Multi-Component Mocking

### Pattern 1: Nested Context Manager Mocking

For tests that require multiple components to be mocked simultaneously:

```python
@pytest.mark.asyncio
async def test_full_framework_initialization(self, controller, llm_config):
    """
    Test complete framework initialization with all components mocked.
    
    This pattern is used when testing the MainController initialization
    process that creates and configures multiple dependent components.
    """
    
    # Nested context managers ensure all components are mocked consistently
    with patch('autogen_framework.main_controller.MemoryManager') as mock_memory_cls, \
         patch('autogen_framework.main_controller.AgentManager') as mock_agent_cls, \
         patch('autogen_framework.main_controller.ShellExecutor') as mock_shell_cls:
        
        # Configure MemoryManager mock
        # The memory manager handles context loading and persistence
        mock_memory_instance = Mock()
        mock_memory_instance.load_memory.return_value = {
            "global": {"patterns": "test patterns"},
            "projects": {"current": "test project"}
        }
        mock_memory_instance.get_memory_stats.return_value = {
            "total_files": 5,
            "total_size": 1024
        }
        mock_memory_cls.return_value = mock_memory_instance
        
        # Configure AgentManager mock
        # The agent manager coordinates between PlanAgent, DesignAgent, and ImplementAgent
        mock_agent_instance = Mock()
        mock_agent_instance.setup_agents.return_value = True  # Successful agent setup
        mock_agent_instance.get_agent_status.return_value = {
            "plan_agent": {"status": "ready", "initialized": True},
            "design_agent": {"status": "ready", "initialized": True},
            "implement_agent": {"status": "ready", "initialized": True}
        }
        mock_agent_instance.update_agent_memory = Mock()  # Memory synchronization
        mock_agent_cls.return_value = mock_agent_instance
        
        # Configure ShellExecutor mock
        # The shell executor handles command execution during implementation
        mock_shell_instance = Mock()
        mock_shell_instance.get_execution_stats.return_value = {
            "total_executions": 0,
            "success_rate": 1.0
        }
        mock_shell_cls.return_value = mock_shell_instance
        
        # Execute initialization
        result = controller.initialize_framework(llm_config)
        
        # Verify initialization success
        assert result is True
        assert controller.memory_manager is mock_memory_instance
        assert controller.agent_manager is mock_agent_instance
        assert controller.shell_executor is mock_shell_instance
        
        # Verify component setup was called
        mock_memory_instance.load_memory.assert_called_once()
        mock_agent_instance.setup_agents.assert_called_once_with(llm_config)
        mock_agent_instance.update_agent_memory.assert_called_once()
```

### Pattern 2: Component Interaction Mocking

For testing interactions between components:

```python
@pytest.mark.asyncio
async def test_component_interaction_workflow(self, controller):
    """
    Test workflow that involves interaction between multiple components.
    
    This pattern mocks the interaction flow between AgentManager,
    MemoryManager, and ShellExecutor during task execution.
    """
    
    # Mock the components with interaction tracking
    with patch.object(controller, 'agent_manager') as mock_agent, \
         patch.object(controller, 'memory_manager') as mock_memory, \
         patch.object(controller, 'shell_executor') as mock_shell:
        
        # Setup interaction chain
        # 1. Agent coordination triggers memory updates
        # 2. Memory updates trigger shell commands
        # 3. Shell results update agent state
        
        async def mock_coordinate_with_side_effects(task_type, context):
            """Mock coordination that triggers component interactions."""
            
            # Simulate agent requesting memory context
            memory_context = mock_memory.load_memory()
            
            # Simulate agent executing shell commands
            if task_type == "task_execution":
                shell_result = await mock_shell.execute_command("test command")
                
                # Update memory with execution results
                mock_memory.save_memory("execution_log", shell_result)
            
            return {"success": True, "task_type": task_type}
        
        # Configure mocks with realistic responses
        mock_agent.coordinate_agents = AsyncMock(side_effect=mock_coordinate_with_side_effects)
        mock_memory.load_memory.return_value = {"context": "test"}
        mock_memory.save_memory = Mock()
        mock_shell.execute_command = AsyncMock(return_value={"output": "success"})
        
        # Execute workflow
        result = await mock_agent.coordinate_agents("task_execution", {"workspace": "/test"})
        
        # Verify component interactions occurred
        assert result["success"] is True
        mock_memory.load_memory.assert_called_once()
        mock_shell.execute_command.assert_called_once_with("test command")
        mock_memory.save_memory.assert_called_once_with("execution_log", {"output": "success"})
```

## Async Mock Configurations

### Pattern 1: Complex Async Mock Chains

For testing async workflows with multiple dependent calls:

```python
@pytest.mark.asyncio
async def test_async_workflow_chain(self, agent):
    """
    Test complex async workflow with dependent operations.
    
    This pattern handles scenarios where async operations depend on
    the results of previous async operations.
    """
    
    # Mock async file operations
    agent._read_file_content = AsyncMock()
    agent._write_file_content = AsyncMock()
    agent.generate_response = AsyncMock()
    
    # Setup async operation chain
    # 1. Read requirements file
    # 2. Read design file  
    # 3. Generate response based on both files
    # 4. Write result to tasks file
    
    async def mock_file_reader(file_path):
        """Mock file reader that returns different content based on path."""
        if "requirements.md" in file_path:
            return "# Requirements\nUser authentication required"
        elif "design.md" in file_path:
            return "# Design\nREST API with JWT tokens"
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    async def mock_response_generator(prompt):
        """Mock LLM response generator."""
        if "requirements" in prompt and "design" in prompt:
            return "# Tasks\n- [ ] Implement JWT authentication\n- [ ] Create user endpoints"
        else:
            return "# Tasks\n- [ ] Generic task"
    
    # Apply async mocks
    agent._read_file_content.side_effect = mock_file_reader
    agent.generate_response.side_effect = mock_response_generator
    agent._write_file_content.return_value = True
    
    # Execute async workflow
    result = await agent.generate_task_list(
        "/work/design.md", 
        "/work/requirements.md", 
        "/work"
    )
    
    # Verify async call chain
    assert agent._read_file_content.call_count == 2
    assert agent.generate_response.call_count == 1
    assert agent._write_file_content.call_count == 1
    
    # Verify call order and parameters
    read_calls = agent._read_file_content.call_args_list
    assert "/work/design.md" in str(read_calls[0])
    assert "/work/requirements.md" in str(read_calls[1])
    
    # Verify generated response was written
    write_call = agent._write_file_content.call_args_list[0]
    assert "JWT authentication" in str(write_call)
```

### Pattern 2: Async Exception Handling

For testing error handling in async workflows:

```python
@pytest.mark.asyncio
async def test_async_error_handling_and_recovery(self, agent):
    """
    Test async error handling with retry mechanisms.
    
    This pattern tests how components handle async failures
    and implement recovery strategies.
    """
    
    # Mock async operations with failure scenarios
    agent.generate_response = AsyncMock()
    agent._write_file_content = AsyncMock()
    
    # Setup failure and recovery sequence
    # First call fails, second call succeeds
    agent.generate_response.side_effect = [
        Exception("LLM service temporarily unavailable"),  # First call fails
        "# Tasks\n- [ ] Recovered task generation"          # Second call succeeds
    ]
    
    # Mock retry mechanism
    async def mock_generate_with_retry(prompt, max_retries=2):
        """Mock method that implements retry logic."""
        for attempt in range(max_retries):
            try:
                return await agent.generate_response(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                # Wait before retry (mocked)
                await asyncio.sleep(0.1)
    
    # Test error recovery
    result = await mock_generate_with_retry("Generate tasks")
    
    # Verify retry occurred and eventually succeeded
    assert result == "# Tasks\n- [ ] Recovered task generation"
    assert agent.generate_response.call_count == 2
    
    # Test permanent failure
    agent.generate_response.side_effect = Exception("Permanent failure")
    
    with pytest.raises(Exception, match="Permanent failure"):
        await mock_generate_with_retry("Generate tasks", max_retries=1)
```

## State-Dependent Mocking

### Pattern 1: Workflow State Tracking

For testing components that maintain workflow state:

```python
class WorkflowStateMock:
    """Mock that tracks workflow state transitions."""
    
    def __init__(self):
        self.current_phase = "idle"
        self.approved_phases = set()
        self.generated_files = {}
        self.state_history = []
    
    def transition_to_phase(self, phase):
        """Simulate phase transition with validation."""
        self.state_history.append({
            "from": self.current_phase,
            "to": phase,
            "timestamp": "mock_time"
        })
        
        # Validate transition rules
        if phase == "design" and "requirements" not in self.approved_phases:
            raise ValueError("Cannot transition to design without approved requirements")
        
        self.current_phase = phase
    
    def approve_phase(self, phase):
        """Simulate phase approval."""
        if phase != self.current_phase:
            raise ValueError(f"Cannot approve {phase}, currently in {self.current_phase}")
        
        self.approved_phases.add(phase)
        return {"approved": True, "phase": phase}
    
    async def coordinate_agents(self, task_type, context):
        """Mock coordination that respects workflow state."""
        if task_type == "requirements_generation":
            self.transition_to_phase("requirements")
            file_path = f"{context['workspace']}/requirements.md"
            self.generated_files["requirements"] = file_path
            return {"success": True, "requirements_path": file_path}
        
        elif task_type == "design_generation":
            self.transition_to_phase("design")
            file_path = f"{context['workspace']}/design.md"
            self.generated_files["design"] = file_path
            return {"success": True, "design_path": file_path}
        
        return {"success": False, "error": f"Unknown task type: {task_type}"}

@pytest.mark.asyncio
async def test_workflow_state_management(self, controller):
    """Test workflow state transitions and validation."""
    
    # Apply stateful mock
    state_mock = WorkflowStateMock()
    controller.agent_manager = state_mock
    
    # Test valid workflow progression
    req_result = await state_mock.coordinate_agents("requirements_generation", {"workspace": "/test"})
    assert req_result["success"] is True
    assert state_mock.current_phase == "requirements"
    
    # Approve requirements
    approval = state_mock.approve_phase("requirements")
    assert approval["approved"] is True
    assert "requirements" in state_mock.approved_phases
    
    # Test design phase after approval
    design_result = await state_mock.coordinate_agents("design_generation", {"workspace": "/test"})
    assert design_result["success"] is True
    assert state_mock.current_phase == "design"
    
    # Verify state history
    assert len(state_mock.state_history) == 2
    assert state_mock.state_history[0]["from"] == "idle"
    assert state_mock.state_history[0]["to"] == "requirements"
    assert state_mock.state_history[1]["from"] == "requirements"
    assert state_mock.state_history[1]["to"] == "design"
```

## Mock Verification Patterns

### Pattern 1: Complex Call Verification

For verifying complex mock interactions:

```python
@pytest.mark.asyncio
async def test_complex_mock_verification(self, controller):
    """Test comprehensive mock call verification."""
    
    with patch.object(controller, 'agent_manager') as mock_agent:
        # Setup mock with call tracking
        mock_agent.coordinate_agents = AsyncMock()
        mock_agent.update_agent_memory = Mock()
        
        # Execute multiple operations
        await mock_agent.coordinate_agents("requirements_generation", {"workspace": "/test1"})
        await mock_agent.coordinate_agents("design_generation", {"workspace": "/test1"})
        mock_agent.update_agent_memory({"phase": "requirements"})
        mock_agent.update_agent_memory({"phase": "design"})
        
        # Verify call count and order
        assert mock_agent.coordinate_agents.call_count == 2
        assert mock_agent.update_agent_memory.call_count == 2
        
        # Verify specific calls with exact parameters
        mock_agent.coordinate_agents.assert_has_calls([
            call("requirements_generation", {"workspace": "/test1"}),
            call("design_generation", {"workspace": "/test1"})
        ])
        
        mock_agent.update_agent_memory.assert_has_calls([
            call({"phase": "requirements"}),
            call({"phase": "design"})
        ])
        
        # Verify call arguments using call_args_list
        coord_calls = mock_agent.coordinate_agents.call_args_list
        assert coord_calls[0][0][0] == "requirements_generation"  # First positional arg of first call
        assert coord_calls[1][0][0] == "design_generation"       # First positional arg of second call
        
        # Verify keyword arguments
        assert coord_calls[0][0][1]["workspace"] == "/test1"
        assert coord_calls[1][0][1]["workspace"] == "/test1"
```

### Pattern 2: Mock State Verification

For verifying mock internal state changes:

```python
@pytest.mark.asyncio
async def test_mock_state_verification(self, component):
    """Test verification of mock internal state changes."""
    
    # Create mock with internal state
    mock_memory = Mock()
    mock_memory.memory_cache = {}
    mock_memory.access_count = 0
    
    def mock_save_memory(key, content, category="global"):
        """Mock save that updates internal state."""
        mock_memory.access_count += 1
        if category not in mock_memory.memory_cache:
            mock_memory.memory_cache[category] = {}
        mock_memory.memory_cache[category][key] = content
        return True
    
    def mock_load_memory():
        """Mock load that updates access count."""
        mock_memory.access_count += 1
        return mock_memory.memory_cache
    
    # Apply stateful mocks
    mock_memory.save_memory.side_effect = mock_save_memory
    mock_memory.load_memory.side_effect = mock_load_memory
    component.memory_manager = mock_memory
    
    # Execute operations that should change state
    mock_memory.save_memory("test_key", "test_content", "global")
    mock_memory.save_memory("another_key", "more_content", "projects")
    loaded_memory = mock_memory.load_memory()
    
    # Verify internal state changes
    assert mock_memory.access_count == 3  # 2 saves + 1 load
    assert "global" in mock_memory.memory_cache
    assert "projects" in mock_memory.memory_cache
    assert mock_memory.memory_cache["global"]["test_key"] == "test_content"
    assert mock_memory.memory_cache["projects"]["another_key"] == "more_content"
    
    # Verify returned data matches internal state
    assert loaded_memory == mock_memory.memory_cache
```

## Common Complex Scenarios

### Scenario 1: Full E2E Workflow Simulation

```python
@pytest.mark.asyncio
async def test_full_e2e_workflow_simulation(self, controller, temp_workspace):
    """
    Simulate complete end-to-end workflow with realistic mock responses.
    
    This scenario tests the entire workflow from initial request through
    final implementation, with realistic mock responses at each stage.
    """
    
    # Create comprehensive mock setup
    with patch.object(controller, 'agent_manager') as mock_agent, \
         patch.object(controller, 'memory_manager') as mock_memory, \
         patch.object(controller, 'shell_executor') as mock_shell:
        
        # Setup realistic workflow responses
        workflow_responses = {
            "requirements_generation": {
                "success": True,
                "work_directory": f"{temp_workspace}/user-auth-api",
                "requirements_path": f"{temp_workspace}/user-auth-api/requirements.md",
                "content": "# Requirements\n\n## User Authentication API\n\n- JWT-based authentication\n- User registration and login\n- Password reset functionality"
            },
            "design_generation": {
                "success": True,
                "design_path": f"{temp_workspace}/user-auth-api/design.md",
                "content": "# Technical Design\n\n## Architecture\n\n- REST API with Express.js\n- MongoDB for user storage\n- JWT for session management"
            },
            "task_execution": {
                "success": True,
                "tasks_file": f"{temp_workspace}/user-auth-api/tasks.md",
                "content": "# Implementation Tasks\n\n- [ ] Setup Express.js server\n- [ ] Implement user model\n- [ ] Create authentication middleware"
            }
        }
        
        async def mock_coordinate_realistic(task_type, context):
            """Mock coordination with realistic workflow simulation."""
            if task_type in workflow_responses:
                response = workflow_responses[task_type].copy()
                
                # Simulate file creation
                if "content" in response:
                    mock_memory.save_memory(
                        f"{task_type}_content",
                        response["content"],
                        "workflow"
                    )
                
                return response
            
            return {"success": False, "error": f"Unknown task type: {task_type}"}
        
        # Configure mocks
        mock_agent.coordinate_agents = AsyncMock(side_effect=mock_coordinate_realistic)
        mock_agent.setup_agents.return_value = True
        mock_memory.load_memory.return_value = {"global": {"context": "test"}}
        mock_memory.save_memory = Mock(return_value=True)
        mock_shell.get_execution_stats.return_value = {"ready": True}
        
        # Initialize framework
        assert controller.initialize_framework(Mock()) is True
        
        # Execute full workflow
        # Phase 1: Requirements
        req_result = await controller.process_user_request("Create user authentication API")
        assert req_result["requires_user_approval"] is True
        assert req_result["approval_needed_for"] == "requirements"
        
        # Approve requirements
        controller.approve_phase("requirements", True)
        
        # Phase 2: Design
        design_result = await controller.continue_workflow()
        assert design_result["phase"] == "design"
        assert design_result["requires_approval"] is True
        
        # Approve design
        controller.approve_phase("design", True)
        
        # Phase 3: Tasks
        tasks_result = await controller.continue_workflow()
        assert tasks_result["phase"] == "tasks"
        
        # Verify complete workflow execution
        assert mock_agent.coordinate_agents.call_count == 3
        assert mock_memory.save_memory.call_count >= 3  # At least one save per phase
        
        # Verify workflow progression
        coord_calls = mock_agent.coordinate_agents.call_args_list
        assert coord_calls[0][0][0] == "requirements_generation"
        assert coord_calls[1][0][0] == "design_generation"
        assert coord_calls[2][0][0] == "task_execution"
```

This guide provides comprehensive patterns for setting up complex mock scenarios. Use these patterns as templates for creating robust, maintainable tests that accurately simulate real-world usage of the AutoGen Multi-Agent Framework.