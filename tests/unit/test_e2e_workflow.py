"""
End-to-End Workflow Test for AutoGen Multi-Agent Framework

This test simulates the complete user experience through the command-line interface,
including all phases of the workflow with user approval and revision cycles.

The test covers:
1. Requirements phase: submission, review, revision, approval
2. Design phase: review, revision, approval  
3. Tasks phase: review, revision, approval
4. Final verification that all tasks are properly delivered

This test uses the actual CLI interface to ensure the user experience works correctly.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import subprocess
import time
import threading
import queue
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, Mock

from autogen_framework.main import main
from autogen_framework.main_controller import MainController
from autogen_framework.models import LLMConfig

class TestE2EWorkflow:
    """End-to-end workflow test cases."""
    # Using shared fixtures from conftest.py
    @pytest.fixture
    def mock_test_llm_config(self):
        """Create a mock LLM configuration for testing."""
        return LLMConfig(
            base_url="http://test.local:8888/openai/v1",
            model="test-model",
            api_key="sk-test123"
        )
    
    @pytest.mark.asyncio
    async def test_complete_workflow_with_revisions(self, temp_workspace, mock_test_llm_config):
        """
        Test the complete workflow including all revision cycles.
        
        This test simulates a real user going through:
        1. Initial request submission
        2. Requirements review and revision
        3. Design review and revision
        4. Tasks review and revision
        5. Final verification
        """
        # Initialize the controller directly for more control
        controller = MainController(temp_workspace)
        
        # Mock the agent coordination to simulate realistic responses
        # This complex mock setup simulates the entire framework stack for E2E testing
        with patch.object(controller, 'agent_manager') as mock_agent_manager:
            # Setup mock agent manager with realistic coordination behavior
            # The coordinate_agents method is the core of the workflow - it handles
            # requirements generation, design creation, and task execution
            mock_agent_manager.coordinate_agents = self._create_mock_coordinate_agents()
            mock_agent_manager.setup_agents.return_value = True  # Successful agent initialization
            mock_agent_manager.get_agent_status.return_value = {"status": "ready"}  # Agents ready
            mock_agent_manager.update_agent_memory = Mock()  # Memory update operations
            
            # Initialize framework - this will create the container with managers
            assert controller.initialize_framework(mock_test_llm_config) == True
            
            # Test Phase 1: Initial Request and Requirements Review
            await self._test_requirements_phase_with_revision(controller, temp_workspace)
            
            # Test Phase 2: Design Review with Revision
            await self._test_design_phase_with_revision(controller, temp_workspace)
            
            # Test Phase 3: Tasks Review with Revision
            await self._test_tasks_phase_with_revision(controller, temp_workspace)
            
            # Test Phase 4: Final Verification
            await self._test_final_verification(controller, temp_workspace)
    
    async def _test_requirements_phase_with_revision(self, controller: MainController, workspace: str):
        """Test requirements phase with revision cycle."""
        print("\n=== Testing Requirements Phase with Revision ===")
        
        # Step 1: Submit initial request
        user_request = "Create a REST API for user management with authentication"
        print(f"1. Submitting request: {user_request}")
        
        result = await controller.process_user_request(user_request)
        
        # Verify initial requirements generation
        assert result.get("success") == False  # Should require approval
        assert result.get("requires_user_approval") == True
        assert result.get("approval_needed_for") == "requirements"
        assert "requirements_path" in result
        
        print("✅ Requirements generated and waiting for approval")
        
        # Step 2: Check pending approval
        pending = controller.get_pending_approval()
        assert pending is not None
        assert pending["phase"] == "requirements"
        assert pending["phase_name"] == "Requirements"
        
        print("✅ Pending approval detected correctly")
        
        # Step 3: Simulate user requesting revision
        revision_feedback = "Please add more details about security requirements and API rate limiting"
        print(f"2. Requesting revision: {revision_feedback}")
        
        revision_result = await controller.apply_phase_revision("requirements", revision_feedback)
        
        # Verify revision was applied
        assert revision_result.get("success") == True
        assert "Requirements phase revised successfully" in revision_result.get("message", "")
        
        print("✅ Requirements revision applied successfully")
        
        # Step 4: Approve the revised requirements
        print("3. Approving revised requirements")
        approval_result = controller.approve_phase("requirements", True)
        
        assert approval_result.get("approved") == True
        assert approval_result.get("can_proceed") == True
        
        print("✅ Requirements approved successfully")
        
        # Verify framework status
        status = controller.get_framework_status()
        assert status["approval_status"]["requirements"] == "approved"
        
        print("✅ Requirements phase completed with revision cycle")
    
    async def _test_design_phase_with_revision(self, controller: MainController, workspace: str):
        """Test design phase with revision cycle."""
        print("\n=== Testing Design Phase with Revision ===")
        
        # Step 1: Continue to design phase
        print("1. Continuing to design phase")
        continue_result = await controller.continue_workflow()
        
        assert continue_result.get("phase") == "design"
        assert continue_result.get("requires_approval") == True
        
        print("✅ Design phase initiated")
        
        # Verify that the design result was stored
        design_result = continue_result.get("result", {})
        assert design_result.get("success") == True
        print("✅ Design result stored successfully")
        
        # Step 2: Check pending approval for design
        pending = controller.get_pending_approval()
        assert pending is not None
        assert pending["phase"] == "design"
        
        print("✅ Design approval pending detected")
        
        # Step 3: Request design revision
        design_revision = "Please add more details about database schema and include error handling patterns"
        print(f"2. Requesting design revision: {design_revision}")
        
        revision_result = await controller.apply_phase_revision("design", design_revision)
        
        assert revision_result.get("success") == True
        print("✅ Design revision applied successfully")
        
        # Step 4: Approve the revised design
        print("3. Approving revised design")
        approval_result = controller.approve_phase("design", True)
        
        assert approval_result.get("approved") == True
        print("✅ Design approved successfully")
        
        # Verify status
        status = controller.get_framework_status()
        assert status["approval_status"]["design"] == "approved"
        
        print("✅ Design phase completed with revision cycle")
    
    async def _test_tasks_phase_with_revision(self, controller: MainController, workspace: str):
        """Test tasks phase with revision cycle."""
        print("\n=== Testing Tasks Phase with Revision ===")
        
        # Step 1: Continue to tasks phase
        print("1. Continuing to tasks phase")
        continue_result = await controller.continue_workflow()
        
        assert continue_result.get("phase") == "tasks"
        assert continue_result.get("requires_approval") == True
        
        print("✅ Tasks phase initiated")
        
        # Step 2: Check pending approval for tasks
        pending = controller.get_pending_approval()
        assert pending is not None
        assert pending["phase"] == "tasks"
        
        print("✅ Tasks approval pending detected")
        
        # Step 3: Request tasks revision
        tasks_revision = "Please break down the authentication task into smaller subtasks and add unit testing tasks"
        print(f"2. Requesting tasks revision: {tasks_revision}")
        
        revision_result = await controller.apply_phase_revision("tasks", tasks_revision)
        
        assert revision_result.get("success") == True
        print("✅ Tasks revision applied successfully")
        
        # Step 4: Approve the revised tasks
        print("3. Approving revised tasks")
        approval_result = controller.approve_phase("tasks", True)
        
        assert approval_result.get("approved") == True
        print("✅ Tasks approved successfully")
        
        # Verify status
        status = controller.get_framework_status()
        assert status["approval_status"]["tasks"] == "approved"
        
        print("✅ Tasks phase completed with revision cycle")
    
    async def _test_final_verification(self, controller: MainController, workspace: str):
        """Test final verification that all tasks are properly delivered."""
        print("\n=== Testing Final Verification ===")
        
        # Step 1: Complete the workflow
        print("1. Completing final workflow phase")
        continue_result = await controller.continue_workflow()
        
        assert continue_result.get("workflow_completed") == True
        print("✅ Workflow marked as completed")
        
        # Step 2: Verify all files were created
        workspace_path = Path(workspace)
        
        # Check that work directory was created
        work_dirs = list(workspace_path.glob("*"))
        work_dirs = [d for d in work_dirs if d.is_dir() and d.name != "memory" and d.name != "logs"]
        assert len(work_dirs) > 0, "No work directory created"
        
        work_dir = work_dirs[0]  # Get the first work directory
        print(f"✅ Work directory created: {work_dir.name}")
        
        # Check that all required files exist
        required_files = ["requirements.md", "design.md", "tasks.md"]
        for filename in required_files:
            file_path = work_dir / filename
            assert file_path.exists(), f"Required file {filename} not found"
            
            # Verify file has content
            content = file_path.read_text()
            assert len(content.strip()) > 0, f"File {filename} is empty"
            
            print(f"✅ {filename} exists and has content")
        
        # Step 3: Verify task structure in tasks.md
        tasks_content = (work_dir / "tasks.md").read_text()
        
        # Check for proper task formatting
        assert "# Implementation Plan" in tasks_content
        assert "- [ ]" in tasks_content  # Should have checkbox tasks
        assert "_Requirements:" in tasks_content  # Should reference requirements
        
        print("✅ Tasks file has proper structure")
        
        # Step 4: Verify framework status after completion
        final_status = controller.get_framework_status()
        
        assert final_status["current_workflow"]["active"] == False
        assert len(final_status["approval_status"]) == 0  # Should be cleared after completion
        
        print("✅ Framework status correctly shows completed workflow")
        
        # Step 5: Verify execution log
        execution_log = controller.get_execution_log()
        
        # Should have events for each phase
        event_types = [event["event_type"] for event in execution_log]
        expected_events = ["framework_initialization", "phase_completed", "phase_approval", "phase_revision", "workflow_completed"]
        
        for expected_event in expected_events:
            assert any(expected_event in event_type for event_type in event_types), f"Missing event type: {expected_event}"
        
        print("✅ Execution log contains all expected events")
        
        print("\n🎉 Complete E2E workflow test passed successfully!")
        print("✅ All phases completed with revision cycles")
        print("✅ All files properly generated")
        print("✅ Framework state properly managed")
        print("✅ User approval and revision workflows working correctly")
    
    def _create_mock_coordinate_agents(self):
        """Create a mock coordinate_agents function that simulates realistic agent responses."""
        
        async def mock_coordinate_agents(task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
            """Mock agent coordination with realistic responses."""
            
            if task_type == "requirements_generation":
                # Simulate requirements generation
                work_dir = Path(context["workspace_path"]) / "test-api-project"
                work_dir.mkdir(exist_ok=True)
                
                requirements_path = work_dir / "requirements.md"
                requirements_content = """# Requirements Document

## Introduction

This document outlines the requirements for a REST API for user management with authentication.

## Requirements

### Requirement 1

**User Story:** As a system administrator, I want to manage user accounts, so that I can control access to the system.

#### Acceptance Criteria

1. WHEN a new user is created THEN the system SHALL store user credentials securely
2. WHEN a user logs in THEN the system SHALL authenticate using secure methods
3. WHEN user data is accessed THEN the system SHALL require proper authorization

### Requirement 2

**User Story:** As a developer, I want API endpoints for user operations, so that I can integrate with the system.

#### Acceptance Criteria

1. WHEN API endpoints are called THEN the system SHALL return proper HTTP status codes
2. WHEN invalid data is submitted THEN the system SHALL return validation errors
3. WHEN rate limits are exceeded THEN the system SHALL return 429 status code
"""
                
                requirements_path.write_text(requirements_content)
                
                return {
                    "success": True,
                    "requirements_path": str(requirements_path),
                    "work_directory": str(work_dir)
                }
            
            elif task_type == "design_generation":
                # Simulate design generation
                work_directory = context.get("work_directory")
                if not work_directory:
                    # If work_directory is None, try to find it from requirements_path
                    requirements_path = context.get("requirements_path")
                    if requirements_path:
                        work_directory = str(Path(requirements_path).parent)
                    else:
                        # Fallback: create a default work directory
                        work_directory = str(Path(context.get("workspace_path", "/tmp")) / "test-api-project")
                        Path(work_directory).mkdir(exist_ok=True)
                
                work_dir = Path(work_directory)
                design_path = work_dir / "design.md"
                
                design_content = """# Design Document

## Overview

REST API design for user management with JWT authentication.

## Architecture

```mermaid
graph TD
    A[Client] --> B[API Gateway]
    B --> C[Auth Service]
    B --> D[User Service]
    C --> E[Database]
    D --> E
```

## Components and Interfaces

### Authentication Service
- JWT token generation and validation
- Password hashing with bcrypt
- Rate limiting implementation

### User Service
- CRUD operations for users
- Input validation
- Error handling

## Data Models

### User Model
```json
{
  "id": "string",
  "username": "string", 
  "email": "string",
  "password_hash": "string",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

## Error Handling

- 400: Bad Request for validation errors
- 401: Unauthorized for authentication failures
- 403: Forbidden for authorization failures
- 429: Too Many Requests for rate limiting
- 500: Internal Server Error for system errors

## Testing Strategy

- Unit tests for all service methods
- Integration tests for API endpoints
- Load testing for performance validation
"""
                
                design_path.write_text(design_content)
                
                return {
                    "success": True,
                    "design_path": str(design_path)
                }
            
            elif task_type == "task_generation":
                # Simulate task generation using TasksAgent
                work_directory = context.get("work_directory")
                if not work_directory:
                    # Try to get work directory from design_path
                    design_path = context.get("design_path")
                    if design_path:
                        work_directory = str(Path(design_path).parent)
                    else:
                        # Fallback
                        work_directory = str(Path("/tmp") / "test-api-project")
                        Path(work_directory).mkdir(exist_ok=True)
                
                work_dir = Path(work_directory)
                tasks_path = work_dir / "tasks.md"
                
                tasks_content = """# Implementation Plan

- [ ] 1. Set up project structure and dependencies
  - Create directory structure for API project
  - Initialize package.json or requirements.txt
  - Install necessary dependencies (Express.js/Flask, JWT library, bcrypt)
  - _Requirements: 1.1, 2.1_

- [ ] 2. Implement user data model and validation
  - [ ] 2.1 Create User model with validation
    - Define User schema with required fields
    - Implement password hashing functionality
    - Add email validation and uniqueness constraints
    - _Requirements: 1.1, 1.2_
  
  - [ ] 2.2 Create database connection and migrations
    - Set up database connection configuration
    - Create user table migration
    - Implement database initialization scripts
    - _Requirements: 1.1_

- [ ] 3. Implement authentication service
  - [ ] 3.1 Create JWT token generation and validation
    - Implement JWT signing with secret key
    - Create token validation middleware
    - Add token expiration handling
    - _Requirements: 1.2, 2.1_
  
  - [ ] 3.2 Implement login and registration endpoints
    - Create POST /auth/register endpoint
    - Create POST /auth/login endpoint
    - Add input validation and error handling
    - _Requirements: 1.2, 2.2_

- [ ] 4. Implement user management endpoints
  - [ ] 4.1 Create user CRUD operations
    - Implement GET /users endpoint with authentication
    - Implement PUT /users/:id endpoint for updates
    - Implement DELETE /users/:id endpoint
    - _Requirements: 1.1, 2.1, 2.2_
  
  - [ ] 4.2 Add rate limiting and security middleware
    - Implement rate limiting for all endpoints
    - Add CORS configuration
    - Implement request logging
    - _Requirements: 2.2_

- [ ] 5. Implement comprehensive testing
  - [ ] 5.1 Create unit tests for services
    - Test user model validation
    - Test JWT token generation and validation
    - Test password hashing functionality
    - _Requirements: 1.1, 1.2_
  
  - [ ] 5.2 Create integration tests for API endpoints
    - Test authentication flow end-to-end
    - Test user management operations
    - Test error handling and edge cases
    - _Requirements: 2.1, 2.2_

- [ ] 6. Add error handling and logging
  - Implement centralized error handling middleware
  - Add structured logging for all operations
  - Create error response formatting
  - _Requirements: 2.2_
"""
                
                tasks_path.write_text(tasks_content)
                
                return {
                    "success": True,
                    "tasks_file": str(tasks_path)
                }
            
            elif task_type.endswith("_revision"):
                # Simulate revision handling
                phase = task_type.replace("_revision", "")
                work_dir = Path(context["work_directory"])
                
                if phase == "requirements":
                    file_path = work_dir / "requirements.md"
                    # Simulate adding revision content
                    current_content = file_path.read_text()
                    revised_content = current_content + f"\n\n## Revision Notes\n\nRevision applied: {context['revision_feedback']}\n\n### Additional Security Requirements\n\n- API rate limiting: 100 requests per minute per IP\n- Password complexity requirements\n- Session timeout after 30 minutes of inactivity"
                    file_path.write_text(revised_content)
                    
                    return {
                        "success": True,
                        "requirements_path": str(file_path)
                    }
                
                elif phase == "design":
                    file_path = work_dir / "design.md"
                    current_content = file_path.read_text()
                    revised_content = current_content + f"\n\n## Revision Notes\n\nRevision applied: {context['revision_feedback']}\n\n### Database Schema Details\n\n```sql\nCREATE TABLE users (\n    id UUID PRIMARY KEY,\n    username VARCHAR(50) UNIQUE NOT NULL,\n    email VARCHAR(255) UNIQUE NOT NULL,\n    password_hash VARCHAR(255) NOT NULL,\n    created_at TIMESTAMP DEFAULT NOW(),\n    updated_at TIMESTAMP DEFAULT NOW()\n);\n```\n\n### Error Handling Patterns\n\n- Use middleware for centralized error handling\n- Return consistent error response format\n- Log all errors with correlation IDs"
                    file_path.write_text(revised_content)
                    
                    return {
                        "success": True,
                        "design_path": str(file_path)
                    }
                
                elif phase == "tasks":
                    file_path = work_dir / "tasks.md"
                    current_content = file_path.read_text()
                    revised_content = current_content + f"\n\n## Revision Notes\n\nRevision applied: {context['revision_feedback']}\n\n- [ ] 3.3 Break down authentication into subtasks\n  - [ ] 3.3.1 Implement password validation rules\n  - [ ] 3.3.2 Add login attempt rate limiting\n  - [ ] 3.3.3 Implement session management\n  - _Requirements: 1.2_\n\n- [ ] 7. Add comprehensive unit testing\n  - [ ] 7.1 Test authentication service methods\n  - [ ] 7.2 Test user service CRUD operations\n  - [ ] 7.3 Test middleware functions\n  - _Requirements: 1.1, 1.2, 2.1, 2.2_"
                    file_path.write_text(revised_content)
                    
                    return {
                        "success": True,
                        "tasks_path": str(file_path)
                    }
            
            # Default response for unknown task types
            return {"success": False, "error": f"Unknown task type: {task_type}"}
        
        return mock_coordinate_agents
    
    def test_cli_request_mode_simulation(self, temp_workspace, mock_test_llm_config):
        """
        Test the CLI request mode with simulated user inputs.
        
        This test simulates the actual command-line interaction experience using --request flag.
        """
        from click.testing import CliRunner
        from unittest.mock import patch
        
        runner = CliRunner()
        
        # Mock the necessary components
        with patch('autogen_framework.main.MainController') as mock_controller_class:
            # Create a real controller instance for testing
            real_controller = MainController(temp_workspace)
            
            # Mock the agent coordination
            with patch.object(real_controller, 'agent_manager') as mock_agent_manager:
                mock_agent_manager.coordinate_agents = self._create_mock_coordinate_agents()
                mock_agent_manager.setup_agents.return_value = True
                mock_agent_manager.get_agent_status.return_value = {"status": "ready"}
                mock_agent_manager.update_agent_memory = Mock()
                
                # Initialize the controller - this will create the container with managers
                assert real_controller.initialize_framework(mock_test_llm_config) == True
                
                # Return the real controller when MainController is instantiated
                mock_controller_class.return_value = real_controller
                
                # Test 1: Submit a request via CLI
                result = runner.invoke(main, [
                    '--workspace', temp_workspace,
                    '--request', 'Create a simple web API',
                    '--llm-base-url', 'http://test.local:8888/openai/v1',
                    '--llm-model', 'test-model',
                    '--llm-api-key', 'sk-test123'
                ])
                
                # Verify the CLI executed successfully
                assert result.exit_code == 0
                assert "Framework initialized successfully!" in result.output
                assert "Processing request: Create a simple web API" in result.output
                assert "User approval required for: requirements" in result.output
                
                print("✅ CLI request mode test completed successfully!")
                
                # Test 2: Check status via CLI
                result2 = runner.invoke(main, [
                    '--workspace', temp_workspace,
                    '--status',
                    '--llm-base-url', 'http://test.local:8888/openai/v1',
                    '--llm-model', 'test-model',
                    '--llm-api-key', 'sk-test123'
                ])
                
                assert result2.exit_code == 0
                assert "📊 Framework Status" in result2.output
                
                print("✅ CLI status mode test completed successfully!")

if __name__ == "__main__":
    # Run the test with verbose output
    pytest.main([__file__, "-v", "-s"])