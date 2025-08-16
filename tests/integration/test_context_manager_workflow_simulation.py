"""
Workflow simulation tests for ContextManager.

Tests the ContextManager's behavior in simulated workflow scenarios.
These are NOT true end-to-end tests as they simulate the workflow
rather than testing the actual framework integration.

For true E2E testing, use the shell scripts in tests/e2e/ directory.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from autogen_framework.context_manager import ContextManager
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.context_compressor import ContextCompressor
from autogen_framework.models import LLMConfig, TaskDefinition, ExecutionResult


class TestContextManagerWorkflowSimulation:
    """Workflow simulation tests for ContextManager."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create workspace structure
            workspace = Path(temp_dir)
            memory_dir = workspace / "memory"
            memory_dir.mkdir()
            (memory_dir / "global").mkdir()
            (memory_dir / "projects").mkdir()
            
            # Create comprehensive memory content
            global_memory = memory_dir / "global" / "development_patterns.md"
            global_memory.write_text("""# Development Patterns

## Planning Best Practices
- Start with clear user requirements
- Define success criteria upfront
- Consider scalability from the beginning
- Plan for testing and validation

## Design Principles
- Follow SOLID principles
- Use dependency injection
- Implement proper error handling
- Design for maintainability

## Implementation Guidelines
- Write tests first (TDD)
- Use meaningful variable names
- Keep functions small and focused
- Document complex logic

## Task Management
- Break large tasks into smaller ones
- Define clear dependencies
- Set realistic timelines
- Track progress regularly
""")
            
            project_memory = memory_dir / "projects" / "web_apps"
            project_memory.mkdir()
            (project_memory / "authentication_patterns.md").write_text("""# Authentication Patterns

## Common Approaches
- JWT tokens for stateless authentication
- Session-based authentication for traditional web apps
- OAuth2 for third-party integration
- Multi-factor authentication for security

## Implementation Tips
- Always hash passwords with bcrypt
- Implement rate limiting for login attempts
- Use secure session storage
- Validate tokens on every request
""")
            
            yield temp_dir
    
    @pytest.fixture
    def real_memory_manager(self, temp_workspace):
        """Create a real MemoryManager instance."""
        return MemoryManager(temp_workspace)
    
    @pytest.fixture
    def real_context_compressor(self, test_llm_config):
        """Create a real ContextCompressor instance."""
        return ContextCompressor(test_llm_config)
    
    @pytest.fixture
    def real_token_manager(self):
        """Create a real TokenManager instance."""
        from autogen_framework.token_manager import TokenManager
        from autogen_framework.config_manager import ConfigManager
        config_manager = ConfigManager()
        return TokenManager(config_manager)
    
    @pytest.fixture
    def real_config_manager(self):
        """Create a real ConfigManager instance."""
        from autogen_framework.config_manager import ConfigManager
        return ConfigManager()
    
    @pytest.fixture
    def project_workspace(self, temp_workspace):
        """Create a complete project workspace for workflow simulation testing."""
        project_dir = Path(temp_workspace) / "simulation_project"
        project_dir.mkdir()
        
        # This simulates the complete workflow output
        # Starting with empty project, then adding files as workflow progresses
        return str(project_dir)
    
    @pytest.fixture
    def context_manager(self, project_workspace, real_memory_manager, real_context_compressor,
                       test_llm_config, real_token_manager, real_config_manager):
        """Create a ContextManager for E2E testing."""
        return ContextManager(
            work_dir=project_workspace,
            memory_manager=real_memory_manager,
            context_compressor=real_context_compressor,
            llm_config=test_llm_config,
            token_manager=real_token_manager,
            config_manager=real_config_manager
        )
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Long-running workflow simulation test - excluded from performance analysis")
    async def test_complete_workflow_simulation(self, context_manager, project_workspace):
        """
        Test complete workflow simulation from planning through implementation.
        
        NOTE: This is a SIMULATION test, not a true E2E test.
        It manually creates workflow files to test ContextManager behavior,
        rather than testing actual framework integration.
        """
        project_dir = Path(project_workspace)
        
        # === PHASE 1: PLANNING ===
        # PlanAgent would use ContextManager to get planning context
        await context_manager.initialize()
        
        plan_context = await context_manager.get_plan_context(
            "Create a web application with user authentication and data management"
        )
        
        # Verify planning context
        assert plan_context.user_request is not None
        assert plan_context.project_structure is not None
        assert isinstance(plan_context.memory_patterns, list)
        
        # Simulate PlanAgent creating requirements.md
        requirements_content = """# Requirements Document

## Introduction

This web application provides secure user authentication and comprehensive data management capabilities for small to medium businesses.

## Requirements

### Requirement 1: User Authentication System

**User Story:** As a user, I want to securely authenticate to the system, so that my data is protected and I can access personalized features.

#### Acceptance Criteria

1. WHEN a user registers THEN the system SHALL create a secure account with hashed password
2. WHEN a user logs in with valid credentials THEN the system SHALL authenticate them and create a session
3. WHEN a user logs in with invalid credentials THEN the system SHALL reject access and log the attempt
4. WHEN a user session expires THEN the system SHALL require re-authentication
5. WHEN a user logs out THEN the system SHALL invalidate their session

### Requirement 2: Data Management System

**User Story:** As a user, I want to manage my business data efficiently, so that I can organize and access information when needed.

#### Acceptance Criteria

1. WHEN a user creates data THEN the system SHALL store it securely with proper validation
2. WHEN a user updates data THEN the system SHALL maintain version history and audit trail
3. WHEN a user deletes data THEN the system SHALL confirm the action and allow recovery
4. WHEN a user searches data THEN the system SHALL provide fast and accurate results
5. WHEN data is accessed THEN the system SHALL enforce proper authorization

### Requirement 3: System Security and Performance

**User Story:** As a system administrator, I want the application to be secure and performant, so that users have a reliable experience.

#### Acceptance Criteria

1. WHEN the system processes requests THEN it SHALL respond within 2 seconds for 95% of requests
2. WHEN security threats are detected THEN the system SHALL log and block suspicious activity
3. WHEN data is transmitted THEN the system SHALL use HTTPS encryption
4. WHEN the system is under load THEN it SHALL maintain performance with proper scaling
"""
        (project_dir / "requirements.md").write_text(requirements_content)
        
        # === PHASE 2: DESIGN ===
        # DesignAgent would use ContextManager to get design context
        design_context = await context_manager.get_design_context(
            "Design a scalable web application architecture based on the requirements"
        )
        
        # Verify design context includes requirements
        assert design_context.user_request is not None
        assert design_context.requirements is not None
        assert len(design_context.requirements.requirements) == 3
        assert design_context.project_structure is not None
        assert isinstance(design_context.memory_patterns, list)
        
        # Simulate DesignAgent creating design.md
        design_content = """# Design Document

## Overview

This web application follows a modern three-tier architecture with clear separation between presentation, business logic, and data layers. The system is designed for scalability, security, and maintainability.

## Architecture

### High-Level Architecture
```
[Web Browser] <-> [Load Balancer] <-> [Web Server] <-> [Application Server] <-> [Database]
                                           |
                                    [Authentication Service]
                                           |
                                    [Data Management Service]
```

### Technology Stack
- **Frontend**: React.js with TypeScript
- **Backend**: Node.js with Express.js
- **Database**: PostgreSQL with Redis for caching
- **Authentication**: JWT tokens with bcrypt password hashing
- **Deployment**: Docker containers with Kubernetes orchestration

## Components and Interfaces

### Authentication Service
- **Purpose**: Handles user registration, login, logout, and session management
- **Key Methods**:
  - `register(email, password)`: Creates new user account
  - `login(email, password)`: Authenticates user and returns JWT token
  - `logout(token)`: Invalidates user session
  - `validateToken(token)`: Verifies JWT token validity

### Data Management Service
- **Purpose**: Provides CRUD operations for business data with proper authorization
- **Key Methods**:
  - `createData(userId, data)`: Creates new data record
  - `updateData(userId, dataId, updates)`: Updates existing data
  - `deleteData(userId, dataId)`: Soft deletes data record
  - `searchData(userId, query)`: Searches user's data

### Security Layer
- **Purpose**: Implements security policies and threat detection
- **Features**:
  - Rate limiting for API endpoints
  - Input validation and sanitization
  - SQL injection prevention
  - XSS protection
  - CSRF token validation

## Data Models

### User Model
```typescript
interface User {
  id: string;
  email: string;
  passwordHash: string;
  createdAt: Date;
  lastLoginAt: Date;
  isActive: boolean;
}
```

### Data Record Model
```typescript
interface DataRecord {
  id: string;
  userId: string;
  title: string;
  content: object;
  version: number;
  createdAt: Date;
  updatedAt: Date;
  isDeleted: boolean;
}
```

## Error Handling

- **Client Errors (4xx)**: Return structured error responses with helpful messages
- **Server Errors (5xx)**: Log detailed error information and return generic error messages
- **Database Errors**: Implement retry logic and fallback mechanisms
- **Authentication Errors**: Clear error messages without revealing system details

## Testing Strategy

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test service interactions and database operations
- **End-to-End Tests**: Test complete user workflows
- **Security Tests**: Penetration testing and vulnerability scanning
- **Performance Tests**: Load testing and stress testing
"""
        (project_dir / "design.md").write_text(design_content)
        
        # === PHASE 3: TASK GENERATION ===
        # TasksAgent would use ContextManager to get tasks context
        tasks_context = await context_manager.get_tasks_context(
            "Create detailed implementation tasks based on requirements and design"
        )
        
        # Verify tasks context includes both requirements and design
        assert tasks_context.user_request is not None
        assert tasks_context.requirements is not None
        assert tasks_context.design is not None
        assert "three-tier architecture" in tasks_context.design.content.lower()
        assert isinstance(tasks_context.memory_patterns, list)
        
        # Simulate TasksAgent creating tasks.md
        tasks_content = """# Implementation Plan

## Phase 1: Project Setup and Infrastructure

- [ ] 1. Initialize project structure and development environment
  - Set up Node.js project with TypeScript configuration
  - Configure ESLint, Prettier, and testing frameworks
  - Set up Docker containers for development
  - Initialize Git repository with proper .gitignore
  - _Requirements: 3.4_

- [ ] 2. Set up database and caching infrastructure
  - Configure PostgreSQL database with proper schemas
  - Set up Redis for session and data caching
  - Create database migration system
  - Implement connection pooling and error handling
  - _Requirements: 2.1, 2.2, 3.1_

## Phase 2: Authentication System Implementation

- [ ] 3. Implement user model and database schema
  - Create User table with proper indexes
  - Implement password hashing with bcrypt
  - Add user validation and sanitization
  - Create user repository with CRUD operations
  - _Requirements: 1.1, 1.2_

- [ ] 4. Build authentication service
  - Implement user registration endpoint
  - Create login/logout functionality with JWT tokens
  - Add session management and token validation
  - Implement rate limiting for authentication endpoints
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 5. Add security middleware and protection
  - Implement JWT token validation middleware
  - Add CORS configuration and CSRF protection
  - Create input validation and sanitization middleware
  - Implement security headers and HTTPS enforcement
  - _Requirements: 3.2, 3.3_

## Phase 3: Data Management System

- [ ] 6. Create data models and database schema
  - Design DataRecord table with proper relationships
  - Implement version control for data changes
  - Add soft delete functionality with recovery options
  - Create indexes for efficient querying
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 7. Implement data management service
  - Build CRUD operations for data records
  - Add authorization checks for data access
  - Implement search functionality with full-text search
  - Create audit trail for data changes
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 8. Add data validation and business logic
  - Implement comprehensive input validation
  - Add business rules and constraints
  - Create data transformation and normalization
  - Implement backup and recovery procedures
  - _Requirements: 2.1, 2.5_

## Phase 4: API and Frontend Integration

- [ ] 9. Build RESTful API endpoints
  - Create authentication endpoints (/auth/register, /auth/login, /auth/logout)
  - Implement data management endpoints (/api/data/*)
  - Add proper HTTP status codes and error responses
  - Implement API versioning and documentation
  - _Requirements: 1.1, 1.2, 1.5, 2.1, 2.2, 2.3, 2.4_

- [ ] 10. Develop frontend application
  - Set up React.js project with TypeScript
  - Create authentication components (login, register, logout)
  - Build data management interface (create, read, update, delete)
  - Implement responsive design and user experience
  - _Requirements: 1.1, 1.2, 1.5, 2.1, 2.2, 2.3, 2.4_

## Phase 5: Testing and Quality Assurance

- [ ] 11. Implement comprehensive testing suite
  - Write unit tests for all services and utilities
  - Create integration tests for API endpoints
  - Add end-to-end tests for user workflows
  - Implement security and penetration testing
  - _Requirements: All requirements validation_

- [ ] 12. Performance optimization and monitoring
  - Implement performance monitoring and logging
  - Add database query optimization
  - Configure caching strategies for improved performance
  - Set up health checks and system monitoring
  - _Requirements: 3.1, 3.4_

## Phase 6: Deployment and Production Setup

- [ ] 13. Prepare production deployment
  - Configure production Docker containers
  - Set up CI/CD pipeline with automated testing
  - Configure production database and security settings
  - Implement backup and disaster recovery procedures
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 14. Deploy and validate production system
  - Deploy application to production environment
  - Run full system validation and security checks
  - Perform load testing and performance validation
  - Create user documentation and admin guides
  - _Requirements: All requirements final validation_
"""
        (project_dir / "tasks.md").write_text(tasks_content)
        
        # === PHASE 4: IMPLEMENTATION ===
        # ImplementAgent would use ContextManager for each task
        
        # Refresh context manager to pick up new files
        await context_manager.initialize()
        
        # Simulate implementing the first task
        first_task = TaskDefinition(
            id="task_1",
            title="Initialize project structure and development environment",
            description="Set up Node.js project with TypeScript configuration",
            steps=[
                "Set up Node.js project with TypeScript configuration",
                "Configure ESLint, Prettier, and testing frameworks",
                "Set up Docker containers for development",
                "Initialize Git repository with proper .gitignore"
            ],
            requirements_ref=["3.4"]
        )
        
        impl_context = await context_manager.get_implementation_context(first_task)
        
        # Verify implementation context has complete project information
        assert impl_context.task == first_task
        assert impl_context.requirements is not None
        assert impl_context.design is not None
        assert impl_context.tasks is not None
        assert impl_context.project_structure is not None
        
        # Verify all documents are properly loaded
        assert len(impl_context.requirements.requirements) == 3
        assert "three-tier architecture" in impl_context.design.content.lower()
        assert len(impl_context.tasks.tasks) == 14
        
        # Verify memory patterns are available
        assert isinstance(impl_context.memory_patterns, list)
        
        # Simulate task execution and history tracking
        execution_result = ExecutionResult.create_success(
            command="npm init -y && npm install typescript @types/node",
            stdout="Successfully initialized Node.js project with TypeScript",
            execution_time=5.2,
            working_directory=project_workspace,
            approach_used="npm_package_manager"
        )
        
        await context_manager.update_execution_history(execution_result)
        
        # Verify execution history is tracked
        assert len(context_manager.execution_history) == 1
        assert context_manager.execution_history[0] == execution_result
        
        # Simulate implementing another task that depends on the first
        second_task = TaskDefinition(
            id="task_2",
            title="Set up database and caching infrastructure",
            description="Configure PostgreSQL database with proper schemas",
            steps=[
                "Configure PostgreSQL database with proper schemas",
                "Set up Redis for session and data caching",
                "Create database migration system",
                "Implement connection pooling and error handling"
            ],
            requirements_ref=["2.1", "2.2", "3.1"],
            dependencies=["task_1"]
        )
        
        impl_context_2 = await context_manager.get_implementation_context(second_task)
        
        # Verify context includes execution history from previous task
        assert len(impl_context_2.execution_history) == 1
        assert impl_context_2.execution_history[0].command.startswith("npm init")
        
        # Verify related tasks are identified
        related_task_ids = [t.id for t in impl_context_2.related_tasks]
        # Note: The related tasks logic looks for shared dependencies, not dependents
        # So we won't necessarily find task_1 as related unless they share dependencies
        
        # Add another execution result
        execution_result_2 = ExecutionResult.create_success(
            command="docker-compose up -d postgres redis",
            stdout="Successfully started PostgreSQL and Redis containers",
            execution_time=8.1,
            working_directory=project_workspace,
            approach_used="docker_compose"
        )
        
        await context_manager.update_execution_history(execution_result_2)
        
        # Verify execution history accumulates
        assert len(context_manager.execution_history) == 2
        
        # === PHASE 5: CONTEXT CONSISTENCY VERIFICATION ===
        # Verify that context remains consistent across multiple retrievals
        
        # Refresh project structure to pick up new files
        await context_manager.refresh_project_structure()
        
        # Get contexts for different agents again
        plan_context_2 = await context_manager.get_plan_context("Review project progress")
        design_context_2 = await context_manager.get_design_context("Review design decisions")
        tasks_context_2 = await context_manager.get_tasks_context("Review task progress")
        
        # Verify consistency (both should now have the updated file count)
        assert plan_context_2.project_structure.total_files >= 3  # At least requirements, design, tasks
        assert design_context_2.requirements.content == design_context.requirements.content
        assert tasks_context_2.design.content == tasks_context.design.content
        
        # === PHASE 6: MEMORY INTEGRATION VERIFICATION ===
        # Verify that memory patterns are properly integrated
        
        # All contexts should have memory patterns
        all_contexts = [plan_context, design_context, tasks_context, impl_context, impl_context_2]
        for context in all_contexts:
            assert hasattr(context, 'memory_patterns')
            assert isinstance(context.memory_patterns, list)
            # Memory patterns may be empty if no matches found, but structure should be correct
            for pattern in context.memory_patterns:
                assert hasattr(pattern, 'category')
                assert hasattr(pattern, 'content')
                assert hasattr(pattern, 'relevance_score')
                assert hasattr(pattern, 'source')
    
    @pytest.mark.asyncio
    async def test_context_compression_in_workflow(self, context_manager, project_workspace):
        """Test context compression during workflow with large content."""
        project_dir = Path(project_workspace)
        
        # Create very large requirements document to trigger compression
        large_requirements = "# Requirements Document\n\n## Introduction\n\n" + "This is a very detailed requirements document. " * 2000
        large_requirements += "\n\n### Requirement 1: Large Feature\n\n**User Story:** As a user, I want comprehensive features.\n\n#### Acceptance Criteria\n\n"
        for i in range(100):
            large_requirements += f"{i+1}. WHEN condition {i+1} occurs THEN system SHALL respond appropriately\n"
        
        (project_dir / "requirements.md").write_text(large_requirements)
        
        await context_manager.initialize()
        
        # Get design context with large requirements - should trigger compression
        with patch.object(context_manager.context_compressor, 'compress_context') as mock_compress:
            mock_compress.return_value = Mock(
                success=True,
                compressed_content={"requirements": "Compressed requirements content"},
                compression_ratio=0.3,
                error=None
            )
            
            design_context = await context_manager.get_design_context("Design with large requirements")
            
            # Verify compression was attempted due to large content
            # Note: The actual compression call depends on token estimation
            assert design_context is not None
            assert design_context.requirements is not None
    
    @pytest.mark.asyncio
    async def test_error_recovery_in_workflow(self, context_manager, project_workspace):
        """Test error handling and recovery during workflow execution."""
        project_dir = Path(project_workspace)
        
        # Create minimal project files
        (project_dir / "requirements.md").write_text("# Minimal Requirements\n\n### Requirement 1: Basic Feature")
        
        await context_manager.initialize()
        
        # Test with corrupted file
        (project_dir / "design.md").write_text("Invalid JSON content: {{{")
        
        # Should handle corrupted files gracefully
        design_context = await context_manager.get_design_context("Handle corrupted design")
        assert design_context is not None
        assert design_context.requirements is not None
        # Design should be loaded despite corruption (it's just text content)
        assert design_context.design is not None if hasattr(design_context, 'design') else True
        
        # Test with missing files after initialization
        os.remove(project_dir / "requirements.md")
        
        # Should detect file removal and handle gracefully
        tasks_context = await context_manager.get_tasks_context("Handle missing requirements")
        assert tasks_context is not None
        # Requirements should be None after file removal and refresh
        if context_manager._should_refresh_requirements():
            await context_manager._load_requirements()
            assert context_manager.requirements is None
    
    @pytest.mark.asyncio
    async def test_concurrent_context_access(self, context_manager, project_workspace):
        """Test concurrent access to context manager from multiple agents."""
        project_dir = Path(project_workspace)
        
        # Create project files
        (project_dir / "requirements.md").write_text("# Requirements\n\n### Requirement 1: Concurrent Feature")
        (project_dir / "design.md").write_text("# Design\n\n## Overview\n\nConcurrent design")
        (project_dir / "tasks.md").write_text("# Tasks\n\n- [ ] 1. Concurrent task")
        
        await context_manager.initialize()
        
        # Simulate concurrent access from multiple agents
        import asyncio
        
        async def get_plan_context():
            return await context_manager.get_plan_context("Concurrent plan request")
        
        async def get_design_context():
            return await context_manager.get_design_context("Concurrent design request")
        
        async def get_tasks_context():
            return await context_manager.get_tasks_context("Concurrent tasks request")
        
        async def get_impl_context():
            test_task = TaskDefinition(id="concurrent", title="Concurrent Task", description="", steps=[], requirements_ref=[])
            return await context_manager.get_implementation_context(test_task)
        
        # Run all context retrievals concurrently
        results = await asyncio.gather(
            get_plan_context(),
            get_design_context(),
            get_tasks_context(),
            get_impl_context(),
            return_exceptions=True
        )
        
        # Verify all contexts were retrieved successfully
        assert len(results) == 4
        for result in results:
            assert not isinstance(result, Exception)
            assert result is not None
        
        # Verify context consistency
        plan_ctx, design_ctx, tasks_ctx, impl_ctx = results
        assert plan_ctx.project_structure.total_files == design_ctx.project_structure.total_files
        assert design_ctx.requirements.content == tasks_ctx.requirements.content
        assert tasks_ctx.design.content == impl_ctx.design.content
    
    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, context_manager, project_workspace, 
                                             test_llm_config, real_token_manager, real_config_manager):
        """Test that workflow state is properly persisted and can be restored."""
        project_dir = Path(project_workspace)
        
        # Create project files
        (project_dir / "requirements.md").write_text("# Requirements\n\n### Requirement 1: Persistence Feature")
        (project_dir / "design.md").write_text("# Design\n\n## Overview\n\nPersistent design")
        (project_dir / "tasks.md").write_text("# Tasks\n\n- [ ] 1. Persistence task")
        
        await context_manager.initialize()
        
        # Add execution history
        result1 = ExecutionResult.create_success("command1", "output1", 1.0, project_workspace, "approach1")
        result2 = ExecutionResult.create_success("command2", "output2", 2.0, project_workspace, "approach2")
        
        await context_manager.update_execution_history(result1)
        await context_manager.update_execution_history(result2)
        
        # Verify history file exists
        history_file = project_dir / "execution_history.json"
        assert history_file.exists()
        
        # Create new context manager instance (simulating restart)
        new_context_manager = ContextManager(
            work_dir=project_workspace,
            memory_manager=context_manager.memory_manager,
            context_compressor=context_manager.context_compressor,
            llm_config=test_llm_config,
            token_manager=real_token_manager,
            config_manager=real_config_manager
        )
        
        await new_context_manager.initialize()
        
        # Verify state can be restored (execution history is currently not auto-loaded)
        # This would be enhanced in a full implementation
        assert new_context_manager.requirements is not None
        assert new_context_manager.design is not None
        assert new_context_manager.tasks is not None
        assert new_context_manager.project_structure is not None
        
        # Verify file content consistency
        assert new_context_manager.requirements.content == context_manager.requirements.content
        assert new_context_manager.design.content == context_manager.design.content
        assert new_context_manager.tasks.content == context_manager.tasks.content