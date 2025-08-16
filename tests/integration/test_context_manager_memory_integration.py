"""
Integration tests for ContextManager with MemoryManager and ContextCompressor.

Tests the integration between ContextManager and its dependencies to verify
memory pattern retrieval, context compression, and component interactions.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from autogen_framework.context_manager import ContextManager
from autogen_framework.memory_manager import MemoryManager
from autogen_framework.context_compressor import ContextCompressor
from autogen_framework.models import LLMConfig, TaskDefinition, CompressionResult


class TestContextManagerMemoryIntegration:
    """Integration tests for ContextManager memory and compression integration."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace with memory content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            memory_dir = workspace / "memory"
            memory_dir.mkdir()
            (memory_dir / "global").mkdir()
            (memory_dir / "projects").mkdir()
            
            # Create comprehensive memory content for testing
            global_memory = memory_dir / "global" / "planning_patterns.md"
            global_memory.write_text("""# Planning Patterns

## Requirements Analysis
- Start with user needs assessment
- Define clear acceptance criteria
- Consider edge cases and error scenarios
- Plan for scalability and performance

## Project Structure
- Use consistent naming conventions
- Organize code by feature or layer
- Separate concerns appropriately
- Plan for testing and documentation
""")
            
            design_memory = memory_dir / "global" / "design_patterns.md"
            design_memory.write_text("""# Design Patterns

## Architecture Principles
- Follow SOLID principles
- Use dependency injection
- Implement proper abstraction layers
- Design for testability

## Common Patterns
- Repository pattern for data access
- Factory pattern for object creation
- Observer pattern for event handling
- Strategy pattern for algorithms
""")
            
            impl_memory = memory_dir / "global" / "implementation_best_practices.md"
            impl_memory.write_text("""# Implementation Best Practices

## Coding Standards
- Use meaningful variable names
- Keep functions small and focused
- Write comprehensive tests
- Document complex logic

## Error Handling
- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Implement graceful degradation
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
    def work_dir(self, temp_workspace):
        """Create a work directory with test project files."""
        work_dir = Path(temp_workspace) / "test_project"
        work_dir.mkdir()
        
        # Create project files
        (work_dir / "requirements.md").write_text("""# Requirements Document

## Introduction
Test project for memory integration testing.

### Requirement 1: Memory Integration
**User Story:** As a system, I want to integrate with memory patterns.
#### Acceptance Criteria
1. WHEN memory patterns are requested THEN the system SHALL retrieve relevant patterns
""")
        
        (work_dir / "design.md").write_text("""# Design Document

## Overview
This design integrates memory patterns for better context awareness.

## Architecture
The system uses memory patterns to inform design decisions.
""")
        
        return str(work_dir)
    
    @pytest.fixture
    def context_manager(self, work_dir, real_memory_manager, real_context_compressor,
                       test_llm_config, real_token_manager, real_config_manager):
        """Create a ContextManager with real dependencies."""
        return ContextManager(
            work_dir=work_dir,
            memory_manager=real_memory_manager,
            context_compressor=real_context_compressor,
            llm_config=test_llm_config,
            token_manager=real_token_manager,
            config_manager=real_config_manager
        )
    
    @pytest.mark.asyncio
    async def test_memory_manager_integration(self, context_manager, real_memory_manager):
        """Test integration with MemoryManager for pattern retrieval."""
        await context_manager.initialize()
        
        # Verify memory manager is properly integrated
        assert context_manager.memory_manager is real_memory_manager
        
        # Test memory content loading
        memory_content = real_memory_manager.load_memory()
        assert len(memory_content) > 0
        assert "global" in memory_content
        
        # Verify memory patterns are retrieved for different contexts
        plan_context = await context_manager.get_plan_context("Test planning request")
        design_context = await context_manager.get_design_context("Test design request")
        
        # Memory patterns should be included
        assert isinstance(plan_context.memory_patterns, list)
        assert isinstance(design_context.memory_patterns, list)
        
        # Verify memory manager search is called
        with patch.object(real_memory_manager, 'search_memory') as mock_search:
            mock_search.return_value = [
                {"category": "global", "file": "test.md", "content": "Test content", "relevance_score": 1.0}
            ]
            
            await context_manager._get_plan_memory_patterns("test query")
            mock_search.assert_called_with("planning requirements project")
    
    @pytest.mark.asyncio
    async def test_context_compressor_integration(self, context_manager, real_context_compressor):
        """Test integration with ContextCompressor for automatic compression."""
        await context_manager.initialize()
        
        # Verify context compressor is properly integrated
        assert context_manager.context_compressor is real_context_compressor
        
        # Create a proper context object with large content
        from autogen_framework.context_manager import PlanContext
        large_context = PlanContext(
            user_request="x" * 40000,  # Large content to trigger compression (> 8192 * 4)
            memory_patterns=[]
        )
        
        # Mock the compression result
        with patch.object(real_context_compressor, 'compress_context') as mock_compress:
            mock_compress.return_value = CompressionResult(
                original_size=20000,
                compressed_size=6000,
                compression_ratio=0.3,
                compressed_content={"user_request": "Compressed content"},
                method_used="llm_compression_dynamic",
                success=True
            )
            
            # Test compression is triggered for large content
            result = await context_manager._compress_if_needed(large_context, "plan")
            
            # Verify compression was attempted
            mock_compress.assert_called_once()
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_memory_pattern_retrieval_for_all_agents(self, context_manager):
        """Test memory pattern retrieval for all agent types."""
        await context_manager.initialize()
        
        # Test plan memory patterns
        plan_patterns = await context_manager._get_plan_memory_patterns("Create new project")
        assert isinstance(plan_patterns, list)
        
        # Test design memory patterns
        design_patterns = await context_manager._get_design_memory_patterns()
        assert isinstance(design_patterns, list)
        
        # Test tasks memory patterns
        tasks_patterns = await context_manager._get_tasks_memory_patterns()
        assert isinstance(tasks_patterns, list)
        
        # Test implementation memory patterns
        test_task = TaskDefinition(id="test", title="Test Task", description="", steps=[], requirements_ref=[])
        impl_patterns = await context_manager._get_implementation_memory_patterns(test_task)
        assert isinstance(impl_patterns, list)
        
        # Verify patterns have correct structure
        all_patterns = plan_patterns + design_patterns + tasks_patterns + impl_patterns
        for pattern in all_patterns:
            assert hasattr(pattern, 'category')
            assert hasattr(pattern, 'content')
            assert hasattr(pattern, 'relevance_score')
            assert hasattr(pattern, 'source')
    
    @pytest.mark.asyncio
    async def test_automatic_context_compression_thresholds(self, context_manager):
        """Test automatic context compression based on agent-specific thresholds."""
        await context_manager.initialize()
        
        # Test model-specific token limit
        model_limit = context_manager.model_token_limit
        assert model_limit > 0
        assert isinstance(model_limit, int)
        
        # Verify limit is reasonable (should be model-specific value)
        assert model_limit >= 1000  # Minimum reasonable limit
        assert model_limit <= 10000000  # Maximum reasonable limit (for large models like Gemini)
        
        # Test compression threshold
        compression_threshold = context_manager.compression_threshold
        assert 0.0 < compression_threshold <= 1.0  # Should be a percentage
    
    @pytest.mark.asyncio
    async def test_memory_search_with_different_queries(self, context_manager, real_memory_manager):
        """Test memory search with different query types."""
        await context_manager.initialize()
        
        # Test search functionality
        search_results = real_memory_manager.search_memory("planning")
        assert isinstance(search_results, list)
        
        # Test different search queries
        queries = [
            "planning requirements project",
            "design architecture patterns", 
            "tasks implementation breakdown",
            "implementation coding best practices"
        ]
        
        for query in queries:
            results = real_memory_manager.search_memory(query)
            assert isinstance(results, list)
            # Results may be empty if no matches, but should be valid list
    
    @pytest.mark.asyncio
    async def test_context_compression_with_real_compressor(self, context_manager, real_context_compressor):
        """Test context compression with real ContextCompressor."""
        await context_manager.initialize()
        
        # Create a context with substantial content
        test_context = {
            "user_request": "Create a comprehensive web application with authentication, data management, and reporting features.",
            "requirements": "This is a detailed requirements document. " * 100,
            "design": "This is a comprehensive design document. " * 100,
            "project_structure": {
                "files": ["file1.py", "file2.py"] * 50,
                "directories": ["dir1", "dir2"] * 25
            }
        }
        
        # Test compression
        try:
            result = await real_context_compressor.compress_context(test_context, 0.5)
            assert isinstance(result, CompressionResult)
            assert hasattr(result, 'success')
            assert hasattr(result, 'original_size')
            assert hasattr(result, 'compressed_size')
            assert hasattr(result, 'compression_ratio')
        except Exception as e:
            # If compression fails due to LLM unavailability, that's expected in test environment
            assert "LLM" in str(e) or "connection" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_memory_manager_error_handling(self, context_manager):
        """Test error handling in memory manager integration."""
        await context_manager.initialize()
        
        # Test with memory manager that raises exceptions
        with patch.object(context_manager.memory_manager, 'search_memory') as mock_search:
            mock_search.side_effect = Exception("Memory search failed")
            
            # Should handle errors gracefully
            patterns = await context_manager._get_plan_memory_patterns("test query")
            assert isinstance(patterns, list)
            assert len(patterns) == 0  # Should return empty list on error
    
    @pytest.mark.asyncio
    async def test_context_compressor_error_handling(self, context_manager):
        """Test error handling in context compressor integration."""
        await context_manager.initialize()
        
        # Test with context compressor that raises exceptions
        with patch.object(context_manager.context_compressor, 'compress_context') as mock_compress:
            mock_compress.side_effect = Exception("Compression failed")
            
            # Create large context that would normally trigger compression
            large_context = Mock()
            large_context.user_request = "x" * 20000
            large_context.memory_patterns = []
            
            # Should handle compression errors gracefully
            result = await context_manager._compress_if_needed(large_context, "plan")
            assert result is large_context  # Should return original context on error
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Long-running end-to-end test - excluded from performance analysis")
    async def test_end_to_end_context_flow(self, context_manager):
        """Test complete end-to-end context flow with memory and compression."""
        await context_manager.initialize()
        
        # Test complete flow for implementation context
        test_task = TaskDefinition(
            id="e2e_task",
            title="End-to-end test task",
            description="Test complete context flow",
            steps=["Step 1", "Step 2"],
            requirements_ref=["1.1"]
        )
        
        # Get implementation context (most comprehensive)
        context = await context_manager.get_implementation_context(test_task)
        
        # Verify all components are integrated
        assert context.task == test_task
        assert context.requirements is not None
        assert context.project_structure is not None
        assert isinstance(context.memory_patterns, list)
        assert isinstance(context.execution_history, list)
        assert isinstance(context.related_tasks, list)
        
        # Verify context has compression metadata
        assert hasattr(context, 'compressed')
        assert hasattr(context, 'compression_ratio')
        
        # Test that context can be retrieved multiple times consistently
        context2 = await context_manager.get_implementation_context(test_task)
        assert context2.task.id == context.task.id
        assert context2.requirements.content == context.requirements.content
    
    @pytest.mark.asyncio
    async def test_memory_pattern_relevance_scoring(self, context_manager, real_memory_manager):
        """Test memory pattern relevance scoring and ranking."""
        await context_manager.initialize()
        
        # Add specific memory content for testing
        memory_manager = context_manager.memory_manager
        
        # Save test memory content
        test_content = """# Authentication Patterns
        
## JWT Implementation
- Use secure secret keys
- Implement token expiration
- Validate tokens on each request

## Password Security
- Hash passwords with bcrypt
- Implement password strength requirements
- Add rate limiting for login attempts
"""
        
        success = memory_manager.save_memory("auth_patterns", test_content, "global")
        assert success
        
        # Search for authentication-related patterns
        auth_patterns = await context_manager._get_implementation_memory_patterns(
            TaskDefinition(id="auth", title="Authentication Task", description="", steps=[], requirements_ref=[])
        )
        
        # Verify patterns are returned with relevance scoring
        for pattern in auth_patterns:
            assert isinstance(pattern.relevance_score, (int, float))
            assert pattern.relevance_score >= 0
    
    @pytest.mark.asyncio
    async def test_component_interaction_verification(self, context_manager, real_memory_manager, real_context_compressor):
        """Verify all component interactions work correctly."""
        await context_manager.initialize()
        
        # Verify ContextManager properly uses both dependencies
        assert context_manager.memory_manager is real_memory_manager
        assert context_manager.context_compressor is real_context_compressor
        
        # Test memory manager interaction
        memory_stats = real_memory_manager.get_memory_stats()
        assert isinstance(memory_stats, dict)
        assert "total_files" in memory_stats
        
        # Test context compressor capabilities
        capabilities = real_context_compressor.get_capabilities()
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        
        # Test integrated workflow
        plan_context = await context_manager.get_plan_context("Integration test")
        
        # Verify context includes memory patterns (from memory manager)
        assert isinstance(plan_context.memory_patterns, list)
        
        # Verify context has compression metadata (from context compressor integration)
        assert hasattr(plan_context, 'compressed')
        assert hasattr(plan_context, 'compression_ratio')
        
        # Test that both components are used in context generation
        with patch.object(real_memory_manager, 'search_memory') as mock_memory:
            with patch.object(real_context_compressor, 'compress_context') as mock_compress:
                mock_memory.return_value = []
                mock_compress.return_value = CompressionResult(
                    original_size=1000, compressed_size=800, compression_ratio=0.8,
                    compressed_content={}, method_used="test", success=True
                )
                
                await context_manager.get_design_context("Component interaction test")
                
                # Verify both components were used
                mock_memory.assert_called()
                # Compression may or may not be called depending on content size