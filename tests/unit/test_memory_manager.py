"""
Unit tests for the MemoryManager class.

Tests cover all core functionality including loading, saving, searching,
and organizing memory content.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from autogen_framework.memory_manager import MemoryManager

class TestMemoryManager:
    """Test suite for MemoryManager functionality."""
    # Using shared fixtures from conftest.py
    @pytest.fixture
    def memory_manager(self, temp_workspace):
        """Create a MemoryManager instance with temporary workspace."""
        return MemoryManager(temp_workspace)
    
    @pytest.fixture
    def populated_memory_manager(self, temp_workspace):
        """Create a MemoryManager with some test content."""
        manager = MemoryManager(temp_workspace)
        
        # Add some global memory
        manager.save_memory("python_tips", "# Python Tips\n\nUse list comprehensions for better performance.", "global")
        manager.save_memory("debugging", "# Debugging Strategies\n\n1. Use print statements\n2. Use debugger", "global")
        
        # Add project memory
        manager.save_memory("context", "# Project Context\n\nThis is a web application.", "project", "webapp")
        manager.save_memory("learnings", "# Learnings\n\nLearned about async/await.", "project", "webapp")
        
        # Add root memory
        manager.save_memory("general_notes", "# General Notes\n\nSome general observations.", "root")
        
        return manager
    
    def test_initialization(self, memory_manager, temp_workspace):
        """Test MemoryManager initialization."""
        assert memory_manager.workspace_path == Path(temp_workspace)
        assert memory_manager.memory_path == Path(temp_workspace) / "memory"
        assert memory_manager.global_memory_path == Path(temp_workspace) / "memory" / "global"
        assert memory_manager.project_memory_path == Path(temp_workspace) / "memory" / "projects"
        
        # Check that directories are created
        assert memory_manager.memory_path.exists()
        assert memory_manager.global_memory_path.exists()
        assert memory_manager.project_memory_path.exists()
        
        # Check that README is created
        readme_path = memory_manager.memory_path / "README.md"
        assert readme_path.exists()
        readme_content = readme_path.read_text()
        assert "Memory System" in readme_content
    
    def test_save_global_memory(self, memory_manager):
        """Test saving global memory content."""
        content = "# Test Content\n\nThis is a test."
        result = memory_manager.save_memory("test_key", content, "global")
        
        assert result is True
        
        # Check file was created
        expected_path = memory_manager.global_memory_path / "test_key.md"
        assert expected_path.exists()
        
        # Check content includes metadata
        saved_content = expected_path.read_text()
        assert "# Test Key" in saved_content
        assert "*Category: global*" in saved_content
        assert content in saved_content
    
    def test_save_project_memory(self, memory_manager):
        """Test saving project-specific memory content."""
        content = "# Project Context\n\nThis is project-specific content."
        result = memory_manager.save_memory("context", content, "project", "test_project")
        
        assert result is True
        
        # Check file was created in correct location
        expected_path = memory_manager.project_memory_path / "test_project" / "context.md"
        assert expected_path.exists()
        
        # Check content includes project metadata
        saved_content = expected_path.read_text()
        assert "*Project: test_project*" in saved_content
        assert content in saved_content
    
    def test_save_root_memory(self, memory_manager):
        """Test saving root-level memory content."""
        content = "# Root Content\n\nThis is root-level content."
        result = memory_manager.save_memory("root_notes", content, "root")
        
        assert result is True
        
        # Check file was created
        expected_path = memory_manager.memory_path / "root_notes.md"
        assert expected_path.exists()
        
        saved_content = expected_path.read_text()
        assert "*Category: root*" in saved_content
        assert content in saved_content
    
    def test_save_memory_invalid_category(self, memory_manager):
        """Test saving memory with invalid category."""
        result = memory_manager.save_memory("test", "content", "invalid_category")
        assert result is False
    
    def test_save_project_memory_without_project_name(self, memory_manager):
        """Test saving project memory without providing project name."""
        result = memory_manager.save_memory("test", "content", "project")
        assert result is False
    
    def test_load_empty_memory(self, memory_manager):
        """Test loading memory when no content exists."""
        memory_content = memory_manager.load_memory()
        
        # Should return empty dict or only contain README
        assert isinstance(memory_content, dict)
    
    def test_load_populated_memory(self, populated_memory_manager):
        """Test loading memory with existing content."""
        memory_content = populated_memory_manager.load_memory()
        
        assert "global" in memory_content
        assert "projects" in memory_content
        assert "root" in memory_content
        
        # Check global memory
        global_memory = memory_content["global"]
        assert any("python_tips" in key for key in global_memory.keys())
        assert any("debugging" in key for key in global_memory.keys())
        
        # Check project memory
        project_memory = memory_content["projects"]
        assert "webapp" in project_memory
        webapp_memory = project_memory["webapp"]
        assert any("context" in key for key in webapp_memory.keys())
        assert any("learnings" in key for key in webapp_memory.keys())
        
        # Check root memory
        root_memory = memory_content["root"]
        assert any("general_notes" in key for key in root_memory.keys())
    
    def test_get_system_instructions_empty(self, memory_manager):
        """Test getting system instructions with no memory content."""
        instructions = memory_manager.get_system_instructions()
        
        assert isinstance(instructions, str)
        assert "No memory content available" in instructions
    
    def test_get_system_instructions_populated(self, populated_memory_manager):
        """Test getting system instructions with populated memory."""
        instructions = populated_memory_manager.get_system_instructions()
        
        assert isinstance(instructions, str)
        assert "# System Memory and Context" in instructions
        assert "## Global Knowledge and Best Practices" in instructions
        assert "## Project-Specific Context" in instructions
        assert "## Additional Context" in instructions
        
        # Check that content is included
        assert "Python Tips" in instructions
        assert "Debugging Strategies" in instructions
        assert "Project Context" in instructions
        assert "General Notes" in instructions
    
    def test_search_memory(self, populated_memory_manager):
        """Test searching memory content."""
        # Search for "Python"
        results = populated_memory_manager.search_memory("Python")
        
        assert len(results) > 0
        assert any("python" in result["file"].lower() for result in results)
        
        # Search in specific category
        global_results = populated_memory_manager.search_memory("debugging", "global")
        assert len(global_results) > 0
        assert all(result["category"] == "global" for result in global_results)
        
        # Search for non-existent term
        no_results = populated_memory_manager.search_memory("nonexistent")
        assert len(no_results) == 0
    
    def test_get_memory_stats(self, populated_memory_manager):
        """Test getting memory statistics."""
        stats = populated_memory_manager.get_memory_stats()
        
        assert "total_categories" in stats
        assert "total_files" in stats
        assert "total_size_chars" in stats
        assert "categories" in stats
        
        assert stats["total_categories"] >= 3  # global, projects, root
        assert stats["total_files"] > 0
        assert stats["total_size_chars"] > 0
        
        # Check category-specific stats
        categories = stats["categories"]
        assert "global" in categories
        assert "projects" in categories
        assert "root" in categories
        
        for cat_stats in categories.values():
            assert "files" in cat_stats
            assert "size_chars" in cat_stats
    
    def test_memory_caching(self, populated_memory_manager):
        """Test memory caching functionality."""
        # First load should populate cache
        memory1 = populated_memory_manager.load_memory()
        
        # Second load should use cache (same object)
        memory2 = populated_memory_manager.load_memory()
        
        assert memory1 == memory2
        
        # Clear cache and load again
        populated_memory_manager.clear_cache()
        memory3 = populated_memory_manager.load_memory()
        
        assert memory1 == memory3  # Content should be same, but freshly loaded
    
    def test_export_memory_json(self, populated_memory_manager, temp_workspace):
        """Test exporting memory to JSON format."""
        export_path = Path(temp_workspace) / "exported_memory.json"
        
        result = populated_memory_manager.export_memory(str(export_path), "json")
        assert result is True
        assert export_path.exists()
        
        # Load and verify exported content
        with open(export_path, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        assert "global" in exported_data
        assert "projects" in exported_data
        assert "root" in exported_data
    
    def test_export_memory_markdown(self, populated_memory_manager, temp_workspace):
        """Test exporting memory to Markdown format."""
        export_path = Path(temp_workspace) / "exported_memory.md"
        
        result = populated_memory_manager.export_memory(str(export_path), "markdown")
        assert result is True
        assert export_path.exists()
        
        # Verify exported content
        exported_content = export_path.read_text()
        assert "# System Memory and Context" in exported_content
        assert "Python Tips" in exported_content
    
    def test_export_memory_invalid_format(self, populated_memory_manager, temp_workspace):
        """Test exporting memory with invalid format."""
        export_path = Path(temp_workspace) / "exported_memory.txt"
        
        result = populated_memory_manager.export_memory(str(export_path), "invalid")
        assert result is False
    
    @patch('autogen_framework.memory_manager.logging.getLogger')
    def test_logging_integration(self, mock_logger, memory_manager):
        """Test that logging is properly integrated."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance
        
        # Create new instance to trigger logger setup
        manager = MemoryManager(memory_manager.workspace_path)
        
        # Test that logger is called during operations
        manager.save_memory("test", "content", "global")
        
        # Verify logger was set up
        mock_logger.assert_called_with('autogen_framework.memory_manager')
    
    def test_file_reading_error_handling(self, memory_manager, temp_workspace):
        """Test handling of file reading errors."""
        # Create a file with restricted permissions (if possible)
        test_file = memory_manager.global_memory_path / "restricted.md"
        test_file.write_text("test content")
        
        # Try to make it unreadable (may not work on all systems)
        try:
            test_file.chmod(0o000)
            
            # Load memory should handle the error gracefully
            memory_content = memory_manager.load_memory()
            
            # Should still return a dict, just without the problematic file
            assert isinstance(memory_content, dict)
            
        except (OSError, PermissionError):
            # If we can't change permissions, skip this test
            pytest.skip("Cannot test file permission errors on this system")
        finally:
            # Restore permissions for cleanup
            try:
                test_file.chmod(0o644)
            except (OSError, PermissionError):
                pass
    
    def test_directory_structure_creation(self, temp_workspace):
        """Test that directory structure is created correctly."""
        # Remove the memory directory if it exists
        memory_path = Path(temp_workspace) / "memory"
        if memory_path.exists():
            shutil.rmtree(memory_path)
        
        # Create MemoryManager - should recreate directories
        manager = MemoryManager(temp_workspace)
        
        assert manager.memory_path.exists()
        assert manager.global_memory_path.exists()
        assert manager.project_memory_path.exists()
        
        # Check README creation
        readme_path = manager.memory_path / "README.md"
        assert readme_path.exists()
        
        readme_content = readme_path.read_text()
        assert "Memory System" in readme_content
        assert "global/" in readme_content
        assert "projects/" in readme_content

if __name__ == "__main__":
    pytest.main([__file__])