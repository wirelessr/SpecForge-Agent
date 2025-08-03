"""
Memory management system for the AutoGen multi-agent framework.

This module provides functionality to load, save, and organize memory content
that serves as system instructions and context for AI agents. It supports both
project-specific memory and global reusable knowledge.
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import logging

from .models import MemoryContent, SystemInstructions


class MemoryManager:
    """
    Manages memory content for the multi-agent framework.
    
    The memory system is organized into two main categories:
    1. Project-specific memory: Context and learnings specific to individual projects
    2. Global memory: Reusable knowledge and patterns that apply across projects
    
    Memory content is stored as text files in the memory directory, with optional
    metadata stored as JSON files.
    """
    
    def __init__(self, workspace_path: str):
        """
        Initialize the memory manager.
        
        Args:
            workspace_path: Path to the workspace root directory
        """
        self.workspace_path = Path(workspace_path)
        self.memory_path = self.workspace_path / "memory"
        self.global_memory_path = self.memory_path / "global"
        self.project_memory_path = self.memory_path / "projects"
        
        # Ensure memory directories exist
        self._ensure_memory_directories()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Cache for loaded memory content
        self._memory_cache: Optional[MemoryContent] = None
        self._cache_timestamp: Optional[datetime] = None
    
    def _ensure_memory_directories(self) -> None:
        """Create memory directory structure if it doesn't exist."""
        self.memory_path.mkdir(exist_ok=True)
        self.global_memory_path.mkdir(exist_ok=True)
        self.project_memory_path.mkdir(exist_ok=True)
        
        # Create initial structure files if they don't exist
        readme_path = self.memory_path / "README.md"
        if not readme_path.exists():
            readme_content = """# Memory System

This directory contains the memory system for the AutoGen multi-agent framework.

## Structure

- `global/`: Reusable knowledge and patterns that apply across projects
- `projects/`: Project-specific context and learnings
- `README.md`: This file explaining the memory system

## File Organization

Memory files should be organized by topic and use descriptive names:
- `global/python_best_practices.md`
- `global/debugging_strategies.md`
- `projects/project_name/context.md`
- `projects/project_name/learnings.md`

## Content Format

Memory files should be written in clear, structured markdown with:
- Clear headings and sections
- Specific examples and code snippets
- Context about when and how to apply the knowledge
"""
            readme_path.write_text(readme_content, encoding='utf-8')
    
    def load_memory(self) -> MemoryContent:
        """
        Load all memory content from the memory directory.
        
        Returns:
            Dictionary containing all memory content organized by category and file
        """
        # Check if we can use cached content
        if self._is_cache_valid():
            return self._memory_cache
        
        memory_content = {}
        
        try:
            # Load global memory
            global_memory = self._load_directory_content(self.global_memory_path)
            if global_memory:
                memory_content["global"] = global_memory
            
            # Load project-specific memory
            project_memory = self._load_project_memory()
            if project_memory:
                memory_content["projects"] = project_memory
            
            # Load any additional files in the root memory directory
            root_memory = self._load_directory_content(
                self.memory_path, 
                exclude_dirs={"global", "projects"}
            )
            if root_memory:
                memory_content["root"] = root_memory
            
            # Update cache
            self._memory_cache = memory_content
            self._cache_timestamp = datetime.now()
            
            self.logger.info(f"Loaded memory content from {len(memory_content)} categories")
            return memory_content
            
        except Exception as e:
            self.logger.error(f"Error loading memory content: {e}")
            return {}
    
    def _load_directory_content(self, directory: Path, exclude_dirs: set = None) -> Dict[str, str]:
        """
        Load all text files from a directory.
        
        Args:
            directory: Directory to load content from
            exclude_dirs: Set of directory names to exclude
            
        Returns:
            Dictionary mapping file names to their content
        """
        if not directory.exists():
            return {}
        
        exclude_dirs = exclude_dirs or set()
        content = {}
        
        for item in directory.iterdir():
            if item.is_file() and item.suffix in {'.md', '.txt', '.json'}:
                try:
                    file_content = item.read_text(encoding='utf-8')
                    # Use relative path as key for better organization
                    key = str(item.relative_to(self.memory_path))
                    content[key] = file_content
                except Exception as e:
                    self.logger.warning(f"Could not read file {item}: {e}")
            elif item.is_dir() and item.name not in exclude_dirs:
                # Recursively load subdirectories
                subdir_content = self._load_directory_content(item)
                if subdir_content:
                    content.update(subdir_content)
        
        return content
    
    def _load_project_memory(self) -> Dict[str, Dict[str, str]]:
        """
        Load project-specific memory content.
        
        Returns:
            Dictionary mapping project names to their memory content
        """
        if not self.project_memory_path.exists():
            return {}
        
        projects = {}
        
        for project_dir in self.project_memory_path.iterdir():
            if project_dir.is_dir():
                project_content = self._load_directory_content(project_dir)
                if project_content:
                    projects[project_dir.name] = project_content
        
        return projects
    
    def save_memory(self, key: str, content: str, category: str = "global", 
                   project_name: Optional[str] = None) -> bool:
        """
        Save new memory content to the appropriate location.
        
        Args:
            key: Identifier for the memory content (will be used as filename)
            content: The actual memory content to save
            category: Category of memory ("global", "project", or "root")
            project_name: Project name if category is "project"
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Determine the target directory and file path
            if category == "global":
                target_dir = self.global_memory_path
                file_path = target_dir / f"{key}.md"
            elif category == "project":
                if not project_name:
                    raise ValueError("project_name is required for project category")
                target_dir = self.project_memory_path / project_name
                target_dir.mkdir(exist_ok=True)
                file_path = target_dir / f"{key}.md"
            elif category == "root":
                target_dir = self.memory_path
                file_path = target_dir / f"{key}.md"
            else:
                raise ValueError(f"Invalid category: {category}")
            
            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Add metadata header to content
            timestamp = datetime.now().isoformat()
            formatted_content = f"""# {key.replace('_', ' ').title()}

*Last updated: {timestamp}*
*Category: {category}*
{f'*Project: {project_name}*' if project_name else ''}

{content}
"""
            
            # Write the content
            file_path.write_text(formatted_content, encoding='utf-8')
            
            # Invalidate cache
            self._memory_cache = None
            self._cache_timestamp = None
            
            self.logger.info(f"Saved memory content: {key} in category {category}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving memory content {key}: {e}")
            return False
    
    def get_system_instructions(self) -> SystemInstructions:
        """
        Generate comprehensive system instructions from all memory content.
        
        Returns:
            Formatted string containing all memory content as system instructions
        """
        memory_content = self.load_memory()
        
        # Check if we have any meaningful content (excluding just README)
        meaningful_content = False
        for cat_name, cat_content in memory_content.items():
            if isinstance(cat_content, dict):
                for file_key, content in cat_content.items():
                    if not file_key.endswith("README.md"):
                        meaningful_content = True
                        break
            if meaningful_content:
                break
        
        if not memory_content or not meaningful_content:
            return "No memory content available."
        
        instructions = []
        instructions.append("# System Memory and Context")
        instructions.append("")
        instructions.append("The following information represents accumulated knowledge and context from previous work:")
        instructions.append("")
        
        # Add global memory first (most generally applicable)
        if "global" in memory_content:
            instructions.append("## Global Knowledge and Best Practices")
            instructions.append("")
            for file_key, content in memory_content["global"].items():
                instructions.append(f"### {file_key}")
                instructions.append("")
                instructions.append(content)
                instructions.append("")
        
        # Add project-specific memory
        if "projects" in memory_content:
            instructions.append("## Project-Specific Context")
            instructions.append("")
            for project_name, project_content in memory_content["projects"].items():
                instructions.append(f"### Project: {project_name}")
                instructions.append("")
                for file_key, content in project_content.items():
                    instructions.append(f"#### {file_key}")
                    instructions.append("")
                    instructions.append(content)
                    instructions.append("")
        
        # Add any root-level memory
        if "root" in memory_content:
            instructions.append("## Additional Context")
            instructions.append("")
            for file_key, content in memory_content["root"].items():
                instructions.append(f"### {file_key}")
                instructions.append("")
                instructions.append(content)
                instructions.append("")
        
        return "\n".join(instructions)
    
    def search_memory(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search memory content for specific terms or patterns.
        
        Args:
            query: Search query string
            category: Optional category to limit search to
            
        Returns:
            List of matching memory entries with metadata
        """
        memory_content = self.load_memory()
        results = []
        
        query_lower = query.lower()
        
        for cat_name, cat_content in memory_content.items():
            if category and cat_name != category:
                continue
            
            if isinstance(cat_content, dict):
                # Handle nested project structure
                if cat_name == "projects":
                    for project_name, project_content in cat_content.items():
                        if isinstance(project_content, dict):
                            for file_key, content in project_content.items():
                                if isinstance(content, str) and query_lower in content.lower():
                                    results.append({
                                        "category": f"{cat_name}/{project_name}",
                                        "file": file_key,
                                        "content": content,
                                        "relevance_score": content.lower().count(query_lower)
                                    })
                else:
                    # Handle direct file content
                    for file_key, content in cat_content.items():
                        if isinstance(content, str) and query_lower in content.lower():
                            results.append({
                                "category": cat_name,
                                "file": file_key,
                                "content": content,
                                "relevance_score": content.lower().count(query_lower)
                            })
        
        # Sort by relevance score (number of matches)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary containing memory system statistics
        """
        memory_content = self.load_memory()
        
        stats = {
            "total_categories": len(memory_content),
            "total_files": 0,
            "total_size_chars": 0,
            "categories": {}
        }
        
        for cat_name, cat_content in memory_content.items():
            if isinstance(cat_content, dict):
                cat_files = len(cat_content)
                cat_size = sum(len(content) for content in cat_content.values())
                
                stats["categories"][cat_name] = {
                    "files": cat_files,
                    "size_chars": cat_size
                }
                
                stats["total_files"] += cat_files
                stats["total_size_chars"] += cat_size
        
        return stats
    
    def _is_cache_valid(self) -> bool:
        """Check if the memory cache is still valid."""
        if not self._memory_cache or not self._cache_timestamp:
            return False
        
        # Cache is valid for 5 minutes
        cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
        return cache_age < 300
    
    def clear_cache(self) -> None:
        """Clear the memory cache to force reload on next access."""
        self._memory_cache = None
        self._cache_timestamp = None
        self.logger.info("Memory cache cleared")
    
    def export_memory(self, export_path: str, format: str = "json") -> bool:
        """
        Export all memory content to a file.
        
        Args:
            export_path: Path where to save the exported memory
            format: Export format ("json" or "markdown")
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            memory_content = self.load_memory()
            export_file = Path(export_path)
            
            if format == "json":
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(memory_content, f, indent=2, ensure_ascii=False)
            elif format == "markdown":
                markdown_content = self.get_system_instructions()
                export_file.write_text(markdown_content, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Memory exported to {export_path} in {format} format")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting memory: {e}")
            return False