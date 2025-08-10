"""
Plan Agent for the AutoGen multi-agent framework.

This module implements the PlanAgent class responsible for:
- Parsing user requests and extracting requirements
- Creating work directories for specific tasks
- Generating requirements.md documents
- Integrating with the memory system for context
"""

import os
import re
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .base_agent import BaseLLMAgent, ContextSpec
from ..models import LLMConfig, AgentContext, WorkflowState, WorkflowPhase
from ..memory_manager import MemoryManager


class PlanAgent(BaseLLMAgent):
    """
    Plan Agent responsible for requirement analysis and work directory setup.
    
    This agent handles the initial phase of the workflow by:
    1. Parsing user requests to understand the requirements
    2. Creating appropriately named work directories
    3. Generating comprehensive requirements.md documents
    4. Integrating memory context for better understanding
    """
    
    def __init__(self, llm_config: LLMConfig, memory_manager: MemoryManager, token_manager, context_manager):
        """
        Initialize the Plan Agent.

        Args:
            llm_config: LLM configuration for API connection
            memory_manager: Memory manager instance for context loading
            token_manager: TokenManager instance for token operations (mandatory)
            context_manager: ContextManager instance for context operations (mandatory)
        """
        system_message = self._build_system_message()
        
        super().__init__(
            name="PlanAgent",
            llm_config=llm_config,
            system_message=system_message,
            token_manager=token_manager,
            context_manager=context_manager,
            description="Plan Agent responsible for parsing user requests and generating requirements",
        )
        
        self.memory_manager = memory_manager
        self.workspace_path = Path(memory_manager.workspace_path)
        
        # Load memory context on initialization
        self._load_memory_context()
    
    def _build_system_message(self) -> str:
        """Build the system message for the Plan Agent."""
        return """You are the Plan Agent in an AutoGen multi-agent framework. Your responsibilities are:

1. **Parse User Requests**: Analyze user input to understand their needs and extract key requirements
2. **Create Work Directories**: Generate descriptive directory names and create organized workspaces
3. **Generate Requirements Documents**: Create comprehensive requirements.md files using EARS format

## Key Guidelines:

### Request Parsing
- Extract the core problem or need from user input
- Identify the scope and complexity of the request
- Determine the type of work required (development, debugging, analysis, etc.)
- Consider technical constraints and dependencies

### Work Directory Naming
- Use kebab-case format (lowercase with hyphens)
- Make names descriptive but concise (max 50 characters)
- Include the main action or problem being addressed
- Examples: "fix-authentication-bug", "implement-user-dashboard", "optimize-database-queries"

### Requirements Generation
- Use EARS (Easy Approach to Requirements Syntax) format
- Structure with clear user stories: "As a [role], I want [feature], so that [benefit]"
- Include numbered acceptance criteria using WHEN/THEN/SHALL format
- Consider edge cases, error handling, and success criteria
- Reference any relevant memory context or previous learnings

### Memory Integration
- Use loaded memory context to inform requirements
- Reference similar past projects or solutions
- Apply learned best practices and patterns
- Avoid repeating past mistakes documented in memory

You should be thorough, precise, and consider both functional and non-functional requirements.
"""
    
    def _load_memory_context(self) -> None:
        """Load memory context from the memory manager."""
        try:
            memory_content = self.memory_manager.load_memory()
            self.update_memory_context(memory_content)
            self.logger.info("Loaded memory context for Plan Agent")
        except Exception as e:
            self.logger.warning(f"Could not load memory context: {e}")
    
    def get_context_requirements(self, task_input: Dict[str, Any]) -> Optional['ContextSpec']:
        """Define context requirements for PlanAgent."""
        if task_input.get("user_request"):
            return ContextSpec(context_type="plan")
        return None
    
    async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a planning task.
        
        Args:
            task_input: Dictionary containing:
                - user_request: The user's request string (for new tasks)
                - task_type: Type of task ("revision" for revisions)
                - revision_feedback: Feedback for revisions
                - current_result: Current phase result for revisions
                - work_directory: Work directory path
                - workspace_path: Optional workspace path override
                
        Returns:
            Dictionary containing task results and success status
        """
        try:
            task_type = task_input.get("task_type", "planning")
            
            if task_type == "revision":
                return await self._process_revision_task(task_input)
            else:
                return await self._process_planning_task(task_input)
                
        except Exception as e:
            self.logger.error(f"Error processing task: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process planning task: {e}"
            }
    
    async def _process_planning_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process a regular planning task."""
        user_request = task_input.get("user_request", "")
        if not user_request:
            raise ValueError("user_request is required in task_input")
        
        # Parse the user request
        parsed_request = await self.parse_user_request(user_request)
        
        # Create work directory
        work_directory = await self.create_work_directory(parsed_request["summary"])
        
        # Generate requirements document
        requirements_path = await self.generate_requirements(
            user_request, 
            work_directory, 
            parsed_request
        )
        
        return {
            "work_directory": work_directory,
            "requirements_path": requirements_path,
            "parsed_request": parsed_request,
            "success": True
        }
    
    async def _process_revision_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process a requirements revision task."""
        revision_feedback = task_input.get("revision_feedback", "")
        current_result = task_input.get("current_result", {})
        work_directory = task_input.get("work_directory", "")
        
        if not revision_feedback:
            raise ValueError("revision_feedback is required for revision tasks")
        
        if not work_directory:
            raise ValueError("work_directory is required for revision tasks")
        
        # Get current requirements path
        requirements_path = current_result.get("requirements_path")
        if not requirements_path or not os.path.exists(requirements_path):
            raise ValueError("Current requirements file not found")
        
        # Read current requirements
        with open(requirements_path, 'r', encoding='utf-8') as f:
            current_requirements = f.read()
        
        # Apply revision using LLM
        revised_requirements = await self._apply_requirements_revision(
            current_requirements, 
            revision_feedback
        )
        
        # Save revised requirements
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write(revised_requirements)
        
        self.logger.info(f"Requirements revised based on feedback: {revision_feedback[:100]}...")
        
        return {
            "work_directory": work_directory,
            "requirements_path": requirements_path,
            "revision_applied": True,
            "success": True,
            "message": "Requirements revised successfully"
        }
    
    async def _apply_requirements_revision(self, current_requirements: str, revision_feedback: str) -> str:
        """
        Apply revision feedback to current requirements.
        
        Args:
            current_requirements: Current requirements document content
            revision_feedback: User's revision feedback
            
        Returns:
            Revised requirements document content
        """
        revision_prompt = f"""Please revise the following requirements document based on the user's feedback.

Current Requirements Document:
{current_requirements}

User Feedback:
{revision_feedback}

Please provide the complete revised requirements document that incorporates the user's feedback while maintaining the same structure and format. Make sure to:
1. Keep the same markdown structure
2. Maintain EARS format for acceptance criteria
3. Update or add requirements as needed based on the feedback
4. Ensure all requirements are clear and testable

Revised Requirements Document:"""

        try:
            revised_content = await self.generate_response(revision_prompt)
            
            # Post-process the revised content
            if not revised_content.startswith("# Requirements Document"):
                revised_content = "# Requirements Document\n\n" + revised_content
            
            return revised_content
            
        except Exception as e:
            self.logger.error(f"Error applying requirements revision: {e}")
            # Return original content with a note about the revision attempt
            return f"{current_requirements}\n\n<!-- Revision attempted but failed: {str(e)} -->"
            
        except Exception as e:
            self.logger.error(f"Error processing planning task: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def parse_user_request(self, request: str) -> Dict[str, Any]:
        """
        Parse and analyze a user request to extract key information.
        
        Args:
            request: The user's request string
            
        Returns:
            Dictionary containing parsed request information:
                - summary: Brief summary of the request
                - type: Type of work (development, debugging, analysis, etc.)
                - scope: Estimated scope (small, medium, large)
                - key_requirements: List of identified key requirements
                - technical_context: Any technical details mentioned
                - constraints: Identified constraints or limitations
        """
        prompt = f"""Analyze the following user request and extract key information:

User Request: "{request}"

Please provide a structured analysis in the following format:

## Summary
[One sentence summary of what the user wants]

## Request Type
[Type: development/debugging/analysis/optimization/integration/other]

## Scope Estimate
[Scope: small/medium/large based on complexity]

## Key Requirements
[List 3-5 main requirements identified from the request]

## Technical Context
[Any specific technologies, frameworks, or technical details mentioned]

## Constraints
[Any limitations, deadlines, or constraints mentioned]

## Suggested Directory Name
[Suggest a kebab-case directory name for this request (max 50 chars)]

Provide clear, concise analysis that will help in creating comprehensive requirements."""
        
        try:
            response = await self.generate_response(prompt)
            
            # Parse the structured response
            parsed = self._parse_analysis_response(response)
            
            self.logger.info(f"Parsed user request: {parsed['summary']}")
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error parsing user request: {e}")
            # Return a basic fallback analysis
            return {
                "summary": request[:100] + "..." if len(request) > 100 else request,
                "type": "development",
                "scope": "medium",
                "key_requirements": [request],
                "technical_context": "Not specified",
                "constraints": "None identified",
                "suggested_directory": self._generate_fallback_directory_name(request)
            }
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured analysis response from the LLM."""
        parsed = {
            "summary": "",
            "type": "development",
            "scope": "medium",
            "key_requirements": [],
            "technical_context": "",
            "constraints": "",
            "suggested_directory": ""
        }
        
        # Extract sections using regex patterns
        sections = {
            "summary": r"## Summary\s*\n(.*?)(?=\n##|\Z)",
            "type": r"## Request Type\s*\n(.*?)(?=\n##|\Z)",
            "scope": r"## Scope Estimate\s*\n(.*?)(?=\n##|\Z)",
            "key_requirements": r"## Key Requirements\s*\n(.*?)(?=\n##|\Z)",
            "technical_context": r"## Technical Context\s*\n(.*?)(?=\n##|\Z)",
            "constraints": r"## Constraints\s*\n(.*?)(?=\n##|\Z)",
            "suggested_directory": r"## Suggested Directory Name\s*\n(.*?)(?=\n##|\Z)"
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                
                if key == "type":
                    # Extract just the type value, handle various formats
                    type_match = re.search(r'(?:Type:\s*)?(\w+)', value, re.IGNORECASE)
                    if type_match:
                        parsed[key] = type_match.group(1).lower()
                elif key == "scope":
                    # Extract just the scope value, handle various formats
                    scope_match = re.search(r'(?:Scope:\s*)?(\w+)', value, re.IGNORECASE)
                    if scope_match:
                        parsed[key] = scope_match.group(1).lower()
                elif key == "key_requirements":
                    # Parse list items
                    requirements = []
                    for line in value.split('\n'):
                        line = line.strip()
                        if line and (line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line)):
                            # Remove list markers
                            clean_line = re.sub(r'^[-*\d.)\s]+', '', line).strip()
                            if clean_line:
                                requirements.append(clean_line)
                    parsed[key] = requirements
                elif key == "suggested_directory":
                    # Handle various formats for directory name
                    # Look for kebab-case pattern or extract from brackets
                    bracket_match = re.search(r'\[(.*?)\]', value)
                    if bracket_match:
                        parsed[key] = bracket_match.group(1).strip()
                    else:
                        # Look for kebab-case pattern
                        kebab_match = re.search(r'([a-z]+(?:-[a-z]+)*)', value)
                        if kebab_match:
                            parsed[key] = kebab_match.group(1)
                        else:
                            parsed[key] = value.strip()
                else:
                    parsed[key] = value
        
        return parsed
    
    def _generate_fallback_directory_name(self, request: str) -> str:
        """Generate a fallback directory name from the request."""
        # Extract key words and create a directory name
        words = re.findall(r'\b\w+\b', request.lower())
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those'}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Take first 3-4 meaningful words
        selected_words = meaningful_words[:4]
        if not selected_words or len(selected_words) < 2:
            return f"task-{datetime.now().strftime('%Y%m%d-%H%M')}"
        
        directory_name = '-'.join(selected_words)
        
        # Ensure it's not too long - truncate at word boundary
        if len(directory_name) > 50:
            truncated = directory_name[:50]
            last_hyphen = truncated.rfind('-')
            if last_hyphen > 20:  # Only use word boundary if it's not too short
                directory_name = truncated[:last_hyphen]
            else:
                directory_name = truncated.rstrip('-')
        
        return directory_name
    
    async def create_work_directory(self, request_summary: str) -> str:
        """
        Create a work directory for the current request.
        
        Args:
            request_summary: Summary of the request to base directory name on
            
        Returns:
            Path to the created work directory
        """
        try:
            # Generate directory name from summary using LLM
            directory_name = await self._generate_directory_name(request_summary)
            
            # The ensure_unique_name method already handles uniqueness
            work_directory = self.workspace_path / directory_name
            
            # Create the directory
            work_directory.mkdir(parents=True, exist_ok=True)
            
            # Create basic structure files with placeholder content
            requirements_placeholder = """# Requirements Document

## Introduction

This document will contain the requirements for the project.

## Requirements

Requirements will be added here during the requirements generation phase.
"""
            
            design_placeholder = """# Design Document

## Overview

This document will contain the technical design for the project.

## Architecture

Design details will be added here during the design phase.
"""
            
            tasks_placeholder = """# Implementation Plan

Implementation tasks will be added here during the task planning phase.
"""
            
            (work_directory / "requirements.md").write_text(requirements_placeholder, encoding='utf-8')
            (work_directory / "design.md").write_text(design_placeholder, encoding='utf-8')
            (work_directory / "tasks.md").write_text(tasks_placeholder, encoding='utf-8')
            
            self.logger.info(f"Created work directory: {work_directory}")
            return str(work_directory)
            
        except Exception as e:
            self.logger.error(f"Error creating work directory: {e}")
            raise
    
    async def _generate_directory_name(self, summary: str) -> str:
        """
        Generate a directory name using LLM for better naming.
        
        Args:
            summary: Request summary to base directory name on
            
        Returns:
            Clean, descriptive directory name in kebab-case format
        """
        try:
            # Use LLM to generate a concise, descriptive directory name
            prompt = f"""Generate a concise, descriptive directory name for this request:

Request: "{summary}"

Requirements:
- Use kebab-case format (lowercase with hyphens)
- Maximum 50 characters
- Focus on the main action and object
- Be descriptive but concise
- Examples: "user-auth-system", "fix-database-bug", "optimize-api-performance"

Return only the directory name, nothing else."""

            suggested_name = await self.generate_response(prompt)
            
            # Clean and validate the LLM output
            clean_name = self.clean_directory_name(suggested_name.strip())
            
            # Ensure uniqueness and constraints
            return self.ensure_unique_name(clean_name, max_length=50)
            
        except Exception as e:
            self.logger.warning(f"LLM directory name generation failed: {e}, using fallback")
            # Fallback to the old method if LLM fails
            return self._generate_fallback_directory_name(summary)
    
    def clean_directory_name(self, llm_output: str) -> str:
        """
        Clean and sanitize LLM output for directory naming.
        
        Args:
            llm_output: Raw output from LLM
            
        Returns:
            Cleaned directory name in kebab-case format
        """
        # Extract the actual directory name from various possible formats
        name = llm_output.strip()
        
        # Handle common LLM response patterns
        # Remove quotes, brackets, or other wrapper characters
        name = re.sub(r'^["\'\[\]`]+|["\'\[\]`]+$', '', name)
        
        # Check if it's already in kebab-case format
        if re.match(r'^[a-z]+(?:-[a-z]+)*$', name.lower()):
            name = name.lower()
        else:
            # Convert to kebab-case if not already
            # Remove non-alphanumeric characters except spaces and hyphens
            name = re.sub(r'[^\w\s-]', '', name.lower())
            # Replace spaces and multiple hyphens with single hyphens
            name = re.sub(r'[-\s]+', '-', name)
        
        # Remove leading/trailing hyphens
        name = name.strip('-')
        
        # Ensure it's not empty and has reasonable length
        if not name or len(name) < 3:
            name = f"task-{datetime.now().strftime('%Y%m%d-%H%M')}"
        
        return name
    
    def ensure_unique_name(self, base_name: str, max_length: int = 50) -> str:
        """
        Ensure directory name is unique and meets length constraints.
        
        Args:
            base_name: Base directory name
            max_length: Maximum allowed length
            
        Returns:
            Unique directory name that doesn't conflict with existing directories
        """
        # Truncate if too long, preserving word boundaries
        if len(base_name) > max_length:
            truncated = base_name[:max_length]
            last_hyphen = truncated.rfind('-')
            if last_hyphen > max_length // 2:  # Only use word boundary if reasonable
                base_name = truncated[:last_hyphen]
            else:
                base_name = truncated.rstrip('-')
        
        # Check for existing directories and ensure uniqueness
        original_name = base_name
        counter = 1
        
        while (self.workspace_path / base_name).exists():
            if counter == 1:
                # First conflict: try with timestamp
                timestamp = datetime.now().strftime("%m%d-%H%M")
                base_name = f"{original_name}-{timestamp}"
            else:
                # Subsequent conflicts: use counter
                base_name = f"{original_name}-{counter}"
            
            # Ensure we don't exceed max length with suffix
            if len(base_name) > max_length:
                # Calculate how much space we need for the suffix
                suffix = f"-{counter}" if counter > 1 else f"-{timestamp}"
                available_length = max_length - len(suffix)
                base_name = original_name[:available_length].rstrip('-') + suffix
            
            counter += 1
            if counter > 100:  # Prevent infinite loop
                base_name = f"task-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                break
        
        return base_name
    
    async def generate_requirements(self, user_request: str, work_directory: str, 
                                  parsed_request: Dict[str, Any]) -> str:
        """
        Generate a comprehensive requirements.md document.
        
        Args:
            user_request: Original user request
            work_directory: Path to the work directory
            parsed_request: Parsed request information
            
        Returns:
            Path to the generated requirements.md file
        """
        requirements_path = Path(work_directory) / "requirements.md"
        
        prompt = f"""Generate a comprehensive requirements document for the following user request:

**Original Request:** {user_request}

**Parsed Analysis:**
- Summary: {parsed_request.get('summary', '')}
- Type: {parsed_request.get('type', '')}
- Scope: {parsed_request.get('scope', '')}
- Key Requirements: {', '.join(parsed_request.get('key_requirements', []))}
- Technical Context: {parsed_request.get('technical_context', '')}
- Constraints: {parsed_request.get('constraints', '')}

Please create a requirements document following this exact format:

# Requirements Document

## Introduction

[Write a clear introduction that summarizes the feature/project and its purpose]

## Requirements

### Requirement 1: [Title]

**User Story:** As a [role], I want [feature], so that [benefit]

#### Acceptance Criteria

1. WHEN [event/condition] THEN [system] SHALL [response/behavior]
2. WHEN [event/condition] THEN [system] SHALL [response/behavior]
3. IF [precondition] THEN [system] SHALL [response/behavior]

### Requirement 2: [Title]

**User Story:** As a [role], I want [feature], so that [benefit]

#### Acceptance Criteria

1. WHEN [event/condition] THEN [system] SHALL [response/behavior]
2. WHEN [event/condition] AND [additional condition] THEN [system] SHALL [response/behavior]

[Continue with additional requirements as needed]

## Guidelines:
- Use EARS (Easy Approach to Requirements Syntax) format for acceptance criteria
- Include both functional and non-functional requirements
- Consider error handling, edge cases, and user experience
- Make requirements specific, measurable, and testable
- Reference any technical constraints or dependencies
- Include security, performance, and usability considerations where relevant

Generate a complete, professional requirements document that covers all aspects of the user's request."""
        
        try:
            response = await self.generate_response(prompt)
            
            # Write the requirements document
            requirements_path.write_text(response, encoding='utf-8')
            
            self.logger.info(f"Generated requirements document: {requirements_path}")
            return str(requirements_path)
            
        except Exception as e:
            self.logger.error(f"Error generating requirements: {e}")
            raise
    
    def get_agent_capabilities(self) -> List[str]:
        """Get a list of capabilities that this agent provides."""
        return [
            "Parse and analyze user requests",
            "Extract key requirements from natural language",
            "Create organized work directories",
            "Generate comprehensive requirements documents",
            "Apply EARS format for acceptance criteria",
            "Integrate memory context for better understanding",
            "Estimate project scope and complexity",
            "Identify technical constraints and dependencies"
        ]
    
    async def get_work_directory_suggestions(self, request: str) -> List[str]:
        """
        Get multiple directory name suggestions for a request.
        
        Args:
            request: User request string
            
        Returns:
            List of suggested directory names
        """
        suggestions = []
        
        try:
            # Generate primary suggestion using LLM
            primary = await self._generate_directory_name(request)
            suggestions.append(primary)
        except Exception:
            # Fallback to simple generation
            primary = self._generate_fallback_directory_name(request)
            suggestions.append(primary)
        
        # Generate alternative suggestions using fallback method
        words = re.findall(r'\b\w+\b', request.lower())
        meaningful_words = [w for w in words if len(w) > 2]
        
        if len(meaningful_words) >= 2:
            # Different combinations
            alt1 = self.clean_directory_name('-'.join(meaningful_words[:2]))
            suggestions.append(alt1)
            
            if len(meaningful_words) >= 3:
                alt2 = self.clean_directory_name('-'.join(meaningful_words[1:3]))
                alt3 = self.clean_directory_name('-'.join([meaningful_words[0], meaningful_words[-1]]))
                suggestions.extend([alt2, alt3])
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions and suggestion:
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:5]  # Return max 5 suggestions