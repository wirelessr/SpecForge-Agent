"""
Design Agent for the AutoGen multi-agent framework.

This module implements the DesignAgent class responsible for generating technical
design documents based on approved requirements. The agent creates comprehensive
design documents including architecture diagrams, component interfaces, and
technical specifications.
"""

import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_agent import BaseLLMAgent, ContextSpec
from ..models import LLMConfig, AgentContext


class DesignAgent(BaseLLMAgent):
    """
    AI agent responsible for generating technical design documents.
    
    The DesignAgent takes approved requirements and creates detailed technical
    designs including:
    - Architecture overview
    - Component and interface definitions
    - Data flow diagrams using Mermaid.js
    - Security considerations
    - Testing strategies
    
    It integrates with the memory system to leverage relevant technical knowledge
    and patterns from previous projects.
    """
    
    def __init__(self, llm_config: LLMConfig, memory_context: Optional[Dict[str, Any]] = None, token_manager=None, context_manager=None, config_manager=None):
        """
        Initialize the Design Agent.

        Args:
            llm_config: LLM configuration for API connection
            memory_context: Optional memory context from MemoryManager
            token_manager: TokenManager instance for token operations (mandatory)
            context_manager: ContextManager instance for context operations (mandatory)
            config_manager: ConfigManager instance for model configuration (optional)
        """
        system_message = self._build_system_message()
        
        super().__init__(
            name="DesignAgent",
            llm_config=llm_config,
            system_message=system_message,
            token_manager=token_manager,
            context_manager=context_manager,
            config_manager=config_manager,
            description="AI agent specialized in generating technical design documents from requirements",
        )
        
        # Update memory context if provided
        if memory_context:
            self.update_memory_context(memory_context)
        
        self.logger.info("DesignAgent initialized successfully")
    
    def _build_system_message(self) -> str:
        """
        Build the system message for the Design Agent.
        
        Returns:
            System message string defining the agent's role and capabilities
        """
        return """You are a Design Agent specialized in creating comprehensive technical design documents.

## Your Role
You are responsible for transforming approved requirements into detailed technical designs that serve as blueprints for implementation. Your designs should be thorough, practical, and implementable.

## Core Responsibilities
1. **Architecture Design**: Create clear architectural overviews that define system structure
2. **Component Definition**: Specify components, their interfaces, and interactions
3. **Data Flow Modeling**: Generate Mermaid.js diagrams showing data flow and system interactions
4. **Technical Specifications**: Define technical details, constraints, and implementation approaches
5. **Security Considerations**: Identify and address security requirements and concerns
6. **Testing Strategy**: Define comprehensive testing approaches for the system

## Design Document Structure
Your design documents should follow this structure:
1. **Architectural Overview**: High-level system description and design principles
2. **Data Flow Diagrams**: Mermaid.js diagrams showing system interactions
3. **Components and Interfaces**: Detailed component specifications with code interfaces
4. **Data Models**: Define data structures and their relationships
5. **Security Considerations**: Security requirements and implementation approaches
6. **Testing Strategy**: Unit, integration, and system testing approaches

## Technical Guidelines
- Base designs ONLY on approved requirements and available memory context
- Use clear, implementable technical specifications
- Include practical code examples and interface definitions
- Generate valid Mermaid.js syntax for diagrams
- Consider scalability, maintainability, and performance
- Address error handling and edge cases
- Ensure designs are testable and verifiable

## Mermaid.js Diagram Standards
- Use appropriate diagram types (graph, flowchart, sequence, class)
- Include clear node labels and relationship descriptions
- Ensure diagrams are readable and well-organized
- Use consistent styling and naming conventions

## Integration Requirements
- Leverage memory context for technical patterns and best practices
- Ensure compatibility with existing system components
- Consider deployment and operational requirements
- Address integration points and dependencies

## Quality Standards
- Designs must be complete and implementable
- All requirements must be addressed in the design
- Technical decisions should be justified and documented
- Interfaces should be clearly defined with types and contracts
- Error handling and edge cases must be considered

Remember: Your designs serve as the foundation for implementation. They must be detailed enough for developers to implement without ambiguity while being flexible enough to accommodate reasonable implementation variations."""
    
    def get_context_requirements(self, task_input: Dict[str, Any]) -> Optional['ContextSpec']:
        """Define context requirements for DesignAgent."""
        if task_input.get("user_request"):
            from .base_agent import ContextSpec
            return ContextSpec(context_type="design")
        return None
    
    async def _process_task_impl(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a design task assigned to this agent.
        
        Args:
            task_input: Dictionary containing task parameters including:
                - requirements_path: Path to the requirements document (for new tasks)
                - work_directory: Working directory for the project
                - task_type: Type of task ("revision" for revisions)
                - revision_feedback: Feedback for revisions
                - current_result: Current phase result for revisions
                - memory_context: Optional additional memory context
                
        Returns:
            Dictionary containing task results and metadata
        """
        try:
            task_type = task_input.get("task_type", "design")
            
            if task_type == "revision":
                return await self._process_revision_task(task_input)
            else:
                return await self._process_design_task(task_input)
                
        except Exception as e:
            self.logger.error(f"Error processing design task: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process design task: {e}"
            }
    
    async def _process_design_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process a regular design task."""
        requirements_path = task_input.get("requirements_path")
        work_directory = task_input.get("work_directory")
        additional_context = task_input.get("memory_context", {})
        
        if not requirements_path or not work_directory:
            raise ValueError("requirements_path and work_directory are required")
        
        # Update context with additional information
        if additional_context:
            self.update_context(additional_context)
        
        # Generate the design document
        design_content = await self.generate_design(requirements_path, self.memory_context)
        
        # Save the design document
        design_path = os.path.join(work_directory, "design.md")
        with open(design_path, 'w', encoding='utf-8') as f:
            f.write(design_content)
        
        self.logger.info(f"Design document generated and saved to {design_path}")
        
        return {
            "success": True,
            "design_path": design_path,
            "design_content": design_content,
            "message": "Design document generated successfully"
        }
    
    async def _process_revision_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process a design revision task."""
        revision_feedback = task_input.get("revision_feedback", "")
        current_result = task_input.get("current_result", {})
        work_directory = task_input.get("work_directory", "")
        
        if not revision_feedback:
            raise ValueError("revision_feedback is required for revision tasks")
        
        if not work_directory:
            raise ValueError("work_directory is required for revision tasks")
        
        # Get current design path
        design_path = current_result.get("design_path")
        if not design_path or not os.path.exists(design_path):
            raise ValueError("Current design file not found")
        
        # Read current design
        with open(design_path, 'r', encoding='utf-8') as f:
            current_design = f.read()
        
        # Apply revision using LLM
        revised_design = await self._apply_design_revision(
            current_design, 
            revision_feedback
        )
        
        # Save revised design
        with open(design_path, 'w', encoding='utf-8') as f:
            f.write(revised_design)
        
        self.logger.info(f"Design revised based on feedback: {revision_feedback[:100]}...")
        
        return {
            "work_directory": work_directory,
            "design_path": design_path,
            "revision_applied": True,
            "success": True,
            "message": "Design revised successfully"
        }
    
    async def _apply_design_revision(self, current_design: str, revision_feedback: str) -> str:
        """
        Apply revision feedback to current design.
        
        Args:
            current_design: Current design document content
            revision_feedback: User's revision feedback
            
        Returns:
            Revised design document content
        """
        revision_prompt = f"""Please revise the following technical design document based on the user's feedback.

Current Design Document:
{current_design}

User Feedback:
{revision_feedback}

Please provide the complete revised design document that incorporates the user's feedback while maintaining the same structure and format. Make sure to:
1. Keep the same markdown structure and sections
2. Update technical specifications based on the feedback
3. Maintain valid Mermaid.js diagram syntax if present
4. Update component interfaces and data models as needed
5. Ensure all design decisions are justified and documented
6. Address any new requirements or constraints mentioned in the feedback

Revised Design Document:"""

        try:
            revised_content = await self.generate_response(revision_prompt)
            
            # Post-process the revised content
            revised_content = self._post_process_design(revised_content)
            
            return revised_content
            
        except Exception as e:
            self.logger.error(f"Error applying design revision: {e}")
            # Return original content with a note about the revision attempt
            return f"{current_design}\n\n<!-- Revision attempted but failed: {str(e)} -->"
            
        except Exception as e:
            self.logger.error(f"Error processing design task: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to generate design document: {e}"
            }
    
    async def generate_design(self, requirements_path: str, memory_context: Dict[str, Any]) -> str:
        """
        Generate a comprehensive technical design document based on requirements.
        
        Args:
            requirements_path: Path to the requirements.md file
            memory_context: Memory context from MemoryManager
            
        Returns:
            Complete design document content as markdown string
        """
        try:
            # Read the requirements document
            with open(requirements_path, 'r', encoding='utf-8') as f:
                requirements_content = f.read()
            
            # Prepare context for design generation
            design_context = {
                "requirements_content": requirements_content,
                "requirements_path": requirements_path,
                "memory_available": bool(memory_context),
                "memory_categories": list(memory_context.keys()) if memory_context else []
            }
            
            # Generate design using the LLM
            design_prompt = self._build_design_prompt(requirements_content, memory_context)
            design_content = await self.generate_response(design_prompt, design_context)
            
            # Post-process the design content
            processed_design = self._post_process_design(design_content)
            
            self.logger.info("Design document generated successfully")
            return processed_design
            
        except Exception as e:
            self.logger.error(f"Error generating design: {e}")
            raise
    
    def _build_design_prompt(self, requirements_content: str, memory_context: Dict[str, Any]) -> str:
        """
        Build the prompt for design generation.
        
        Args:
            requirements_content: Content of the requirements document
            memory_context: Available memory context
            
        Returns:
            Formatted prompt for design generation
        """
        prompt_parts = [
            "Please generate a comprehensive technical design document based on the following approved requirements.",
            "",
            "## Requirements Document",
            requirements_content,
            "",
            "## Task Instructions",
            "1. Create a complete design document following the standard structure",
            "2. Include architectural overview with clear design principles",
            "3. Generate Mermaid.js diagrams for data flow and system architecture",
            "4. Define all components with clear interfaces and code examples",
            "5. Specify data models and their relationships",
            "6. Address security considerations and requirements",
            "7. Define comprehensive testing strategies",
            "8. Ensure all requirements are addressed in the design",
            "",
            "## Design Requirements",
            "- Base the design ONLY on the provided requirements",
            "- Use memory context for technical patterns and best practices",
            "- Include practical, implementable specifications",
            "- Generate valid Mermaid.js syntax for all diagrams",
            "- Provide clear code interfaces and examples",
            "- Address error handling and edge cases",
            "- Ensure the design is testable and verifiable",
            "",
        ]
        
        # Add memory context information if available
        if memory_context:
            prompt_parts.extend([
                "## Available Memory Context",
                "You have access to the following categories of technical knowledge:",
                ""
            ])
            
            for category, content in memory_context.items():
                if isinstance(content, dict):
                    prompt_parts.append(f"- **{category}**: {len(content)} files with technical knowledge")
                else:
                    prompt_parts.append(f"- **{category}**: Technical knowledge available")
            
            prompt_parts.extend([
                "",
                "Use this memory context to inform your design decisions and leverage proven patterns.",
                ""
            ])
        
        prompt_parts.extend([
            "## Output Format",
            "Generate a complete design document in markdown format with:",
            "- Clear headings and structure",
            "- Valid Mermaid.js diagrams using proper syntax",
            "- Code examples with proper syntax highlighting",
            "- Comprehensive coverage of all requirements",
            "",
            "Begin generating the design document now:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _post_process_design(self, design_content: str) -> str:
        """
        Post-process the generated design content to ensure quality and consistency.
        
        Args:
            design_content: Raw design content from LLM
            
        Returns:
            Processed and validated design content
        """
        # Remove markdown code block wrapper if present
        design_content = self._extract_from_markdown_block(design_content)
        
        # Validate and fix Mermaid diagrams
        design_content = self._validate_mermaid_diagrams(design_content)
        
        # Ensure proper markdown formatting
        design_content = self._fix_markdown_formatting(design_content)
        
        # Add metadata header if not present
        if not design_content.startswith("# Design Document"):
            header = "# Design Document\n\n"
            design_content = header + design_content
        
        return design_content
    
    def _extract_from_markdown_block(self, content: str) -> str:
        """
        Extract content from markdown code block if it's wrapped in one.
        
        Args:
            content: Content that might be wrapped in markdown code block
            
        Returns:
            Extracted content without markdown wrapper
        """
        # Check if content starts with ```markdown and ends with ```
        if content.strip().startswith('```markdown') and content.strip().endswith('```'):
            # Extract content between the code block markers
            lines = content.strip().split('\n')
            if len(lines) > 2:
                # Remove first line (```markdown) and last line (```)
                extracted_content = '\n'.join(lines[1:-1])
                self.logger.info("Extracted design content from markdown code block")
                return extracted_content
        
        # Check for other markdown block patterns
        if content.strip().startswith('```') and content.strip().endswith('```'):
            lines = content.strip().split('\n')
            if len(lines) > 2:
                # Remove first and last lines
                extracted_content = '\n'.join(lines[1:-1])
                self.logger.info("Extracted design content from generic code block")
                return extracted_content
        
        return content
    
    def _validate_mermaid_diagrams(self, content: str) -> str:
        """
        Validate and fix Mermaid diagram syntax in the design content.
        
        Args:
            content: Design content with potential Mermaid diagrams
            
        Returns:
            Content with validated Mermaid diagrams
        """
        # Find all Mermaid code blocks
        mermaid_pattern = r'```mermaid\n(.*?)\n```'
        matches = re.findall(mermaid_pattern, content, re.DOTALL)
        
        for match in matches:
            # Basic validation - ensure diagram starts with a valid type
            diagram_content = match.strip()
            valid_types = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'erDiagram']
            
            if not any(diagram_content.startswith(diagram_type) for diagram_type in valid_types):
                # Try to fix by adding 'graph TD' if it looks like a graph
                if '-->' in diagram_content or '---' in diagram_content:
                    fixed_diagram = f"graph TD\n    {diagram_content}"
                    content = content.replace(match, fixed_diagram)
                    self.logger.info("Fixed Mermaid diagram by adding graph type")
        
        return content
    
    def _fix_markdown_formatting(self, content: str) -> str:
        """
        Fix common markdown formatting issues in the design content.
        
        Args:
            content: Design content with potential formatting issues
            
        Returns:
            Content with improved markdown formatting
        """
        # Ensure proper spacing around headers (handle headers at start of line)
        content = re.sub(r'(^|\n)(#{1,6})([^\s#][^\n]*)', r'\1\n\2 \3\n', content)
        content = re.sub(r'\n(#{1,6})\s*([^\n]+)\n', r'\n\n\1 \2\n\n', content)
        
        # Ensure code blocks have proper language specification
        content = re.sub(r'```\n(class |def |function |interface )', r'```python\n\1', content)
        content = re.sub(r'```\n(import |from )', r'```python\n\1', content)
        
        # Clean up excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def create_architecture_diagram(self, design_content: str) -> str:
        """
        Generate a Mermaid.js architecture diagram based on design content.
        
        This method can be used to create additional diagrams or extract
        diagram content from the design document.
        
        Args:
            design_content: The design document content
            
        Returns:
            Mermaid.js diagram code
        """
        try:
            # Extract existing Mermaid diagrams from the content
            mermaid_pattern = r'```mermaid\n(.*?)\n```'
            matches = re.findall(mermaid_pattern, design_content, re.DOTALL)
            
            if matches:
                # Return the first architecture-related diagram
                for match in matches:
                    if any(keyword in match.lower() for keyword in ['graph', 'flowchart', 'architecture']):
                        return match.strip()
                
                # If no architecture diagram found, return the first one
                return matches[0].strip()
            
            # If no diagrams found, create a basic one based on content analysis
            return self._generate_basic_architecture_diagram(design_content)
            
        except Exception as e:
            self.logger.error(f"Error creating architecture diagram: {e}")
            return "graph TD\n    A[System] --> B[Components]\n    B --> C[Implementation]"
    
    def _generate_basic_architecture_diagram(self, design_content: str) -> str:
        """
        Generate a basic architecture diagram by analyzing design content.
        
        Args:
            design_content: Design document content to analyze
            
        Returns:
            Basic Mermaid.js diagram
        """
        # Simple analysis to identify key components
        components = []
        
        # Look for class definitions or component mentions
        class_pattern = r'class\s+(\w+)'
        component_pattern = r'##\s+(\w+(?:\s+\w+)*)'
        
        classes = re.findall(class_pattern, design_content)
        sections = re.findall(component_pattern, design_content)
        
        components.extend(classes)
        components.extend([s.replace(' ', '') for s in sections if 'component' in s.lower()])
        
        if not components:
            components = ['System', 'Core', 'Interface']
        
        # Generate a simple flow diagram
        diagram_lines = ['graph TD']
        for i, component in enumerate(components[:5]):  # Limit to 5 components
            if i == 0:
                diagram_lines.append(f'    A[{component}]')
            else:
                prev_letter = chr(ord('A') + i - 1)
                curr_letter = chr(ord('A') + i)
                diagram_lines.append(f'    {prev_letter} --> {curr_letter}[{component}]')
        
        return '\n'.join(diagram_lines)
    
    def get_agent_capabilities(self) -> List[str]:
        """
        Get a list of capabilities that this agent provides.
        
        Returns:
            List of capability descriptions
        """
        return [
            "Generate comprehensive technical design documents",
            "Create architecture overviews and system designs",
            "Generate Mermaid.js diagrams for data flow and architecture",
            "Define component interfaces and specifications",
            "Specify data models and relationships",
            "Address security considerations and requirements",
            "Define testing strategies and approaches",
            "Integrate memory context for technical patterns",
            "Validate and fix diagram syntax",
            "Post-process design content for quality assurance"
        ]
    
    def get_design_templates(self) -> Dict[str, str]:
        """
        Get available design document templates.
        
        Returns:
            Dictionary mapping template names to their content
        """
        return {
            "web_application": self._get_web_app_template(),
            "api_service": self._get_api_service_template(),
            "data_processing": self._get_data_processing_template(),
            "multi_agent_system": self._get_multi_agent_template()
        }
    
    def _get_web_app_template(self) -> str:
        """Get template for web application design."""
        return """# Design Document

## Architectural Overview
[Web application architecture description]

## Data Flow Diagrams
```mermaid
graph TD
    A[Client] --> B[Web Server]
    B --> C[Application Logic]
    C --> D[Database]
```

## Components and Interfaces
### Frontend Components
### Backend Services
### Database Layer

## Data Models
### Entity Definitions
### Relationships

## Security Considerations
### Authentication
### Authorization
### Data Protection

## Testing Strategy
### Unit Testing
### Integration Testing
### End-to-End Testing
"""
    
    def _get_api_service_template(self) -> str:
        """Get template for API service design."""
        return """# Design Document

## Architectural Overview
[API service architecture description]

## Data Flow Diagrams
```mermaid
sequenceDiagram
    Client->>API: Request
    API->>Service: Process
    Service->>Database: Query
    Database-->>Service: Result
    Service-->>API: Response
    API-->>Client: Result
```

## Components and Interfaces
### API Endpoints
### Service Layer
### Data Access Layer

## Data Models
### Request/Response Models
### Domain Models

## Security Considerations
### API Authentication
### Rate Limiting
### Input Validation

## Testing Strategy
### API Testing
### Service Testing
### Integration Testing
"""
    
    def _get_data_processing_template(self) -> str:
        """Get template for data processing system design."""
        return """# Design Document

## Architectural Overview
[Data processing system architecture]

## Data Flow Diagrams
```mermaid
graph LR
    A[Input] --> B[Processing]
    B --> C[Transformation]
    C --> D[Output]
```

## Components and Interfaces
### Data Ingestion
### Processing Pipeline
### Output Generation

## Data Models
### Input Schemas
### Processing Models
### Output Formats

## Security Considerations
### Data Privacy
### Access Control
### Audit Logging

## Testing Strategy
### Data Validation Testing
### Pipeline Testing
### Performance Testing
"""
    
    def _get_multi_agent_template(self) -> str:
        """Get template for multi-agent system design."""
        return """# Design Document

## Architectural Overview
[Multi-agent system architecture]

## Data Flow Diagrams
```mermaid
graph TD
    A[Agent Manager] --> B[Agent 1]
    A --> C[Agent 2]
    A --> D[Agent 3]
    B --> E[Shared Memory]
    C --> E
    D --> E
```

## Components and Interfaces
### Agent Definitions
### Communication Protocols
### Coordination Mechanisms

## Data Models
### Agent States
### Message Formats
### Shared Data Structures

## Security Considerations
### Agent Authentication
### Message Security
### Resource Access Control

## Testing Strategy
### Agent Testing
### Communication Testing
### System Integration Testing
"""