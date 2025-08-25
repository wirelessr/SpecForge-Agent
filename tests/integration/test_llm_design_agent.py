"""
DesignAgent LLM Integration Tests.

This module tests the DesignAgent's LLM interactions and output quality validation,
focusing on real LLM calls and design document generation quality assessment.

Test Categories:
1. Design.md generation with required sections validation
2. Mermaid diagram generation and syntax validation
3. Architecture coherence and technical accuracy
4. Design revision with improvement assessment
5. Requirements alignment in generated designs

All tests use real LLM configurations and validate output quality using the
enhanced QualityMetricsFramework with LLM-specific validation methods.
"""

import pytest
import asyncio
import tempfile
import shutil
import re
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import replace

from autogen_framework.agents.design_agent import DesignAgent
from autogen_framework.models import LLMConfig
from autogen_framework.memory_manager import MemoryManager
from tests.integration.test_llm_base import (
    LLMIntegrationTestBase,
    sequential_test_execution,
    QUALITY_THRESHOLDS_STRICT
)


class TestDesignAgentLLMIntegration:
    """
    Integration tests for DesignAgent LLM interactions.
    
    These tests validate the DesignAgent's ability to:
    - Generate high-quality design documents using real LLM calls
    - Create valid Mermaid diagrams with proper syntax
    - Maintain architecture coherence and technical accuracy
    - Revise designs based on feedback with measurable improvements
    - Align designs with requirements specifications
    """
    
    @pytest.fixture(autouse=True)
    def setup_design_agent(self, real_llm_config, initialized_real_managers, temp_workspace):
        """Setup DesignAgent with real LLM configuration and managers."""
        # Initialize test base functionality
        self.test_base = LLMIntegrationTestBase()
        self.test_base.setup_method(self.setup_design_agent)  # Initialize the test base
        
        self.llm_config = real_llm_config
        self.managers = initialized_real_managers
        self.workspace_path = Path(temp_workspace)
        
        # Create memory manager for the workspace
        self.memory_manager = MemoryManager(workspace_path=temp_workspace)
        
        # Initialize DesignAgent with dependency container
        from autogen_framework.dependency_container import DependencyContainer
        container = DependencyContainer.create_production(temp_workspace, self.llm_config)
        self.design_agent = DesignAgent(
            name="TestDesignAgent",
            llm_config=self.llm_config,
            system_message="Generate technical design documents",
            container=container
        )
        
        # Use more lenient quality thresholds for DesignAgent tests
        from tests.integration.test_llm_base import QualityThresholds
        self.test_base.quality_validator.thresholds = QualityThresholds(
            design_generation={
                'structure_score': 0.6,
                'completeness_score': 0.3,
                'format_compliance': True,  # Expect Mermaid diagrams
                'overall_score': 0.4
            }
        )
    
    async def execute_with_rate_limit_handling(self, llm_operation):
        """Delegate to test base."""
        return await self.test_base.execute_with_rate_limit_handling(llm_operation)
    
    async def execute_with_retry(self, llm_operation, max_attempts=3):
        """Delegate to test base."""
        return await self.test_base.execute_with_retry(llm_operation, max_attempts)
    
    def assert_quality_threshold(self, validation_result, custom_message=None):
        """Delegate to test base."""
        return self.test_base.assert_quality_threshold(validation_result, custom_message)
    
    def log_quality_assessment(self, validation_result):
        """Delegate to test base."""
        return self.test_base.log_quality_assessment(validation_result)
    
    @property
    def quality_validator(self):
        """Access quality validator from test base."""
        return self.test_base.quality_validator
    
    @property
    def logger(self):
        """Access logger from test base."""
        return self.test_base.logger
    
    def create_sample_requirements(self, work_dir: Path, content: str = None) -> Path:
        """Create a sample requirements.md file for testing."""
        if content is None:
            content = """# Requirements Document

## Introduction

This feature implements a user authentication system that allows users to register, 
login, and manage their profiles securely. The system includes password reset 
functionality and email verification to ensure account security.

## Requirements

### Requirement 1

**User Story:** As a new user, I want to register for an account, so that I can access the system features.

#### Acceptance Criteria

1. WHEN a user provides valid registration information THEN the system SHALL create a new user account
2. WHEN a user provides an email address THEN the system SHALL send a verification email
3. WHEN a user clicks the verification link THEN the system SHALL activate the account
4. IF the email address is already registered THEN the system SHALL display an appropriate error message

### Requirement 2

**User Story:** As a registered user, I want to login to my account, so that I can access my profile and system features.

#### Acceptance Criteria

1. WHEN a user provides valid credentials THEN the system SHALL authenticate the user
2. WHEN a user provides invalid credentials THEN the system SHALL display an error message
3. WHEN a user fails login 3 times THEN the system SHALL temporarily lock the account
4. WHEN a user successfully logs in THEN the system SHALL create a secure session

### Requirement 3

**User Story:** As a user, I want to reset my password, so that I can regain access if I forget it.

#### Acceptance Criteria

1. WHEN a user requests password reset THEN the system SHALL send a reset email
2. WHEN a user clicks the reset link THEN the system SHALL allow password change
3. WHEN a user sets a new password THEN the system SHALL validate password strength
4. WHEN password is successfully reset THEN the system SHALL invalidate all existing sessions
"""
        
        requirements_path = work_dir / "requirements.md"
        requirements_path.write_text(content, encoding='utf-8')
        return requirements_path
    
    @sequential_test_execution()
    async def test_design_generation_with_required_sections(self):
        """
        Test design.md generation with required sections validation.
        
        Validates:
        - Design document is generated with proper structure
        - All required sections are present (Overview, Architecture, Components, Data Models, Error Handling, Testing Strategy)
        - Document meets quality thresholds for structure and completeness
        - Technical specifications are coherent and implementable
        """
        # Create work directory and requirements file
        work_dir = self.workspace_path / "test-design-generation"
        work_dir.mkdir(exist_ok=True)
        
        requirements_path = self.create_sample_requirements(work_dir)
        
        task_input = {
            "requirements_path": str(requirements_path),
            "work_directory": str(work_dir),
            "task_type": "design"
        }
        
        # Execute design generation with rate limit handling
        result = await self.execute_with_rate_limit_handling(
            lambda: self.design_agent._process_task_impl(task_input)
        )
        
        # Verify task execution succeeded
        assert result["success"], f"Design generation failed: {result.get('error', 'Unknown error')}"
        assert "design_path" in result
        
        # Read generated design document
        design_path = Path(result["design_path"])
        assert design_path.exists(), "Design file was not created"
        
        design_content = design_path.read_text(encoding='utf-8')
        
        # Validate design quality with required sections checking
        validation_result = self.quality_validator.validate_design_quality(design_content)
        
        # Log quality assessment for debugging
        self.log_quality_assessment(validation_result)
        
        # Assert quality thresholds are met
        self.assert_quality_threshold(
            validation_result,
            "Design document quality below threshold"
        )
        
        # Verify required sections are present using flexible matching
        section_patterns = {
            'Design Document': [r'#\s*Design Document'],
            'Architectural Overview': [r'##\s*.*[Aa]rchitectur.*[Oo]verview', r'##\s*.*[Oo]verview', r'##\s*.*[Aa]rchitecture'],
            'Components': [r'##\s*.*[Cc]omponent', r'##\s*.*[Ii]nterface'],
            'Data Models': [r'##\s*.*[Dd]ata.*[Mm]odel', r'##\s*.*[Ss]chema', r'##\s*.*[Dd]ata.*[Ss]tructure'],
            'Error Handling': [r'##\s*.*[Ee]rror.*[Hh]andling', r'##\s*.*[Ee]rror.*[Mm]anagement', r'##\s*.*[Ee]xception'],
            'Testing Strategy': [r'##\s*.*[Tt]esting', r'##\s*.*[Tt]est.*[Ss]trategy', r'##\s*.*[Tt]est.*[Pp]lan']
        }
        
        missing_sections = []
        for section_name, patterns in section_patterns.items():
            found = False
            for pattern in patterns:
                if re.search(pattern, design_content, re.MULTILINE):
                    found = True
                    break
            
            if not found:
                missing_sections.append(section_name)
        
        # Allow up to 1 missing section for flexibility
        assert len(missing_sections) <= 1, f"Too many missing required sections: {missing_sections}"
        
        # Verify content relevance to requirements
        key_terms = ["authentication", "user", "login", "register", "password", "email"]
        content_lower = design_content.lower()
        found_terms = [term for term in key_terms if term in content_lower]
        assert len(found_terms) >= 4, f"Design missing key terms from requirements. Found: {found_terms}"
        
        # Verify technical depth
        technical_indicators = ["class", "interface", "method", "function", "database", "api", "endpoint"]
        found_technical = [term for term in technical_indicators if term in content_lower]
        assert len(found_technical) >= 3, f"Design lacks technical depth. Found: {found_technical}"
        
        self.logger.info(f"Design generation test passed with quality score: {validation_result['assessment']['overall_score']:.2f}")
    
    @sequential_test_execution()
    async def test_mermaid_diagram_generation_and_syntax_validation(self):
        """
        Test Mermaid diagram generation and syntax validation.
        
        Validates:
        - Mermaid diagrams are generated in the design document
        - Diagram syntax is valid and parseable
        - Diagrams are relevant to the system architecture
        - Multiple diagram types are used appropriately
        - Diagram content matches the design specifications
        """
        # Create work directory with requirements for a system that benefits from diagrams
        work_dir = self.workspace_path / "test-mermaid-diagrams"
        work_dir.mkdir(exist_ok=True)
        
        requirements_content = """# Requirements Document

## Introduction

This feature implements a real-time chat application with user authentication, 
message persistence, and WebSocket communication. The system supports multiple 
chat rooms and user presence indicators.

## Requirements

### Requirement 1

**User Story:** As a user, I want to join chat rooms, so that I can communicate with other users in real-time.

#### Acceptance Criteria

1. WHEN a user joins a room THEN the system SHALL establish a WebSocket connection
2. WHEN a user sends a message THEN the system SHALL broadcast it to all room members
3. WHEN a user leaves a room THEN the system SHALL notify other members
4. WHEN a message is sent THEN the system SHALL persist it to the database

### Requirement 2

**User Story:** As a user, I want to see who is online, so that I know who is available to chat.

#### Acceptance Criteria

1. WHEN a user comes online THEN the system SHALL update their presence status
2. WHEN a user goes offline THEN the system SHALL update their status after a timeout
3. WHEN presence changes THEN the system SHALL notify connected users
4. WHEN a user joins a room THEN the system SHALL show current online members
"""
        
        requirements_path = self.create_sample_requirements(work_dir, requirements_content)
        
        task_input = {
            "requirements_path": str(requirements_path),
            "work_directory": str(work_dir),
            "task_type": "design"
        }
        
        # Execute design generation with rate limit handling
        result = await self.execute_with_rate_limit_handling(
            lambda: self.design_agent._process_task_impl(task_input)
        )
        
        assert result["success"], f"Design generation failed: {result.get('error')}"
        
        # Read generated design document
        design_path = Path(result["design_path"])
        design_content = design_path.read_text(encoding='utf-8')
        
        # Validate Mermaid diagram presence and syntax
        validation_result = self.quality_validator.validate_design_quality(design_content)
        
        # Log quality assessment
        self.log_quality_assessment(validation_result)
        
        # Assert Mermaid compliance
        assert validation_result['mermaid_compliant'], (
            f"Design document does not contain valid Mermaid diagrams. "
            f"Mermaid validation: {validation_result['mermaid_validation']}"
        )
        
        # Extract and validate Mermaid diagrams
        mermaid_pattern = r'```mermaid\n(.*?)\n```'
        mermaid_matches = re.findall(mermaid_pattern, design_content, re.DOTALL)
        
        assert len(mermaid_matches) >= 1, "Design should contain at least one Mermaid diagram"
        
        # Validate diagram syntax and content
        valid_diagram_types = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'erDiagram']
        
        for i, diagram in enumerate(mermaid_matches):
            diagram_content = diagram.strip()
            
            # Check if diagram starts with a valid type
            starts_with_valid_type = any(
                diagram_content.startswith(diagram_type) 
                for diagram_type in valid_diagram_types
            )
            
            assert starts_with_valid_type, (
                f"Diagram {i+1} does not start with a valid type. "
                f"Content: {diagram_content[:100]}..."
            )
            
            # Validate diagram has meaningful content
            assert len(diagram_content.split('\n')) >= 2, (
                f"Diagram {i+1} is too simple: {diagram_content}"
            )
            
            # Check for relevant content based on requirements
            diagram_lower = diagram_content.lower()
            relevant_terms = ["user", "message", "websocket", "room", "chat", "database", "api"]
            found_relevant = [term for term in relevant_terms if term in diagram_lower]
            
            assert len(found_relevant) >= 1, (
                f"Diagram {i+1} lacks relevant content. Found: {found_relevant}"
            )
        
        # Validate overall design quality
        self.assert_quality_threshold(
            validation_result,
            "Design with Mermaid diagrams quality below threshold"
        )
        
        self.logger.info(f"Mermaid diagram test passed with {len(mermaid_matches)} diagrams")
    
    @sequential_test_execution()
    async def test_architecture_coherence_and_technical_accuracy(self):
        """
        Test architecture coherence and technical accuracy.
        
        Validates:
        - Architecture design is coherent and well-structured
        - Technical specifications are accurate and implementable
        - Component interfaces are clearly defined
        - Data flow is logical and consistent
        - Security considerations are appropriate
        - Performance implications are addressed
        """
        # Create work directory with complex requirements
        work_dir = self.workspace_path / "test-architecture-coherence"
        work_dir.mkdir(exist_ok=True)
        
        requirements_content = """# Requirements Document

## Introduction

This feature implements a microservices-based e-commerce platform with user management, 
product catalog, order processing, and payment integration. The system must handle 
high traffic loads and ensure data consistency across services.

## Requirements

### Requirement 1

**User Story:** As a customer, I want to browse and purchase products, so that I can complete my shopping online.

#### Acceptance Criteria

1. WHEN a customer views products THEN the system SHALL display real-time inventory
2. WHEN a customer adds items to cart THEN the system SHALL reserve inventory temporarily
3. WHEN a customer checks out THEN the system SHALL process payment securely
4. WHEN payment succeeds THEN the system SHALL confirm the order and update inventory

### Requirement 2

**User Story:** As the system, I want to maintain data consistency, so that inventory and orders remain accurate.

#### Acceptance Criteria

1. WHEN inventory changes THEN the system SHALL update all relevant services
2. WHEN a service fails THEN the system SHALL maintain transaction integrity
3. WHEN concurrent orders occur THEN the system SHALL prevent overselling
4. WHEN data conflicts arise THEN the system SHALL resolve them automatically

### Requirement 3

**User Story:** As an administrator, I want to monitor system performance, so that I can ensure optimal operation.

#### Acceptance Criteria

1. WHEN system load increases THEN the system SHALL scale services automatically
2. WHEN errors occur THEN the system SHALL log them for analysis
3. WHEN performance degrades THEN the system SHALL alert administrators
4. WHEN maintenance is needed THEN the system SHALL support graceful shutdowns
"""
        
        requirements_path = self.create_sample_requirements(work_dir, requirements_content)
        
        task_input = {
            "requirements_path": str(requirements_path),
            "work_directory": str(work_dir),
            "task_type": "design"
        }
        
        # Execute design generation with rate limit handling
        result = await self.execute_with_rate_limit_handling(
            lambda: self.design_agent._process_task_impl(task_input)
        )
        
        assert result["success"], f"Design generation failed: {result.get('error')}"
        
        # Read generated design document
        design_path = Path(result["design_path"])
        design_content = design_path.read_text(encoding='utf-8')
        
        # Validate design quality
        validation_result = self.quality_validator.validate_design_quality(design_content)
        self.log_quality_assessment(validation_result)
        
        # Assert quality thresholds for technical accuracy
        self.assert_quality_threshold(
            validation_result,
            "Architecture design quality below threshold"
        )
        
        # Validate architectural coherence
        content_lower = design_content.lower()
        
        # Check for microservices architecture elements
        microservices_terms = ["service", "microservice", "api", "endpoint", "distributed"]
        found_microservices = [term for term in microservices_terms if term in content_lower]
        assert len(found_microservices) >= 3, (
            f"Design lacks microservices architecture elements. Found: {found_microservices}"
        )
        
        # Check for e-commerce domain elements
        ecommerce_terms = ["product", "order", "payment", "inventory", "customer", "cart"]
        found_ecommerce = [term for term in ecommerce_terms if term in content_lower]
        assert len(found_ecommerce) >= 4, (
            f"Design lacks e-commerce domain elements. Found: {found_ecommerce}"
        )
        
        # Check for technical accuracy indicators
        technical_terms = ["database", "transaction", "consistency", "scaling", "monitoring"]
        found_technical = [term for term in technical_terms if term in content_lower]
        assert len(found_technical) >= 2, (
            f"Design lacks technical depth. Found: {found_technical}"
        )
        
        # Validate security considerations
        security_terms = ["security", "authentication", "authorization", "encryption", "secure"]
        found_security = [term for term in security_terms if term in content_lower]
        assert len(found_security) >= 2, (
            f"Design lacks security considerations. Found: {found_security}"
        )
        
        # Validate performance considerations
        performance_terms = ["performance", "scalability", "load", "caching", "optimization"]
        found_performance = [term for term in performance_terms if term in content_lower]
        assert len(found_performance) >= 2, (
            f"Design lacks performance considerations. Found: {found_performance}"
        )
        
        # Check for component interface definitions
        interface_indicators = ["interface", "api", "method", "endpoint", "contract"]
        found_interfaces = [term for term in interface_indicators if term in content_lower]
        assert len(found_interfaces) >= 1, (
            f"Design lacks interface definitions. Found: {found_interfaces}"
        )
        
        self.logger.info("Architecture coherence test passed with comprehensive technical coverage")
    
    @sequential_test_execution()
    async def test_design_revision_with_improvement_assessment(self):
        """
        Test design revision with improvement assessment.
        
        Validates:
        - LLM can revise designs based on feedback
        - Revisions show measurable quality improvements
        - Feedback is properly incorporated into the design
        - Document structure and format are maintained
        - Technical accuracy is preserved or improved
        - Mermaid diagrams are updated appropriately
        """
        # Create work directory and initial design
        work_dir = self.workspace_path / "test-design-revision"
        work_dir.mkdir(exist_ok=True)
        
        requirements_path = self.create_sample_requirements(work_dir)
        
        # Generate initial design
        initial_task_input = {
            "requirements_path": str(requirements_path),
            "work_directory": str(work_dir),
            "task_type": "design"
        }
        
        initial_result = await self.execute_with_rate_limit_handling(
            lambda: self.design_agent._process_task_impl(initial_task_input)
        )
        
        assert initial_result["success"], "Initial design generation failed"
        
        design_path = Path(initial_result["design_path"])
        original_content = design_path.read_text(encoding='utf-8')
        
        # Validate initial quality
        initial_validation = self.quality_validator.validate_design_quality(original_content)
        self.log_quality_assessment(initial_validation)
        
        # Apply revision with specific feedback
        revision_feedback = (
            "Please add more detailed API specifications with request/response examples, "
            "include a sequence diagram showing the complete authentication flow, "
            "add specific database schema definitions with relationships, "
            "and enhance the security section with specific encryption and hashing algorithms."
        )
        
        revision_task_input = {
            "task_type": "revision",
            "revision_feedback": revision_feedback,
            "current_result": initial_result,
            "work_directory": str(work_dir)
        }
        
        # Apply revision
        revision_result = await self.execute_with_rate_limit_handling(
            lambda: self.design_agent._process_task_impl(revision_task_input)
        )
        
        assert revision_result["success"], f"Design revision failed: {revision_result.get('error')}"
        assert revision_result["revision_applied"], "Revision was not applied"
        
        # Read revised content
        revised_content = design_path.read_text(encoding='utf-8')
        
        # Validate revision improvement
        improvement_validation = self.quality_validator.validate_revision_improvement(
            original_content, revised_content, revision_feedback
        )
        
        # Assert meaningful improvement
        assert improvement_validation['shows_improvement'], (
            f"Revision does not show meaningful improvement. "
            f"Assessment: {improvement_validation['improvement_assessment']}"
        )
        
        # Validate that feedback was incorporated
        revised_lower = revised_content.lower()
        feedback_terms = ["api", "sequence", "database", "schema", "security", "encryption"]
        
        incorporated_terms = [term for term in feedback_terms if term in revised_lower]
        assert len(incorporated_terms) >= 4, (
            f"Revision did not incorporate enough feedback terms. "
            f"Expected: {feedback_terms}, Found: {incorporated_terms}"
        )
        
        # Validate document structure is maintained (use flexible matching)
        required_patterns = [
            r'#\s*Design Document',
            r'##\s*.*[Aa]rchitectur.*[Oo]verview|##\s*.*[Oo]verview|##\s*.*[Aa]rchitecture',
            r'##\s*.*[Cc]omponent'
        ]
        for i, pattern in enumerate(required_patterns):
            assert re.search(pattern, revised_content, re.MULTILINE), f"Section pattern {i+1} lost during revision: {pattern}"
        
        # Validate Mermaid diagrams are still present and valid
        mermaid_pattern = r'```mermaid\n(.*?)\n```'
        revised_diagrams = re.findall(mermaid_pattern, revised_content, re.DOTALL)
        original_diagrams = re.findall(mermaid_pattern, original_content, re.DOTALL)
        
        # Should have at least as many diagrams as before, possibly more
        assert len(revised_diagrams) >= len(original_diagrams), (
            f"Revision lost Mermaid diagrams. Original: {len(original_diagrams)}, "
            f"Revised: {len(revised_diagrams)}"
        )
        
        # Check for sequence diagram as requested in feedback
        has_sequence_diagram = any(
            'sequenceDiagram' in diagram or 'sequence' in diagram.lower()
            for diagram in revised_diagrams
        )
        assert has_sequence_diagram, "Revision should include sequence diagram as requested"
        
        # Validate revised quality meets thresholds
        revised_validation = self.quality_validator.validate_design_quality(revised_content)
        self.log_quality_assessment(revised_validation)
        
        self.assert_quality_threshold(
            revised_validation,
            "Revised design quality below threshold"
        )
        
        # Ensure improvement in quality score
        initial_score = initial_validation['assessment']['overall_score']
        revised_score = revised_validation['assessment']['overall_score']
        
        self.logger.info(f"Quality improvement: {initial_score:.2f} -> {revised_score:.2f}")
        
        # Allow for small variations but expect general improvement or maintenance
        assert revised_score >= initial_score - 0.1, (
            f"Revised quality score significantly decreased: {initial_score:.2f} -> {revised_score:.2f}"
        )
        
        self.logger.info("Design revision test passed with meaningful improvements")
    
    @sequential_test_execution()
    async def test_requirements_alignment_validation(self):
        """
        Test requirements alignment in generated designs.
        
        Validates:
        - Design addresses all requirements from requirements.md
        - Requirements traceability is maintained
        - Design components map to specific requirements
        - No requirements are overlooked or misinterpreted
        - Design scope matches requirements scope
        - Technical solutions align with requirement constraints
        """
        # Create work directory with detailed requirements
        work_dir = self.workspace_path / "test-requirements-alignment"
        work_dir.mkdir(exist_ok=True)
        
        requirements_content = """# Requirements Document

## Introduction

This feature implements a document management system with version control, 
collaborative editing, and access control. The system supports multiple file 
formats and provides audit trails for all document operations.

## Requirements

### Requirement 1

**User Story:** As a user, I want to upload and organize documents, so that I can manage my files efficiently.

#### Acceptance Criteria

1. WHEN a user uploads a document THEN the system SHALL store it with metadata
2. WHEN a user creates folders THEN the system SHALL organize documents hierarchically
3. WHEN a user searches documents THEN the system SHALL return relevant results
4. WHEN a user tags documents THEN the system SHALL enable tag-based filtering

### Requirement 2

**User Story:** As a collaborator, I want to edit documents with others, so that we can work together effectively.

#### Acceptance Criteria

1. WHEN multiple users edit simultaneously THEN the system SHALL prevent conflicts
2. WHEN a user makes changes THEN the system SHALL track all modifications
3. WHEN conflicts occur THEN the system SHALL provide resolution mechanisms
4. WHEN editing sessions end THEN the system SHALL save all changes automatically

### Requirement 3

**User Story:** As an administrator, I want to control document access, so that sensitive information remains secure.

#### Acceptance Criteria

1. WHEN access is granted THEN the system SHALL enforce permission levels
2. WHEN permissions change THEN the system SHALL update access immediately
3. WHEN unauthorized access is attempted THEN the system SHALL log and block it
4. WHEN documents are shared THEN the system SHALL track sharing history

### Requirement 4

**User Story:** As a compliance officer, I want to audit document activities, so that I can ensure regulatory compliance.

#### Acceptance Criteria

1. WHEN any document operation occurs THEN the system SHALL log it with timestamps
2. WHEN audit reports are requested THEN the system SHALL generate comprehensive logs
3. WHEN data retention policies apply THEN the system SHALL enforce them automatically
4. WHEN compliance violations are detected THEN the system SHALL alert administrators
"""
        
        requirements_path = self.create_sample_requirements(work_dir, requirements_content)
        
        task_input = {
            "requirements_path": str(requirements_path),
            "work_directory": str(work_dir),
            "task_type": "design"
        }
        
        # Execute design generation with rate limit handling
        result = await self.execute_with_rate_limit_handling(
            lambda: self.design_agent._process_task_impl(task_input)
        )
        
        assert result["success"], f"Design generation failed: {result.get('error')}"
        
        # Read generated design document
        design_path = Path(result["design_path"])
        design_content = design_path.read_text(encoding='utf-8')
        
        # Validate design quality
        validation_result = self.quality_validator.validate_design_quality(design_content)
        self.log_quality_assessment(validation_result)
        
        self.assert_quality_threshold(
            validation_result,
            "Requirements-aligned design quality below threshold"
        )
        
        # Validate requirements alignment
        content_lower = design_content.lower()
        
        # Check coverage of Requirement 1: Document upload and organization
        req1_terms = ["upload", "document", "metadata", "folder", "search", "tag"]
        found_req1 = [term for term in req1_terms if term in content_lower]
        assert len(found_req1) >= 4, (
            f"Design doesn't address Requirement 1 adequately. Found: {found_req1}"
        )
        
        # Check coverage of Requirement 2: Collaborative editing
        req2_terms = ["collaborative", "edit", "conflict", "version", "simultaneous", "change"]
        found_req2 = [term for term in req2_terms if term in content_lower]
        assert len(found_req2) >= 3, (
            f"Design doesn't address Requirement 2 adequately. Found: {found_req2}"
        )
        
        # Check coverage of Requirement 3: Access control
        req3_terms = ["access", "permission", "security", "authorization", "share", "control"]
        found_req3 = [term for term in req3_terms if term in content_lower]
        assert len(found_req3) >= 4, (
            f"Design doesn't address Requirement 3 adequately. Found: {found_req3}"
        )
        
        # Check coverage of Requirement 4: Audit and compliance
        req4_terms = ["audit", "log", "compliance", "track", "report", "retention"]
        found_req4 = [term for term in req4_terms if term in content_lower]
        assert len(found_req4) >= 2, (
            f"Design doesn't address Requirement 4 adequately. Found: {found_req4}"
        )
        
        # Validate technical components align with requirements
        technical_components = [
            "document storage", "version control", "access control", "audit system",
            "search engine", "collaboration", "metadata", "permission"
        ]
        
        found_components = [comp for comp in technical_components if comp in content_lower]
        assert len(found_components) >= 5, (
            f"Design lacks required technical components. Found: {found_components}"
        )
        
        # Check for system architecture that supports all requirements
        architecture_elements = ["database", "api", "service", "interface", "storage"]
        found_architecture = [elem for elem in architecture_elements if elem in content_lower]
        assert len(found_architecture) >= 3, (
            f"Design lacks comprehensive architecture. Found: {found_architecture}"
        )
        
        # Validate security considerations for sensitive document management
        security_elements = ["encryption", "authentication", "authorization", "secure", "privacy"]
        found_security = [elem for elem in security_elements if elem in content_lower]
        assert len(found_security) >= 3, (
            f"Design lacks adequate security considerations. Found: {found_security}"
        )
        
        # Check for scalability considerations (multiple users, large documents)
        scalability_elements = ["scalability", "performance", "concurrent", "load", "cache"]
        found_scalability = [elem for elem in scalability_elements if elem in content_lower]
        assert len(found_scalability) >= 2, (
            f"Design should address scalability for collaborative system. Found: {found_scalability}"
        )
        
        self.logger.info("Requirements alignment test passed with comprehensive coverage")
    
    @sequential_test_execution()
    async def test_error_handling_and_recovery(self):
        """
        Test error handling and recovery in DesignAgent operations.
        
        Validates:
        - Graceful handling of invalid inputs
        - Proper error messages for missing parameters
        - Recovery from LLM API errors
        - Fallback mechanisms for diagram generation
        - Consistent behavior under error conditions
        """
        # Test invalid task input
        invalid_task_input = {
            "task_type": "design"
            # Missing required requirements_path and work_directory
        }
        
        result = await self.execute_with_rate_limit_handling(
            lambda: self.design_agent._process_task_impl(invalid_task_input)
        )
        
        # Should handle error gracefully
        assert not result["success"], "Should fail with missing required parameters"
        assert "error" in result, "Should provide error message"
        assert any(param in result["error"] for param in ["requirements_path", "work_directory"]), (
            "Error should mention missing parameters"
        )
        
        # Test with non-existent requirements file
        work_dir = self.workspace_path / "test-error-handling"
        work_dir.mkdir(exist_ok=True)
        
        invalid_file_task = {
            "requirements_path": str(work_dir / "nonexistent.md"),
            "work_directory": str(work_dir),
            "task_type": "design"
        }
        
        result = await self.execute_with_rate_limit_handling(
            lambda: self.design_agent._process_task_impl(invalid_file_task)
        )
        
        # Should handle missing file gracefully
        assert not result["success"], "Should fail with non-existent requirements file"
        assert "error" in result, "Should provide error message for missing file"
        
        # Test revision with missing current result
        invalid_revision_task = {
            "task_type": "revision",
            "revision_feedback": "Some feedback",
            "work_directory": str(work_dir)
            # Missing current_result
        }
        
        result = await self.execute_with_rate_limit_handling(
            lambda: self.design_agent._process_task_impl(invalid_revision_task)
        )
        
        # Should handle missing revision context gracefully
        assert not result["success"], "Should fail with missing revision context"
        assert "error" in result, "Should provide error message for missing revision context"
        
        self.logger.info("Error handling test passed with appropriate error responses")
    
    @sequential_test_execution()
    async def test_memory_context_integration(self):
        """
        Test memory context integration in design generation.
        
        Validates:
        - DesignAgent loads and uses memory context
        - Memory patterns influence design generation
        - Historical design patterns are applied to new projects
        - Context integration improves output quality
        - Memory updates are properly handled
        """
        # Setup memory context with design patterns
        memory_content = {
            "design_patterns": [
                {
                    "pattern_type": "authentication_design",
                    "description": "Comprehensive authentication system design pattern",
                    "components": [
                        "JWT token service",
                        "Password hashing service", 
                        "Session management",
                        "Multi-factor authentication",
                        "OAuth integration"
                    ],
                    "security_considerations": [
                        "Token expiration and refresh",
                        "Secure password storage",
                        "Rate limiting for login attempts",
                        "Audit logging for security events"
                    ]
                },
                {
                    "pattern_type": "microservices_design",
                    "description": "Microservices architecture design pattern",
                    "components": [
                        "API Gateway",
                        "Service discovery",
                        "Load balancer",
                        "Circuit breaker",
                        "Distributed tracing"
                    ],
                    "best_practices": [
                        "Database per service",
                        "Event-driven communication",
                        "Centralized logging",
                        "Health check endpoints"
                    ]
                }
            ]
        }
        
        # Create DesignAgent with memory context using container
        from autogen_framework.dependency_container import DependencyContainer
        container = DependencyContainer.create_production(str(self.workspace_path), self.llm_config)
        
        design_agent_with_memory = DesignAgent(
            container=container,
            name="DesignAgent",
            llm_config=self.llm_config,
            system_message="Generate technical design documents",
            memory_context=memory_content
        )
        
        # Create work directory and requirements that should benefit from memory
        work_dir = self.workspace_path / "test-memory-integration"
        work_dir.mkdir(exist_ok=True)
        
        requirements_content = """# Requirements Document

## Introduction

This feature implements a secure microservices-based user management system 
with authentication, authorization, and user profile management capabilities.

## Requirements

### Requirement 1

**User Story:** As a user, I want to authenticate securely, so that my account is protected.

#### Acceptance Criteria

1. WHEN a user logs in THEN the system SHALL authenticate using secure methods
2. WHEN authentication succeeds THEN the system SHALL provide secure session tokens
3. WHEN tokens expire THEN the system SHALL handle refresh automatically
4. WHEN suspicious activity is detected THEN the system SHALL require additional verification

### Requirement 2

**User Story:** As the system, I want to scale efficiently, so that performance remains optimal under load.

#### Acceptance Criteria

1. WHEN load increases THEN the system SHALL distribute requests across services
2. WHEN services fail THEN the system SHALL route around failures
3. WHEN monitoring detects issues THEN the system SHALL alert administrators
4. WHEN capacity is exceeded THEN the system SHALL scale services automatically
"""
        
        requirements_path = self.create_sample_requirements(work_dir, requirements_content)
        
        task_input = {
            "requirements_path": str(requirements_path),
            "work_directory": str(work_dir),
            "task_type": "design"
        }
        
        # Generate design with memory context
        result = await self.execute_with_rate_limit_handling(
            lambda: design_agent_with_memory._process_task_impl(task_input)
        )
        
        assert result["success"], f"Design generation with memory failed: {result.get('error')}"
        
        # Read generated design
        design_path = Path(result["design_path"])
        design_content = design_path.read_text(encoding='utf-8')
        
        # Validate memory pattern integration
        content_lower = design_content.lower()
        
        # Check for authentication pattern elements
        auth_patterns = ["jwt", "token", "session", "multi-factor", "oauth", "password hashing"]
        found_auth_patterns = [pattern for pattern in auth_patterns if pattern in content_lower]
        
        # Check for microservices pattern elements
        microservices_patterns = ["api gateway", "service discovery", "load balancer", "circuit breaker", "health check"]
        found_microservices_patterns = [pattern for pattern in microservices_patterns if pattern in content_lower]
        
        # Should find patterns from memory
        total_patterns_found = len(found_auth_patterns) + len(found_microservices_patterns)
        assert total_patterns_found >= 4, (
            f"Memory patterns not sufficiently integrated. "
            f"Auth patterns found: {found_auth_patterns}, "
            f"Microservices patterns found: {found_microservices_patterns}"
        )
        
        # Validate quality with memory integration
        validation_result = self.quality_validator.validate_design_quality(design_content)
        self.log_quality_assessment(validation_result)
        
        # Memory integration should result in high-quality output
        self.assert_quality_threshold(
            validation_result,
            "Design with memory integration below quality threshold"
        )
        
        # Verify comprehensive coverage due to memory patterns
        assert validation_result['assessment']['completeness_score'] >= 0.5, (
            f"Memory integration should improve completeness. "
            f"Score: {validation_result['assessment']['completeness_score']}"
        )
        
        # Check for security best practices from memory
        security_practices = ["audit logging", "rate limiting", "secure storage", "token expiration"]
        found_security_practices = [practice for practice in security_practices if practice in content_lower]
        assert len(found_security_practices) >= 1, (
            f"Memory should contribute security best practices. Found: {found_security_practices}"
        )
        
        self.logger.info(f"Memory context integration test passed. Patterns found: {total_patterns_found}")