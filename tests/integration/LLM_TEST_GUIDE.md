# LLM Integration Test Guide

## Overview

This guide provides comprehensive documentation for LLM integration tests in the AutoGen multi-agent framework. These tests focus specifically on validating LLM interactions and output quality rather than functional testing, ensuring our agents produce high-quality outputs using real LLM calls.

## Test Organization Structure

### Test Categories by LLM Interaction Patterns

```
tests/integration/
├── LLM-focused tests (test_llm_*.py)
│   ├── Document Generation Tests
│   │   ├── test_llm_plan_agent.py          # Requirements.md generation
│   │   ├── test_llm_design_agent.py        # Design.md generation  
│   │   ├── test_llm_tasks_agent.py         # Tasks.md generation
│   │   └── test_llm_document_consistency.py # Cross-document alignment
│   ├── Intelligent Operations Tests
│   │   ├── test_llm_task_decomposer.py     # Task breakdown intelligence
│   │   ├── test_llm_error_recovery.py      # Error analysis & strategy generation
│   │   └── test_llm_interactive_features.py # Command enhancement & feedback
│   └── Context Management Tests
│       ├── test_llm_context_compressor.py  # Content summarization
│       └── test_llm_memory_integration.py  # Memory & system message building
├── Infrastructure
│   └── test_llm_base.py                    # Base classes & utilities
└── Legacy integration tests (non-LLM focused)
    ├── test_real_*.py                      # Real component integration
    ├── test_*_integration.py               # Component interaction tests
    └── test_*_workflow_*.py                # Workflow integration tests
```

## Test Naming Conventions

### LLM Test File Naming Pattern
- **Format**: `test_llm_{component}_{capability}.py`
- **Examples**:
  - `test_llm_plan_agent.py` - PlanAgent LLM interactions
  - `test_llm_error_recovery.py` - ErrorRecovery LLM intelligence
  - `test_llm_document_consistency.py` - Cross-document LLM validation

### Test Method Naming Pattern
- **Format**: `test_{llm_behavior}_{validation_aspect}`
- **Examples**:
  - `test_requirements_generation_ears_format_compliance()`
  - `test_error_analysis_categorization_accuracy()`
  - `test_design_revision_improvement_assessment()`

### Test Class Naming Pattern
- **Format**: `Test{Component}LLMIntegration`
- **Examples**:
  - `TestPlanAgentLLMIntegration`
  - `TestErrorRecoveryLLMIntegration`
  - `TestDocumentConsistencyLLMValidation`

## LLM Interaction Categories

### 1. Document Generation Tests

#### Purpose
Validate that agents can generate high-quality structured documents using LLM calls.

#### Test Files
- `test_llm_plan_agent.py` - Requirements document generation
- `test_llm_design_agent.py` - Design document generation
- `test_llm_tasks_agent.py` - Task list generation
- `test_llm_document_consistency.py` - Cross-document alignment

#### Key LLM Behaviors Tested
- **Content Structure**: Proper formatting, sections, hierarchical organization
- **Format Compliance**: EARS format for requirements, Mermaid syntax for designs
- **Content Quality**: Completeness, coherence, technical accuracy
- **Revision Capability**: Meaningful improvements based on feedback
- **Context Integration**: Proper use of memory and project context

#### Example Test Structure
```python
class TestPlanAgentLLMIntegration:
    """Tests PlanAgent's LLM-based document generation capabilities."""
    
    async def test_requirements_generation_ears_format_compliance(self):
        """Validate that LLM generates requirements in proper EARS format."""
        # Test LLM's ability to structure requirements correctly
        
    async def test_requirements_revision_improvement_assessment(self):
        """Validate that LLM can meaningfully improve requirements based on feedback."""
        # Test LLM's ability to understand and implement feedback
```

### 2. Intelligent Operations Tests

#### Purpose
Validate that agents can perform intelligent analysis and decision-making using LLM calls.

#### Test Files
- `test_llm_task_decomposer.py` - Task breakdown intelligence
- `test_llm_error_recovery.py` - Error analysis and strategy generation
- `test_llm_interactive_features.py` - Command enhancement and feedback processing

#### Key LLM Behaviors Tested
- **Analysis Accuracy**: Correct categorization and root cause identification
- **Strategy Generation**: Viable alternative approaches with logical ranking
- **Complexity Assessment**: Accurate difficulty evaluation with confidence scoring
- **Pattern Recognition**: Learning from successful recovery patterns
- **Decision Making**: Context-aware choices and conditional logic

#### Example Test Structure
```python
class TestTaskDecomposerLLMIntegration:
    """Tests TaskDecomposer's LLM-based intelligent task breakdown."""
    
    async def test_task_breakdown_complexity_analysis_accuracy(self):
        """Validate that LLM accurately assesses task complexity."""
        # Test LLM's analytical capabilities
        
    async def test_command_sequence_generation_logical_flow(self):
        """Validate that LLM generates logically sequenced commands."""
        # Test LLM's planning and sequencing abilities
```

### 3. Context Management Tests

#### Purpose
Validate that agents can effectively manage and utilize context using LLM calls.

#### Test Files
- `test_llm_context_compressor.py` - Content summarization
- `test_llm_memory_integration.py` - Memory and system message building

#### Key LLM Behaviors Tested
- **Content Summarization**: Coherent compression while preserving essential information
- **Context Integration**: Proper incorporation of memory and current context
- **Information Retention**: Maintaining critical details during compression
- **Context Formatting**: Appropriate structure for agent consumption
- **Memory Utilization**: Effective use of historical patterns

#### Example Test Structure
```python
class TestContextCompressorLLMIntegration:
    """Tests ContextCompressor's LLM-based content summarization."""
    
    async def test_content_summarization_coherence_preservation(self):
        """Validate that LLM maintains coherence during compression."""
        # Test LLM's summarization quality
        
    async def test_essential_information_retention_accuracy(self):
        """Validate that LLM preserves critical information."""
        # Test LLM's information prioritization
```

## Quality Validation Framework

### Quality Assessment Methodology

#### Quality Dimensions
1. **Structure Score** (0.0-1.0): Proper formatting and organization
2. **Completeness Score** (0.0-1.0): All required elements present
3. **Accuracy Score** (0.0-1.0): Technical correctness and validity
4. **Coherence Score** (0.0-1.0): Logical consistency and flow
5. **Overall Score** (0.0-1.0): Weighted combination of all dimensions

#### Quality Thresholds by Test Type
```python
QUALITY_THRESHOLDS = {
    'requirements_generation': {
        'structure_score': 0.8,
        'completeness_score': 0.85,
        'accuracy_score': 0.75,
        'coherence_score': 0.8
    },
    'design_generation': {
        'structure_score': 0.85,
        'completeness_score': 0.8,
        'accuracy_score': 0.8,
        'coherence_score': 0.85
    },
    'task_decomposition': {
        'structure_score': 0.9,
        'completeness_score': 0.85,
        'accuracy_score': 0.9,
        'coherence_score': 0.8
    },
    'error_analysis': {
        'structure_score': 0.75,
        'completeness_score': 0.8,
        'accuracy_score': 0.85,
        'coherence_score': 0.75
    }
}
```

### Validation Methods

#### Document Format Validation
- **EARS Format**: Validate requirements follow "WHEN/IF...THEN...SHALL" structure
- **Mermaid Syntax**: Validate diagram syntax and renderability
- **Task Structure**: Validate sequential numbering and requirement references
- **Section Completeness**: Validate all required sections are present

#### Content Quality Validation
- **Technical Accuracy**: Validate technical soundness of generated content
- **Logical Consistency**: Validate internal coherence and flow
- **Requirement Alignment**: Validate alignment between documents
- **Improvement Assessment**: Validate meaningful enhancement after revision

## Test Execution Patterns

### Sequential Execution for Rate Limiting

#### Rate Limit Handling
```python
class RateLimitHandler:
    """Handles API rate limiting with bash sleep commands."""
    
    def is_rate_limit_error(self, error: Exception) -> bool:
        """Detect if error is due to API rate limiting (429 status)."""
        return "429" in str(error) or "rate limit" in str(error).lower()
    
    async def handle_rate_limit(self, error: Exception) -> None:
        """Handle rate limiting by executing bash sleep command."""
        delay_seconds = self._extract_delay_from_error(error) or 60
        process = await asyncio.create_subprocess_exec(
            'sleep', str(delay_seconds),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        await process.communicate()
```

#### Sequential Test Execution
```python
@sequential_test_execution
class TestLLMComponent:
    """Tests are automatically executed sequentially to avoid rate limits."""
    
    async def test_first_llm_operation(self):
        """First test in sequence."""
        pass
    
    async def test_second_llm_operation(self):
        """Second test in sequence (waits for first to complete)."""
        pass
```

### Test Data Management

#### Test Scenarios
```python
@dataclass
class LLMTestScenario:
    """Configuration for LLM integration test scenarios."""
    name: str
    description: str
    input_data: Dict[str, Any]
    expected_quality_thresholds: Dict[str, float]
    validation_criteria: List[str]
    timeout_seconds: int = 300
```

#### Baseline Management
- **Quality Baselines**: Maintain baseline quality scores for regression testing
- **Test Data**: Use consistent test scenarios across different test runs
- **Result Tracking**: Track quality improvements over time

## Test Infrastructure

### Base Test Classes

#### LLMIntegrationTestBase
```python
class LLMIntegrationTestBase:
    """Base class for all LLM integration tests."""
    
    def setup_method(self, real_llm_config, temp_workspace):
        """Setup using existing fixtures for credentials and workspace."""
        self.real_llm_config = real_llm_config
        self.test_workspace = temp_workspace
        self.quality_validator = LLMQualityValidator()
        self.rate_limit_handler = RateLimitHandler()
    
    async def execute_with_rate_limit_handling(self, llm_operation: Callable) -> Any:
        """Execute LLM operation with automatic rate limit handling."""
        # Implementation handles rate limiting automatically
```

#### LLMQualityValidator
```python
class LLMQualityValidator:
    """Helper class for LLM integration tests using enhanced quality framework."""
    
    def validate_llm_output(self, content: str, output_type: str) -> Dict[str, Any]:
        """Validate LLM output using enhanced existing framework."""
        return self.quality_metrics.assess_llm_document_quality(content, output_type)
    
    def validate_ears_format(self, requirements_content: str) -> bool:
        """Validate EARS format compliance."""
        # Implementation validates EARS structure
    
    def validate_mermaid_syntax(self, design_content: str) -> bool:
        """Validate Mermaid diagram syntax."""
        # Implementation validates Mermaid syntax
```

### Fixture Usage

#### Required Fixtures
- `real_llm_config`: Real LLM configuration from `.env.integration`
- `temp_workspace`: Temporary workspace for test execution
- `initialized_real_managers`: Real manager instances for integration

#### Fixture Integration
```python
@pytest.fixture(autouse=True)
def setup_llm_test(self, real_llm_config, initialized_real_managers, temp_workspace):
    """Setup LLM test with real configuration and managers."""
    self.llm_config = real_llm_config
    self.managers = initialized_real_managers
    self.workspace = temp_workspace
    self.quality_validator = LLMQualityValidator()
```

## Test Maintenance Procedures

### Regular Maintenance Tasks

#### Quality Threshold Updates
1. **Monthly Review**: Review quality thresholds based on test results
2. **Baseline Updates**: Update quality baselines when improvements are made
3. **Threshold Adjustment**: Adjust thresholds based on LLM model changes

#### Test Data Management
1. **Scenario Updates**: Update test scenarios to reflect new requirements
2. **Data Cleanup**: Remove outdated test data and scenarios
3. **Result Archival**: Archive historical test results for trend analysis

#### Documentation Updates
1. **Pattern Documentation**: Document new LLM interaction patterns
2. **Best Practice Updates**: Update best practices based on experience
3. **Troubleshooting Guides**: Maintain troubleshooting information

### Troubleshooting Common Issues

#### Rate Limiting Issues
- **Symptoms**: 429 errors, "rate limit exceeded" messages
- **Solution**: Ensure sequential test execution, increase delays if needed
- **Prevention**: Use `@sequential_test_execution` decorator

#### Quality Threshold Failures
- **Symptoms**: Tests failing on quality assessments
- **Investigation**: Check if LLM model changed, review quality criteria
- **Solution**: Adjust thresholds or improve prompts

#### Context Integration Issues
- **Symptoms**: Tests failing on context-dependent operations
- **Investigation**: Verify context manager setup, check memory integration
- **Solution**: Ensure proper context initialization and data flow

### Performance Monitoring

#### Test Execution Metrics
- **Execution Time**: Monitor test execution duration trends
- **Success Rate**: Track test success rates over time
- **Quality Scores**: Monitor quality score trends and improvements

#### LLM Performance Metrics
- **Response Quality**: Track LLM response quality over time
- **Error Rates**: Monitor LLM error rates and types
- **Token Usage**: Track token consumption patterns

## Best Practices

### Test Design Principles

#### Focus on LLM Behavior
- Test what the LLM actually does, not just component functionality
- Validate intelligent behavior, not just successful execution
- Assess output quality, not just output presence

#### Use Real Configurations
- Always use `real_llm_config` fixture for LLM tests
- Test with actual LLM models and configurations
- Avoid mocking LLM interactions in integration tests

#### Quality-First Approach
- Define quality criteria before writing tests
- Use consistent quality assessment methods
- Track quality improvements over time

### Code Organization

#### File Structure
- Group related LLM tests in single files
- Use clear, descriptive file and class names
- Separate LLM tests from functional tests

#### Test Structure
- Use consistent test method naming
- Include comprehensive docstrings
- Group related test methods in logical order

#### Documentation
- Document LLM behaviors being tested
- Include examples of expected outputs
- Maintain up-to-date troubleshooting guides

### Integration Guidelines

#### Fixture Usage
- Use existing fixtures consistently
- Don't create duplicate fixture functionality
- Leverage shared test infrastructure

#### Quality Framework Integration
- Use enhanced QualityMetricsFramework methods
- Don't duplicate quality assessment logic
- Maintain consistency with existing quality standards

#### Error Handling
- Handle rate limiting gracefully
- Provide clear error messages
- Include debugging information in failures

This guide serves as the comprehensive reference for all LLM integration testing in the AutoGen multi-agent framework. Follow these patterns and procedures to ensure consistent, high-quality LLM testing that validates real intelligent behavior rather than just functional correctness.