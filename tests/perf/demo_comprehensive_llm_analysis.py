#!/usr/bin/env python3
"""
Comprehensive demo of LLM API profiling integrated with performance analysis.

This script demonstrates the complete workflow of LLM API call profiling
integrated with existing cProfile analysis and bottleneck identification.
"""

import asyncio
import time
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from profiler import PerformanceProfiler
from bottleneck_analyzer import BottleneckAnalyzer
from llm_profiler import LLMAPICall, LLMProfilingResult
from models import TestIdentifier, ProfilingResult, ProfilerType, CProfileResult


def create_comprehensive_test_scenario():
    """Create a comprehensive test scenario with both cProfile and LLM data."""
    
    # Test identifier
    test_identifier = TestIdentifier(
        file_path="tests/integration/test_implement_agent_complex.py",
        class_name="TestImplementAgentComplex",
        method_name="test_execute_multi_step_task_with_error_recovery"
    )
    
    # Create realistic LLM API calls
    base_time = time.time()
    api_calls = [
        # TaskDecomposer - Initial task analysis
        LLMAPICall(
            timestamp=base_time,
            component='TaskDecomposer',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Analyze the complexity of this multi-step task: Create a data processing pipeline that reads CSV files, validates data, applies transformations, and generates reports. The pipeline should handle missing files, invalid data formats, and memory constraints.',
            prompt_size_chars=245,
            prompt_size_tokens=62,
            response_text='This is a high-complexity task requiring multiple components:\n1. File I/O operations\n2. Data validation logic\n3. Transformation algorithms\n4. Error handling mechanisms\n5. Memory management\n6. Report generation\n\nEstimated implementation time: 45-60 minutes',
            response_size_chars=267,
            response_size_tokens=68,
            total_time=3.2,
            network_time=0.6,
            processing_time=2.6,
            status_code=200,
            success=True
        ),
        
        # TaskDecomposer - Command sequence generation
        LLMAPICall(
            timestamp=base_time + 3.5,
            component='TaskDecomposer',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Generate a detailed shell command sequence for implementing the data processing pipeline:\n\nRequirements:\n- Create Python modules for each component\n- Set up proper directory structure\n- Include test files and sample data\n- Add configuration files\n- Implement logging and error handling\n\nProject structure should follow best practices for maintainability.',
            prompt_size_chars=398,
            prompt_size_tokens=98,
            response_text='# Data Processing Pipeline Implementation\n\n## 1. Project Structure Setup\nmkdir -p data_pipeline/{src,tests,config,data,logs}\ntouch data_pipeline/src/{__init__.py,reader.py,validator.py,transformer.py,reporter.py}\ntouch data_pipeline/tests/{__init__.py,test_reader.py,test_validator.py,test_transformer.py,test_reporter.py}\ntouch data_pipeline/config/{settings.yaml,logging.conf}\ntouch data_pipeline/data/{sample.csv,schema.json}\n\n## 2. Core Implementation\necho "import csv, logging" > data_pipeline/src/reader.py\necho "class DataReader:" >> data_pipeline/src/reader.py\necho "    def read_csv(self, filepath): pass" >> data_pipeline/src/reader.py\n\n## 3. Error Handling Setup\necho "import logging" > data_pipeline/src/validator.py\necho "class DataValidator:" >> data_pipeline/src/validator.py\necho "    def validate_schema(self, data): pass" >> data_pipeline/src/validator.py',
            response_size_chars=892,
            response_size_tokens=225,
            total_time=4.8,
            network_time=0.8,
            processing_time=4.0,
            status_code=200,
            success=True
        ),
        
        # First execution attempt - leads to error
        # ErrorRecovery - Analyzing FileNotFoundError
        LLMAPICall(
            timestamp=base_time + 12.0,
            component='ErrorRecovery',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Analyze this error and provide recovery strategies:\n\nError: FileNotFoundError: [Errno 2] No such file or directory: \'data_pipeline/data/input.csv\'\nContext: Attempting to read CSV file for data processing pipeline\nStack trace shows error in DataReader.read_csv() method\nExpected file path: data_pipeline/data/input.csv\nCurrent working directory: /home/user/projects',
            prompt_size_chars=356,
            prompt_size_tokens=89,
            response_text='Error Analysis: Missing input file\n\nRoot Cause: The expected input file \'input.csv\' does not exist in the data directory.\n\nRecovery Strategies (ranked by likelihood of success):\n\n1. **Check file path and create missing file** (90% success rate)\n   - Verify the data directory exists\n   - Create a sample input.csv file with proper headers\n   - Ensure file permissions are correct\n\n2. **Use alternative input source** (75% success rate)\n   - Look for other CSV files in the directory\n   - Use sample.csv if available\n   - Prompt user for correct file path\n\n3. **Generate synthetic data** (60% success rate)\n   - Create sample data matching expected schema\n   - Use for testing and development purposes\n\nRecommended: Try strategy 1 first, fallback to strategy 2 if needed.',
            response_size_chars=756,
            response_size_tokens=189,
            total_time=5.4,
            network_time=0.9,
            processing_time=4.5,
            status_code=200,
            success=True
        ),
        
        # ErrorRecovery - Generating specific recovery commands
        LLMAPICall(
            timestamp=base_time + 18.0,
            component='ErrorRecovery',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Generate specific shell commands to implement the file recovery strategy:\n\nStrategy: Create missing input.csv file with sample data\nRequired schema: id,name,age,department,salary\nSample data should include 10-15 realistic records\nFile should be saved to: data_pipeline/data/input.csv',
            prompt_size_chars=298,
            prompt_size_tokens=74,
            response_text='# File Recovery Commands\n\n# 1. Ensure directory exists\nmkdir -p data_pipeline/data\n\n# 2. Create input.csv with sample data\ncat > data_pipeline/data/input.csv << EOF\nid,name,age,department,salary\n1,John Smith,28,Engineering,75000\n2,Sarah Johnson,32,Marketing,68000\n3,Mike Chen,25,Engineering,72000\n4,Lisa Brown,29,Sales,65000\n5,David Wilson,35,Management,85000\n6,Emma Davis,27,Design,70000\n7,James Miller,31,Engineering,78000\n8,Anna Garcia,26,Marketing,63000\n9,Robert Taylor,33,Sales,71000\n10,Jennifer Lee,30,Engineering,76000\n11,Thomas Anderson,28,Design,69000\n12,Maria Rodriguez,34,Management,82000\nEOF\n\n# 3. Verify file creation\nls -la data_pipeline/data/input.csv\nhead -5 data_pipeline/data/input.csv',
            response_size_chars=742,
            response_size_tokens=186,
            total_time=3.9,
            network_time=0.7,
            processing_time=3.2,
            status_code=200,
            success=True
        ),
        
        # Second execution attempt - new error
        # ErrorRecovery - Analyzing validation error (slow call)
        LLMAPICall(
            timestamp=base_time + 25.0,
            component='ErrorRecovery',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Analyze this complex validation error and provide comprehensive recovery strategies:\n\nError: ValidationError: Data validation failed\nDetails:\n- Row 3: Invalid age value \'25abc\' (expected integer)\n- Row 7: Missing required field \'department\'\n- Row 9: Salary value \'71000.50\' exceeds maximum allowed precision\n- Row 12: Invalid department \'Managment\' (typo, should be \'Management\')\n\nValidation rules:\n- age: integer between 18-65\n- department: one of [Engineering, Marketing, Sales, Management, Design]\n- salary: integer between 30000-150000\n- All fields required\n\nContext: This is a critical data processing pipeline for payroll system. Data integrity is essential. The pipeline processes 10,000+ records daily. Need robust error handling that can:\n1. Identify all validation errors\n2. Provide clear error messages\n3. Suggest corrections\n4. Allow partial processing of valid records\n5. Generate detailed error reports\n\nConsider performance implications and user experience.',
            prompt_size_chars=1089,
            prompt_size_tokens=272,
            response_text='Comprehensive Validation Error Analysis\n\n## Error Classification\n\n### Type 1: Data Type Errors (Row 3)\n- Issue: \'25abc\' cannot be converted to integer\n- Severity: High (blocks processing)\n- Auto-fix potential: Low (requires human judgment)\n\n### Type 2: Missing Data Errors (Row 7)\n- Issue: Required field \'department\' is empty\n- Severity: High (violates schema)\n- Auto-fix potential: Medium (could use default or prompt)\n\n### Type 3: Precision Errors (Row 9)\n- Issue: Decimal salary when integer expected\n- Severity: Medium (data loss on conversion)\n- Auto-fix potential: High (can round or truncate)\n\n### Type 4: Reference Data Errors (Row 12)\n- Issue: Typo in department name\n- Severity: Medium (invalid reference)\n- Auto-fix potential: High (fuzzy matching possible)\n\n## Recovery Strategies\n\n### Strategy 1: Intelligent Data Cleaning (Recommended)\n```python\ndef clean_and_validate(row):\n    errors = []\n    \n    # Fix age field\n    if not str(row[\'age\']).isdigit():\n        # Extract numeric part\n        age_clean = re.sub(r\'[^0-9]\', \'\', str(row[\'age\']))\n        if age_clean and 18 <= int(age_clean) <= 65:\n            row[\'age\'] = int(age_clean)\n            errors.append(f"Age corrected from {original} to {age_clean}")\n        else:\n            errors.append(f"Invalid age: {row[\'age\']} - requires manual review")\n    \n    # Handle missing department\n    if not row.get(\'department\'):\n        # Could infer from salary range or set default\n        if int(row.get(\'salary\', 0)) > 80000:\n            row[\'department\'] = \'Management\'\n            errors.append("Department inferred as Management based on salary")\n    \n    # Fix salary precision\n    if \'.\' in str(row[\'salary\']):\n        row[\'salary\'] = int(float(row[\'salary\']))\n        errors.append("Salary rounded to nearest integer")\n    \n    # Fix department typos using fuzzy matching\n    valid_depts = [\'Engineering\', \'Marketing\', \'Sales\', \'Management\', \'Design\']\n    if row[\'department\'] not in valid_depts:\n        best_match = find_closest_match(row[\'department\'], valid_depts)\n        if similarity_score > 0.8:\n            row[\'department\'] = best_match\n            errors.append(f"Department corrected: {original} -> {best_match}")\n    \n    return row, errors\n```\n\n### Strategy 2: Partial Processing with Error Reporting\n- Process valid records immediately\n- Quarantine invalid records for manual review\n- Generate detailed error report with suggested fixes\n- Provide data quality metrics\n\n### Strategy 3: Interactive Correction Mode\n- Pause processing on errors\n- Present correction suggestions to user\n- Allow batch approval of similar fixes\n- Learn from user corrections for future processing\n\n## Implementation Commands\n\n```bash\n# Create enhanced validator with error recovery\ncat > data_pipeline/src/enhanced_validator.py << \'EOF\'\nimport re\nfrom difflib import get_close_matches\nfrom typing import List, Tuple, Dict\n\nclass EnhancedDataValidator:\n    def __init__(self):\n        self.valid_departments = [\'Engineering\', \'Marketing\', \'Sales\', \'Management\', \'Design\']\n        self.error_log = []\n    \n    def validate_and_clean(self, data: List[Dict]) -> Tuple[List[Dict], List[str]]:\n        cleaned_data = []\n        errors = []\n        \n        for i, row in enumerate(data, 1):\n            try:\n                cleaned_row, row_errors = self._clean_row(row, i)\n                cleaned_data.append(cleaned_row)\n                errors.extend(row_errors)\n            except Exception as e:\n                errors.append(f"Row {i}: Critical error - {str(e)}")\n        \n        return cleaned_data, errors\n    \n    def _clean_row(self, row: Dict, row_num: int) -> Tuple[Dict, List[str]]:\n        errors = []\n        cleaned = row.copy()\n        \n        # Age validation and cleaning\n        age_str = str(row.get(\'age\', \'\'))\n        if not age_str.isdigit():\n            numeric_age = re.sub(r\'[^0-9]\', \'\', age_str)\n            if numeric_age and 18 <= int(numeric_age) <= 65:\n                cleaned[\'age\'] = int(numeric_age)\n                errors.append(f"Row {row_num}: Age corrected from \'{age_str}\' to {numeric_age}")\n            else:\n                errors.append(f"Row {row_num}: Invalid age \'{age_str}\' - manual review required")\n                return None, errors\n        \n        # Department validation with fuzzy matching\n        dept = row.get(\'department\', \'\')\n        if not dept:\n            # Infer department from salary\n            salary = int(row.get(\'salary\', 0))\n            if salary > 80000:\n                cleaned[\'department\'] = \'Management\'\n                errors.append(f"Row {row_num}: Department inferred as Management (salary: {salary})")\n            else:\n                cleaned[\'department\'] = \'Engineering\'  # Default\n                errors.append(f"Row {row_num}: Department set to default (Engineering)")\n        elif dept not in self.valid_departments:\n            matches = get_close_matches(dept, self.valid_departments, n=1, cutoff=0.6)\n            if matches:\n                cleaned[\'department\'] = matches[0]\n                errors.append(f"Row {row_num}: Department corrected from \'{dept}\' to \'{matches[0]}\'")\n            else:\n                errors.append(f"Row {row_num}: Invalid department \'{dept}\' - manual review required")\n        \n        # Salary precision fix\n        salary_str = str(row.get(\'salary\', \'\'))\n        if \'.\' in salary_str:\n            cleaned[\'salary\'] = int(float(salary_str))\n            errors.append(f"Row {row_num}: Salary rounded from {salary_str} to {cleaned[\'salary\']}")\n        \n        return cleaned, errors\nEOF\n\n# Create error reporting module\ncat > data_pipeline/src/error_reporter.py << \'EOF\'\nimport json\nfrom datetime import datetime\nfrom typing import List, Dict\n\nclass ErrorReporter:\n    def generate_report(self, errors: List[str], total_records: int) -> Dict:\n        report = {\n            \'timestamp\': datetime.now().isoformat(),\n            \'total_records\': total_records,\n            \'error_count\': len(errors),\n            \'success_rate\': ((total_records - len(errors)) / total_records * 100) if total_records > 0 else 0,\n            \'errors\': errors,\n            \'recommendations\': self._generate_recommendations(errors)\n        }\n        return report\n    \n    def _generate_recommendations(self, errors: List[str]) -> List[str]:\n        recommendations = []\n        \n        age_errors = [e for e in errors if \'age\' in e.lower()]\n        if len(age_errors) > 3:\n            recommendations.append("Consider implementing stricter age validation at data entry point")\n        \n        dept_errors = [e for e in errors if \'department\' in e.lower()]\n        if len(dept_errors) > 2:\n            recommendations.append("Implement dropdown selection for department field to prevent typos")\n        \n        return recommendations\nEOF\n\n# Update main pipeline to use enhanced validation\necho "Enhanced validation and error recovery implemented successfully"\necho "Files created: enhanced_validator.py, error_reporter.py"\necho "Next: Integrate with main pipeline and test with sample data"\n```\n\n## Performance Considerations\n\n- **Batch Processing**: Process records in chunks of 1000 for better memory usage\n- **Parallel Validation**: Use multiprocessing for large datasets\n- **Caching**: Cache fuzzy matching results for repeated corrections\n- **Streaming**: For very large files, use streaming validation to avoid memory issues\n\n## Monitoring and Alerting\n\n- Set up alerts for error rates > 5%\n- Track data quality trends over time\n- Monitor processing performance and adjust batch sizes\n- Generate daily data quality reports for stakeholders',
            response_size_chars=6247,
            response_size_tokens=1562,
            total_time=12.8,  # Very slow call - complex analysis
            network_time=2.1,
            processing_time=10.7,
            status_code=200,
            success=True
        ),
        
        # ImplementAgent - Coordinating final execution
        LLMAPICall(
            timestamp=base_time + 40.0,
            component='ImplementAgent',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Coordinate the final execution of the data processing pipeline with all error recovery mechanisms in place:\n\n1. File creation completed\n2. Enhanced validation implemented\n3. Error reporting configured\n\nNext steps: Execute pipeline, monitor for issues, validate output quality',
            prompt_size_chars=318,
            prompt_size_tokens=79,
            response_text='Final Execution Coordination Plan:\n\n## Phase 1: Pre-execution Validation\n- Verify all input files exist and are readable\n- Check enhanced validator configuration\n- Initialize error reporting system\n- Set up monitoring and logging\n\n## Phase 2: Pipeline Execution\n- Execute data reader with error handling\n- Apply enhanced validation with auto-correction\n- Process valid records through transformation pipeline\n- Generate comprehensive error reports\n\n## Phase 3: Quality Assurance\n- Validate output data integrity\n- Review error correction accuracy\n- Generate data quality metrics\n- Prepare summary report for stakeholders\n\nExecution Status: Ready to proceed with enhanced error recovery',
            response_size_chars=678,
            response_size_tokens=170,
            total_time=2.8,
            network_time=0.5,
            processing_time=2.3,
            status_code=200,
            success=True
        ),
        
        # ContextManager - Loading comprehensive context
        LLMAPICall(
            timestamp=base_time + 44.0,
            component='ContextManager',
            request_url='https://api.openai.com/v1/chat/completions',
            request_method='POST',
            prompt_text='Compress and summarize the complete project context for efficient processing:\n\nProject: Data Processing Pipeline with Error Recovery\nComponents: DataReader, EnhancedValidator, ErrorReporter, Transformer, Reporter\nCurrent Status: All components implemented, error recovery tested\nError History: FileNotFoundError (resolved), ValidationError (resolved with enhanced validation)\nPerformance Metrics: Processing 12 records, 4 errors auto-corrected, 100% success rate after recovery\nNext Phase: Production deployment with monitoring\nKey Learnings: Fuzzy matching effective for typo correction, salary inference works well\nConfiguration: Enhanced validation enabled, error reporting active, logging configured',
            prompt_size_chars=692,
            prompt_size_tokens=173,
            response_text='Project Context Summary:\n\n**Status**: Production-ready data processing pipeline\n**Architecture**: Modular design with enhanced error recovery\n**Key Components**: Reader → Enhanced Validator → Transformer → Reporter\n**Error Recovery**: Auto-correction for common issues (typos, formatting, missing data)\n**Quality Metrics**: 100% processing success rate with intelligent error handling\n**Performance**: Optimized for 10K+ daily records with batch processing\n**Monitoring**: Comprehensive error reporting and data quality tracking\n**Deployment Ready**: All components tested and integrated\n\n**Critical Success Factors**:\n- Fuzzy matching for department name corrections\n- Salary-based department inference for missing data\n- Intelligent age field cleaning with regex\n- Comprehensive error logging and reporting\n\n**Next Steps**: Deploy to production with monitoring dashboard',
            response_size_chars=798,
            response_size_tokens=200,
            total_time=3.6,
            network_time=1.8,  # Higher network latency
            processing_time=1.8,
            status_code=200,
            success=True
        )
    ]
    
    # Create corresponding cProfile data
    function_stats = {
        'tests/integration/test_implement_agent_complex.py:75(test_execute_multi_step_task_with_error_recovery)': {
            'cumulative_time': 35.2,
            'total_time': 0.8,
            'call_count': 1,
            'filename': 'tests/integration/test_implement_agent_complex.py',
            'line_number': 75,
            'function_name': 'test_execute_multi_step_task_with_error_recovery'
        },
        'autogen_framework/agents/implement_agent.py:150(execute_task)': {
            'cumulative_time': 34.1,
            'total_time': 1.2,
            'call_count': 1,
            'filename': 'autogen_framework/agents/implement_agent.py',
            'line_number': 150,
            'function_name': 'execute_task'
        },
        'autogen_framework/agents/task_decomposer.py:90(decompose_task)': {
            'cumulative_time': 8.2,
            'total_time': 0.2,
            'call_count': 1,
            'filename': 'autogen_framework/agents/task_decomposer.py',
            'line_number': 90,
            'function_name': 'decompose_task'
        },
        'autogen_framework/agents/task_decomposer.py:120(generate_command_sequence)': {
            'cumulative_time': 4.9,
            'total_time': 0.1,
            'call_count': 1,
            'filename': 'autogen_framework/agents/task_decomposer.py',
            'line_number': 120,
            'function_name': 'generate_command_sequence'
        },
        'autogen_framework/agents/error_recovery.py:80(analyze_error)': {
            'cumulative_time': 5.6,
            'total_time': 0.2,
            'call_count': 1,
            'filename': 'autogen_framework/agents/error_recovery.py',
            'line_number': 80,
            'function_name': 'analyze_error'
        },
        'autogen_framework/agents/error_recovery.py:110(generate_recovery_strategies)': {
            'cumulative_time': 13.1,
            'total_time': 0.3,
            'call_count': 1,
            'filename': 'autogen_framework/agents/error_recovery.py',
            'line_number': 110,
            'function_name': 'generate_recovery_strategies'
        },
        'autogen_framework/agents/context_manager.py:200(get_implementation_context)': {
            'cumulative_time': 3.8,
            'total_time': 0.2,
            'call_count': 1,
            'filename': 'autogen_framework/agents/context_manager.py',
            'line_number': 200,
            'function_name': 'get_implementation_context'
        },
        'openai/api_resources/chat_completion.py:200(create)': {
            'cumulative_time': 32.5,  # Total LLM API time
            'total_time': 32.5,
            'call_count': 7,
            'filename': 'openai/api_resources/chat_completion.py',
            'line_number': 200,
            'function_name': 'create'
        },
        'autogen_framework/shell_executor.py:50(execute_command)': {
            'cumulative_time': 2.1,
            'total_time': 2.1,
            'call_count': 8,
            'filename': 'autogen_framework/shell_executor.py',
            'line_number': 50,
            'function_name': 'execute_command'
        }
    }
    
    top_functions = [
        {
            'name': 'tests/integration/test_implement_agent_complex.py:75(test_execute_multi_step_task_with_error_recovery)',
            'cumulative_time': 35.2,
            'total_time': 0.8,
            'call_count': 1,
            'percentage': 100.0
        },
        {
            'name': 'autogen_framework/agents/implement_agent.py:150(execute_task)',
            'cumulative_time': 34.1,
            'total_time': 1.2,
            'call_count': 1,
            'percentage': 96.9
        },
        {
            'name': 'openai/api_resources/chat_completion.py:200(create)',
            'cumulative_time': 32.5,
            'total_time': 32.5,
            'call_count': 7,
            'percentage': 92.3
        },
        {
            'name': 'autogen_framework/agents/error_recovery.py:110(generate_recovery_strategies)',
            'cumulative_time': 13.1,
            'total_time': 0.3,
            'call_count': 1,
            'percentage': 37.2
        }
    ]
    
    cprofile_result = CProfileResult(
        test_identifier=test_identifier,
        profile_file_path="/tmp/complex_test_profile.prof",
        total_time=35.2,
        function_stats=function_stats,
        top_functions=top_functions,
        call_count=150
    )
    
    llm_result = LLMProfilingResult(
        test_identifier=test_identifier,
        api_calls=api_calls
    )
    
    profiling_result = ProfilingResult(
        test_identifier=test_identifier,
        profiler_used=ProfilerType.CPROFILE,
        cprofile_result=cprofile_result,
        llm_profiling_result=llm_result,
        total_profiling_time=36.0
    )
    
    return profiling_result


async def demo_comprehensive_analysis():
    """Demonstrate comprehensive LLM and performance analysis."""
    print("🔍 Comprehensive LLM + Performance Analysis Demo")
    print("=" * 60)
    
    # Create comprehensive test scenario
    profiling_result = create_comprehensive_test_scenario()
    
    print(f"📊 Test Case: {profiling_result.test_identifier.full_name}")
    print("-" * 60)
    
    # Initialize analyzer
    analyzer = BottleneckAnalyzer()
    
    # Perform comprehensive analysis
    reports = analyzer.analyze_profiling_results_with_llm([profiling_result])
    report = reports[0]
    
    print("📈 Overall Performance Summary:")
    print(f"  Total Execution Time: {profiling_result.cprofile_result.total_time:.1f}s")
    print(f"  Total LLM Time: {profiling_result.llm_profiling_result.total_llm_time:.1f}s")
    print(f"  LLM Percentage: {(profiling_result.llm_profiling_result.total_llm_time / profiling_result.cprofile_result.total_time * 100):.1f}%")
    print()
    
    # LLM Analysis
    llm_result = profiling_result.llm_profiling_result
    print("🤖 LLM API Call Analysis:")
    print(f"  Total API Calls: {llm_result.total_api_calls}")
    print(f"  Average Call Time: {llm_result.average_call_time:.2f}s")
    print(f"  Slowest Call: {llm_result.slowest_call_time:.2f}s")
    print(f"  Network vs Processing: {llm_result.network_percentage:.1f}% network, {100-llm_result.network_percentage:.1f}% processing")
    print()
    
    # Component Breakdown
    print("🔧 Component Performance Breakdown:")
    for component, total_time in llm_result.component_total_times.items():
        call_count = llm_result.component_call_counts.get(component, 0)
        avg_time = total_time / call_count if call_count > 0 else 0.0
        percentage = (total_time / llm_result.total_llm_time * 100) if llm_result.total_llm_time > 0 else 0.0
        
        print(f"  {component}:")
        print(f"    Calls: {call_count}")
        print(f"    Total Time: {total_time:.2f}s ({percentage:.1f}% of LLM time)")
        print(f"    Avg Time: {avg_time:.2f}s")
    print()
    
    # Slow Calls Analysis
    if llm_result.slow_calls:
        print("🐌 Slow API Calls Analysis (>5s):")
        for i, call in enumerate(llm_result.slow_calls, 1):
            print(f"  {i}. {call.component} - {call.total_time:.2f}s")
            print(f"     Prompt Size: {call.prompt_size_chars} chars ({call.prompt_size_tokens} tokens)")
            print(f"     Response Size: {call.response_size_chars} chars ({call.response_size_tokens} tokens)")
            print(f"     Network: {call.network_time:.2f}s, Processing: {call.processing_time:.2f}s")
            print(f"     Generation Rate: {call.chars_per_second:.1f} chars/sec")
            print(f"     Prompt Preview: {call.prompt_text[:80]}...")
            print()
    
    # Bottleneck Analysis
    print("⚠️  Identified Bottlenecks:")
    for i, bottleneck in enumerate(report.component_bottlenecks, 1):
        print(f"  {i}. {bottleneck.component_name}")
        print(f"     Time Spent: {bottleneck.time_spent:.2f}s ({bottleneck.percentage_of_total:.1f}%)")
        print(f"     Optimization Potential: {bottleneck.optimization_potential}")
        if bottleneck.function_calls:
            print(f"     Key Functions: {len(bottleneck.function_calls)} function calls analyzed")
    print()
    
    # Optimization Recommendations
    print("💡 Optimization Recommendations:")
    high_priority = [r for r in report.optimization_recommendations if r.expected_impact == 'high']
    medium_priority = [r for r in report.optimization_recommendations if r.expected_impact == 'medium']
    
    if high_priority:
        print("  🔴 High Priority:")
        for i, rec in enumerate(high_priority, 1):
            print(f"    {i}. {rec.component}: {rec.issue_description}")
            print(f"       → {rec.recommendation}")
            print(f"       Impact: {rec.expected_impact}, Effort: {rec.implementation_effort}")
        print()
    
    if medium_priority:
        print("  🟡 Medium Priority:")
        for i, rec in enumerate(medium_priority, 1):
            print(f"    {i}. {rec.component}: {rec.issue_description}")
            print(f"       → {rec.recommendation}")
            print(f"       Impact: {rec.expected_impact}, Effort: {rec.implementation_effort}")
        print()
    
    # Detailed LLM Call Timeline
    print("📅 LLM Call Timeline:")
    for i, call in enumerate(llm_result.api_calls, 1):
        status = "✅" if call.success else "❌"
        relative_time = call.timestamp - llm_result.api_calls[0].timestamp
        print(f"  {i}. [{relative_time:5.1f}s] {status} {call.component} - {call.total_time:.2f}s")
        print(f"     {call.prompt_size_chars:3d} chars → {call.response_size_chars:3d} chars")
        if call.is_slow_call:
            print(f"     ⚠️  SLOW CALL - Consider optimization")
    print()
    
    # Performance Insights
    print("🎯 Key Performance Insights:")
    
    # Identify dominant component
    max_component = max(llm_result.component_total_times.items(), key=lambda x: x[1])
    print(f"  • {max_component[0]} dominates LLM usage ({max_component[1]:.1f}s, {max_component[1]/llm_result.total_llm_time*100:.1f}%)")
    
    # Check for correlation between prompt size and response time
    if abs(llm_result.prompt_size_correlation) > 0.5:
        correlation_strength = "strong" if abs(llm_result.prompt_size_correlation) > 0.7 else "moderate"
        correlation_direction = "positive" if llm_result.prompt_size_correlation > 0 else "negative"
        print(f"  • {correlation_strength.title()} {correlation_direction} correlation between prompt size and response time (r={llm_result.prompt_size_correlation:.3f})")
    
    # Network latency analysis
    if llm_result.network_percentage > 25:
        print(f"  • High network latency detected ({llm_result.network_percentage:.1f}% of total LLM time)")
        print(f"    Consider: closer API endpoints, connection pooling, or request batching")
    
    # Error recovery analysis
    error_recovery_time = llm_result.component_total_times.get('ErrorRecovery', 0)
    if error_recovery_time > 5.0:
        error_recovery_calls = llm_result.component_call_counts.get('ErrorRecovery', 0)
        print(f"  • Significant error recovery overhead ({error_recovery_time:.1f}s, {error_recovery_calls} calls)")
        print(f"    Consider: improving initial task decomposition to reduce errors")
    
    # Task decomposition efficiency
    decomposer_time = llm_result.component_total_times.get('TaskDecomposer', 0)
    decomposer_calls = llm_result.component_call_counts.get('TaskDecomposer', 0)
    if decomposer_calls > 1:
        avg_decomposer_time = decomposer_time / decomposer_calls
        print(f"  • Task decomposition required {decomposer_calls} iterations (avg {avg_decomposer_time:.1f}s each)")
        if avg_decomposer_time > 3.0:
            print(f"    Consider: more specific initial prompts or decomposition caching")
    
    print()
    print("✨ Comprehensive analysis completed!")
    print()
    print("📋 Summary Report:")
    print(f"  • Total execution time dominated by LLM calls ({(llm_result.total_llm_time/profiling_result.cprofile_result.total_time*100):.1f}%)")
    print(f"  • {len(llm_result.slow_calls)} slow calls identified for optimization")
    print(f"  • {len([r for r in report.optimization_recommendations if r.expected_impact == 'high'])} high-priority optimization opportunities")
    print(f"  • Error recovery system handled {llm_result.component_call_counts.get('ErrorRecovery', 0)} error scenarios successfully")


if __name__ == "__main__":
    asyncio.run(demo_comprehensive_analysis())