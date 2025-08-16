"""
Configuration settings for performance analysis tools.

This module centralizes all configuration parameters used throughout
the performance analysis infrastructure.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List


class PerformanceConfig:
    """Configuration settings for performance analysis."""
    
    # Test execution settings
    DEFAULT_TIMEOUT_SECONDS = 60
    INTEGRATION_TEST_DIR = "tests/integration"
    PERFORMANCE_OUTPUT_DIR = "artifacts/performance"
    
    # Profiling settings
    CPROFILE_OUTPUT_DIR = "artifacts/performance/profiles"
    FLAMEGRAPH_OUTPUT_DIR = "artifacts/performance/flamegraphs"
    REPORTS_OUTPUT_DIR = "artifacts/performance/reports"
    
    # Component identification patterns for ImplementAgent analysis
    COMPONENT_PATTERNS = {
        'TaskDecomposer': [
            'task_decomposer',
            'decompose_task',
            'TaskDecomposer',
            'generate_command_sequence',
            'analyze_complexity'
        ],
        'ShellExecutor': [
            'shell_executor',
            'execute_command',
            'ShellExecutor',
            'run_command',
            'execute_plan'
        ],
        'ErrorRecovery': [
            'error_recovery',
            'recover_from_error',
            'ErrorRecovery',
            'analyze_error',
            'generate_strategies'
        ],
        'ContextManager': [
            'context_manager',
            'get_context',
            'ContextManager',
            'load_context',
            'get_implementation_context'
        ],
        'ImplementAgent': [
            'implement_agent',
            'ImplementAgent',
            'execute_task',
            'process_task'
        ]
    }
    
    # LLM API call patterns for profiling
    LLM_API_PATTERNS = [
        'openai',
        'anthropic',
        'generate_response',
        'chat_completion',
        'api_call',
        'llm_request'
    ]
    
    # Test infrastructure patterns to separate from ImplementAgent code
    TEST_PATTERNS = [
        'pytest',
        'test_',
        'setup',
        'teardown',
        'fixture',
        'conftest',
        'mock'
    ]
    
    # Profiling tool availability
    PREFERRED_PROFILER = "cProfile"  # Built-in, deterministic
    SECONDARY_PROFILER = "py-spy"    # For flame graphs
    
    # Analysis thresholds
    SLOW_TEST_THRESHOLD_SECONDS = 10.0
    SIGNIFICANT_BOTTLENECK_PERCENTAGE = 10.0
    HIGH_IMPACT_TIME_THRESHOLD = 5.0  # seconds
    
    # Report generation settings
    MAX_TOP_FUNCTIONS = 20
    MAX_RECOMMENDATIONS = 10
    FLAMEGRAPH_WIDTH = 1200
    FLAMEGRAPH_HEIGHT = 800
    
    @classmethod
    def get_output_dir(cls, subdir: str = "") -> Path:
        """Get output directory path, creating if necessary."""
        base_dir = Path(cls.PERFORMANCE_OUTPUT_DIR)
        if subdir:
            output_dir = base_dir / subdir
        else:
            output_dir = base_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    @classmethod
    def get_profile_output_path(cls, test_name: str) -> Path:
        """Get output path for cProfile results."""
        profiles_dir = cls.get_output_dir("profiles")
        return profiles_dir / f"{test_name}.prof"
    
    @classmethod
    def get_flamegraph_output_path(cls, test_name: str) -> Path:
        """Get output path for flame graph SVG."""
        flamegraphs_dir = cls.get_output_dir("flamegraphs")
        return flamegraphs_dir / f"{test_name}_flamegraph.svg"
    
    @classmethod
    def get_report_output_path(cls, report_name: str) -> Path:
        """Get output path for analysis reports."""
        reports_dir = cls.get_output_dir("reports")
        return reports_dir / f"{report_name}.html"
    
    @classmethod
    def is_integration_test_file(cls, file_path: str) -> bool:
        """Check if file is an integration test."""
        path = Path(file_path)
        return (cls.INTEGRATION_TEST_DIR in str(path) and 
                path.name.startswith('test_') and 
                path.suffix == '.py')
    
    @classmethod
    def should_profile_test(cls, execution_time: float) -> bool:
        """Determine if test should be profiled based on execution time."""
        return execution_time >= cls.SLOW_TEST_THRESHOLD_SECONDS


# Environment-specific settings
class EnvironmentConfig:
    """Environment-specific configuration settings."""
    
    @staticmethod
    def get_python_executable() -> str:
        """Get Python executable path."""
        return os.environ.get('PYTHON_EXECUTABLE', 'python')
    
    @staticmethod
    def get_pytest_executable() -> str:
        """Get pytest executable path."""
        return os.environ.get('PYTEST_EXECUTABLE', 'pytest')
    
    @staticmethod
    def is_pyspy_available() -> bool:
        """Check if py-spy is available in the environment."""
        import shutil
        return shutil.which('py-spy') is not None
    
    @staticmethod
    def get_max_workers() -> int:
        """Get maximum number of worker processes for parallel execution."""
        return int(os.environ.get('PERF_MAX_WORKERS', '4'))
    
    @staticmethod
    def get_timeout_multiplier() -> float:
        """Get timeout multiplier for slower environments."""
        return float(os.environ.get('PERF_TIMEOUT_MULTIPLIER', '1.0'))


# Validation functions
def validate_config() -> List[str]:
    """Validate configuration and return any issues found."""
    issues = []
    
    # Check if integration test directory exists
    if not Path(PerformanceConfig.INTEGRATION_TEST_DIR).exists():
        issues.append(f"Integration test directory not found: {PerformanceConfig.INTEGRATION_TEST_DIR}")
    
    # Check if output directories can be created
    try:
        PerformanceConfig.get_output_dir()
    except Exception as e:
        issues.append(f"Cannot create output directory: {e}")
    
    # Check Python executable
    python_exe = EnvironmentConfig.get_python_executable()
    if not shutil.which(python_exe):
        issues.append(f"Python executable not found: {python_exe}")
    
    # Check pytest executable  
    pytest_exe = EnvironmentConfig.get_pytest_executable()
    if not shutil.which(pytest_exe):
        issues.append(f"Pytest executable not found: {pytest_exe}")
    
    return issues