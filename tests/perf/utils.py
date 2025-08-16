"""
Utility functions for performance analysis tools.

This module provides common functionality used across the performance
analysis infrastructure.
"""

import asyncio
import time
import subprocess
import signal
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Pattern
from contextlib import asynccontextmanager

try:
    from .models import TestIdentifier, ExecutionResult, TestStatus
    from .config import PerformanceConfig, EnvironmentConfig
except ImportError:
    # Handle direct execution
    from models import TestIdentifier, ExecutionResult, TestStatus
    from config import PerformanceConfig, EnvironmentConfig


class TestDiscovery:
    """Utilities for discovering integration tests."""
    
    @staticmethod
    def discover_integration_tests() -> List[TestIdentifier]:
        """Discover all integration test files and methods."""
        integration_dir = Path(PerformanceConfig.INTEGRATION_TEST_DIR)
        if not integration_dir.exists():
            return []
        
        test_identifiers = []
        
        # Find all test files
        for test_file in integration_dir.rglob("test_*.py"):
            if test_file.name == "__init__.py":
                continue
            
            # Handle both absolute and relative paths
            try:
                if test_file.is_absolute():
                    # Try to make it relative to current working directory
                    try:
                        relative_path = str(test_file.relative_to(Path.cwd()))
                    except ValueError:
                        # If that fails, use relative to integration dir
                        relative_path = str(Path(PerformanceConfig.INTEGRATION_TEST_DIR) / test_file.name)
                else:
                    relative_path = str(test_file)
            except Exception:
                # Fallback to just the filename with integration dir prefix
                relative_path = str(Path(PerformanceConfig.INTEGRATION_TEST_DIR) / test_file.name)
            
            # Add file-level identifier
            test_identifiers.append(TestIdentifier(
                file_path=relative_path,
                class_name=None,
                method_name=None
            ))
            
            # Parse file for test methods and classes
            try:
                test_methods = TestDiscovery._parse_test_methods(test_file)
                for class_name, method_name in test_methods:
                    test_identifiers.append(TestIdentifier(
                        file_path=relative_path,
                        class_name=class_name,
                        method_name=method_name
                    ))
            except Exception:
                # If parsing fails, just use file-level identifier
                continue
        
        return test_identifiers
    
    @staticmethod
    def _parse_test_methods(file_path: Path) -> List[Tuple[Optional[str], str]]:
        """Parse test file to extract test methods and classes."""
        test_methods = []
        current_class = None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find test classes and methods
            lines = content.split('\n')
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                
                # Check for test class (any class starting with Test)
                if line.startswith('class ') and 'Test' in line:
                    class_match = re.match(r'^class\s+(\w*Test\w*).*:', line)
                    if class_match:
                        current_class = class_match.group(1)
                        continue
                
                # Check for test method (indented def test_*)
                if re.match(r'^\s+.*def\s+test_\w+', line):
                    method_match = re.match(r'^\s+.*def\s+(test_\w+)', line)
                    if method_match:
                        method_name = method_match.group(1)
                        test_methods.append((current_class, method_name))
                
                # Reset class context if we hit a new class at root level
                if line.startswith('class ') and not ('Test' in line):
                    current_class = None
                
                # Reset class context if we hit a function at root level
                if line.startswith('def ') and not line.startswith('    '):
                    current_class = None
        
        except Exception:
            # Return empty list if parsing fails
            pass
        
        return test_methods


class CommandBuilder:
    """Utilities for building test execution commands."""
    
    @staticmethod
    def build_pytest_command(test_identifier: TestIdentifier, 
                           timeout: Optional[int] = None,
                           verbose: bool = True,
                           capture_output: bool = True) -> List[str]:
        """Build pytest command for executing a specific test."""
        cmd = [EnvironmentConfig.get_pytest_executable()]
        
        # Add test specification
        cmd.append(test_identifier.full_name)
        
        # Add common options
        if verbose:
            cmd.append('-v')
        
        if capture_output:
            cmd.extend(['-s', '--tb=short'])
        
        # Note: Timeout is handled at the process level via asyncio.wait_for()
        # rather than using pytest-timeout plugin to avoid dependency issues
        
        # Disable warnings for cleaner output
        cmd.append('--disable-warnings')
        
        return cmd
    
    @staticmethod
    def build_cprofile_command(test_identifier: TestIdentifier,
                             profile_output_path: Path) -> List[str]:
        """Build command for running test with cProfile."""
        python_exe = EnvironmentConfig.get_python_executable()
        pytest_cmd = CommandBuilder.build_pytest_command(test_identifier)
        
        # Build cProfile command
        cmd = [
            python_exe, '-m', 'cProfile',
            '-o', str(profile_output_path),
            '-m', 'pytest'
        ]
        
        # Add pytest arguments (skip the 'pytest' executable)
        cmd.extend(pytest_cmd[1:])
        
        return cmd


class ProcessManager:
    """Utilities for managing test execution processes."""
    
    @staticmethod
    async def execute_with_timeout(command: List[str], 
                                 timeout_seconds: float,
                                 cwd: Optional[str] = None) -> ExecutionResult:
        """Execute command with timeout and capture results."""
        start_time = time.time()
        timeout_occurred = False
        
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout_seconds
                )
                exit_code = process.returncode
                
            except asyncio.TimeoutError:
                # Kill the process on timeout
                timeout_occurred = True
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass
                
                stdout = b"Process timed out"
                stderr = b"Process killed due to timeout"
                exit_code = -1
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            return ExecutionResult(
                command=' '.join(command),
                exit_code=exit_code,
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                execution_time=execution_time,
                timeout_occurred=timeout_occurred,
                start_time=start_time,
                end_time=end_time
            )
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            return ExecutionResult(
                command=' '.join(command),
                exit_code=-1,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                execution_time=execution_time,
                timeout_occurred=False,
                start_time=start_time,
                end_time=end_time
            )


class PatternMatcher:
    """Utilities for matching function patterns in profiling data."""
    
    def __init__(self):
        """Initialize pattern matchers."""
        self._component_patterns = self._compile_patterns(
            PerformanceConfig.COMPONENT_PATTERNS
        )
        self._llm_patterns = self._compile_pattern_list(
            PerformanceConfig.LLM_API_PATTERNS
        )
        self._test_patterns = self._compile_pattern_list(
            PerformanceConfig.TEST_PATTERNS
        )
    
    def _compile_patterns(self, pattern_dict: Dict[str, List[str]]) -> Dict[str, List[Pattern]]:
        """Compile string patterns to regex patterns."""
        compiled = {}
        for component, patterns in pattern_dict.items():
            compiled[component] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return compiled
    
    def _compile_pattern_list(self, patterns: List[str]) -> List[Pattern]:
        """Compile list of string patterns to regex patterns."""
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def match_component(self, function_name: str) -> Optional[str]:
        """Match function name to ImplementAgent component."""
        for component, patterns in self._component_patterns.items():
            for pattern in patterns:
                if pattern.search(function_name):
                    return component
        return None
    
    def is_llm_call(self, function_name: str) -> bool:
        """Check if function is an LLM API call."""
        return any(pattern.search(function_name) for pattern in self._llm_patterns)
    
    def is_test_infrastructure(self, function_name: str) -> bool:
        """Check if function is part of test infrastructure."""
        return any(pattern.search(function_name) for pattern in self._test_patterns)


class FileManager:
    """Utilities for managing performance analysis files."""
    
    @staticmethod
    def ensure_output_directories():
        """Ensure all required output directories exist."""
        PerformanceConfig.get_output_dir("profiles")
        PerformanceConfig.get_output_dir("flamegraphs") 
        PerformanceConfig.get_output_dir("reports")
    
    @staticmethod
    def cleanup_old_files(max_age_days: int = 7):
        """Clean up old performance analysis files."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        for subdir in ["profiles", "flamegraphs", "reports"]:
            output_dir = PerformanceConfig.get_output_dir(subdir)
            for file_path in output_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                    except Exception:
                        pass  # Ignore cleanup errors
    
    @staticmethod
    def get_unique_filename(base_path: Path) -> Path:
        """Get unique filename by adding counter if file exists."""
        if not base_path.exists():
            return base_path
        
        counter = 1
        while True:
            stem = base_path.stem
            suffix = base_path.suffix
            new_name = f"{stem}_{counter}{suffix}"
            new_path = base_path.parent / new_name
            
            if not new_path.exists():
                return new_path
            
            counter += 1


class StatusConverter:
    """Utilities for converting between different status representations."""
    
    @staticmethod
    def execution_result_to_test_status(result: ExecutionResult) -> TestStatus:
        """Convert ExecutionResult to TestStatus."""
        if result.timeout_occurred:
            return TestStatus.TIMEOUT
        elif result.exit_code == 0:
            return TestStatus.PASSED
        elif result.exit_code != 0:
            # Check stderr for specific error types
            stderr_lower = result.stderr.lower()
            if 'error' in stderr_lower or 'exception' in stderr_lower:
                return TestStatus.ERROR
            else:
                return TestStatus.FAILED
        else:
            return TestStatus.ERROR
    
    @staticmethod
    def pytest_output_to_status(stdout: str, stderr: str, exit_code: int) -> TestStatus:
        """Convert pytest output to TestStatus."""
        if exit_code == 0:
            return TestStatus.PASSED
        
        # Check for specific pytest patterns
        output = (stdout + stderr).lower()
        
        if 'failed' in output:
            return TestStatus.FAILED
        elif 'error' in output:
            return TestStatus.ERROR
        elif 'skipped' in output:
            return TestStatus.SKIPPED
        else:
            return TestStatus.ERROR


# Context managers for resource management
@asynccontextmanager
async def temporary_process_group():
    """Context manager for creating a new process group."""
    original_pgrp = os.getpgrp()
    try:
        # Create new process group
        os.setpgrp()
        yield
    finally:
        # Restore original process group
        try:
            os.setpgrp(original_pgrp)
        except:
            pass  # Ignore errors during cleanup