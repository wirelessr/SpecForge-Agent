"""
Quality Metrics Framework for ImplementAgent evaluation.

This module provides comprehensive quality assessment capabilities for measuring
the output quality of ImplementAgent task execution. It includes objective scoring
for functionality, maintainability, standards compliance, test coverage, and documentation.
"""

import json
import os
import re
import ast
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum


class QualityMetric(Enum):
    """Enumeration of quality metrics."""
    FUNCTIONALITY = "functionality"
    MAINTAINABILITY = "maintainability"
    STANDARDS_COMPLIANCE = "standards_compliance"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION = "documentation"


@dataclass
class QualityScore:
    """
    Represents a quality score for a specific metric.
    
    Scores are on a 1-10 scale with specific criteria for each level.
    """
    metric: QualityMetric
    score: float  # 1-10 scale
    max_score: float = 10.0
    details: Dict[str, Any] = field(default_factory=dict)
    criteria_met: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def percentage(self) -> float:
        """Get score as percentage."""
        return (self.score / self.max_score) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric": self.metric.value,
            "score": self.score,
            "max_score": self.max_score,
            "percentage": self.percentage,
            "details": self.details,
            "criteria_met": self.criteria_met,
            "criteria_failed": self.criteria_failed,
            "recommendations": self.recommendations
        }


@dataclass
class QualityReport:
    """
    Comprehensive quality assessment report for a task execution.
    """
    task_id: str
    task_title: str
    execution_timestamp: str
    scores: Dict[QualityMetric, QualityScore] = field(default_factory=dict)
    overall_score: float = 0.0
    execution_success: bool = False
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    commands_executed: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    
    def add_score(self, score: QualityScore) -> None:
        """Add a quality score to the report."""
        self.scores[score.metric] = score
        self._calculate_overall_score()
    
    def _calculate_overall_score(self) -> None:
        """Calculate weighted overall score."""
        if not self.scores:
            self.overall_score = 0.0
            return
        
        # Weighted scoring system
        weights = {
            QualityMetric.FUNCTIONALITY: 0.3,
            QualityMetric.MAINTAINABILITY: 0.2,
            QualityMetric.STANDARDS_COMPLIANCE: 0.2,
            QualityMetric.TEST_COVERAGE: 0.15,
            QualityMetric.DOCUMENTATION: 0.15
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric, score in self.scores.items():
            weight = weights.get(metric, 0.1)
            total_weighted_score += score.score * weight
            total_weight += weight
        
        self.overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "task_title": self.task_title,
            "execution_timestamp": self.execution_timestamp,
            "overall_score": self.overall_score,
            "execution_success": self.execution_success,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "commands_executed": self.commands_executed,
            "execution_time": self.execution_time,
            "scores": {metric.value: score.to_dict() for metric, score in self.scores.items()},
            "metadata": {
                "framework_version": "1.0.0",
                "evaluation_duration": 0.0,  # Will be set during evaluation
                "work_directory": "",  # Will be set during evaluation
                "agent_version": "current",
                "llm_model": "unknown",
                "environment": "test"
            },
            "comparison": {
                "baseline_report_id": None,
                "improvement_over_baseline": None,
                "metric_improvements": {},
                "regression_detected": False
            }
        }


class FunctionalityEvaluator:
    """Evaluates functionality - does the code work as intended."""
    
    def evaluate(self, work_dir: Path, task_def: Any, execution_result: Dict[str, Any]) -> QualityScore:
        """
        Evaluate functionality score.
        
        Criteria:
        - 9-10: Code executes successfully, meets all requirements, handles edge cases
        - 7-8: Code executes successfully, meets most requirements
        - 5-6: Code executes with minor issues, meets basic requirements
        - 3-4: Code has execution issues but partially works
        - 1-2: Code fails to execute or doesn't meet basic requirements
        """
        score = QualityScore(metric=QualityMetric.FUNCTIONALITY, score=5.0)
        
        # Check if execution was successful
        execution_success = execution_result.get('success', False)
        if not execution_success:
            score.score = 2.0
            score.criteria_failed.append("Task execution failed")
            score.recommendations.append("Fix execution errors and ensure task completes successfully")
            return score
        
        # Check if expected files were created
        files_created = self._check_expected_files(work_dir, task_def)
        score.details['files_created'] = files_created
        
        # Check if code can be executed (for Python files)
        syntax_valid = self._check_syntax_validity(work_dir)
        score.details['syntax_valid'] = syntax_valid
        
        # Check if basic requirements are met
        requirements_met = self._check_requirements_fulfillment(work_dir, task_def)
        score.details['requirements_met'] = requirements_met
        
        # Calculate score based on criteria
        base_score = 5.0  # Start with middle score
        
        if files_created:
            base_score += 1.0
            score.criteria_met.append("Expected files were created")
        else:
            score.criteria_failed.append("Expected files were not created")
        
        if syntax_valid:
            base_score += 1.5
            score.criteria_met.append("Code has valid syntax")
        else:
            base_score -= 2.0
            score.criteria_failed.append("Code has syntax errors")
        
        if requirements_met:
            base_score += 1.5
            score.criteria_met.append("Basic requirements are fulfilled")
        else:
            score.criteria_failed.append("Basic requirements are not met")
        
        score.score = max(1.0, min(10.0, base_score))
        
        if score.score < 5.0:
            score.recommendations.append("Fix syntax errors and ensure basic functionality works")
        elif score.score < 7.0:
            score.recommendations.append("Improve error handling and edge case coverage")
        
        return score
    
    def _check_expected_files(self, work_dir: Path, task_def: Any) -> bool:
        """Check if expected files were created based on task description."""
        # Simple heuristic: look for common file patterns in task description
        description = getattr(task_def, 'description', '') + ' '.join(getattr(task_def, 'steps', []))
        
        # Look for file mentions in description
        file_patterns = re.findall(r'(\w+\.\w+)', description)
        
        if not file_patterns:
            # If no specific files mentioned, check if any files were created
            return len(list(work_dir.glob('*'))) > 3  # More than just requirements.md, design.md, tasks.md
        
        # Check if mentioned files exist
        for pattern in file_patterns:
            if (work_dir / pattern).exists():
                return True
        
        return False
    
    def _check_syntax_validity(self, work_dir: Path) -> bool:
        """Check if Python files have valid syntax."""
        python_files = list(work_dir.glob('**/*.py'))
        
        if not python_files:
            return True  # No Python files to check
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError:
                return False
            except Exception:
                continue  # Skip files that can't be read
        
        return True
    
    def _check_requirements_fulfillment(self, work_dir: Path, task_def: Any) -> bool:
        """Check if basic requirements from task definition are met."""
        # This is a simplified check - in practice, this would be more sophisticated
        requirements_refs = getattr(task_def, 'requirements_ref', [])
        
        if not requirements_refs:
            return True  # No specific requirements to check
        
        # Check if requirements.md exists and contains referenced requirements
        req_file = work_dir / 'requirements.md'
        if not req_file.exists():
            return False
        
        try:
            req_content = req_file.read_text(encoding='utf-8')
            # Simple check: see if requirement references appear in requirements file
            for req_ref in requirements_refs:
                if req_ref not in req_content:
                    return False
            return True
        except Exception:
            return False


class MaintainabilityEvaluator:
    """Evaluates code maintainability - readability, structure, comments."""
    
    def evaluate(self, work_dir: Path, task_def: Any, execution_result: Dict[str, Any]) -> QualityScore:
        """
        Evaluate maintainability score.
        
        Criteria:
        - 9-10: Excellent structure, comprehensive comments, clear naming
        - 7-8: Good structure, adequate comments, mostly clear naming
        - 5-6: Acceptable structure, some comments, reasonable naming
        - 3-4: Poor structure, minimal comments, unclear naming
        - 1-2: Very poor structure, no comments, confusing code
        """
        score = QualityScore(metric=QualityMetric.MAINTAINABILITY, score=5.0)
        
        # Analyze Python files for maintainability
        python_files = list(work_dir.glob('**/*.py'))
        
        if not python_files:
            score.score = 5.0  # Neutral score if no Python files
            score.details['python_files_found'] = False
            return score
        
        score.details['python_files_found'] = True
        score.details['files_analyzed'] = len(python_files)
        
        # Check various maintainability aspects
        comment_ratio = self._calculate_comment_ratio(python_files)
        function_complexity = self._analyze_function_complexity(python_files)
        naming_quality = self._analyze_naming_quality(python_files)
        structure_quality = self._analyze_structure_quality(python_files)
        
        score.details.update({
            'comment_ratio': comment_ratio,
            'avg_function_complexity': function_complexity,
            'naming_quality': naming_quality,
            'structure_quality': structure_quality
        })
        
        # Calculate score based on criteria
        base_score = 5.0
        
        # Comment ratio scoring
        if comment_ratio > 0.15:  # >15% comments
            base_score += 1.5
            score.criteria_met.append("Good comment coverage")
        elif comment_ratio > 0.05:  # >5% comments
            base_score += 0.5
            score.criteria_met.append("Adequate comment coverage")
        else:
            base_score -= 1.0
            score.criteria_failed.append("Insufficient comments")
        
        # Function complexity scoring
        if function_complexity < 5:
            base_score += 1.0
            score.criteria_met.append("Low function complexity")
        elif function_complexity > 10:
            base_score -= 1.0
            score.criteria_failed.append("High function complexity")
        
        # Naming quality scoring
        if naming_quality > 0.8:
            base_score += 1.0
            score.criteria_met.append("Good naming conventions")
        elif naming_quality < 0.5:
            base_score -= 1.0
            score.criteria_failed.append("Poor naming conventions")
        
        # Structure quality scoring
        if structure_quality > 0.7:
            base_score += 0.5
            score.criteria_met.append("Good code structure")
        elif structure_quality < 0.4:
            base_score -= 0.5
            score.criteria_failed.append("Poor code structure")
        
        score.score = max(1.0, min(10.0, base_score))
        
        # Add recommendations
        if comment_ratio < 0.1:
            score.recommendations.append("Add more comments to explain complex logic")
        if function_complexity > 8:
            score.recommendations.append("Break down complex functions into smaller ones")
        if naming_quality < 0.6:
            score.recommendations.append("Use more descriptive variable and function names")
        
        return score
    
    def _calculate_comment_ratio(self, python_files: List[Path]) -> float:
        """Calculate ratio of comment lines to total lines."""
        total_lines = 0
        comment_lines = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                total_lines += len(lines)
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        comment_lines += 1
            except Exception:
                continue
        
        return comment_lines / total_lines if total_lines > 0 else 0.0
    
    def _analyze_function_complexity(self, python_files: List[Path]) -> float:
        """Analyze average function complexity (simplified cyclomatic complexity)."""
        complexities = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        complexities.append(complexity)
            except Exception:
                continue
        
        return sum(complexities) / len(complexities) if complexities else 1.0
    
    def _calculate_cyclomatic_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate simplified cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _analyze_naming_quality(self, python_files: List[Path]) -> float:
        """Analyze quality of naming conventions."""
        good_names = 0
        total_names = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_names += 1
                        if self._is_good_name(node.name):
                            good_names += 1
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        total_names += 1
                        if self._is_good_name(node.id):
                            good_names += 1
            except Exception:
                continue
        
        return good_names / total_names if total_names > 0 else 0.5
    
    def _is_good_name(self, name: str) -> bool:
        """Check if a name follows good naming conventions."""
        # Avoid single letter names (except common ones like i, j, x, y)
        if len(name) == 1 and name not in ['i', 'j', 'x', 'y']:
            return False
        
        # Avoid generic names
        generic_names = {'temp', 'tmp', 'data', 'var', 'obj', 'item'}
        if name.lower() in generic_names:
            return False
        
        # Check for descriptive length
        if len(name) < 3 and name not in ['id', 'db', 'ui', 'os', 'io']:
            return False
        
        return True
    
    def _analyze_structure_quality(self, python_files: List[Path]) -> float:
        """Analyze overall code structure quality."""
        structure_score = 0.0
        total_files = len(python_files)
        
        for py_file in python_files:
            file_score = 0.0
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Check for docstrings
                if ast.get_docstring(tree):
                    file_score += 0.3
                
                # Check for proper imports organization
                imports_at_top = True
                found_non_import = False
                for node in tree.body:
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if found_non_import:
                            imports_at_top = False
                            break
                    else:
                        found_non_import = True
                
                if imports_at_top:
                    file_score += 0.2
                
                # Check for reasonable file length
                lines = content.count('\n')
                if lines < 500:  # Not too long
                    file_score += 0.3
                
                # Check for function organization
                functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
                if functions:
                    file_score += 0.2
                
                structure_score += min(1.0, file_score)
            except Exception:
                continue
        
        return structure_score / total_files if total_files > 0 else 0.5


class StandardsComplianceEvaluator:
    """Evaluates adherence to coding standards and conventions."""
    
    def evaluate(self, work_dir: Path, task_def: Any, execution_result: Dict[str, Any]) -> QualityScore:
        """
        Evaluate standards compliance score.
        
        Criteria:
        - 9-10: Excellent adherence to PEP 8, proper imports, consistent style
        - 7-8: Good adherence with minor violations
        - 5-6: Acceptable adherence with some violations
        - 3-4: Poor adherence with many violations
        - 1-2: Very poor adherence, inconsistent style
        """
        score = QualityScore(metric=QualityMetric.STANDARDS_COMPLIANCE, score=5.0)
        
        python_files = list(work_dir.glob('**/*.py'))
        
        if not python_files:
            score.score = 5.0  # Neutral score if no Python files
            score.details['python_files_found'] = False
            return score
        
        # Run basic style checks
        pep8_violations = self._check_pep8_compliance(python_files)
        import_organization = self._check_import_organization(python_files)
        line_length_violations = self._check_line_length(python_files)
        indentation_consistency = self._check_indentation_consistency(python_files)
        
        score.details.update({
            'pep8_violations': pep8_violations,
            'import_organization_score': import_organization,
            'line_length_violations': line_length_violations,
            'indentation_consistent': indentation_consistency
        })
        
        # Calculate score based on violations
        base_score = 8.0  # Start with good score
        
        # PEP 8 violations
        if pep8_violations == 0:
            score.criteria_met.append("No PEP 8 violations found")
        elif pep8_violations < 5:
            base_score -= 1.0
            score.criteria_failed.append(f"Minor PEP 8 violations ({pep8_violations})")
        else:
            base_score -= 2.0
            score.criteria_failed.append(f"Multiple PEP 8 violations ({pep8_violations})")
        
        # Import organization
        if import_organization > 0.8:
            score.criteria_met.append("Good import organization")
        elif import_organization < 0.5:
            base_score -= 1.0
            score.criteria_failed.append("Poor import organization")
        
        # Line length
        if line_length_violations == 0:
            score.criteria_met.append("No line length violations")
        elif line_length_violations > 10:
            base_score -= 1.0
            score.criteria_failed.append(f"Many long lines ({line_length_violations})")
        
        # Indentation
        if indentation_consistency:
            score.criteria_met.append("Consistent indentation")
        else:
            base_score -= 1.5
            score.criteria_failed.append("Inconsistent indentation")
        
        score.score = max(1.0, min(10.0, base_score))
        
        # Add recommendations
        if pep8_violations > 0:
            score.recommendations.append("Fix PEP 8 style violations")
        if line_length_violations > 5:
            score.recommendations.append("Break long lines to improve readability")
        if not indentation_consistency:
            score.recommendations.append("Use consistent indentation (spaces or tabs, not mixed)")
        
        return score
    
    def _check_pep8_compliance(self, python_files: List[Path]) -> int:
        """Check for basic PEP 8 violations."""
        violations = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    # Check for trailing whitespace
                    if line.rstrip() != line:
                        violations += 1
                    
                    # Check for multiple spaces after comma
                    if re.search(r',\s{2,}', line):
                        violations += 1
                    
                    # Check for space before comma
                    if re.search(r'\s,', line):
                        violations += 1
                    
                    # Check for missing space after comma
                    if re.search(r',[^\s\]]', line):
                        violations += 1
            except Exception:
                continue
        
        return violations
    
    def _check_import_organization(self, python_files: List[Path]) -> float:
        """Check import organization quality."""
        good_files = 0
        total_files = len(python_files)
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Check if imports are at the top
                imports_at_top = True
                found_non_import = False
                
                for node in tree.body:
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if found_non_import:
                            imports_at_top = False
                            break
                    elif not isinstance(node, (ast.Expr, ast.Assign)) or not ast.get_docstring(tree):
                        found_non_import = True
                
                if imports_at_top:
                    good_files += 1
            except Exception:
                continue
        
        return good_files / total_files if total_files > 0 else 0.0
    
    def _check_line_length(self, python_files: List[Path]) -> int:
        """Check for lines exceeding reasonable length."""
        violations = 0
        max_length = 88  # Black's default
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if len(line.rstrip()) > max_length:
                        violations += 1
            except Exception:
                continue
        
        return violations
    
    def _check_indentation_consistency(self, python_files: List[Path]) -> bool:
        """Check for consistent indentation."""
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for mixed tabs and spaces
                has_tabs = '\t' in content
                has_spaces = re.search(r'^ +', content, re.MULTILINE)
                
                if has_tabs and has_spaces:
                    return False
            except Exception:
                continue
        
        return True


class TestCoverageEvaluator:
    """Evaluates test coverage and test quality."""
    
    def evaluate(self, work_dir: Path, task_def: Any, execution_result: Dict[str, Any]) -> QualityScore:
        """
        Evaluate test coverage score.
        
        Criteria:
        - 9-10: Comprehensive tests, high coverage, good test structure
        - 7-8: Good test coverage, adequate test structure
        - 5-6: Basic tests present, moderate coverage
        - 3-4: Minimal tests, poor coverage
        - 1-2: No tests or very poor test coverage
        """
        score = QualityScore(metric=QualityMetric.TEST_COVERAGE, score=5.0)
        
        # Find test files
        test_files = list(work_dir.glob('**/test_*.py')) + list(work_dir.glob('**/*_test.py'))
        python_files = list(work_dir.glob('**/*.py'))
        
        # Exclude test files from main code files
        main_files = [f for f in python_files if f not in test_files]
        
        score.details.update({
            'test_files_found': len(test_files),
            'main_files_found': len(main_files),
            'test_to_code_ratio': len(test_files) / len(main_files) if main_files else 0
        })
        
        if not test_files:
            score.score = 1.0
            score.criteria_failed.append("No test files found")
            score.recommendations.append("Create test files to verify functionality")
            return score
        
        # Analyze test quality
        test_functions = self._count_test_functions(test_files)
        assertion_count = self._count_assertions(test_files)
        test_structure_quality = self._analyze_test_structure(test_files)
        
        score.details.update({
            'test_functions': test_functions,
            'total_assertions': assertion_count,
            'avg_assertions_per_test': assertion_count / test_functions if test_functions else 0,
            'test_structure_quality': test_structure_quality
        })
        
        # Calculate score
        base_score = 3.0  # Start low since tests exist
        
        # Test file ratio
        test_ratio = len(test_files) / len(main_files) if main_files else 1.0
        if test_ratio >= 0.5:  # At least half as many test files as main files
            base_score += 2.0
            score.criteria_met.append("Good test file coverage")
        elif test_ratio >= 0.2:
            base_score += 1.0
            score.criteria_met.append("Adequate test file coverage")
        else:
            score.criteria_failed.append("Insufficient test file coverage")
        
        # Test function count
        if test_functions >= 5:
            base_score += 2.0
            score.criteria_met.append("Multiple test functions")
        elif test_functions >= 2:
            base_score += 1.0
            score.criteria_met.append("Basic test functions present")
        else:
            score.criteria_failed.append("Too few test functions")
        
        # Assertion quality
        avg_assertions = assertion_count / test_functions if test_functions else 0
        if avg_assertions >= 3:
            base_score += 1.5
            score.criteria_met.append("Good assertion coverage")
        elif avg_assertions >= 1:
            base_score += 0.5
            score.criteria_met.append("Basic assertions present")
        else:
            score.criteria_failed.append("Insufficient assertions")
        
        # Test structure
        if test_structure_quality > 0.7:
            base_score += 1.0
            score.criteria_met.append("Good test structure")
        elif test_structure_quality < 0.4:
            base_score -= 0.5
            score.criteria_failed.append("Poor test structure")
        
        score.score = max(1.0, min(10.0, base_score))
        
        # Add recommendations
        if test_functions < 3:
            score.recommendations.append("Add more test functions to cover different scenarios")
        if avg_assertions < 2:
            score.recommendations.append("Add more assertions to verify behavior thoroughly")
        if test_ratio < 0.3:
            score.recommendations.append("Create more test files to match code structure")
        
        return score
    
    def _count_test_functions(self, test_files: List[Path]) -> int:
        """Count test functions in test files."""
        count = 0
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        count += 1
            except Exception:
                continue
        
        return count
    
    def _count_assertions(self, test_files: List[Path]) -> int:
        """Count assertion statements in test files."""
        count = 0
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple regex count for assert statements
                assertions = re.findall(r'\bassert\b', content)
                count += len(assertions)
            except Exception:
                continue
        
        return count
    
    def _analyze_test_structure(self, test_files: List[Path]) -> float:
        """Analyze test structure quality."""
        structure_scores = []
        
        for test_file in test_files:
            file_score = 0.0
            
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Check for test class organization
                has_test_classes = any(isinstance(node, ast.ClassDef) for node in tree.body)
                if has_test_classes:
                    file_score += 0.3
                
                # Check for setup/teardown methods
                has_setup = 'setUp' in content or 'setup' in content or '@pytest.fixture' in content
                if has_setup:
                    file_score += 0.2
                
                # Check for docstrings in test functions
                test_functions = [node for node in ast.walk(tree) 
                                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')]
                
                documented_tests = sum(1 for func in test_functions if ast.get_docstring(func))
                if test_functions:
                    file_score += 0.3 * (documented_tests / len(test_functions))
                
                # Check for imports (pytest, unittest, etc.)
                has_test_imports = any('test' in line or 'pytest' in line or 'unittest' in line 
                                     for line in content.split('\n')[:20])
                if has_test_imports:
                    file_score += 0.2
                
                structure_scores.append(min(1.0, file_score))
            except Exception:
                continue
        
        return sum(structure_scores) / len(structure_scores) if structure_scores else 0.0


class DocumentationEvaluator:
    """Evaluates documentation quality."""
    
    def evaluate(self, work_dir: Path, task_def: Any, execution_result: Dict[str, Any]) -> QualityScore:
        """
        Evaluate documentation score.
        
        Criteria:
        - 9-10: Comprehensive documentation, README, docstrings, comments
        - 7-8: Good documentation with README and most functions documented
        - 5-6: Basic documentation, some docstrings or README
        - 3-4: Minimal documentation, few docstrings
        - 1-2: No documentation or very poor documentation
        """
        score = QualityScore(metric=QualityMetric.DOCUMENTATION, score=5.0)
        
        # Check for README files
        readme_files = list(work_dir.glob('README*')) + list(work_dir.glob('readme*'))
        has_readme = len(readme_files) > 0
        
        # Check Python file documentation
        python_files = list(work_dir.glob('**/*.py'))
        docstring_coverage = self._calculate_docstring_coverage(python_files)
        comment_density = self._calculate_comment_density(python_files)
        
        score.details.update({
            'has_readme': has_readme,
            'readme_files': [str(f.name) for f in readme_files],
            'python_files_count': len(python_files),
            'docstring_coverage': docstring_coverage,
            'comment_density': comment_density
        })
        
        # Calculate score
        base_score = 2.0  # Start low
        
        # README presence
        if has_readme:
            base_score += 3.0
            score.criteria_met.append("README file present")
            
            # Check README quality
            readme_quality = self._analyze_readme_quality(readme_files[0])
            base_score += readme_quality * 1.0
            score.details['readme_quality'] = readme_quality
        else:
            score.criteria_failed.append("No README file found")
            score.recommendations.append("Create a README file to explain the project")
        
        # Docstring coverage
        if docstring_coverage > 0.8:
            base_score += 2.5
            score.criteria_met.append("Excellent docstring coverage")
        elif docstring_coverage > 0.5:
            base_score += 1.5
            score.criteria_met.append("Good docstring coverage")
        elif docstring_coverage > 0.2:
            base_score += 0.5
            score.criteria_met.append("Basic docstring coverage")
        else:
            score.criteria_failed.append("Poor docstring coverage")
            score.recommendations.append("Add docstrings to functions and classes")
        
        # Comment density
        if comment_density > 0.1:
            base_score += 1.5
            score.criteria_met.append("Good comment density")
        elif comment_density > 0.05:
            base_score += 0.5
            score.criteria_met.append("Adequate comment density")
        else:
            score.criteria_failed.append("Insufficient comments")
            score.recommendations.append("Add more comments to explain complex logic")
        
        score.score = max(1.0, min(10.0, base_score))
        
        return score
    
    def _calculate_docstring_coverage(self, python_files: List[Path]) -> float:
        """Calculate percentage of functions/classes with docstrings."""
        total_items = 0
        documented_items = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Check module docstring
                if ast.get_docstring(tree):
                    documented_items += 1
                total_items += 1
                
                # Check functions and classes
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        total_items += 1
                        if ast.get_docstring(node):
                            documented_items += 1
            except Exception:
                continue
        
        return documented_items / total_items if total_items > 0 else 0.0
    
    def _calculate_comment_density(self, python_files: List[Path]) -> float:
        """Calculate ratio of comment lines to total lines."""
        total_lines = 0
        comment_lines = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                total_lines += len(lines)
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        comment_lines += 1
            except Exception:
                continue
        
        return comment_lines / total_lines if total_lines > 0 else 0.0
    
    def _analyze_readme_quality(self, readme_file: Path) -> float:
        """Analyze README file quality."""
        try:
            content = readme_file.read_text(encoding='utf-8')
            
            quality_score = 0.0
            
            # Check for common sections
            sections = ['installation', 'usage', 'example', 'description', 'requirements']
            for section in sections:
                if section.lower() in content.lower():
                    quality_score += 0.15
            
            # Check for code examples
            if '```' in content or '    ' in content:  # Code blocks
                quality_score += 0.2
            
            # Check for reasonable length
            if len(content) > 200:
                quality_score += 0.1
            
            return min(1.0, quality_score)
        except Exception:
            return 0.0


class QualityMetricsFramework:
    """
    Main framework for evaluating ImplementAgent output quality.
    
    Provides comprehensive quality assessment with objective scoring
    across multiple dimensions.
    """
    
    def __init__(self):
        self.evaluators = {
            QualityMetric.FUNCTIONALITY: FunctionalityEvaluator(),
            QualityMetric.MAINTAINABILITY: MaintainabilityEvaluator(),
            QualityMetric.STANDARDS_COMPLIANCE: StandardsComplianceEvaluator(),
            QualityMetric.TEST_COVERAGE: TestCoverageEvaluator(),
            QualityMetric.DOCUMENTATION: DocumentationEvaluator()
        }
    
    def evaluate_task_execution(self, work_dir: Path, task_def: Any, 
                              execution_result: Dict[str, Any]) -> QualityReport:
        """
        Evaluate the quality of a task execution.
        
        Args:
            work_dir: Directory where task was executed
            task_def: Task definition that was executed
            execution_result: Result of task execution
            
        Returns:
            QualityReport with comprehensive quality assessment
        """
        evaluation_start = datetime.now()
        
        report = QualityReport(
            task_id=getattr(task_def, 'id', 'unknown'),
            task_title=getattr(task_def, 'title', 'Unknown Task'),
            execution_timestamp=datetime.now().isoformat(),
            execution_success=execution_result.get('success', False),
            execution_time=execution_result.get('execution_time', 0.0)
        )
        
        # Collect file information
        if work_dir.exists():
            all_files = list(work_dir.glob('**/*'))
            report.files_created = [str(f.relative_to(work_dir)) for f in all_files if f.is_file()]
        
        # Collect command information
        report.commands_executed = execution_result.get('commands_executed', [])
        
        # Evaluate each quality metric
        for metric, evaluator in self.evaluators.items():
            try:
                score = evaluator.evaluate(work_dir, task_def, execution_result)
                report.add_score(score)
            except Exception as e:
                # Create a failed score if evaluation fails
                failed_score = QualityScore(metric=metric, score=1.0)
                failed_score.criteria_failed.append(f"Evaluation failed: {str(e)}")
                failed_score.recommendations.append("Fix evaluation errors")
                report.add_score(failed_score)
        
        return report
    
    def save_report(self, report: QualityReport, output_dir: Path, work_dir: Path = None) -> Path:
        """
        Save quality report to JSON file.
        
        Args:
            report: Quality report to save
            output_dir: Directory to save report in
            work_dir: Work directory for metadata (optional)
            
        Returns:
            Path to saved report file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quality_report_{report.task_id}_{timestamp}.json"
        report_path = output_dir / filename
        
        # Prepare report data with metadata
        report_data = report.to_dict()
        
        # Update metadata
        report_data["metadata"].update({
            "evaluation_duration": 0.0,  # Could be calculated if needed
            "work_directory": str(work_dir) if work_dir else "",
            "framework_version": "1.0.0",
            "agent_version": "current",
            "llm_model": "unknown",  # Could be extracted from execution result
            "environment": "test"
        })
        
        # Save report as JSON
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return report_path
    
    def compare_reports(self, baseline_report: QualityReport, 
                       current_report: QualityReport) -> Dict[str, Any]:
        """
        Compare two quality reports to show improvement/regression.
        
        Args:
            baseline_report: Baseline quality report
            current_report: Current quality report to compare
            
        Returns:
            Comparison results with improvements and regressions
        """
        comparison = {
            'overall_improvement': current_report.overall_score - baseline_report.overall_score,
            'metric_comparisons': {},
            'improvements': [],
            'regressions': []
        }
        
        for metric in QualityMetric:
            if metric in baseline_report.scores and metric in current_report.scores:
                baseline_score = baseline_report.scores[metric].score
                current_score = current_report.scores[metric].score
                improvement = current_score - baseline_score
                
                comparison['metric_comparisons'][metric.value] = {
                    'baseline': baseline_score,
                    'current': current_score,
                    'improvement': improvement
                }
                
                if improvement > 0.5:
                    comparison['improvements'].append({
                        'metric': metric.value,
                        'improvement': improvement,
                        'description': f"{metric.value} improved by {improvement:.1f} points"
                    })
                elif improvement < -0.5:
                    comparison['regressions'].append({
                        'metric': metric.value,
                        'regression': improvement,
                        'description': f"{metric.value} regressed by {abs(improvement):.1f} points"
                    })
        
        return comparison