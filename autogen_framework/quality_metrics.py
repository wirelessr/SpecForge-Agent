"""
Quality Metrics Framework for ImplementAgent evaluation.

This module provides comprehensive quality assessment capabilities for measuring
the output quality of ImplementAgent task execution. It includes objective scoring
for functionality, maintainability, standards compliance, test coverage, and documentation.

Enhanced with LLM-specific validation methods for requirements documents (EARS format),
design documents (Mermaid syntax), task structure validation, and revision improvement assessment.
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
    
    Enhanced with LLM-specific validation methods for document quality assessment.
    """
    
    def __init__(self):
        self.evaluators = {
            QualityMetric.FUNCTIONALITY: FunctionalityEvaluator(),
            QualityMetric.MAINTAINABILITY: MaintainabilityEvaluator(),
            QualityMetric.STANDARDS_COMPLIANCE: StandardsComplianceEvaluator(),
            QualityMetric.TEST_COVERAGE: TestCoverageEvaluator(),
            QualityMetric.DOCUMENTATION: DocumentationEvaluator()
        }
    
    def assess_llm_document_quality(self, content: str, document_type: str) -> Dict[str, Any]:
        """
        Assess LLM-generated document quality.
        
        Args:
            content: Document content to assess
            document_type: Type of document ('requirements', 'design', 'tasks')
            
        Returns:
            Dictionary containing quality assessment results
        """
        assessment = {
            'document_type': document_type,
            'content_length': len(content),
            'structure_score': 0.0,
            'completeness_score': 0.0,
            'format_compliance': False,
            'issues': [],
            'strengths': []
        }
        
        if document_type == 'requirements':
            assessment.update(self._assess_requirements_quality(content))
        elif document_type == 'design':
            assessment.update(self._assess_design_quality(content))
        elif document_type == 'tasks':
            assessment.update(self._assess_tasks_quality(content))
        
        # Calculate overall score
        assessment['overall_score'] = (
            assessment['structure_score'] + assessment['completeness_score']
        ) / 2.0
        
        return assessment
    
    def validate_ears_format(self, requirements_content: str) -> Dict[str, Any]:
        """
        Validate EARS (Easy Approach to Requirements Syntax) format compliance.
        
        Args:
            requirements_content: Requirements document content
            
        Returns:
            Dictionary with EARS format validation results
        """
        validation = {
            'compliance_score': 0.0,
            'ears_patterns_found': 0,
            'user_stories_found': 0,
            'acceptance_criteria_found': 0,
            'issues': [],
            'strengths': []
        }
        
        # Check for EARS patterns
        ears_patterns = [
            r'WHEN\s+.*\s+THEN\s+.*\s+SHALL\s+.*',
            r'IF\s+.*\s+THEN\s+.*\s+SHALL\s+.*',
            r'WHERE\s+.*\s+THEN\s+.*\s+SHALL\s+.*'
        ]
        
        for pattern in ears_patterns:
            matches = re.findall(pattern, requirements_content, re.IGNORECASE)
            validation['ears_patterns_found'] += len(matches)
        
        # Check for user stories format
        user_story_pattern = r'As\s+a\s+.*,\s+I\s+want\s+.*,\s+so\s+that\s+.*'
        user_stories = re.findall(user_story_pattern, requirements_content, re.IGNORECASE)
        validation['user_stories_found'] = len(user_stories)
        
        # Check for acceptance criteria sections
        acceptance_criteria_pattern = r'#+\s*Acceptance\s+Criteria'
        validation['acceptance_criteria_found'] = len(
            re.findall(acceptance_criteria_pattern, requirements_content, re.IGNORECASE)
        )
        
        # Calculate compliance score
        total_elements = (
            validation['ears_patterns_found'] + 
            validation['user_stories_found'] + 
            validation['acceptance_criteria_found']
        )
        
        if total_elements >= 5:  # Good coverage
            validation['compliance_score'] = 1.0
            validation['strengths'].append('Excellent EARS format compliance')
        elif total_elements >= 3:  # Adequate coverage
            validation['compliance_score'] = 0.7
            validation['strengths'].append('Good EARS format compliance')
        elif total_elements >= 1:  # Minimal coverage
            validation['compliance_score'] = 0.4
            validation['issues'].append('Limited EARS format usage')
        else:  # No EARS format
            validation['compliance_score'] = 0.0
            validation['issues'].append('EARS format not used')
        
        return validation
    
    def validate_mermaid_syntax(self, design_content: str) -> Dict[str, Any]:
        """
        Validate Mermaid diagram syntax in design documents.
        
        Args:
            design_content: Design document content
            
        Returns:
            Dictionary with Mermaid syntax validation results
        """
        validation = {
            'mermaid_diagrams_found': 0,
            'valid_diagrams': 0,
            'diagram_types': [],
            'syntax_valid': False,
            'issues': [],
            'strengths': []
        }
        
        # Look for Mermaid code blocks
        mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', design_content, re.DOTALL)
        validation['mermaid_diagrams_found'] = len(mermaid_blocks)
        
        if not mermaid_blocks:
            validation['issues'].append('No Mermaid diagrams found')
            return validation
        
        # Basic syntax validation for common Mermaid diagram types
        diagram_types = {
            'graph': r'^graph\s+(TD|TB|BT|RL|LR)',
            'flowchart': r'^flowchart\s+(TD|TB|BT|RL|LR)',
            'sequence': r'^sequenceDiagram',
            'class': r'^classDiagram',
            'state': r'^stateDiagram',
            'er': r'^erDiagram'
        }
        
        for block in mermaid_blocks:
            block = block.strip()
            
            for diagram_name, pattern in diagram_types.items():
                if re.match(pattern, block, re.IGNORECASE):
                    validation['valid_diagrams'] += 1
                    validation['diagram_types'].append(diagram_name)
                    break
        
        validation['syntax_valid'] = validation['valid_diagrams'] > 0
        
        if validation['syntax_valid']:
            validation['strengths'].append(f'Valid Mermaid diagrams found: {", ".join(validation["diagram_types"])}')
        else:
            validation['issues'].append('No valid Mermaid diagram syntax found')
        
        return validation
    
    def assess_task_structure(self, tasks_content: str) -> Dict[str, Any]:
        """
        Assess task list structure and numbering.
        
        Args:
            tasks_content: Tasks document content
            
        Returns:
            Dictionary containing task structure assessment
        """
        assessment = {
            'sequential_numbering': False,
            'requirement_references': False,
            'actionable_tasks': False,
            'task_count': 0,
            'structure_score': 0.0,
            'tasks_with_requirements': 0,
            'issues': [],
            'strengths': []
        }
        
        # Find task items (checkbox format)
        task_pattern = r'- \[[ x]\] (\d+\.?\d*\.?)\s+(.*)'
        tasks = re.findall(task_pattern, tasks_content)
        
        assessment['task_count'] = len(tasks)
        
        if tasks:
            # Check sequential numbering
            numbers = []
            for task_num, _ in tasks:
                try:
                    numbers.append(float(task_num.rstrip('.')))
                except ValueError:
                    continue
            
            if numbers:
                is_sequential = all(
                    numbers[i] <= numbers[i + 1] for i in range(len(numbers) - 1)
                )
                assessment['sequential_numbering'] = is_sequential
                
                if is_sequential:
                    assessment['strengths'].append('Tasks are sequentially numbered')
                else:
                    assessment['issues'].append('Task numbering is not sequential')
            
            # Check for requirement references
            req_refs = re.findall(r'_Requirements?:\s*([^_]+)_', tasks_content)
            assessment['tasks_with_requirements'] = len(req_refs)
            assessment['requirement_references'] = len(req_refs) > 0
            
            if assessment['requirement_references']:
                assessment['strengths'].append(f'{len(req_refs)} tasks reference requirements')
            else:
                assessment['issues'].append('Tasks do not reference requirements')
            
            # Check for actionable language
            actionable_verbs = ['create', 'implement', 'modify', 'test', 'validate', 'build', 'write', 'add', 'update', 'fix']
            actionable_count = 0
            
            for _, task_desc in tasks:
                if any(verb in task_desc.lower() for verb in actionable_verbs):
                    actionable_count += 1
            
            assessment['actionable_tasks'] = actionable_count / len(tasks) > 0.7
            
            if assessment['actionable_tasks']:
                assessment['strengths'].append('Most tasks use actionable language')
            else:
                assessment['issues'].append('Tasks lack actionable language')
            
            # Calculate structure score
            structure_factors = [
                assessment['sequential_numbering'],
                assessment['requirement_references'],
                assessment['actionable_tasks']
            ]
            assessment['structure_score'] = sum(structure_factors) / len(structure_factors)
        
        return assessment
    
    def assess_revision_improvement(self, original: str, revised: str, feedback: str) -> Dict[str, Any]:
        """
        Assess quality improvement after revision.
        
        Args:
            original: Original document content
            revised: Revised document content
            feedback: Feedback that was provided
            
        Returns:
            Dictionary containing revision improvement assessment
        """
        assessment = {
            'content_expanded': False,
            'feedback_addressed': False,
            'structure_improved': False,
            'quality_increased': False,
            'improvement_score': 0.0,
            'changes_made': [],
            'issues': []
        }
        
        # Check if content was expanded
        original_length = len(original.split())
        revised_length = len(revised.split())
        
        if revised_length > original_length * 1.1:  # At least 10% increase
            assessment['content_expanded'] = True
            assessment['changes_made'].append(f'Content expanded by {revised_length - original_length} words')
        
        # Check if feedback keywords appear in revised version
        feedback_keywords = re.findall(r'\b\w+\b', feedback.lower())
        revised_lower = revised.lower()
        
        addressed_keywords = sum(1 for keyword in feedback_keywords if keyword in revised_lower)
        if addressed_keywords > len(feedback_keywords) * 0.3:  # At least 30% of keywords addressed
            assessment['feedback_addressed'] = True
            assessment['changes_made'].append('Feedback keywords incorporated')
        
        # Check structural improvements (more sections, better formatting)
        original_sections = len(re.findall(r'^#+\s+', original, re.MULTILINE))
        revised_sections = len(re.findall(r'^#+\s+', revised, re.MULTILINE))
        
        if revised_sections > original_sections:
            assessment['structure_improved'] = True
            assessment['changes_made'].append('Document structure enhanced')
        
        # Calculate improvement score
        improvements = [
            assessment['content_expanded'],
            assessment['feedback_addressed'],
            assessment['structure_improved']
        ]
        
        assessment['improvement_score'] = sum(improvements) / len(improvements)
        assessment['quality_increased'] = assessment['improvement_score'] > 0.5
        
        if not assessment['quality_increased']:
            assessment['issues'].append('Revision shows minimal improvement')
        
        return assessment
    
    def _assess_requirements_quality(self, content: str) -> Dict[str, Any]:
        """Assess requirements document quality."""
        assessment = {
            'issues': [],
            'strengths': []
        }
        
        # Check EARS format compliance
        ears_validation = self.validate_ears_format(content)
        assessment['format_compliance'] = ears_validation['compliance_score'] > 0.5
        
        # Check for required sections
        required_sections = ['introduction', 'requirements']
        sections_found = []
        
        for section in required_sections:
            if re.search(rf'^#+\s*{section}', content, re.IGNORECASE | re.MULTILINE):
                sections_found.append(section)
        
        assessment['structure_score'] = len(sections_found) / len(required_sections)
        
        # Check completeness
        user_stories = ears_validation['user_stories_found']
        acceptance_criteria = ears_validation['ears_formatted_requirements']
        
        assessment['completeness_score'] = min(1.0, (user_stories + acceptance_criteria) / 10.0)
        
        if assessment['format_compliance']:
            assessment['strengths'].append('EARS format properly used')
        else:
            assessment['issues'].append('EARS format not followed')
        
        assessment['issues'].extend(ears_validation['issues'])
        assessment['strengths'].extend(ears_validation['strengths'])
        
        return assessment
    
    def _assess_design_quality(self, content: str) -> Dict[str, Any]:
        """Assess design document quality."""
        assessment = {
            'issues': [],
            'strengths': []
        }
        
        # Check Mermaid syntax
        mermaid_validation = self.validate_mermaid_syntax(content)
        assessment['format_compliance'] = mermaid_validation['syntax_valid']
        
        # Check for required sections with alternative names
        section_patterns = {
            'overview': ['overview', 'architectural overview'],
            'architecture': ['architecture', 'architectural overview', 'system architecture'],
            'components': ['components', 'component', 'interfaces'],
            'data models': ['data model', 'data structure', 'schema'],
            'error handling': ['error handling', 'error management', 'exception'],
            'testing strategy': ['testing', 'test strategy', 'test plan']
        }
        
        sections_found = []
        
        for section, patterns in section_patterns.items():
            found = False
            for pattern in patterns:
                if re.search(rf'^#+\s*.*{pattern}', content, re.IGNORECASE | re.MULTILINE):
                    sections_found.append(section)
                    found = True
                    break
        
        assessment['structure_score'] = len(sections_found) / len(section_patterns)
        
        # Check completeness (presence of technical details)
        technical_indicators = ['class', 'function', 'interface', 'api', 'database', 'component']
        technical_count = sum(1 for indicator in technical_indicators if indicator in content.lower())
        
        assessment['completeness_score'] = min(1.0, technical_count / 10.0)
        
        if assessment['format_compliance']:
            assessment['strengths'].append('Mermaid diagrams included')
        else:
            assessment['issues'].append('No valid Mermaid diagrams found')
        
        assessment['issues'].extend(mermaid_validation['issues'])
        assessment['strengths'].extend(mermaid_validation['strengths'])
        
        return assessment
    
    def _assess_tasks_quality(self, content: str) -> Dict[str, Any]:
        """Assess tasks document quality."""
        task_assessment = self.assess_task_structure(content)
        
        # Check if we have requirement references (use tasks_with_requirements > 0 as proxy)
        has_requirement_references = task_assessment.get('tasks_with_requirements', 0) > 0
        
        assessment = {
            'format_compliance': task_assessment['sequential_numbering'] and has_requirement_references,
            'structure_score': task_assessment['structure_score'],
            'completeness_score': 0.0
        }
        
        # Completeness score based on task count and detail
        task_count = task_assessment.get('total_tasks', task_assessment.get('task_count', 0))
        assessment['completeness_score'] = min(1.0, task_count / 15.0)  # Expect ~15 tasks for good coverage
        
        assessment['issues'] = task_assessment['issues']
        assessment['strengths'] = task_assessment['strengths']
        
        return assessment
    
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
    
    # LLM-specific validation methods for integration tests
    
    def validate_ears_format(self, requirements_content: str) -> Dict[str, Any]:
        """
        Validate EARS (Easy Approach to Requirements Syntax) format compliance.
        
        EARS format requirements should follow patterns like:
        - WHEN [event] THEN [system] SHALL [response]
        - IF [precondition] THEN [system] SHALL [response]
        - WHERE [feature] IS [value] THE [system] SHALL [response]
        - WHILE [state] THE [system] SHALL [response]
        
        Args:
            requirements_content: Content of requirements.md file
            
        Returns:
            Dict with validation results including compliance score and issues
        """
        validation_result = {
            'ears_compliant': False,
            'compliance_score': 0.0,
            'total_requirements': 0,
            'ears_formatted_requirements': 0,
            'user_stories_found': 0,
            'acceptance_criteria_found': 0,
            'issues': [],
            'strengths': []
        }
        
        try:
            # Check for user stories format
            user_story_pattern = r'(?i)\*\*user\s+story:\*\*\s*as\s+a\s+.+?,\s*i\s+want\s+.+?,\s*so\s+that\s+.+?'
            user_stories = re.findall(user_story_pattern, requirements_content, re.MULTILINE | re.DOTALL)
            validation_result['user_stories_found'] = len(user_stories)
            
            # Check for EARS format patterns
            ears_patterns = [
                r'(?i)when\s+.+?\s+then\s+.+?\s+shall\s+.+?',
                r'(?i)if\s+.+?\s+then\s+.+?\s+shall\s+.+?',
                r'(?i)where\s+.+?\s+is\s+.+?\s+the\s+.+?\s+shall\s+.+?',
                r'(?i)while\s+.+?\s+the\s+.+?\s+shall\s+.+?'
            ]
            
            ears_requirements = 0
            for pattern in ears_patterns:
                matches = re.findall(pattern, requirements_content, re.MULTILINE | re.DOTALL)
                ears_requirements += len(matches)
            
            validation_result['ears_formatted_requirements'] = ears_requirements
            
            # Count total requirements (numbered items under acceptance criteria)
            acceptance_criteria_sections = re.findall(
                r'(?i)acceptance\s+criteria.*?(?=\n\s*(?:###|\*\*|$))', 
                requirements_content, 
                re.MULTILINE | re.DOTALL
            )
            
            total_requirements = 0
            for section in acceptance_criteria_sections:
                numbered_items = re.findall(r'^\s*\d+\.', section, re.MULTILINE)
                total_requirements += len(numbered_items)
            
            # If no acceptance criteria sections found, look for numbered items after "Acceptance Criteria"
            if total_requirements == 0:
                # Look for numbered items anywhere in the document
                all_numbered_items = re.findall(r'^\s*\d+\.\s+', requirements_content, re.MULTILINE)
                total_requirements = len(all_numbered_items)
            
            validation_result['total_requirements'] = total_requirements
            validation_result['acceptance_criteria_found'] = len(acceptance_criteria_sections)
            
            # Calculate compliance score
            if total_requirements > 0:
                compliance_ratio = ears_requirements / total_requirements
                validation_result['compliance_score'] = min(1.0, compliance_ratio)
                validation_result['ears_compliant'] = compliance_ratio >= 0.7  # 70% threshold
            
            # Add strengths and issues
            if validation_result['user_stories_found'] > 0:
                validation_result['strengths'].append(f"Found {validation_result['user_stories_found']} properly formatted user stories")
            else:
                validation_result['issues'].append("No user stories found in proper format")
            
            if validation_result['ears_formatted_requirements'] > 0:
                validation_result['strengths'].append(f"Found {validation_result['ears_formatted_requirements']} EARS-formatted requirements")
            else:
                validation_result['issues'].append("No EARS-formatted requirements found")
            
            if validation_result['acceptance_criteria_found'] > 0:
                validation_result['strengths'].append(f"Found {validation_result['acceptance_criteria_found']} acceptance criteria sections")
            else:
                validation_result['issues'].append("No acceptance criteria sections found")
            
            if validation_result['compliance_score'] < 0.5:
                validation_result['issues'].append("Low EARS format compliance - consider reformatting requirements")
            
        except Exception as e:
            validation_result['issues'].append(f"Error during EARS validation: {str(e)}")
        
        return validation_result
    
    def validate_mermaid_syntax(self, design_content: str) -> Dict[str, Any]:
        """
        Validate Mermaid diagram syntax in design documents.
        
        Checks for proper Mermaid diagram blocks and basic syntax validation.
        
        Args:
            design_content: Content of design.md file
            
        Returns:
            Dict with validation results including syntax validity and diagram count
        """
        validation_result = {
            'mermaid_diagrams_found': 0,
            'valid_diagrams': 0,
            'syntax_valid': True,
            'diagram_types': [],
            'issues': [],
            'strengths': []
        }
        
        try:
            # Find Mermaid code blocks
            mermaid_blocks = re.findall(
                r'```mermaid\s*\n(.*?)\n```', 
                design_content, 
                re.MULTILINE | re.DOTALL
            )
            
            validation_result['mermaid_diagrams_found'] = len(mermaid_blocks)
            
            if len(mermaid_blocks) == 0:
                validation_result['issues'].append("No Mermaid diagrams found in design document")
                return validation_result
            
            # Validate each diagram
            valid_count = 0
            for i, diagram in enumerate(mermaid_blocks):
                diagram_validation = self._validate_single_mermaid_diagram(diagram)
                
                if diagram_validation['valid']:
                    valid_count += 1
                    validation_result['diagram_types'].append(diagram_validation['type'])
                else:
                    validation_result['issues'].extend([
                        f"Diagram {i+1}: {issue}" for issue in diagram_validation['issues']
                    ])
            
            validation_result['valid_diagrams'] = valid_count
            validation_result['syntax_valid'] = valid_count == len(mermaid_blocks)
            
            # Add strengths and issues
            if validation_result['valid_diagrams'] > 0:
                validation_result['strengths'].append(f"Found {validation_result['valid_diagrams']} valid Mermaid diagrams")
            
            if len(set(validation_result['diagram_types'])) > 1:
                validation_result['strengths'].append(f"Multiple diagram types used: {', '.join(set(validation_result['diagram_types']))}")
            
            if not validation_result['syntax_valid']:
                validation_result['issues'].append("Some Mermaid diagrams have syntax errors")
            
        except Exception as e:
            validation_result['issues'].append(f"Error during Mermaid validation: {str(e)}")
            validation_result['syntax_valid'] = False
        
        return validation_result
    
    def _validate_single_mermaid_diagram(self, diagram_content: str) -> Dict[str, Any]:
        """
        Validate a single Mermaid diagram for basic syntax correctness.
        
        Args:
            diagram_content: Content of a single Mermaid diagram
            
        Returns:
            Dict with validation results for the diagram
        """
        result = {
            'valid': False,
            'type': 'unknown',
            'issues': []
        }
        
        try:
            lines = [line.strip() for line in diagram_content.strip().split('\n') if line.strip()]
            
            if not lines:
                result['issues'].append("Empty diagram")
                return result
            
            # Detect diagram type from first line
            first_line = lines[0].lower()
            diagram_types = {
                'graph': ['graph', 'flowchart'],
                'sequenceDiagram': ['sequencediagram'],
                'classDiagram': ['classdiagram'],
                'stateDiagram': ['statediagram', 'statediagram-v2'],
                'erDiagram': ['erdiagram'],
                'gitgraph': ['gitgraph'],
                'pie': ['pie'],
                'gantt': ['gantt']
            }
            
            detected_type = 'unknown'
            for diagram_type, keywords in diagram_types.items():
                if any(keyword in first_line for keyword in keywords):
                    detected_type = diagram_type
                    break
            
            result['type'] = detected_type
            
            # Basic syntax validation
            if detected_type == 'graph':
                result['valid'] = self._validate_graph_diagram(lines)
                if not result['valid']:
                    result['issues'].append("Invalid graph diagram syntax")
            elif detected_type == 'sequenceDiagram':
                result['valid'] = self._validate_sequence_diagram(lines)
                if not result['valid']:
                    result['issues'].append("Invalid sequence diagram syntax")
            else:
                # For other types, just check if it has content beyond the declaration
                result['valid'] = len(lines) > 1
                if not result['valid']:
                    result['issues'].append("Diagram has no content beyond type declaration")
            
        except Exception as e:
            result['issues'].append(f"Syntax validation error: {str(e)}")
        
        return result
    
    def _validate_graph_diagram(self, lines: List[str]) -> bool:
        """Validate graph/flowchart diagram syntax."""
        try:
            # Check for valid graph declaration
            first_line = lines[0].lower()
            if not any(keyword in first_line for keyword in ['graph', 'flowchart']):
                return False
            
            # Check for node connections (arrows)
            has_connections = any('-->' in line or '---' in line or '-.-' in line for line in lines[1:])
            
            # Check for node definitions
            has_nodes = any(re.search(r'[A-Za-z0-9_]+\[.*?\]', line) for line in lines[1:])
            
            return has_connections or has_nodes
        except:
            return False
    
    def _validate_sequence_diagram(self, lines: List[str]) -> bool:
        """Validate sequence diagram syntax."""
        try:
            # Check for participant definitions or interactions
            has_participants = any('participant' in line.lower() for line in lines[1:])
            has_interactions = any('->>' in line or '-->' in line or '->' in line for line in lines[1:])
            
            return has_participants or has_interactions
        except:
            return False
    
    def assess_task_structure(self, tasks_content: str) -> Dict[str, Any]:
        """
        Assess task list structure and numbering in tasks.md files.
        
        Validates sequential numbering, requirement references, and task actionability.
        
        Args:
            tasks_content: Content of tasks.md file
            
        Returns:
            Dict with structure assessment results
        """
        assessment_result = {
            'structure_valid': False,
            'structure_score': 0.0,
            'total_tasks': 0,
            'numbered_tasks': 0,
            'tasks_with_requirements': 0,
            'actionable_tasks': 0,
            'sequential_numbering': True,
            'issues': [],
            'strengths': []
        }
        
        try:
            # Find all task items (numbered checkboxes)
            task_pattern = r'^\s*-\s*\[\s*[x\-\s]*\]\s*(\d+(?:\.\d+)*)\.\s*(.+?)$'
            tasks = re.findall(task_pattern, tasks_content, re.MULTILINE)
            
            assessment_result['total_tasks'] = len(tasks)
            assessment_result['numbered_tasks'] = len(tasks)
            
            if len(tasks) == 0:
                assessment_result['issues'].append("No numbered task items found")
                return assessment_result
            
            # Check sequential numbering
            task_numbers = []
            for task_num, task_desc in tasks:
                try:
                    # Handle both simple (1, 2, 3) and hierarchical (1.1, 1.2, 2.1) numbering
                    if '.' in task_num:
                        # Hierarchical numbering
                        parts = [int(x) for x in task_num.split('.')]
                        task_numbers.append(parts)
                    else:
                        # Simple numbering
                        task_numbers.append([int(task_num)])
                except ValueError:
                    assessment_result['issues'].append(f"Invalid task number format: {task_num}")
                    assessment_result['sequential_numbering'] = False
            
            # Validate sequential numbering
            if task_numbers:
                # Sort and check for proper sequence
                sorted_numbers = sorted(task_numbers)
                if task_numbers != sorted_numbers:
                    assessment_result['sequential_numbering'] = False
                    assessment_result['issues'].append("Tasks are not in sequential order")
            
            # Check for requirement references
            tasks_with_reqs = 0
            actionable_tasks = 0
            
            # Look for requirement references in task descriptions and details
            for task_num, task_desc in tasks:
                # Find the full task block including details
                task_block_pattern = rf'^\s*-\s*\[\s*[x\-\s]*\]\s*{re.escape(task_num)}\.\s*{re.escape(task_desc)}.*?(?=^\s*-\s*\[|$)'
                task_block_match = re.search(task_block_pattern, tasks_content, re.MULTILINE | re.DOTALL)
                
                if task_block_match:
                    task_block = task_block_match.group(0)
                    
                    # Check for requirement references (flexible patterns)
                    req_patterns = [
                        r'_Requirements?:\s*([0-9., ]+)_',  # _Requirements: X.Y_
                        r'-\s*Requirements?:\s*([^\n]+)',   # - Requirements: X.Y
                        r'Requirements?:\s*([^\n,]+)'       # Requirements: X.Y (general)
                    ]
                    
                    has_req_ref = False
                    for pattern in req_patterns:
                        req_refs = re.findall(pattern, task_block, re.IGNORECASE)
                        if req_refs:
                            has_req_ref = True
                            break
                    
                    if has_req_ref:
                        tasks_with_reqs += 1
                    
                    # Check for actionability (contains action verbs and specific targets)
                    action_verbs = ['create', 'implement', 'add', 'modify', 'update', 'delete', 'test', 'validate', 'fix', 'enhance']
                    file_references = re.findall(r'`[^`]+\.(py|md|json|yaml|yml|txt)`', task_block)
                    
                    has_action = any(verb in task_desc.lower() for verb in action_verbs)
                    has_target = len(file_references) > 0 or any(keyword in task_block.lower() for keyword in ['file', 'class', 'function', 'method'])
                    
                    if has_action and has_target:
                        actionable_tasks += 1
            
            assessment_result['tasks_with_requirements'] = tasks_with_reqs
            assessment_result['actionable_tasks'] = actionable_tasks
            
            # Calculate structure score
            score_components = []
            
            # Sequential numbering (25%)
            if assessment_result['sequential_numbering']:
                score_components.append(0.25)
                assessment_result['strengths'].append("Tasks are sequentially numbered")
            else:
                assessment_result['issues'].append("Tasks are not sequentially numbered")
            
            # Requirement references (25%)
            req_ratio = tasks_with_reqs / len(tasks) if tasks else 0
            score_components.append(0.25 * req_ratio)
            if req_ratio > 0.8:
                assessment_result['strengths'].append("Most tasks reference requirements")
            elif req_ratio < 0.5:
                assessment_result['issues'].append("Many tasks lack requirement references")
            
            # Actionability (25%)
            action_ratio = actionable_tasks / len(tasks) if tasks else 0
            score_components.append(0.25 * action_ratio)
            if action_ratio > 0.8:
                assessment_result['strengths'].append("Most tasks are actionable with clear targets")
            elif action_ratio < 0.5:
                assessment_result['issues'].append("Many tasks lack clear actionable descriptions")
            
            # Task count appropriateness (25%)
            if 5 <= len(tasks) <= 20:
                score_components.append(0.25)
                assessment_result['strengths'].append("Appropriate number of tasks")
            elif len(tasks) < 5:
                score_components.append(0.1)
                assessment_result['issues'].append("Too few tasks - may lack detail")
            else:
                score_components.append(0.15)
                assessment_result['issues'].append("Many tasks - consider grouping or simplification")
            
            assessment_result['structure_score'] = sum(score_components)
            assessment_result['structure_valid'] = assessment_result['structure_score'] >= 0.7
            
        except Exception as e:
            assessment_result['issues'].append(f"Error during task structure assessment: {str(e)}")
        
        return assessment_result
    
    def assess_revision_improvement(self, original_content: str, revised_content: str, 
                                  feedback: str, document_type: str = 'unknown') -> Dict[str, Any]:
        """
        Assess quality improvement after document revision based on feedback.
        
        Compares original and revised content to determine if revisions meaningfully
        address the provided feedback and improve document quality.
        
        Args:
            original_content: Original document content
            revised_content: Revised document content
            feedback: Feedback that was provided for revision
            document_type: Type of document (requirements, design, tasks)
            
        Returns:
            Dict with improvement assessment results
        """
        assessment_result = {
            'improvement_detected': False,
            'improvement_score': 0.0,
            'content_changed': False,
            'feedback_addressed': False,
            'quality_improved': False,
            'changes_summary': {
                'lines_added': 0,
                'lines_removed': 0,
                'lines_modified': 0,
                'sections_added': 0,
                'sections_modified': 0
            },
            'feedback_analysis': {
                'feedback_points_identified': 0,
                'feedback_points_addressed': 0,
                'specific_improvements': []
            },
            'issues': [],
            'strengths': []
        }
        
        try:
            # Check if content actually changed
            if original_content.strip() == revised_content.strip():
                assessment_result['issues'].append("No changes detected between original and revised content")
                return assessment_result
            
            assessment_result['content_changed'] = True
            
            # Analyze content changes
            changes = self._analyze_content_changes(original_content, revised_content)
            assessment_result['changes_summary'] = changes
            
            # Analyze feedback addressing
            feedback_analysis = self._analyze_feedback_addressing(original_content, revised_content, feedback)
            assessment_result['feedback_analysis'] = feedback_analysis
            assessment_result['feedback_addressed'] = feedback_analysis['feedback_points_addressed'] > 0
            
            # Assess quality improvement based on document type
            quality_improvement = self._assess_document_quality_improvement(
                original_content, revised_content, document_type
            )
            assessment_result['quality_improved'] = quality_improvement['improved']
            
            # Calculate overall improvement score
            score_components = []
            
            # Content change significance (20%)
            change_significance = min(1.0, (changes['lines_added'] + changes['lines_modified']) / 10)
            score_components.append(0.2 * change_significance)
            
            # Feedback addressing (40%)
            if feedback_analysis['feedback_points_identified'] > 0:
                feedback_ratio = feedback_analysis['feedback_points_addressed'] / feedback_analysis['feedback_points_identified']
                score_components.append(0.4 * feedback_ratio)
            else:
                score_components.append(0.2)  # Partial credit if no specific feedback points identified
            
            # Quality improvement (40%)
            if quality_improvement['improved']:
                score_components.append(0.4 * quality_improvement['improvement_ratio'])
            
            assessment_result['improvement_score'] = sum(score_components)
            assessment_result['improvement_detected'] = assessment_result['improvement_score'] >= 0.6
            
            # Add strengths and issues
            if assessment_result['content_changed']:
                assessment_result['strengths'].append("Content was meaningfully revised")
            
            if assessment_result['feedback_addressed']:
                assessment_result['strengths'].append(f"Addressed {feedback_analysis['feedback_points_addressed']} feedback points")
            else:
                assessment_result['issues'].append("Feedback points were not clearly addressed")
            
            if quality_improvement['improved']:
                assessment_result['strengths'].append("Document quality improved after revision")
                assessment_result['strengths'].extend(quality_improvement['improvements'])
            else:
                assessment_result['issues'].append("No clear quality improvement detected")
            
            if changes['lines_added'] == 0 and changes['lines_modified'] == 0:
                assessment_result['issues'].append("Only content removal detected - no additions or improvements")
            
        except Exception as e:
            assessment_result['issues'].append(f"Error during revision assessment: {str(e)}")
        
        return assessment_result
    
    def _analyze_content_changes(self, original: str, revised: str) -> Dict[str, int]:
        """Analyze the changes between original and revised content."""
        original_lines = original.split('\n')
        revised_lines = revised.split('\n')
        
        changes = {
            'lines_added': 0,
            'lines_removed': 0,
            'lines_modified': 0,
            'sections_added': 0,
            'sections_modified': 0
        }
        
        # Simple diff analysis
        original_set = set(original_lines)
        revised_set = set(revised_lines)
        
        changes['lines_added'] = len(revised_set - original_set)
        changes['lines_removed'] = len(original_set - revised_set)
        
        # Count modified lines (approximation)
        common_lines = len(original_set & revised_set)
        total_original = len(original_lines)
        total_revised = len(revised_lines)
        
        changes['lines_modified'] = max(0, min(total_original, total_revised) - common_lines)
        
        # Count section changes (headers)
        original_sections = re.findall(r'^#+\s+(.+)$', original, re.MULTILINE)
        revised_sections = re.findall(r'^#+\s+(.+)$', revised, re.MULTILINE)
        
        changes['sections_added'] = len(set(revised_sections) - set(original_sections))
        changes['sections_modified'] = len(set(original_sections) & set(revised_sections))
        
        return changes
    
    def _analyze_feedback_addressing(self, original: str, revised: str, feedback: str) -> Dict[str, Any]:
        """Analyze how well the revision addresses the provided feedback."""
        analysis = {
            'feedback_points_identified': 0,
            'feedback_points_addressed': 0,
            'specific_improvements': []
        }
        
        # Extract feedback points (sentences ending with periods, exclamations, or questions)
        feedback_sentences = re.split(r'[.!?]+', feedback)
        feedback_points = [s.strip() for s in feedback_sentences if len(s.strip()) > 10]
        
        analysis['feedback_points_identified'] = len(feedback_points)
        
        # Check if feedback points are addressed
        for point in feedback_points:
            # Extract key terms from feedback point
            key_terms = re.findall(r'\b[a-zA-Z]{3,}\b', point.lower())
            key_terms = [term for term in key_terms if term not in ['the', 'and', 'for', 'are', 'you', 'can', 'should', 'will', 'this', 'that']]
            
            if not key_terms:
                continue
            
            # Check if these terms appear more in revised than original
            original_mentions = sum(original.lower().count(term) for term in key_terms)
            revised_mentions = sum(revised.lower().count(term) for term in key_terms)
            
            if revised_mentions > original_mentions:
                analysis['feedback_points_addressed'] += 1
                analysis['specific_improvements'].append(f"Addressed feedback about: {point[:50]}...")
        
        return analysis
    
    def _assess_document_quality_improvement(self, original: str, revised: str, document_type: str) -> Dict[str, Any]:
        """Assess quality improvement based on document type."""
        improvement = {
            'improved': False,
            'improvement_ratio': 0.0,
            'improvements': []
        }
        
        if document_type.lower() == 'requirements':
            # Check EARS format improvement
            original_ears = self.validate_ears_format(original)
            revised_ears = self.validate_ears_format(revised)
            
            if revised_ears['compliance_score'] > original_ears['compliance_score']:
                improvement['improved'] = True
                improvement['improvements'].append("EARS format compliance improved")
            
            if revised_ears['user_stories_found'] > original_ears['user_stories_found']:
                improvement['improved'] = True
                improvement['improvements'].append("More user stories added")
            
            improvement['improvement_ratio'] = max(0, revised_ears['compliance_score'] - original_ears['compliance_score'])
        
        elif document_type.lower() == 'design':
            # Check Mermaid diagram improvement
            original_mermaid = self.validate_mermaid_syntax(original)
            revised_mermaid = self.validate_mermaid_syntax(revised)
            
            if revised_mermaid['mermaid_diagrams_found'] > original_mermaid['mermaid_diagrams_found']:
                improvement['improved'] = True
                improvement['improvements'].append("More Mermaid diagrams added")
            
            if revised_mermaid['valid_diagrams'] > original_mermaid['valid_diagrams']:
                improvement['improved'] = True
                improvement['improvements'].append("Mermaid diagram quality improved")
            
            # Check for required sections
            required_sections = ['overview', 'architecture', 'components', 'data models', 'error handling', 'testing']
            original_sections = sum(1 for section in required_sections if section in original.lower())
            revised_sections = sum(1 for section in required_sections if section in revised.lower())
            
            if revised_sections > original_sections:
                improvement['improved'] = True
                improvement['improvements'].append("More required sections added")
            
            improvement['improvement_ratio'] = (revised_sections - original_sections) / len(required_sections)
        
        elif document_type.lower() == 'tasks':
            # Check task structure improvement
            original_structure = self.assess_task_structure(original)
            revised_structure = self.assess_task_structure(revised)
            
            if revised_structure['structure_score'] > original_structure['structure_score']:
                improvement['improved'] = True
                improvement['improvements'].append("Task structure quality improved")
            
            if revised_structure['tasks_with_requirements'] > original_structure['tasks_with_requirements']:
                improvement['improved'] = True
                improvement['improvements'].append("More tasks linked to requirements")
            
            improvement['improvement_ratio'] = max(0, revised_structure['structure_score'] - original_structure['structure_score'])
        
        else:
            # General quality assessment
            original_length = len(original.split())
            revised_length = len(revised.split())
            
            if revised_length > original_length * 1.1:  # At least 10% more content
                improvement['improved'] = True
                improvement['improvements'].append("Content expanded significantly")
                improvement['improvement_ratio'] = min(1.0, (revised_length - original_length) / original_length)
        
        return improvement