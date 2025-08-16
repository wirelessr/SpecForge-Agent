# Performance Analysis CLI

This document describes the command-line interface for performance analysis tools that help identify and optimize ImplementAgent performance bottlenecks.

## Overview

The performance analysis CLI provides comprehensive tools for:
- **Timing Analysis**: Measure execution time of integration tests individually
- **Full Profiling**: Apply cProfile and py-spy profiling to slow tests
- **Selective Profiling**: Profile specific tests by name
- **Bottleneck Analysis**: Identify component-level performance issues
- **Report Generation**: Create HTML reports with optimization recommendations
- **Comparison Reports**: Compare performance between different runs

## Installation and Setup

### Prerequisites

1. **Python 3.8+** with required packages:
   ```bash
   pip install pytest asyncio pathlib
   ```

2. **Optional: py-spy** for flame graph generation:
   ```bash
   pip install py-spy
   ```
   Note: py-spy may require root permissions on some systems.

3. **Integration Tests**: Ensure integration tests exist in `tests/integration/`

### Verification

Check that the CLI is working:
```bash
python -m tests.perf.cli --help
```

Or using the standalone script:
```bash
python perf_analysis.py --help
```

## Usage

### Basic Commands

#### 1. Timing Analysis
Analyze execution time of all integration tests:
```bash
# Basic timing analysis
python -m tests.perf.cli timing

# With custom timeout and pattern filter
python -m tests.perf.cli timing --timeout 120 --pattern "test_real_*"

# Save to specific file
python -m tests.perf.cli timing --output-file timing_results.json
```

#### 2. Full Profiling Analysis
Complete performance analysis with profiling:
```bash
# Basic full profiling (profiles top 10 slowest tests)
python -m tests.perf.cli profile

# Limit to 5 tests, disable flame graphs
python -m tests.perf.cli profile --max-tests 5 --no-flamegraphs

# Custom slow test threshold
python -m tests.perf.cli profile --slow-threshold 15.0
```

#### 3. Selective Profiling
Profile specific tests by name:
```bash
# Profile specific tests
python -m tests.perf.cli selective test_real_agent_manager test_real_main_controller

# Without flame graphs
python -m tests.perf.cli selective --no-flamegraphs test_specific_method
```

#### 4. List Available Tests
Discover available integration tests:
```bash
# List all tests
python -m tests.perf.cli list

# Filter by pattern
python -m tests.perf.cli list --pattern "test_real_*"
```

#### 5. Compare Reports
Generate comparison between two analysis runs:
```bash
python -m tests.perf.cli compare baseline_report_data.json current_report_data.json
```

### Global Options

All commands support these global options:
```bash
# Verbose logging
python -m tests.perf.cli --verbose timing

# Custom output directory
python -m tests.perf.cli --output-dir /tmp/reports profile

# Combined options
python -m tests.perf.cli --verbose --output-dir ./reports profile --max-tests 3
```

### Alternative Usage

Use the standalone script for simpler syntax:
```bash
# Instead of: python -m tests.perf.cli timing
python perf_analysis.py timing

# Instead of: python -m tests.perf.cli profile --max-tests 5
python perf_analysis.py profile --max-tests 5
```

## Command Reference

### `timing` - Timing Analysis

Measures execution time of integration tests individually to identify slow tests.

**Options:**
- `--timeout, -t`: Custom timeout per test in seconds (default: 60)
- `--pattern, -p`: Filter tests by name pattern
- `--output-file`: Save timing report to specific JSON file

**Output:**
- Timing summary with statistics
- List of slowest tests
- JSON report file in `artifacts/performance/reports/`

**Example:**
```bash
python -m tests.perf.cli timing --timeout 90 --pattern "test_real_*"
```

### `profile` - Full Profiling Analysis

Complete performance analysis including timing, profiling, and bottleneck analysis.

**Options:**
- `--max-tests, -m`: Maximum slow tests to profile (default: 10)
- `--slow-threshold, -s`: Custom threshold for slow tests in seconds
- `--no-flamegraphs`: Disable py-spy flame graph generation

**Output:**
- Comprehensive HTML report with all analysis
- cProfile data files
- Flame graph SVG files (if py-spy available)
- Optimization recommendations

**Example:**
```bash
python -m tests.perf.cli profile --max-tests 5 --slow-threshold 10.0
```

### `selective` - Selective Profiling

Profile specific tests chosen by name rather than by timing analysis.

**Arguments:**
- `test_names`: One or more test names to profile

**Options:**
- `--no-flamegraphs`: Disable flame graph generation

**Output:**
- HTML report for selected tests
- Profiling data for each selected test

**Example:**
```bash
python -m tests.perf.cli selective test_real_agent_manager test_context_manager
```

### `compare` - Comparison Reports

Generate comparison between two performance analysis runs.

**Arguments:**
- `baseline_report`: Path to baseline report JSON data file
- `current_report`: Path to current report JSON data file

**Output:**
- HTML comparison report showing performance changes
- Trend analysis and recommendations

**Example:**
```bash
python -m tests.perf.cli compare artifacts/performance/reports/baseline_data.json artifacts/performance/reports/current_data.json
```

### `list` - List Tests

Discover and display available integration tests.

**Options:**
- `--pattern, -p`: Filter tests by name pattern

**Output:**
- List of available tests organized by file

**Example:**
```bash
python -m tests.perf.cli list --pattern "test_real_*"
```

## Output Files

The CLI generates several types of output files:

### Directory Structure
```
artifacts/performance/
├── profiles/           # cProfile data files (.prof)
├── flamegraphs/        # py-spy flame graphs (.svg)
└── reports/           # HTML reports and JSON data
    ├── timing_report_*.json
    ├── performance_report_*.html
    ├── performance_report_*_data.json
    └── comparison_report_*.html
```

### Report Types

1. **Timing Reports** (`timing_report_*.json`)
   - Raw timing data for all tests
   - Used for identifying slow tests
   - Can be loaded for further analysis

2. **Comprehensive Reports** (`performance_report_*.html`)
   - Complete analysis with timing, profiling, and recommendations
   - Interactive HTML with embedded flame graphs
   - Includes component-level bottleneck analysis

3. **Comparison Reports** (`comparison_report_*.html`)
   - Side-by-side comparison of two analysis runs
   - Performance trend analysis
   - Regression and improvement identification

## Configuration

### Environment Variables

- `PYTHON_EXECUTABLE`: Python executable path (default: `python`)
- `PYTEST_EXECUTABLE`: Pytest executable path (default: `pytest`)
- `PERF_MAX_WORKERS`: Maximum parallel workers (default: `4`)
- `PERF_TIMEOUT_MULTIPLIER`: Timeout multiplier for slow environments (default: `1.0`)

### Configuration Files

The CLI uses configuration from `tests/perf/config.py`:
- Test discovery patterns
- Output directories
- Profiling thresholds
- Component identification patterns

## Troubleshooting

### Common Issues

1. **No integration tests found**
   ```
   Error: Integration test directory not found: tests/integration
   ```
   **Solution**: Ensure integration tests exist in `tests/integration/` directory

2. **py-spy permission errors**
   ```
   Error: py-spy requires root permissions
   ```
   **Solution**: Run with `sudo` or use `--no-flamegraphs` option

3. **Timeout errors**
   ```
   Warning: Test timed out after 60 seconds
   ```
   **Solution**: Increase timeout with `--timeout` option

4. **No slow tests found**
   ```
   Warning: No slow tests found for profiling
   ```
   **Solution**: Lower threshold with `--slow-threshold` or check test execution times

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
python -m tests.perf.cli --verbose timing
```

### Validation

Check configuration and environment:
```bash
python -c "from tests.perf.config import validate_config; print(validate_config())"
```

## Examples

### Complete Workflow Example

1. **Discover available tests:**
   ```bash
   python -m tests.perf.cli list --pattern "test_real_*"
   ```

2. **Run timing analysis:**
   ```bash
   python -m tests.perf.cli timing --timeout 120
   ```

3. **Profile slow tests:**
   ```bash
   python -m tests.perf.cli profile --max-tests 5
   ```

4. **Generate comparison after optimization:**
   ```bash
   # After making optimizations, run profiling again
   python -m tests.perf.cli profile --max-tests 5
   
   # Compare results
   python -m tests.perf.cli compare baseline_report_data.json current_report_data.json
   ```

### Continuous Integration Example

```bash
#!/bin/bash
# CI script for performance monitoring

# Run timing analysis
python -m tests.perf.cli timing --output-file ci_timing_report.json

# Profile if slow tests found
python -m tests.perf.cli profile --max-tests 3 --no-flamegraphs

# Compare with baseline if available
if [ -f baseline_report_data.json ]; then
    python -m tests.perf.cli compare baseline_report_data.json performance_report_*_data.json
fi
```

## Demo

Run the interactive demo to see all CLI features:
```bash
python -m tests.perf.demo_cli
```

The demo will show:
- Available commands and syntax
- Interactive examples of each analysis type
- Sample output and reports

## Integration with Development Workflow

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
python -m tests.perf.cli timing --timeout 30 --pattern "test_real_*"
```

### Performance Regression Detection
```bash
# Run before and after changes
python -m tests.perf.cli profile --max-tests 3
# Make changes
python -m tests.perf.cli profile --max-tests 3
python -m tests.perf.cli compare baseline_data.json current_data.json
```

### Optimization Workflow
1. Identify slow tests with timing analysis
2. Profile slow tests to find bottlenecks
3. Apply optimizations based on recommendations
4. Verify improvements with comparison reports
5. Update baseline for future comparisons