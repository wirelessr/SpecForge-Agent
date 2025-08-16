# Performance Analysis Infrastructure

This directory contains the performance analysis tools for optimizing ImplementAgent execution. The infrastructure follows a systematic approach to identify and analyze performance bottlenecks.

## Architecture Overview

### Phase 1: Integration Test Timing Analysis
- **TestExecutor**: Discovers and executes integration tests with timeout handling
- **TestTimingAnalyzer**: Orchestrates timing analysis and generates comprehensive reports

### Phase 2: Targeted Performance Profiling  
- **PerformanceProfiler**: Applies cProfile and py-spy profiling to slow tests
- **BottleneckAnalyzer**: Analyzes profiling results to identify optimization targets

## Components

### Core Components
- `models.py` - Data models for all performance analysis structures
- `test_executor.py` - Integration test discovery and execution (Task 2)
- `timing_analyzer.py` - Comprehensive timing analysis orchestration (Task 3)
- `profiler.py` - cProfile and py-spy integration (Tasks 4-5)
- `bottleneck_analyzer.py` - Profiling result analysis (Task 6)

### Analysis Focus
- **Integration Tests Only**: Focuses specifically on `tests/integration/` directory
- **ImplementAgent Components**: TaskDecomposer, ShellExecutor, ErrorRecovery timing
- **LLM API Profiling**: Request/response timing and prompt analysis
- **Context Loading**: Performance impact of different context sizes

## Usage Workflow

1. **Run Timing Analysis**: Identify slow integration tests
2. **Apply Targeted Profiling**: Profile the slowest tests with cProfile/py-spy
3. **Analyze Bottlenecks**: Categorize time spent in test vs ImplementAgent code
4. **Generate Recommendations**: Provide specific optimization strategies

## Data Models

### Test Identification
- `TestIdentifier`: Unique test identification
- `TestTimingResult`: Individual test timing data
- `FileTimingResult`: File-level timing aggregation
- `TimingReport`: Comprehensive timing analysis

### Profiling Results
- `CProfileResult`: Deterministic profiling data
- `PySpyResult`: Sampling profiler flame graphs
- `ProfilingResult`: Combined profiling analysis

### Bottleneck Analysis
- `ComponentTimings`: ImplementAgent component breakdown
- `TimeCategories`: Test overhead vs implementation time
- `ComponentBottleneck`: Identified performance issues
- `BottleneckReport`: Complete analysis with recommendations

## Requirements Mapping

- **Requirement 1.1**: Individual integration test timing with timeout handling
- **Requirement 1.4**: Comprehensive timing reports for analysis prioritization
- **Requirement 2.1-2.4**: Detailed profiling with cProfile and py-spy
- **Requirement 3.1-3.5**: Bottleneck analysis and optimization recommendations
- **Requirement 4.1-4.5**: Component-level ImplementAgent profiling

## Next Steps

This infrastructure provides the foundation for implementing:
1. TestExecutor for integration test discovery and execution
2. TestTimingAnalyzer for orchestrated timing analysis
3. PerformanceProfiler for detailed profiling capabilities
4. BottleneckAnalyzer for result interpretation and recommendations