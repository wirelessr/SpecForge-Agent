#!/usr/bin/env python3
"""
Demo script for the performance analysis CLI.

This script demonstrates the various CLI commands and their usage
for performance analysis of integration tests.
"""

import asyncio
import logging
import sys
from pathlib import Path

try:
    from .cli import PerformanceAnalysisCLI
    from .config import PerformanceConfig
except ImportError:
    from cli import PerformanceAnalysisCLI
    from config import PerformanceConfig


async def demo_timing_analysis():
    """Demonstrate timing-only analysis."""
    print("\n" + "="*60)
    print("DEMO: Timing-Only Analysis")
    print("="*60)
    
    try:
        cli = PerformanceAnalysisCLI()
        
        print("Running timing analysis on integration tests...")
        print("This will discover and time all integration tests individually.")
        
        # Run timing analysis with a shorter timeout for demo
        result_path = await cli.run_timing_analysis(
            timeout=30,  # Shorter timeout for demo
            test_pattern=None  # Analyze all tests
        )
        
        print(f"\nTiming analysis completed!")
        print(f"Report saved to: {result_path}")
        
        return result_path
        
    except Exception as e:
        print(f"Demo timing analysis failed: {e}")
        return None


async def demo_selective_profiling():
    """Demonstrate selective profiling of specific tests."""
    print("\n" + "="*60)
    print("DEMO: Selective Profiling")
    print("="*60)
    
    try:
        cli = PerformanceAnalysisCLI()
        
        # First, list available tests to show what we can profile
        print("Available integration tests:")
        cli.list_available_tests(pattern="test_real")
        
        # Select a few tests for profiling (these may not exist in demo)
        test_names = [
            "test_real_agent_manager",
            "test_real_main_controller"
        ]
        
        print(f"\nRunning selective profiling on: {test_names}")
        print("This will profile only the specified tests with detailed analysis.")
        
        result_path = await cli.run_selective_profiling(
            test_names=test_names,
            include_flamegraphs=True
        )
        
        print(f"\nSelective profiling completed!")
        print(f"Report saved to: {result_path}")
        
        return result_path
        
    except Exception as e:
        print(f"Demo selective profiling failed: {e}")
        return None


async def demo_full_profiling():
    """Demonstrate full profiling analysis."""
    print("\n" + "="*60)
    print("DEMO: Full Profiling Analysis")
    print("="*60)
    
    try:
        cli = PerformanceAnalysisCLI()
        
        print("Running full profiling analysis...")
        print("This will:")
        print("1. Analyze timing of all integration tests")
        print("2. Identify the slowest tests")
        print("3. Profile slow tests with cProfile and py-spy")
        print("4. Analyze bottlenecks and generate recommendations")
        print("5. Create comprehensive HTML report")
        
        result_path = await cli.run_full_profiling(
            max_tests=3,  # Limit to 3 slowest tests for demo
            include_flamegraphs=True
        )
        
        print(f"\nFull profiling analysis completed!")
        print(f"Comprehensive report saved to: {result_path}")
        
        return result_path
        
    except Exception as e:
        print(f"Demo full profiling failed: {e}")
        return None


def demo_list_tests():
    """Demonstrate listing available tests."""
    print("\n" + "="*60)
    print("DEMO: List Available Tests")
    print("="*60)
    
    try:
        cli = PerformanceAnalysisCLI()
        
        print("All available integration tests:")
        cli.list_available_tests()
        
        print("\nFiltered tests (pattern: 'test_real'):")
        cli.list_available_tests(pattern="test_real")
        
    except Exception as e:
        print(f"Demo list tests failed: {e}")


async def demo_comparison_report():
    """Demonstrate comparison report generation."""
    print("\n" + "="*60)
    print("DEMO: Comparison Report")
    print("="*60)
    
    print("Comparison reports require two existing analysis reports.")
    print("This demo shows how to generate them once you have baseline and current reports.")
    
    # Show example command
    print("\nExample usage:")
    print("python -m tests.perf.cli compare baseline_report_data.json current_report_data.json")
    
    # Check if sample reports exist
    reports_dir = Path("artifacts/performance/reports")
    if reports_dir.exists():
        json_files = list(reports_dir.glob("*_data.json"))
        if len(json_files) >= 2:
            print(f"\nFound {len(json_files)} report data files:")
            for json_file in json_files[:5]:  # Show first 5
                print(f"  - {json_file.name}")
            
            print("\nYou can generate a comparison report using any two of these files.")
        else:
            print("\nNo existing report data files found for comparison.")
            print("Run timing or profiling analysis first to generate baseline reports.")
    else:
        print("\nReports directory not found. Run analysis commands first.")


def demo_cli_commands():
    """Demonstrate CLI command syntax."""
    print("\n" + "="*60)
    print("DEMO: CLI Command Examples")
    print("="*60)
    
    print("The performance analysis CLI supports several commands:")
    print()
    
    print("1. Timing Analysis:")
    print("   python -m tests.perf.cli timing")
    print("   python -m tests.perf.cli timing --timeout 120 --pattern 'test_real_*'")
    print()
    
    print("2. Full Profiling:")
    print("   python -m tests.perf.cli profile")
    print("   python -m tests.perf.cli profile --max-tests 5 --no-flamegraphs")
    print()
    
    print("3. Selective Profiling:")
    print("   python -m tests.perf.cli selective test_real_agent_manager")
    print("   python -m tests.perf.cli selective test_method_1 test_method_2")
    print()
    
    print("4. List Tests:")
    print("   python -m tests.perf.cli list")
    print("   python -m tests.perf.cli list --pattern 'test_real_*'")
    print()
    
    print("5. Compare Reports:")
    print("   python -m tests.perf.cli compare baseline.json current.json")
    print()
    
    print("6. Global Options:")
    print("   python -m tests.perf.cli --verbose --output-dir /tmp/reports timing")
    print()
    
    print("Alternative usage with standalone script:")
    print("   python perf_analysis.py timing")
    print("   python perf_analysis.py profile --max-tests 3")


async def run_demo():
    """Run the complete CLI demonstration."""
    print("Performance Analysis CLI Demonstration")
    print("=====================================")
    
    # Setup logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Show CLI command examples
    demo_cli_commands()
    
    # Show available tests
    demo_list_tests()
    
    # Ask user which demos to run
    print("\n" + "="*60)
    print("INTERACTIVE DEMO OPTIONS")
    print("="*60)
    print("Choose which analysis demos to run:")
    print("1. Timing Analysis (fast)")
    print("2. Selective Profiling (medium)")
    print("3. Full Profiling Analysis (slow)")
    print("4. Comparison Report (requires existing reports)")
    print("5. All demos")
    print("0. Skip interactive demos")
    
    try:
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '0':
            print("Skipping interactive demos.")
            return
        
        elif choice == '1':
            await demo_timing_analysis()
        
        elif choice == '2':
            await demo_selective_profiling()
        
        elif choice == '3':
            await demo_full_profiling()
        
        elif choice == '4':
            await demo_comparison_report()
        
        elif choice == '5':
            print("Running all demos...")
            timing_report = await demo_timing_analysis()
            await demo_selective_profiling()
            profiling_report = await demo_full_profiling()
            await demo_comparison_report()
            
            if timing_report and profiling_report:
                print(f"\nDemo completed! Generated reports:")
                print(f"- Timing report: {timing_report}")
                print(f"- Profiling report: {profiling_report}")
        
        else:
            print("Invalid choice. Exiting demo.")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Demo failed: {e}")
    
    print("\n" + "="*60)
    print("Demo completed. Check the artifacts/performance/ directory for generated reports.")
    print("="*60)


if __name__ == '__main__':
    asyncio.run(run_demo())