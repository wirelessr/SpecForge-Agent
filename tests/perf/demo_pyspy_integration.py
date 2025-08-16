#!/usr/bin/env python3
"""
Demonstration of py-spy integration in PerformanceProfiler.

This script demonstrates all aspects of the py-spy integration including:
- py-spy availability detection
- Process-based profiling with py-spy for flame graph generation
- SVG flame graph generation and storage
- Fallback handling when py-spy is not available or lacks permissions
- Integration with cProfile for comprehensive profiling
"""

import asyncio
import sys
from pathlib import Path

from profiler import PerformanceProfiler
from models import TestIdentifier, TestTimingResult, TestStatus
from config import PerformanceConfig


async def demonstrate_pyspy_integration():
    """Demonstrate complete py-spy integration functionality."""
    print("🔥 py-spy Integration Demonstration")
    print("=" * 50)
    
    # Initialize profiler
    profiler = PerformanceProfiler()
    
    # 1. Show py-spy availability
    print(f"\n1. py-spy Availability:")
    print(f"   Available: {profiler.pyspy_available}")
    if hasattr(profiler, 'pyspy_needs_root'):
        print(f"   Needs root: {profiler.pyspy_needs_root}")
    if hasattr(profiler, 'pyspy_format'):
        print(f"   Format: {profiler.pyspy_format}")
        print(f"   Extension: {profiler.pyspy_extension}")
    
    # Check sudo availability
    sudo_available = profiler.can_use_pyspy_with_sudo()
    print(f"   Sudo available: {sudo_available}")
    
    # 2. Create test scenarios
    test_scenarios = [
        TestIdentifier(
            file_path='tests/integration/test_context_compressor_dynamic_config.py',
            class_name='TestContextCompressorDynamicConfig',
            method_name='test_fallback_to_defaults'
        ),
        TestIdentifier(
            file_path='tests/integration/test_context_compressor_dynamic_config.py',
            method_name='test_model_pattern_matching'
        )
    ]
    
    print(f"\n2. Test Scenarios:")
    for i, test_id in enumerate(test_scenarios, 1):
        print(f"   {i}. {test_id.full_name}")
    
    # 3. Demonstrate profiling workflow
    print(f"\n3. Profiling Workflow Demonstration:")
    
    for i, test_id in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {i}: {test_id.method_name} ---")
        
        try:
            # Use integrated profiling (cProfile + py-spy)
            result = await profiler.profile_specific_test(test_id)
            
            print(f"✓ Profiling completed successfully")
            print(f"  Total time: {result.total_profiling_time:.2f}s")
            print(f"  Profiler: {result.profiler_used}")
            
            # Show cProfile results
            if result.cprofile_result:
                cprofile = result.cprofile_result
                print(f"  cProfile:")
                print(f"    Total time: {cprofile.total_time:.2f}s")
                print(f"    Function calls: {cprofile.call_count:,}")
                print(f"    Profile file: {Path(cprofile.profile_file_path).name}")
                
                # Show top functions
                if cprofile.top_functions:
                    print(f"    Top functions:")
                    for j, func in enumerate(cprofile.top_functions[:3], 1):
                        print(f"      {j}. {func['name'][:60]}...")
                        print(f"         {func['cumulative_time']:.3f}s ({func['percentage']:.1f}%)")
            
            # Show py-spy results
            if result.pyspy_result:
                pyspy = result.pyspy_result
                print(f"  py-spy:")
                print(f"    Flame graph: {Path(pyspy.flamegraph_path).name}")
                print(f"    Sample count: {pyspy.sample_count}")
                print(f"    Duration: {pyspy.sampling_duration:.2f}s")
                print(f"    Samples/sec: {pyspy.samples_per_second:.1f}")
            else:
                print(f"  py-spy: Not available (fallback used)")
            
            # Show component timings
            if result.component_timings:
                print(f"  Component breakdown:")
                total_component_time = sum(
                    time for component, time in result.component_timings.items()
                    if component in ['TaskDecomposer', 'ShellExecutor', 'ErrorRecovery', 
                                   'ContextManager', 'ImplementAgent']
                )
                test_time = result.component_timings.get('Test_Infrastructure', 0.0)
                
                if total_component_time > 0 or test_time > 0:
                    total_categorized = total_component_time + test_time
                    impl_pct = (total_component_time / total_categorized * 100) if total_categorized > 0 else 0
                    test_pct = (test_time / total_categorized * 100) if total_categorized > 0 else 0
                    
                    print(f"    ImplementAgent: {impl_pct:.1f}% ({total_component_time:.3f}s)")
                    print(f"    Test Infrastructure: {test_pct:.1f}% ({test_time:.3f}s)")
                    
                    # Show individual components with significant time
                    for component, timing in result.component_timings.items():
                        if timing > 0.001 and component not in ['Test_Infrastructure', 'Other']:
                            pct = (timing / total_categorized * 100) if total_categorized > 0 else 0
                            print(f"      {component}: {pct:.1f}% ({timing:.3f}s)")
            
        except Exception as e:
            print(f"✗ Profiling failed: {e}")
    
    # 4. Show output files
    print(f"\n4. Generated Files:")
    
    # Check profiles directory
    profiles_dir = PerformanceConfig.get_output_dir("profiles")
    profile_files = list(profiles_dir.glob("*.prof"))
    if profile_files:
        print(f"   cProfile files ({len(profile_files)}):")
        for profile_file in profile_files[-3:]:  # Show last 3
            size = profile_file.stat().st_size
            print(f"     {profile_file.name} ({size:,} bytes)")
    
    # Check flamegraphs directory
    flamegraphs_dir = PerformanceConfig.get_output_dir("flamegraphs")
    flamegraph_files = list(flamegraphs_dir.glob("*"))
    if flamegraph_files:
        print(f"   Flame graph files ({len(flamegraph_files)}):")
        for flamegraph_file in flamegraph_files[-3:]:  # Show last 3
            size = flamegraph_file.stat().st_size
            file_type = "SVG" if flamegraph_file.suffix == '.svg' else "Text"
            print(f"     {flamegraph_file.name} ({size:,} bytes, {file_type})")
    
    # 5. Summary and recommendations
    print(f"\n5. Summary and Recommendations:")
    
    if profiler.pyspy_available:
        if getattr(profiler, 'pyspy_needs_root', False):
            print("   ⚠ py-spy requires root permissions on this system")
            print("   💡 Run with 'sudo' for full py-spy functionality")
            print("   ✓ Fallback flame graphs are generated automatically")
        else:
            print("   ✓ py-spy is fully functional")
            print("   🔥 SVG flame graphs will be generated")
        
        if sudo_available:
            print("   ✓ sudo is available for py-spy")
            print("   💡 Consider running: sudo python -m pytest tests/perf/demo_pyspy_integration.py")
        else:
            print("   ⚠ sudo not available or requires password")
    else:
        print("   ❌ py-spy not available")
        print("   💡 Install with: pip install py-spy")
    
    print("\n   📊 cProfile provides detailed function-level analysis")
    print("   🔥 py-spy provides visual flame graphs for performance hotspots")
    print("   🔄 Both profilers work together for comprehensive analysis")
    
    print(f"\n🎉 py-spy integration demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demonstrate_pyspy_integration())