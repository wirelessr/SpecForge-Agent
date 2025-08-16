#!/usr/bin/env python3
"""
Test script to verify py-spy integration in PerformanceProfiler.

This script tests all aspects of the py-spy integration including:
- py-spy availability detection
- Process-based profiling with py-spy
- SVG flame graph generation and storage
- Fallback handling when py-spy is not available
"""

import asyncio
import sys
import os
import tempfile
import shutil
from pathlib import Path

from .profiler import PerformanceProfiler
from .models import TestIdentifier, ProfilerType
from .config import PerformanceConfig, EnvironmentConfig


async def test_pyspy_availability():
    """Test py-spy availability detection."""
    print("=== Testing py-spy Availability ===")
    
    # Test environment config
    available = EnvironmentConfig.is_pyspy_available()
    print(f"py-spy available (EnvironmentConfig): {available}")
    
    # Test profiler detection
    profiler = PerformanceProfiler()
    print(f"py-spy available (PerformanceProfiler): {profiler.pyspy_available}")
    
    # Check actual py-spy executable
    pyspy_path = shutil.which('py-spy')
    print(f"py-spy executable path: {pyspy_path}")
    
    return available and profiler.pyspy_available


async def test_simple_pyspy_profiling():
    """Test basic py-spy profiling functionality."""
    print("\n=== Testing Simple py-spy Profiling ===")
    
    profiler = PerformanceProfiler()
    
    if not profiler.pyspy_available:
        print("py-spy not available, skipping test")
        return False
    
    # Create a simple test identifier for a quick test
    test_id = TestIdentifier(
        file_path='tests/integration/test_context_compressor_dynamic_config.py',
        class_name='TestContextCompressorDynamicConfig',
        method_name='test_fallback_to_defaults'
    )
    
    try:
        print(f"Profiling test: {test_id.full_name}")
        result = await profiler.profile_with_pyspy(test_id)
        
        print(f"✓ py-spy profiling successful!")
        print(f"  Flame graph path: {result.flamegraph_path}")
        print(f"  Sample count: {result.sample_count}")
        print(f"  Sampling duration: {result.sampling_duration:.2f}s")
        print(f"  Process ID: {result.process_id}")
        
        # Check if flame graph file exists
        if Path(result.flamegraph_path).exists():
            file_size = Path(result.flamegraph_path).stat().st_size
            print(f"  Flame graph file size: {file_size} bytes")
            return True
        else:
            print(f"✗ Flame graph file not found: {result.flamegraph_path}")
            return False
            
    except RuntimeError as e:
        if "root permissions" in str(e):
            print(f"⚠ py-spy requires root permissions: {e}")
            print("  This is expected on macOS. Testing fallback functionality...")
            
            # Test fallback flame graph creation
            try:
                fallback_path = profiler.create_fallback_flame_graph(test_id)
                if fallback_path and Path(fallback_path).exists():
                    print(f"✓ Fallback flame graph created: {fallback_path}")
                    return True
                else:
                    print("✗ Fallback flame graph creation failed")
                    return False
            except Exception as fallback_error:
                print(f"✗ Fallback creation failed: {fallback_error}")
                return False
        else:
            print(f"✗ Unexpected RuntimeError: {e}")
            return False
            
    except Exception as e:
        print(f"✗ py-spy profiling failed: {e}")
        return False


async def test_flame_graph_generation():
    """Test flame graph generation from py-spy data."""
    print("\n=== Testing Flame Graph Generation ===")
    
    profiler = PerformanceProfiler()
    
    if not profiler.pyspy_available:
        print("py-spy not available, skipping test")
        return False
    
    # Create a test identifier
    test_id = TestIdentifier(
        file_path='tests/integration/test_context_compressor_dynamic_config.py',
        method_name='test_model_pattern_matching'
    )
    
    try:
        # Profile with py-spy
        pyspy_result = await profiler.profile_with_pyspy(test_id)
        
        # Test flame graph generation method
        flame_graph_path = profiler.generate_flame_graph_from_pyspy(pyspy_result)
        
        print(f"✓ Flame graph generation successful!")
        print(f"  Generated path: {flame_graph_path}")
        
        # Verify the file exists and has content
        if Path(flame_graph_path).exists():
            file_size = Path(flame_graph_path).stat().st_size
            print(f"  File size: {file_size} bytes")
            
            # Check if it's a valid SVG file
            with open(flame_graph_path, 'r') as f:
                content = f.read(100)  # Read first 100 chars
                if '<svg' in content.lower():
                    print(f"  ✓ Valid SVG format detected")
                    return True
                else:
                    print(f"  ✗ Invalid SVG format")
                    return False
        else:
            print(f"  ✗ Flame graph file not found")
            return False
            
    except RuntimeError as e:
        if "root permissions" in str(e):
            print(f"⚠ py-spy requires root permissions: {e}")
            print("  Testing fallback flame graph generation...")
            
            # Test fallback flame graph creation
            try:
                fallback_path = profiler.create_fallback_flame_graph(test_id)
                if fallback_path and Path(fallback_path).exists():
                    print(f"✓ Fallback flame graph created: {fallback_path}")
                    
                    # Check if it's a valid text file
                    with open(fallback_path, 'r') as f:
                        content = f.read(200)
                        if 'Fallback Flame Graph' in content:
                            print(f"  ✓ Valid fallback format detected")
                            return True
                        else:
                            print(f"  ✗ Invalid fallback format")
                            return False
                else:
                    print("✗ Fallback flame graph creation failed")
                    return False
            except Exception as fallback_error:
                print(f"✗ Fallback creation failed: {fallback_error}")
                return False
        else:
            print(f"✗ Unexpected RuntimeError: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Flame graph generation failed: {e}")
        return False


async def test_fallback_handling():
    """Test fallback handling when py-spy is not available."""
    print("\n=== Testing Fallback Handling ===")
    
    # Create a profiler instance
    profiler = PerformanceProfiler()
    
    # Temporarily disable py-spy
    original_available = profiler.pyspy_available
    profiler.pyspy_available = False
    
    test_id = TestIdentifier(
        file_path='tests/integration/test_context_compressor_dynamic_config.py',
        method_name='test_fallback_to_defaults'
    )
    
    try:
        # This should raise an error
        await profiler.profile_with_pyspy(test_id)
        print("✗ Expected RuntimeError but profiling succeeded")
        return False
        
    except RuntimeError as e:
        if "py-spy is not available" in str(e):
            print("✓ Correct fallback behavior - RuntimeError raised")
            return True
        else:
            print(f"✗ Unexpected RuntimeError: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Unexpected exception: {e}")
        return False
        
    finally:
        # Restore original availability
        profiler.pyspy_available = original_available


async def test_integrated_profiling():
    """Test integrated profiling with both cProfile and py-spy."""
    print("\n=== Testing Integrated Profiling ===")
    
    profiler = PerformanceProfiler()
    
    if not profiler.pyspy_available:
        print("py-spy not available, skipping integrated test")
        return False
    
    test_id = TestIdentifier(
        file_path='tests/integration/test_context_compressor_dynamic_config.py',
        method_name='test_context_compressor_with_real_config_manager'
    )
    
    try:
        # Use the profile_specific_test method which combines both profilers
        result = await profiler.profile_specific_test(test_id)
        
        print(f"✓ Integrated profiling successful!")
        print(f"  Profiler used: {result.profiler_used}")
        print(f"  Total profiling time: {result.total_profiling_time:.2f}s")
        
        # Check cProfile result
        if result.cprofile_result:
            print(f"  ✓ cProfile data available")
            print(f"    Total time: {result.cprofile_result.total_time:.2f}s")
            print(f"    Call count: {result.cprofile_result.call_count}")
        
        # Check py-spy result
        if result.pyspy_result:
            print(f"  ✓ py-spy data available")
            print(f"    Flame graph: {result.pyspy_result.flamegraph_path}")
            print(f"    Sample count: {result.pyspy_result.sample_count}")
        
        # Check component timings
        if result.component_timings:
            print(f"  ✓ Component timings extracted:")
            for component, timing in result.component_timings.items():
                if timing > 0:
                    print(f"    {component}: {timing:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Integrated profiling failed: {e}")
        return False


async def main():
    """Run all py-spy integration tests."""
    print("Testing py-spy Integration for PerformanceProfiler")
    print("=" * 50)
    
    tests = [
        ("py-spy Availability", test_pyspy_availability),
        ("Simple py-spy Profiling", test_simple_pyspy_profiling),
        ("Flame Graph Generation", test_flame_graph_generation),
        ("Fallback Handling", test_fallback_handling),
        ("Integrated Profiling", test_integrated_profiling),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All py-spy integration tests passed!")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)