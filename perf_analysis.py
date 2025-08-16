#!/usr/bin/env python3
"""
Standalone performance analysis tool.

This script provides easy access to the performance analysis CLI
without needing to use the module syntax.

Usage:
    python perf_analysis.py [command] [options]
    
Examples:
    python perf_analysis.py timing
    python perf_analysis.py profile --max-tests 5
    python perf_analysis.py selective test_real_agent_manager
    python perf_analysis.py list --pattern "test_real_*"
"""

import sys
import asyncio
from pathlib import Path

# Add tests/perf to path for imports
sys.path.insert(0, str(Path(__file__).parent / "tests" / "perf"))

from cli import main

if __name__ == '__main__':
    asyncio.run(main())