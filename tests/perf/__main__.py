#!/usr/bin/env python3
"""
Entry point for running performance analysis CLI as a module.

Usage:
    python -m tests.perf [command] [options]
"""

import asyncio
from .cli import main

if __name__ == '__main__':
    asyncio.run(main())