#!/usr/bin/env python3
"""
OpenRAG Enhanced - Test Runner

Runs all tests for the OpenRAG contribution modules:
- Hallucination detection tests
- Storage tests
- API router tests

Usage:
    python scripts/run_tests.py              # Run all tests
    python scripts/run_tests.py --unit       # Run unit tests only
    python scripts/run_tests.py --quick      # Run quick tests (no models)
    python scripts/run_tests.py --coverage   # Run with coverage report
"""

import subprocess
import sys
import argparse
import os


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60 + "\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run OpenRAG Enhanced tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests (skip slow tests)")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--file", "-f", help="Run specific test file")

    args = parser.parse_args()

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        cmd.append("-v")

    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=openrag", "--cov=src", "--cov-report=html", "--cov-report=term"])

    # Select tests
    if args.file:
        cmd.append(args.file)
    elif args.unit:
        cmd.append("tests/")
        cmd.extend(["-m", "not integration"])
    elif args.integration:
        cmd.append("tests/")
        cmd.extend(["-m", "integration"])
    else:
        cmd.append("tests/")

    # Skip slow tests if quick mode
    if args.quick:
        cmd.extend(["-m", "not slow and not integration"])

    # Run tests
    success = run_command(cmd, "Running tests")

    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print('='*60)

    if args.coverage:
        print("\n📊 Coverage report generated at: htmlcov/index.html")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
