#!/usr/bin/env python3
"""
Test runner script for Collision Parts Prediction system.
Provides convenient commands for running different test suites.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    if description:
        print(f"\n🔄 {description}")
        print("-" * 60)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    else:
        print(f"✅ {description or 'Command'} completed successfully")
        return True


def run_unit_tests(verbose=False):
    """Run unit tests only."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "not slow and not integration"]
    if verbose:
        cmd.extend(["-v", "--tb=short"])
    cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    return run_command(cmd, "Running unit tests")


def run_integration_tests(verbose=False):
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "integration"]
    if verbose:
        cmd.extend(["-v", "--tb=short"])
    cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    return run_command(cmd, "Running integration tests")


def run_api_tests(verbose=False):
    """Run API tests."""
    cmd = ["python", "-m", "pytest", "tests/test_infer_api.py"]
    if verbose:
        cmd.extend(["-v", "--tb=short"])
    cmd.extend(["--cov=src.infer"])
    
    return run_command(cmd, "Running API tests")


def run_slow_tests(verbose=False):
    """Run slow tests (performance, model loading, etc.)."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "slow"]
    if verbose:
        cmd.extend(["-v", "--tb=short"])
    
    return run_command(cmd, "Running slow tests")


def run_all_tests(verbose=False):
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    if verbose:
        cmd.extend(["-v", "--tb=short"])
    cmd.extend([
        "--cov=src", 
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--junit-xml=reports/junit.xml"
    ])
    
    return run_command(cmd, "Running all tests")


def run_coverage_report():
    """Generate detailed coverage report."""
    cmd = [
        "python", "-m", "pytest", "tests/",
        "-m", "not slow",  # Skip slow tests for coverage
        "--cov=src",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=70"
    ]
    
    success = run_command(cmd, "Generating coverage report")
    
    if success:
        print(f"\n📊 Coverage report generated:")
        print(f"   - HTML: file://{Path.cwd()}/htmlcov/index.html")
        print(f"   - XML: {Path.cwd()}/coverage.xml")
    
    return success


def run_quality_checks():
    """Run code quality checks."""
    checks = [
        (["python", "-m", "black", "--check", "src/", "tests/"], "Black formatting check"),
        (["python", "-m", "isort", "--check-only", "src/", "tests/"], "Import sorting check"),
        (["python", "-m", "ruff", "check", "src/", "tests/"], "Ruff linting check"),
        (["python", "-m", "bandit", "-r", "src/"], "Security check with Bandit"),
    ]
    
    all_passed = True
    for cmd, description in checks:
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def fix_quality_issues():
    """Fix code quality issues automatically."""
    fixes = [
        (["python", "-m", "black", "src/", "tests/"], "Fixing formatting with Black"),
        (["python", "-m", "isort", "src/", "tests/"], "Fixing imports with isort"),
        (["python", "-m", "ruff", "check", "--fix", "src/", "tests/"], "Fixing linting issues with Ruff"),
    ]
    
    all_passed = True
    for cmd, description in fixes:
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def install_test_dependencies():
    """Install test dependencies."""
    cmd = ["python", "-m", "pip", "install", "-r", "requirements.txt"]
    return run_command(cmd, "Installing test dependencies")


def setup_pre_commit():
    """Set up pre-commit hooks."""
    commands = [
        (["python", "-m", "pip", "install", "pre-commit"], "Installing pre-commit"),
        (["pre-commit", "install"], "Installing pre-commit hooks"),
        (["pre-commit", "run", "--all-files"], "Running pre-commit on all files"),
    ]
    
    all_passed = True
    for cmd, description in commands:
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for Collision Parts Prediction system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py unit              # Run unit tests only
  python run_tests.py integration       # Run integration tests
  python run_tests.py api               # Run API tests
  python run_tests.py all -v            # Run all tests with verbose output
  python run_tests.py coverage          # Generate coverage report
  python run_tests.py quality           # Run code quality checks
  python run_tests.py fix               # Fix code quality issues
  python run_tests.py setup             # Install dependencies and setup
        """
    )
    
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "api", "slow", "all", "coverage", "quality", "fix", "setup"],
        help="Type of tests to run or action to perform"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    print(f"🚀 Collision Parts Prediction Test Runner")
    print(f"Task: {args.test_type}")
    print("=" * 60)
    
    success = False
    
    if args.test_type == "unit":
        success = run_unit_tests(args.verbose)
    elif args.test_type == "integration":
        success = run_integration_tests(args.verbose)
    elif args.test_type == "api":
        success = run_api_tests(args.verbose)
    elif args.test_type == "slow":
        success = run_slow_tests(args.verbose)
    elif args.test_type == "all":
        success = run_all_tests(args.verbose)
    elif args.test_type == "coverage":
        success = run_coverage_report()
    elif args.test_type == "quality":
        success = run_quality_checks()
    elif args.test_type == "fix":
        success = fix_quality_issues()
    elif args.test_type == "setup":
        success = install_test_dependencies() and setup_pre_commit()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tasks completed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tasks failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()