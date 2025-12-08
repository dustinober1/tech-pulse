#!/usr/bin/env python3
"""
Test runner for Tech-Pulse data_loader module.
This script runs all unit tests and saves results to test_results/ folder.
"""

import unittest
import sys
import os
from datetime import datetime
import io
from contextlib import redirect_stdout, redirect_stderr

# Add the parent directory to the path to import data_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests():
    """Run all tests and capture output"""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Create StringIO objects to capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Run tests with captured output
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        runner = unittest.TextTestRunner(
            stream=sys.stdout,
            verbosity=2,
            buffer=True
        )
        result = runner.run(suite)

    return result, stdout_capture.getvalue(), stderr_capture.getvalue()


def save_test_results(result, stdout_output, stderr_output):
    """Save test results to test_results/ folder"""
    # Create test_results directory if it doesn't exist
    test_results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'test_results'
    )
    os.makedirs(test_results_dir, exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_file = os.path.join(test_results_dir, f'test_results_{timestamp}.txt')

    with open(results_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TECH-PULSE DATA LOADER TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tests Run: {result.testsRun}\n")
        f.write(f"Failures: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}\n")
        f.write(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%\n")
        f.write("\n")

        # Overall test result
        if result.wasSuccessful():
            f.write("OVERALL RESULT: ✓ ALL TESTS PASSED\n\n")
        else:
            f.write("OVERALL RESULT: ✗ SOME TESTS FAILED\n\n")

        # Test output
        if stdout_output:
            f.write("STANDARD OUTPUT:\n")
            f.write("-" * 40 + "\n")
            f.write(stdout_output)
            f.write("\n" + "-" * 40 + "\n\n")

        # Error output
        if stderr_output:
            f.write("ERROR OUTPUT:\n")
            f.write("-" * 40 + "\n")
            f.write(stderr_output)
            f.write("\n" + "-" * 40 + "\n\n")

        # Failures
        if result.failures:
            f.write("FAILURES:\n")
            f.write("-" * 40 + "\n")
            for test, traceback in result.failures:
                f.write(f"FAIL: {test}\n")
                f.write(f"{traceback}\n")
                f.write("-" * 40 + "\n")
            f.write("\n")

        # Errors
        if result.errors:
            f.write("ERRORS:\n")
            f.write("-" * 40 + "\n")
            for test, traceback in result.errors:
                f.write(f"ERROR: {test}\n")
                f.write(f"{traceback}\n")
                f.write("-" * 40 + "\n")
            f.write("\n")

        # Test summary by class
        f.write("TEST SUMMARY BY CLASS:\n")
        f.write("-" * 40 + "\n")
        test_classes = {}

        for test_name, test_result in result.failures + result.errors:
            # Handle test name parsing more robustly
            if hasattr(test_name, 'id'):
                # For unittest.TestCase items
                test_id = test_name.id()
                class_name = test_id.split('.')[-2] if '.' in test_id else 'Unknown'
            elif isinstance(test_name, str):
                # For string representations
                class_name = test_name.split('(')[1].split(')')[0] if '(' in test_name else 'Unknown'
            else:
                class_name = 'Unknown'
            if class_name not in test_classes:
                test_classes[class_name] = {'passed': 0, 'failed': 0}
            test_classes[class_name]['failed'] += 1

        # Count passed tests (approximate)
        total_tests_per_class = {
            'TestFetchStoryIds': 4,
            'TestFetchStoryDetails': 3,
            'TestExtractStoryData': 6,
            'TestProcessStoriesToDataframe': 4,
            'TestFetchHnData': 5
        }

        for class_name, total in total_tests_per_class.items():
            failed = test_classes.get(class_name, {}).get('failed', 0)
            passed = total - failed
            test_classes[class_name] = {'passed': passed, 'failed': failed}

        for class_name, counts in sorted(test_classes.items()):
            total = counts['passed'] + counts['failed']
            status = "✓" if counts['failed'] == 0 else "✗"
            f.write(f"{status} {class_name}: {counts['passed']}/{total} passed\n")

        f.write("\n" + "=" * 80 + "\n")

    # Save summary JSON
    summary_file = os.path.join(test_results_dir, f'test_summary_{timestamp}.json')

    import json
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
        'was_successful': result.wasSuccessful(),
        'test_classes': test_classes
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    # Create latest symlink (Unix systems only)
    latest_file = os.path.join(test_results_dir, 'latest_test_results.txt')
    try:
        if os.path.exists(latest_file):
            os.remove(latest_file)
        os.symlink(results_file, latest_file)
    except (OSError, NotImplementedError):
        # Symlink creation failed, skip it
        pass

    return results_file, summary_file


def main():
    """Main function to run tests and save results"""
    print("Running Tech-Pulse Data Loader Tests...")
    print("=" * 60)

    # Run tests
    result, stdout_output, stderr_output = run_tests()

    # Save results
    results_file, summary_file = save_test_results(result, stdout_output, stderr_output)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("  Result: ✓ ALL TESTS PASSED")
    else:
        print("  Result: ✗ SOME TESTS FAILED")

    print(f"\nDetailed results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)