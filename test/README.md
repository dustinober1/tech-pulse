# Tech-Pulse Test Suite

This directory contains the comprehensive test suite for the Tech-Pulse data loader module.

## Test Structure

- `test_data_loader.py` - Unit tests for all data_loader functions
- `run_tests.py` - Test runner with detailed reporting and result saving
- `__init__.py` - Package initialization file

## Running Tests

### Option 1: Using the test runner (recommended)
```bash
python test/run_tests.py
```

This will:
- Run all unit tests with verbose output
- Save detailed results to `test_results/` directory
- Create timestamped test reports and summaries
- Display a summary in the terminal

### Option 2: Using unittest directly
```bash
python -m unittest test.test_data_loader
```

### Option 3: Running a specific test class
```bash
python -m unittest test.test_data_loader.TestFetchHnData
```

### Option 4: Running a specific test method
```bash
python -m unittest test.test_data_loader.TestFetchHnData.test_fetch_hn_data_success
```

## Test Coverage

The test suite covers:

1. **fetch_story_ids** - Fetching top story IDs from Hacker News
2. **fetch_story_details** - Fetching individual story details
3. **extract_story_data** - Data extraction and transformation
4. **process_stories_to_dataframe** - DataFrame creation and sorting
5. **fetch_hn_data** - Integration testing of the complete workflow

## Test Categories

- **Success scenarios**: Normal operation with mocked API responses
- **Error handling**: Network errors, JSON errors, missing data
- **Edge cases**: Empty inputs, missing fields, None values
- **Parameter testing**: Custom limits, URLs, and configurations

## Test Results

Results are automatically saved to the `test_results/` directory:
- `test_results_YYYYMMDD_HHMMSS.txt` - Detailed test output
- `test_summary_YYYYMMDD_HHMMSS.json` - JSON summary with metrics
- `latest_test_results.txt` - Symlink to the most recent test results

## Current Status

- **Tests**: 22 unit tests
- **Coverage**: 100% success rate
- **Test Classes**: 5 (one for each major function)
- **Last Run**: Check test_results for latest status

## Adding New Tests

When adding new functions to `data_loader.py`:
1. Create corresponding test methods in the appropriate test class
2. Follow the existing naming convention (`test_function_name_scenario`)
3. Include both success and error scenarios
4. Update this README if adding new test classes

## Requirements

- Python 3.7+
- pandas
- requests
- unittest (included in Python standard library)