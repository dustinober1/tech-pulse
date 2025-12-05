# Tech-Pulse Scripts

This directory contains utility scripts for the Tech-Pulse project.

## verify_deployment.py

A deployment verification script that checks if a Streamlit app is accessible and responding correctly.

### Features

- HTTP health check with configurable timeout
- Comprehensive error handling (connection errors, timeouts, HTTP errors)
- Detailed logging with timestamps
- Command-line interface with argument parsing
- Default URL for Tech-Pulse deployment
- Verbose mode for debugging
- Proper exit codes (0 for success, 1 for failure)

### Usage

```bash
# Verify default deployment
python scripts/verify_deployment.py

# Verify specific URL
python scripts/verify_deployment.py https://tech-pulse.streamlit.app

# Verify with custom timeout
python scripts/verify_deployment.py --timeout 10 https://tech-pulse.streamlit.app

# Enable verbose output
python scripts/verify_deployment.py -v https://tech-pulse.streamlit.app

# Show help
python scripts/verify_deployment.py --help
```

### Testing

Unit tests are located in `/test/test_verify_deployment.py`. Run them with:

```bash
python -m unittest test.test_verify_deployment -v
```

### Exit Codes

- `0`: Deployment verified successfully (HTTP 200)
- `1`: Deployment verification failed (connection error, timeout, non-200 status)

### Examples

```bash
# Success case
$ python scripts/verify_deployment.py https://www.google.com
2025-12-04 20:26:13 - INFO - ============================================================
2025-12-04 20:26:13 - INFO - Tech-Pulse Deployment Verification
2025-12-04 20:26:13 - INFO - ============================================================
2025-12-04 20:26:13 - INFO - Verifying deployment at: https://www.google.com
2025-12-04 20:26:13 - INFO - ✓ Deployment verified successfully!
2025-12-04 20:26:13 - INFO -   Status Code: 200
2025-12-04 20:26:13 - INFO -   URL: https://www.google.com
2025-12-04 20:26:13 - INFO - ============================================================
2025-12-04 20:26:13 - INFO - Result: DEPLOYMENT VERIFIED ✓
2025-12-04 20:26:13 - INFO - ============================================================

# Failure case
$ python scripts/verify_deployment.py https://nonexistent-app.streamlit.app
2025-12-04 20:26:16 - INFO - ============================================================
2025-12-04 20:26:16 - INFO - Tech-Pulse Deployment Verification
2025-12-04 20:26:16 - INFO - ============================================================
2025-12-04 20:26:16 - INFO - Verifying deployment at: https://nonexistent-app.streamlit.app
2025-12-04 20:26:17 - ERROR - ✗ Deployment check failed!
2025-12-04 20:26:17 - ERROR -   Status Code: 404
2025-12-04 20:26:17 - ERROR -   URL: https://nonexistent-app.streamlit.app
2025-12-04 20:26:17 - INFO - ============================================================
2025-12-04 20:26:17 - ERROR - Result: DEPLOYMENT VERIFICATION FAILED ✗
2025-12-04 20:26:17 - INFO - ============================================================
```
