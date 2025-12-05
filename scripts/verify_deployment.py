#!/usr/bin/env python3
"""
Deployment Verification Script for Tech-Pulse Streamlit App

This script verifies that a deployed Streamlit application is accessible
and responding correctly. It performs HTTP health checks and reports the
deployment status.

Usage:
    python verify_deployment.py [URL]

    If no URL is provided, defaults to https://tech-pulse.streamlit.app

Example:
    python verify_deployment.py https://tech-pulse.streamlit.app
    python verify_deployment.py  # Uses default URL
"""

import sys
import logging
import argparse
from typing import Optional
import requests
from requests.exceptions import (
    ConnectionError,
    Timeout,
    RequestException,
    HTTPError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default deployment URL
DEFAULT_URL = "https://tech-pulse.streamlit.app"

# Request timeout in seconds
REQUEST_TIMEOUT = 30


def verify_deployment(url: str) -> bool:
    """
    Verify that a deployed Streamlit app is accessible and responding.

    This function performs an HTTP GET request to the specified URL and
    checks if it returns a successful status code (200 OK). It handles
    various error conditions gracefully and logs appropriate messages.

    Args:
        url (str): The URL of the deployed application to verify.
                   Should be a complete URL including protocol (http/https).

    Returns:
        bool: True if the deployment is accessible and responding with
              HTTP 200, False otherwise.

    Examples:
        >>> verify_deployment("https://tech-pulse.streamlit.app")
        True

        >>> verify_deployment("https://nonexistent-app.streamlit.app")
        False

    Note:
        - The function uses a 30-second timeout for requests
        - Redirects are followed automatically
        - SSL verification is enabled by default
    """
    # Validate URL is not empty
    if not url or not url.strip():
        logger.error("URL cannot be empty")
        return False

    url = url.strip()

    # Basic URL validation
    if not url.startswith(('http://', 'https://')):
        logger.error(f"Invalid URL format: {url}. URL must start with http:// or https://")
        return False

    logger.info(f"Verifying deployment at: {url}")

    try:
        # Perform HTTP GET request
        response = requests.get(url, timeout=REQUEST_TIMEOUT)

        # Check status code
        if response.status_code == 200:
            logger.info(f"✓ Deployment verified successfully!")
            logger.info(f"  Status Code: {response.status_code}")
            logger.info(f"  URL: {url}")
            return True
        else:
            logger.error(f"✗ Deployment check failed!")
            logger.error(f"  Status Code: {response.status_code}")
            logger.error(f"  URL: {url}")
            return False

    except ConnectionError as e:
        logger.error(f"✗ Connection error: Unable to connect to {url}")
        logger.error(f"  Details: {str(e)}")
        return False

    except Timeout as e:
        logger.error(f"✗ Timeout error: Request timed out after {REQUEST_TIMEOUT} seconds")
        logger.error(f"  URL: {url}")
        logger.error(f"  Details: {str(e)}")
        return False

    except HTTPError as e:
        logger.error(f"✗ HTTP error occurred")
        logger.error(f"  URL: {url}")
        logger.error(f"  Details: {str(e)}")
        return False

    except RequestException as e:
        logger.error(f"✗ Request error occurred")
        logger.error(f"  URL: {url}")
        logger.error(f"  Details: {str(e)}")
        return False

    except Exception as e:
        logger.error(f"✗ Unexpected error occurred")
        logger.error(f"  URL: {url}")
        logger.error(f"  Error type: {type(e).__name__}")
        logger.error(f"  Details: {str(e)}")
        return False


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Verify Streamlit app deployment status',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s https://tech-pulse.streamlit.app
  %(prog)s https://my-custom-app.streamlit.app
        """
    )

    parser.add_argument(
        'url',
        nargs='?',
        default=DEFAULT_URL,
        help=f'URL of the deployed application (default: {DEFAULT_URL})'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=REQUEST_TIMEOUT,
        help=f'Request timeout in seconds (default: {REQUEST_TIMEOUT})'
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the deployment verification script.

    Parses command line arguments, performs deployment verification,
    and returns an appropriate exit code.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()

    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    # Update timeout if specified
    global REQUEST_TIMEOUT
    if args.timeout:
        REQUEST_TIMEOUT = args.timeout
        logger.debug(f"Using custom timeout: {REQUEST_TIMEOUT} seconds")

    logger.info("=" * 60)
    logger.info("Tech-Pulse Deployment Verification")
    logger.info("=" * 60)

    # Perform verification
    success = verify_deployment(args.url)

    logger.info("=" * 60)

    if success:
        logger.info("Result: DEPLOYMENT VERIFIED ✓")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("Result: DEPLOYMENT VERIFICATION FAILED ✗")
        logger.info("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
