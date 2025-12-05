"""
Network utilities for resilient HTTP requests.

Provides timeout and retry logic for network operations
to improve application stability.
"""

import requests
import asyncio
import time
from typing import Optional, Dict, Any, Union
import logging

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)


def resilient_request(url: str,
                     method: str = 'GET',
                     timeout: int = 10,
                     max_retries: int = 3,
                     retry_delay: float = 1.0,
                     **kwargs) -> Optional[requests.Response]:
    """
    Make an HTTP request with timeout and retry logic.

    Args:
        url: URL to request
        method: HTTP method (GET, POST, etc.)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        **kwargs: Additional arguments for requests

    Returns:
        requests.Response if successful, None otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=timeout,
                **kwargs
            )
            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            if attempt < max_retries:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue

    logger.error(f"Failed after {max_retries + 1} attempts: {url}")
    return None


async def resilient_aiohttp_request(session,
                                   url: str,
                                   method: str = 'GET',
                                   timeout: int = 10,
                                   max_retries: int = 3,
                                   retry_delay: float = 1.0,
                                   **kwargs) -> Optional[Dict[str, Any]]:
    """
    Make an async HTTP request with aiohttp with timeout and retry logic.

    Args:
        session: aiohttp ClientSession
        url: URL to request
        method: HTTP method (GET, POST, etc.)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        **kwargs: Additional arguments for aiohttp

    Returns:
        Dict with response data if successful, None otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            async with session.request(
                method=method,
                url=url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                **kwargs
            ) as response:
                response.raise_for_status()
                content = await response.json() if response.content_type == 'application/json' else await response.text()
                return {
                    'status': response.status,
                    'content': content,
                    'headers': dict(response.headers)
                }

        except asyncio.TimeoutError:
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            if attempt < max_retries:
                await asyncio.sleep(retry_delay * (2 ** attempt))
                continue

        except Exception as e:
            logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)
                continue

    logger.error(f"Failed after {max_retries + 1} attempts: {url}")
    return None