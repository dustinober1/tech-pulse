"""
Test suite for requirements.txt validation.

This test file follows TDD methodology to ensure requirements.txt:
- Exists in the project root
- Contains properly formatted package specifications
- Includes all critical packages needed for the project
- Excludes standard library packages
- Has version specifications for all packages

Author: Tech-Pulse Team
Date: 2025-12-04
"""

import unittest
import os
import re
from pathlib import Path


class TestRequirements(unittest.TestCase):
    """Test cases for requirements.txt file validation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - locate requirements.txt file."""
        # Get the project root directory (parent of test directory)
        cls.test_dir = Path(__file__).parent
        cls.project_root = cls.test_dir.parent
        cls.requirements_path = cls.project_root / 'requirements.txt'

        # Define critical packages required for the project
        cls.critical_packages = {
            'streamlit',
            'pandas',
            'plotly',
            'requests',
            'nltk',
            'bertopic',
            'scikit-learn',
            'hdbscan',
            'sentence-transformers',
            'pyarrow',
            'tabulate'
        }

        # Standard library packages that should NOT be in requirements.txt
        cls.standard_library_packages = {
            'os',
            'sys',
            'time',
            'json',
            'datetime',
            'collections',
            'itertools',
            'functools',
            'typing',
            'pathlib',
            're',
            'unittest',
            'logging',
            'io',
            'csv',
            'math',
            'random',
            'subprocess',
            'threading',
            'multiprocessing'
        }

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists in the project root."""
        self.assertTrue(
            self.requirements_path.exists(),
            f"requirements.txt not found at {self.requirements_path}"
        )
        self.assertTrue(
            self.requirements_path.is_file(),
            f"{self.requirements_path} exists but is not a file"
        )

    def test_requirements_file_format(self):
        """Test that each line in requirements.txt follows proper pip format.

        Valid formats:
        - package==version
        - package>=version
        - package<=version
        - package~=version
        - package>version
        - package<version

        Invalid:
        - Blank lines (except at start/end)
        - Comment-only lines
        - Lines without version specifications
        - Improperly formatted lines
        """
        self.assertTrue(self.requirements_path.exists(), "requirements.txt must exist")

        with open(self.requirements_path, 'r') as f:
            lines = f.readlines()

        # Pattern for valid pip package specification
        # Matches: package>=1.2.3, package==1.2.3, package~=1.2, etc.
        valid_pattern = re.compile(
            r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?(==|>=|<=|>|<|~=)\d+(\.\d+)*(\.\*)?([a-zA-Z0-9._-]*)?$'
        )

        errors = []
        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Skip completely empty lines
            if not stripped:
                errors.append(f"Line {line_num}: Blank line found - '{line.rstrip()}'")
                continue

            # Check for comment lines
            if stripped.startswith('#'):
                errors.append(f"Line {line_num}: Comment line found - '{stripped}'")
                continue

            # Check for inline comments
            if '#' in stripped:
                errors.append(f"Line {line_num}: Inline comment found - '{stripped}'")
                continue

            # Validate format
            if not valid_pattern.match(stripped):
                errors.append(f"Line {line_num}: Invalid format - '{stripped}'")

        self.assertEqual(
            len(errors), 0,
            f"requirements.txt has formatting issues:\n" + "\n".join(errors)
        )

    def test_required_packages_present(self):
        """Test that all critical packages are present in requirements.txt.

        Critical packages:
        - streamlit: Dashboard framework
        - pandas: Data manipulation
        - plotly: Interactive visualizations
        - requests: HTTP requests for data fetching
        - nltk: Natural language processing
        - bertopic: Topic modeling
        - scikit-learn: Machine learning utilities
        - hdbscan: Clustering algorithm
        - sentence-transformers: Text embeddings
        - pyarrow: Efficient data storage
        - tabulate: Table formatting
        """
        self.assertTrue(self.requirements_path.exists(), "requirements.txt must exist")

        with open(self.requirements_path, 'r') as f:
            content = f.read()

        # Extract package names from requirements.txt
        # Handle formats like: package>=1.2.3, package==1.2.3, etc.
        package_pattern = re.compile(r'^([a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]?)', re.MULTILINE)
        found_packages = set()

        for line in content.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                match = package_pattern.match(stripped)
                if match:
                    # Normalize package names (handle both hyphens and underscores)
                    package_name = match.group(1).lower().replace('_', '-')
                    found_packages.add(package_name)

        # Normalize critical package names for comparison
        normalized_critical = {pkg.lower().replace('_', '-') for pkg in self.critical_packages}

        missing_packages = normalized_critical - found_packages

        self.assertEqual(
            len(missing_packages), 0,
            f"Missing critical packages in requirements.txt: {sorted(missing_packages)}\n"
            f"Found packages: {sorted(found_packages)}"
        )

    def test_no_standard_library_packages(self):
        """Test that no standard library packages are listed in requirements.txt.

        Standard library packages should not be in requirements.txt because:
        - They are included with Python installation
        - Including them can cause pip install errors
        - They don't have version numbers in the standard library

        Common standard library packages to avoid:
        os, sys, time, json, datetime, collections, re, unittest, etc.
        """
        self.assertTrue(self.requirements_path.exists(), "requirements.txt must exist")

        with open(self.requirements_path, 'r') as f:
            content = f.read()

        # Extract package names from requirements.txt
        package_pattern = re.compile(r'^([a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]?)', re.MULTILINE)
        found_stdlib_packages = []

        for line in content.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                match = package_pattern.match(stripped)
                if match:
                    package_name = match.group(1).lower()
                    if package_name in self.standard_library_packages:
                        found_stdlib_packages.append(package_name)

        self.assertEqual(
            len(found_stdlib_packages), 0,
            f"Standard library packages found in requirements.txt: {found_stdlib_packages}\n"
            f"These packages are part of Python's standard library and should not be listed."
        )

    def test_version_specifications(self):
        """Test that all packages have version specifications.

        Every package should have a version specification using:
        - >= (minimum version, recommended)
        - == (exact version)
        - ~= (compatible version)
        - Other valid version specifiers

        This ensures:
        - Reproducible builds
        - Dependency compatibility
        - Clear version requirements
        """
        self.assertTrue(self.requirements_path.exists(), "requirements.txt must exist")

        with open(self.requirements_path, 'r') as f:
            lines = f.readlines()

        # Pattern to match package with version specifier
        versioned_pattern = re.compile(
            r'^([a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]?)(==|>=|<=|>|<|~=)\d+(\.\d+)*'
        )

        packages_without_version = []

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue

            # Check if line has version specifier
            if not versioned_pattern.match(stripped):
                packages_without_version.append(f"Line {line_num}: {stripped}")

        self.assertEqual(
            len(packages_without_version), 0,
            f"Packages without version specifications:\n" + "\n".join(packages_without_version) +
            f"\n\nAll packages must have version specifications (use >= for minimum version)"
        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
