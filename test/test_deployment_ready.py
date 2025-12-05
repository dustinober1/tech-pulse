"""
Deployment Readiness Tests
==========================
Comprehensive validation suite to ensure Tech-Pulse dashboard is ready for deployment.
Tests verify all critical files, configurations, and deployment prerequisites.

Author: Tech-Pulse Team
Date: 2025-12-04
"""

import unittest
import os
import subprocess
import re
from pathlib import Path


class TestDeploymentReady(unittest.TestCase):
    """Test suite for validating deployment readiness of Tech-Pulse dashboard."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - run once before all tests."""
        # Define project root directory
        cls.project_root = Path("/Users/dustinober/tech-pulse")

        # Define critical file paths
        cls.app_py = cls.project_root / "app.py"
        cls.data_loader_py = cls.project_root / "data_loader.py"
        cls.dashboard_config_py = cls.project_root / "dashboard_config.py"
        cls.cache_manager_py = cls.project_root / "cache_manager.py"
        cls.requirements_txt = cls.project_root / "requirements.txt"
        cls.gitignore = cls.project_root / ".gitignore"

    def test_app_py_exists(self):
        """Test 1: Verify app.py exists."""
        self.assertTrue(
            self.app_py.exists(),
            f"app.py not found at {self.app_py}"
        )
        self.assertTrue(
            self.app_py.is_file(),
            f"{self.app_py} exists but is not a file"
        )

    def test_app_py_has_main(self):
        """Test 2: Check that app.py has streamlit code or main entry point."""
        self.assertTrue(self.app_py.exists(), "app.py must exist")

        with open(self.app_py, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for Streamlit page config (required for Streamlit apps)
        has_page_config = 'st.set_page_config' in content

        # Check for main entry point
        has_main_block = 'if __name__ == "__main__"' in content

        # At least one should be present
        self.assertTrue(
            has_page_config or has_main_block,
            "app.py must contain st.set_page_config or if __name__ == '__main__' block"
        )

        # For a Streamlit app, page config is strongly recommended
        self.assertTrue(
            has_page_config,
            "app.py should call st.set_page_config for proper Streamlit configuration"
        )

    def test_data_loader_exists(self):
        """Test 3: Verify data_loader.py exists."""
        self.assertTrue(
            self.data_loader_py.exists(),
            f"data_loader.py not found at {self.data_loader_py}"
        )
        self.assertTrue(
            self.data_loader_py.is_file(),
            f"{self.data_loader_py} exists but is not a file"
        )

    def test_dashboard_config_exists(self):
        """Test 4: Verify dashboard_config.py exists."""
        self.assertTrue(
            self.dashboard_config_py.exists(),
            f"dashboard_config.py not found at {self.dashboard_config_py}"
        )
        self.assertTrue(
            self.dashboard_config_py.is_file(),
            f"{self.dashboard_config_py} exists but is not a file"
        )

    def test_cache_manager_exists(self):
        """Test 5: Verify cache_manager.py exists."""
        self.assertTrue(
            self.cache_manager_py.exists(),
            f"cache_manager.py not found at {self.cache_manager_py}"
        )
        self.assertTrue(
            self.cache_manager_py.is_file(),
            f"{self.cache_manager_py} exists but is not a file"
        )

    def test_requirements_complete(self):
        """Test 6: Check that requirements.txt contains all necessary packages."""
        self.assertTrue(
            self.requirements_txt.exists(),
            f"requirements.txt not found at {self.requirements_txt}"
        )

        with open(self.requirements_txt, 'r', encoding='utf-8') as f:
            requirements_content = f.read().lower()

        # Define required packages
        required_packages = [
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
        ]

        # Check each required package
        missing_packages = []
        for package in required_packages:
            if package not in requirements_content:
                missing_packages.append(package)

        self.assertEqual(
            len(missing_packages), 0,
            f"requirements.txt is missing the following packages: {', '.join(missing_packages)}"
        )

    def test_no_absolute_local_paths(self):
        """Test 7: Scan app.py and data_loader.py for hardcoded absolute paths."""
        files_to_check = [self.app_py, self.data_loader_py]

        # Patterns that indicate hardcoded local paths
        path_patterns = [
            r'/Users/[^/]+/',  # macOS user paths
            r'/home/[^/]+/',   # Linux user paths
            r'C:\\Users\\',    # Windows user paths
        ]

        issues_found = []

        for file_path in files_to_check:
            if not file_path.exists():
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check each pattern
            for pattern in path_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Filter out comments and docstrings if needed
                    # For now, just report all matches
                    issues_found.append(
                        f"{file_path.name} contains hardcoded path pattern: {pattern}"
                    )

        self.assertEqual(
            len(issues_found), 0,
            f"Hardcoded absolute paths found:\n" + "\n".join(issues_found)
        )

    def test_streamlit_page_config(self):
        """Test 8: Check that app.py calls st.set_page_config."""
        self.assertTrue(self.app_py.exists(), "app.py must exist")

        with open(self.app_py, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn(
            'st.set_page_config',
            content,
            "app.py must call st.set_page_config for proper Streamlit configuration"
        )

        # Verify it's called with proper syntax (basic check)
        # Should be st.set_page_config(...) not just in a comment
        self.assertTrue(
            re.search(r'st\.set_page_config\s*\(', content),
            "st.set_page_config should be called with parentheses"
        )

    def test_gitignore_ready(self):
        """Test 9: Verify .gitignore contains necessary entries."""
        self.assertTrue(
            self.gitignore.exists(),
            f".gitignore not found at {self.gitignore}"
        )

        with open(self.gitignore, 'r', encoding='utf-8') as f:
            gitignore_content = f.read()

        # Define required gitignore entries
        required_entries = [
            'venv/',
            '__pycache__/',
            '.env',
            'cache/'
        ]

        # Check each required entry
        missing_entries = []
        for entry in required_entries:
            if entry not in gitignore_content:
                missing_entries.append(entry)

        self.assertEqual(
            len(missing_entries), 0,
            f".gitignore is missing the following entries: {', '.join(missing_entries)}"
        )

    def test_git_clean(self):
        """Test 10: Run 'git status --porcelain' and verify no uncommitted changes."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Get output and filter out allowed untracked files
            output_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []

            # Filter out acceptable untracked files (like this test file or plans)
            # Untracked files start with '??'
            acceptable_patterns = [
                'plans/',
                'test/test_deployment_ready.py',
                'test_results/'
            ]

            problematic_changes = []
            for line in output_lines:
                if not line:
                    continue

                # Check if it's an acceptable untracked file
                is_acceptable = False
                for pattern in acceptable_patterns:
                    if pattern in line:
                        is_acceptable = True
                        break

                if not is_acceptable:
                    problematic_changes.append(line)

            self.assertEqual(
                len(problematic_changes), 0,
                f"Git working directory has uncommitted changes:\n" +
                "\n".join(problematic_changes) +
                "\n\nPlease commit or stash these changes before deployment."
            )

        except subprocess.TimeoutExpired:
            self.fail("git status command timed out")
        except FileNotFoundError:
            self.fail("git command not found - is git installed?")

    def test_remote_exists(self):
        """Test 11: Run 'git remote -v' and verify origin remote exists with github.com URL."""
        try:
            result = subprocess.run(
                ['git', 'remote', '-v'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10
            )

            output = result.stdout.strip()

            # Check that output contains 'origin'
            self.assertIn(
                'origin',
                output,
                "Git remote 'origin' not found. Please add a remote repository."
            )

            # Check that origin points to github.com
            self.assertIn(
                'github.com',
                output,
                "Git remote 'origin' should point to a github.com URL"
            )

            # Verify both fetch and push are configured
            lines = output.split('\n')
            has_fetch = any('origin' in line and 'fetch' in line for line in lines)
            has_push = any('origin' in line and 'push' in line for line in lines)

            self.assertTrue(
                has_fetch,
                "Git remote 'origin' is not configured for fetch"
            )
            self.assertTrue(
                has_push,
                "Git remote 'origin' is not configured for push"
            )

        except subprocess.TimeoutExpired:
            self.fail("git remote command timed out")
        except FileNotFoundError:
            self.fail("git command not found - is git installed?")


def suite():
    """Create test suite for deployment readiness."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestDeploymentReady))
    return test_suite


if __name__ == '__main__':
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
