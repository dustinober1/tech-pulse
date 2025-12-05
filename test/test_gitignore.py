import unittest
import os
from pathlib import Path


class TestGitignore(unittest.TestCase):
    """Test suite for validating .gitignore configuration."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all test methods."""
        # Get the project root directory (parent of test directory)
        cls.project_root = Path(__file__).parent.parent
        cls.gitignore_path = cls.project_root / '.gitignore'

    def _read_gitignore_patterns(self):
        """
        Helper method to read and return all patterns from .gitignore file.
        Returns a set of non-empty, non-comment lines.
        """
        if not self.gitignore_path.exists():
            return set()

        patterns = set()
        with open(self.gitignore_path, 'r') as f:
            for line in f:
                # Strip whitespace
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    patterns.add(line)
        return patterns

    def test_gitignore_exists(self):
        """Test that .gitignore file exists in project root."""
        self.assertTrue(
            self.gitignore_path.exists(),
            f".gitignore file not found at {self.gitignore_path}"
        )
        self.assertTrue(
            self.gitignore_path.is_file(),
            f".gitignore exists but is not a file at {self.gitignore_path}"
        )

    def test_venv_excluded(self):
        """Test that venv/ pattern is present in .gitignore."""
        patterns = self._read_gitignore_patterns()
        # Check for various common venv patterns
        venv_patterns = {'venv/', 'venv', 'venv/*', 'env/', 'env'}
        found = any(pattern in patterns for pattern in venv_patterns)
        self.assertTrue(
            found,
            f"Virtual environment pattern (venv/ or similar) not found in .gitignore. "
            f"Found patterns: {patterns}"
        )

    def test_pycache_excluded(self):
        """Test that __pycache__/ pattern is present in .gitignore."""
        patterns = self._read_gitignore_patterns()
        # Check for various common pycache patterns
        pycache_patterns = {'__pycache__/', '__pycache__', '__pycache__/*', '*/__pycache__/'}
        found = any(pattern in patterns for pattern in pycache_patterns)
        self.assertTrue(
            found,
            f"Python cache pattern (__pycache__/ or similar) not found in .gitignore. "
            f"Found patterns: {patterns}"
        )

    def test_env_file_excluded(self):
        """Test that .env pattern is present in .gitignore."""
        patterns = self._read_gitignore_patterns()
        # Check for various common .env patterns
        env_patterns = {'.env', '.env.*', '*.env', '.env.local', '.env.production'}
        found = any(pattern in patterns for pattern in env_patterns)
        self.assertTrue(
            found,
            f"Environment file pattern (.env or similar) not found in .gitignore. "
            f"Found patterns: {patterns}"
        )

    def test_ds_store_excluded(self):
        """Test that .DS_Store pattern is present in .gitignore."""
        patterns = self._read_gitignore_patterns()
        # Check for various common .DS_Store patterns
        ds_store_patterns = {'.DS_Store', '**/.DS_Store', '*/.DS_Store', '*.DS_Store'}
        found = any(pattern in patterns for pattern in ds_store_patterns)
        self.assertTrue(
            found,
            f"macOS .DS_Store pattern not found in .gitignore. "
            f"Found patterns: {patterns}"
        )

    def test_cache_directory_excluded(self):
        """Test that cache/ pattern is present in .gitignore."""
        patterns = self._read_gitignore_patterns()
        # Check for various common cache patterns
        cache_patterns = {'cache/', 'cache', 'cache/*', '.cache/', '.cache', '*cache/'}
        found = any(pattern in patterns for pattern in cache_patterns)
        self.assertTrue(
            found,
            f"Cache directory pattern (cache/ or similar) not found in .gitignore. "
            f"Found patterns: {patterns}"
        )


if __name__ == '__main__':
    unittest.main()
