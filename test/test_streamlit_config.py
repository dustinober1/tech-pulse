"""
Test suite for Streamlit configuration files.

This module contains unit tests to verify that the Streamlit configuration
is properly set up, including directory structure, configuration files,
and gitignore settings.
"""

import unittest
import os
import tomllib
from pathlib import Path


class TestStreamlitConfig(unittest.TestCase):
    """Test cases for Streamlit configuration setup."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.streamlit_dir = self.project_root / ".streamlit"
        self.config_file = self.streamlit_dir / "config.toml"
        self.secrets_example_file = self.streamlit_dir / "secrets.toml.example"
        self.gitignore_file = self.project_root / ".gitignore"

    def test_streamlit_directory_exists(self):
        """Test that the .streamlit directory exists in the project root."""
        self.assertTrue(
            self.streamlit_dir.exists(),
            f".streamlit directory does not exist at {self.streamlit_dir}"
        )
        self.assertTrue(
            self.streamlit_dir.is_dir(),
            f".streamlit exists but is not a directory at {self.streamlit_dir}"
        )

    def test_config_toml_exists(self):
        """Test that the config.toml file exists in the .streamlit directory."""
        self.assertTrue(
            self.config_file.exists(),
            f"config.toml file does not exist at {self.config_file}"
        )
        self.assertTrue(
            self.config_file.is_file(),
            f"config.toml exists but is not a file at {self.config_file}"
        )

    def test_config_toml_valid_syntax(self):
        """Test that config.toml has valid TOML syntax and can be parsed."""
        self.assertTrue(
            self.config_file.exists(),
            f"config.toml file does not exist at {self.config_file}"
        )

        try:
            with open(self.config_file, "rb") as f:
                config_data = tomllib.load(f)

            # Verify it's a dictionary (valid TOML structure)
            self.assertIsInstance(
                config_data,
                dict,
                "Parsed config.toml should be a dictionary"
            )

            # Optional: Verify expected sections exist
            # This can be expanded based on your config requirements
            self.assertIsNotNone(
                config_data,
                "config.toml should contain valid TOML data"
            )

        except tomllib.TOMLDecodeError as e:
            self.fail(f"config.toml has invalid TOML syntax: {e}")
        except Exception as e:
            self.fail(f"Error reading config.toml: {e}")

    def test_secrets_example_exists(self):
        """Test that the secrets.toml.example file exists in the .streamlit directory."""
        self.assertTrue(
            self.secrets_example_file.exists(),
            f"secrets.toml.example file does not exist at {self.secrets_example_file}"
        )
        self.assertTrue(
            self.secrets_example_file.is_file(),
            f"secrets.toml.example exists but is not a file at {self.secrets_example_file}"
        )

    def test_secrets_toml_in_gitignore(self):
        """Test that secrets.toml pattern is in the .gitignore file."""
        self.assertTrue(
            self.gitignore_file.exists(),
            f".gitignore file does not exist at {self.gitignore_file}"
        )

        try:
            with open(self.gitignore_file, "r", encoding="utf-8") as f:
                gitignore_content = f.read()

            # Check for various possible patterns that would ignore secrets.toml
            patterns_to_check = [
                "secrets.toml",
                ".streamlit/secrets.toml",
                "*.toml" if ".streamlit/secrets.toml" not in gitignore_content else None
            ]

            found = False
            for pattern in patterns_to_check:
                if pattern and pattern in gitignore_content:
                    found = True
                    break

            self.assertTrue(
                found,
                "secrets.toml pattern not found in .gitignore. "
                "Expected to find 'secrets.toml' or '.streamlit/secrets.toml' pattern."
            )

        except Exception as e:
            self.fail(f"Error reading .gitignore: {e}")


if __name__ == "__main__":
    unittest.main()
