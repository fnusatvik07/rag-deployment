"""Linting tests — runs ruff on the entire codebase."""

import subprocess


def test_ruff_lint():
    """All source files must pass ruff linting rules."""
    result = subprocess.run(
        ["ruff", "check", "app/", "main.py", "tests/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Ruff linting failed:\n{result.stdout}\n{result.stderr}"


def test_ruff_format():
    """All source files must conform to ruff formatting."""
    result = subprocess.run(
        ["ruff", "format", "--check", "app/", "main.py", "tests/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Ruff formatting issues:\n{result.stdout}\n{result.stderr}"
