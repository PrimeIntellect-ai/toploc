# Contributing to Toploc

Thank you for your interest in contributing to Toploc!
We welcome all contributions, including:
- Bug reports
- Feature requests
- Code contributions
- Documentation

# Reporting Bugs

If you encounter a bug, please report it by opening an issue on GitHub.
To help us fix the bug, please include:
- A clear and concise description of the bug
- The version of Toploc you are using
- Any error messages or stack traces
- Steps to reproduce the bug

# Feature Requests

If you have a feature request, please open an issue on GitHub.
To help us understand your request, please include:
- A clear and concise description of the feature
- Any examples of how the feature should work

# Code Contributions

If you'd like to contribute code, please open a pull request on GitHub.
To help us understand your contribution, please include:
- A clear and concise description of the changes you've made
- Any examples of how to test the changes

# Developing

## Development Setup

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

2. Setup virtual environment:
This can take awhile because of the C extensions.
```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --dev
```

## Running Tests

Run the test suite:
```bash
uv run pytest tests
```

Run single file:
```bash
uv run pytest tests/test_utils.py
```

Run single test:
```bash
uv run pytest tests/test_utils.py::test_get_fp32_parts
```

Run coverage:
```bash
uv run pytest --cov=toploc --cov-report=term-missing --cov-report=html
```

## Code Quality

Install pre-commit hooks:
```bash
uv run pre-commit install
```

Run linting and formatting on all files:
```bash
pre-commit run --all-files
```
