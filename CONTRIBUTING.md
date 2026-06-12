# Contributing

Thanks for your interest in contributing to Adaptive Computing.

## Getting Started

1. Fork the repository and create a feature branch.
2. Install dependencies and run tests locally.
3. Open a pull request with a clear description of the change.

## Development Checklist

1. Run tests:

```bash
python -m pytest
```

2. Run lint checks:

```bash
ruff check --select=E9,F63,F7,F82 --target-version=py311 .
ruff check --target-version=py311 .
```

## Pull Request Guidelines

1. Keep pull requests focused and small when possible.
2. Add or update tests for behavior changes.
3. Update documentation for user-facing changes.
4. Ensure CI passes before requesting review.
