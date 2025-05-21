# Contributing to LexoRead

First off, thank you for considering contributing to LexoRead! This project exists to help make reading more accessible for individuals with dyslexia and reading impairments, and your help is essential to making this vision a reality.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Process](#development-process)
  - [Setting Up Your Environment](#setting-up-your-environment)
  - [Testing](#testing)
  - [Style Guidelines](#style-guidelines)
- [Project Structure](#project-structure)
- [Communication](#communication)

## Code of Conduct

This project and everyone participating in it are governed by the [LexoRead Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

If you find a bug in the project, please create an issue on GitHub with the following information:

- A clear title and description
- Steps to reproduce the bug
- Expected behavior and actual behavior
- Screenshots if applicable
- Environment details (OS, browser, etc.)

### Suggesting Features

We welcome suggestions for new features! When suggesting a feature, please:

- Provide a clear title and detailed description
- Explain why this feature would be useful to the project
- Consider how the feature would work with existing features
- If possible, outline how the feature might be implemented

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run the tests to ensure everything still works
5. Submit a pull request with a clear description of the changes

For major changes, please open an issue first to discuss what you would like to change.

## Development Process

### Setting Up Your Environment

1. Clone the repository
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Testing

We use pytest for testing. To run the tests:

```bash
pytest
```

Ensure that all tests pass before submitting a pull request.

### Style Guidelines

- Python code should follow PEP 8
- JavaScript code should follow the ESLint configuration
- Use descriptive variable names and add comments for complex logic
- Write docstrings for all functions, classes, and modules
- Update documentation when necessary

## Project Structure

```
lexoread/
│
├── api/               # Backend API code
├── datasets/          # Dataset preparation scripts
├── docs/              # Documentation
├── frontend/          # Frontend application
├── models/            # ML model implementations
├── notebooks/         # Research notebooks
├── scripts/           # Utility scripts
└── tests/             # Test suite
```

Please respect this structure when adding new files or directories.

## Communication

- GitHub Issues: For bug reports, feature requests, and discussions
- Pull Requests: For code contributions
- Community Chat: Join our community chat (link in README)

## Specialized Contributions

### Datasets

If you're contributing to the datasets:

- Follow the structure in the `datasets/` directory
- Document data sources and licenses
- Add preprocessing scripts if necessary
- Update the relevant README.md file

### ML Models

If you're contributing ML models:

- Include training scripts
- Document hyperparameters and architecture
- Provide evaluation metrics
- Consider model size and performance constraints

### Frontend

If you're contributing to the frontend:

- Follow accessibility best practices
- Test on different browsers and devices
- Maintain consistent design language
- Consider dyslexia-friendly UI principles

## Accessibility Focus

Remember that LexoRead is an accessibility-focused project. All contributions should consider the needs of users with dyslexia and reading impairments. This includes:

- Following text readability guidelines
- Supporting screen readers and assistive technologies
- Allowing customization of text display
- Providing keyboard navigation

Thank you for helping to make reading more accessible for everyone!
