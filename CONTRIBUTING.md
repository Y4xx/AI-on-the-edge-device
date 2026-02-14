# Contributing to Digit Recognition CNN Model

First off, thank you for considering contributing to this project! It's people like you that make this project better.

## Code of Conduct

This project and everyone participating in it is governed by common sense and mutual respect. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, error messages)
- **Describe the behavior you observed** and what you expected
- **Include environment details** (OS, Python version, dependency versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List any similar features** in other projects if applicable

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the coding guidelines below
3. **Add tests** if you've added code that should be tested
4. **Update documentation** if you've changed functionality
5. **Ensure tests pass** by running `python run_tests.py`
6. **Make sure your code follows** the project's coding style
7. **Write a clear commit message** describing your changes
8. **Submit your pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/AI-on-the-edge-device.git
cd AI-on-the-edge-device

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Run tests
python run_tests.py
```

## Coding Guidelines

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Keep functions focused and single-purpose
- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)

### Documentation
- Add docstrings to all functions, classes, and modules
- Use Google-style docstrings format
- Keep docstrings clear and concise
- Update README.md and USAGE.md for user-facing changes

### Example Docstring
```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Short description of what the function does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg2 is negative
    """
    pass
```

### Type Hints
- Use type hints for function arguments and return values
- Import types from `typing` module when needed
```python
from typing import List, Dict, Optional, Tuple

def process_data(data: List[np.ndarray], config: Dict[str, Any]) -> Optional[np.ndarray]:
    pass
```

### Error Handling
- Use specific exceptions rather than generic ones
- Add meaningful error messages
- Log errors appropriately
```python
import logging

logger = logging.getLogger(__name__)

try:
    result = process_data(data)
except FileNotFoundError as e:
    logger.error(f"Data file not found: {e}")
    raise
```

### Testing
- Write unit tests for new functionality
- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Aim for good test coverage

```python
import unittest

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_specific_behavior(self):
        """Test a specific behavior."""
        result = my_function("input")
        self.assertEqual(result, expected_value)
```

### Commit Messages
- Use clear and descriptive commit messages
- Start with a verb in present tense (Add, Fix, Update, etc.)
- Keep the first line under 72 characters
- Add detailed description if needed

```
Add batch prediction support to inference module

- Implement batch_predict method in TFLiteInference class
- Add error handling for missing files
- Update documentation with usage examples
```

## Project Structure

When adding new files, follow this structure:

```
src/
├── data/           # Data loading and preprocessing
├── models/         # Model architecture and training
└── utils/          # Utility functions

tests/              # Unit tests
```

## Areas for Contribution

We welcome contributions in these areas:

### High Priority
- [ ] Add more model architectures (ResNet, MobileNet, etc.)
- [ ] Implement model pruning and optimization
- [ ] Add data pipeline optimization
- [ ] Create web interface for inference
- [ ] Add more comprehensive tests

### Medium Priority
- [ ] Add Docker support
- [ ] Implement distributed training
- [ ] Add real-time camera inference
- [ ] Create model visualization tools
- [ ] Add more data augmentation techniques

### Documentation
- [ ] Add more usage examples
- [ ] Create video tutorials
- [ ] Translate documentation to other languages
- [ ] Add architecture diagrams
- [ ] Improve API documentation

## Recognition

Contributors will be recognized in:
- README.md Contributors section
- Release notes
- Git commit history

## Questions?

Feel free to open an issue with your question or contact the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing! 🎉
