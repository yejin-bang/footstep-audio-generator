# Unit Tests

This directory contains unit tests for the Footstep Audio Pipeline.

## Running Tests

### Install Test Dependencies

```bash
# Install pytest and related tools
pip install pytest pytest-cov

# Or uncomment test tools in requirements.txt and run:
# pip install -r requirements.txt
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html
```

### Run Specific Tests

```bash
# Run tests for a specific module
pytest tests/test_config.py

# Run tests for a specific class
pytest tests/test_backends.py::TestMockBackend

# Run a specific test
pytest tests/test_backends.py::TestMockBackend::test_mock_backend_generate_returns_correct_format
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run tests that don't require API
pytest -m "not requires_api"

# Run fast tests only
pytest -m "not slow"
```

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── README.md                # This file
├── pytest.ini               # Pytest configuration (in project root)
├── fixtures/                # Test fixtures (test videos, audio, etc.)
├── test_config.py           # Tests for configuration module
├── test_backends.py         # Tests for audio backends
├── test_logger.py           # Tests for logging module
└── test_*.py                # Additional test modules
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests for pipeline flow
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.requires_gpu` - Tests requiring GPU access
- `@pytest.mark.requires_api` - Tests requiring API credentials

## Writing New Tests

### Example Test

```python
import pytest
from src.my_module import my_function

@pytest.mark.unit
class TestMyFunction:
    def test_basic_functionality(self):
        result = my_function("input")
        assert result == "expected_output"

    def test_error_handling(self):
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Best Practices

1. **Use descriptive test names**: `test_mock_backend_generates_stereo_audio`
2. **One assertion per test** (when possible)
3. **Use fixtures for setup/teardown**
4. **Mark tests appropriately** (`@pytest.mark.unit`, etc.)
5. **Test both happy path and error cases**

## Current Coverage

To generate a coverage report:

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ --cov=src
```

## Future Improvements

- [ ] Add integration tests for full pipeline
- [ ] Add tests for video_validator module
- [ ] Add tests for footstep_detector module
- [ ] Add tests for spatial_audio_processor module
- [ ] Increase test coverage to >80%
- [ ] Add property-based testing with Hypothesis
- [ ] Add performance benchmarks
