# Testing Infrastructure - AI Image Upscaling

## Overview

Testing infrastructure cho AI Image Upscaling feature, hỗ trợ cả **unit tests** và **property-based tests** sử dụng Hypothesis.

## Structure

```
tests/
├── conftest.py              # Shared fixtures và configuration
├── test_utils.py            # Test utilities và Hypothesis strategies
├── test_fixtures.py         # Tests để verify fixtures hoạt động
├── core/                    # Tests cho core modules (config, exceptions)
├── routers/                 # Tests cho API endpoints
└── services/                # Tests cho business logic services
```

## Test Fixtures

### Image Generation Fixtures

#### `create_test_image(width, height, mode, color)`
Factory fixture để tạo PIL Images với parameters tùy chỉnh.

```python
def test_example(create_test_image):
    image = create_test_image(100, 100, 'RGB')
    assert image.size == (100, 100)
```

#### `create_test_image_bytes(width, height, mode, format, color)`
Factory fixture để tạo image bytes cho upload simulation.

```python
def test_example(create_test_image_bytes):
    image_bytes = create_test_image_bytes(100, 100, 'RGB', 'PNG')
    assert len(image_bytes) > 0
```

#### Pre-configured Image Fixtures
- `sample_image_small`: 100x100 RGB
- `sample_image_medium`: 1920x1080 RGB
- `sample_image_large`: 3840x2160 RGB (4K)
- `sample_image_rgba`: 100x100 RGBA
- `sample_image_grayscale`: 100x100 L (grayscale)
- `sample_image_cmyk`: 100x100 CMYK

### UploadFileInfo Fixtures

#### `create_upload_file_info(filename, content_type, width, height, mode, format)`
Factory fixture để tạo UploadFileInfo objects.

```python
def test_example(create_upload_file_info):
    file_info = create_upload_file_info(
        filename='test.jpg',
        content_type='image/jpeg',
        width=1920,
        height=1080
    )
    assert file_info.size > 0
```

#### `valid_upload_file_info`
Pre-configured valid UploadFileInfo (100x100 PNG).

### ModelManager Mock Fixtures

#### `mock_model_manager`
Mock ModelManager để avoid loading real AI model trong tests.

```python
def test_example(mock_model_manager, sample_image_small):
    # Upscale without loading real model
    result = mock_model_manager.upscale(sample_image_small, outscale=4)
    assert result.size == (400, 400)
```

#### `mock_model_manager_not_loaded`
Mock ModelManager trong not-loaded state (để test lazy loading).

#### `mock_model_download_failure`
Mock ModelManager với download failure (để test error handling).

### File Size Fixtures

#### `create_file_with_size(size_bytes)`
Factory fixture để tạo files với specific byte sizes.

```python
def test_example(create_file_with_size):
    file_10mb = create_file_with_size(10 * 1024 * 1024)
    assert len(file_10mb) == 10485760
```

#### Pre-configured Size Fixtures
- `file_exactly_10mb`: Exactly 10 MiB (10,485,760 bytes)
- `file_over_10mb`: 10 MiB + 1 byte
- `file_under_10mb`: 5 MiB

### Other Fixtures

#### `client`
FastAPI TestClient để test API endpoints.

```python
def test_example(client):
    response = client.get("/health")
    assert response.status_code == 200
```

#### `reset_model_manager_singleton` (autouse)
Automatically reset ModelManager singleton giữa các tests.

## Test Utilities

### Image Generation

```python
from tests.test_utils import generate_test_image, generate_test_image_bytes

# Generate PIL Image
image = generate_test_image(width=100, height=100, mode='RGB')

# Generate image bytes
image_bytes = generate_test_image_bytes(width=100, height=100, format='PNG')
```

### Aspect Ratio Utilities

```python
from tests.test_utils import (
    calculate_aspect_ratio,
    verify_aspect_ratio_preserved,
    aspect_ratios_equal
)

# Calculate aspect ratio
ratio = calculate_aspect_ratio(1920, 1080)  # 1.777...

# Verify aspect ratio preserved
is_preserved = verify_aspect_ratio_preserved(
    original_width=100,
    original_height=100,
    final_width=200,
    final_height=200
)  # True

# Compare aspect ratios with tolerance
are_equal = aspect_ratios_equal(1.777, 1.778, tolerance=0.01)  # True
```

### Scale Factor Utilities

```python
from tests.test_utils import (
    calculate_overall_scale_factor,
    calculate_resize_scale
)

# Calculate overall scale factor (for response header)
overall_scale = calculate_overall_scale_factor(
    original_width=100,
    original_height=100,
    final_width=400,
    final_height=400
)  # 4.0

# Calculate resize scale (for fit-within operation)
resize_scale = calculate_resize_scale(
    current_width=1000,
    current_height=1000,
    target_width=2560,
    target_height=1440
)  # 1.44
```

### Assertion Helpers

```python
from tests.test_utils import (
    assert_valid_png_bytes,
    assert_valid_image_dimensions,
    assert_dimensions_within_target
)

# Assert valid PNG bytes
assert_valid_png_bytes(image_bytes)

# Assert valid dimensions
assert_valid_image_dimensions(width=100, height=100)

# Assert dimensions within target
assert_dimensions_within_target(
    width=2000,
    height=1000,
    target_width=2560,
    target_height=1440
)
```

## Hypothesis Strategies

### Basic Strategies

```python
from tests.test_utils import (
    image_width_strategy,
    image_height_strategy,
    image_mode_strategy,
    resolution_strategy,
    file_size_strategy
)

from hypothesis import given

@given(
    width=image_width_strategy,
    height=image_height_strategy,
    mode=image_mode_strategy
)
def test_example(width, height, mode):
    image = generate_test_image(width, height, mode)
    assert image.size == (width, height)
```

### Composite Strategies

```python
from tests.test_utils import (
    image_dimensions_strategy,
    small_image_dimensions_strategy,
    large_image_dimensions_strategy,
    test_image_strategy,
    upload_file_info_strategy
)

from hypothesis import given

@given(dimensions=image_dimensions_strategy())
def test_example(dimensions):
    width, height = dimensions
    # Test with random dimensions
    pass

@given(image=test_image_strategy())
def test_example(image):
    # Test with random PIL Image
    assert isinstance(image, Image.Image)
```

### Validation Strategies

```python
from tests.test_utils import (
    valid_content_type_strategy,
    invalid_content_type_strategy,
    valid_extension_strategy,
    invalid_extension_strategy,
    valid_file_size_strategy,
    invalid_file_size_strategy
)

@given(content_type=invalid_content_type_strategy)
def test_rejects_invalid_content_type(content_type):
    # Test validation rejects invalid content types
    pass
```

## Running Tests

### Run All Tests

```bash
# Run all tests
pytest backend/tests/

# Run with verbose output
pytest backend/tests/ -v

# Run with coverage
pytest backend/tests/ --cov=app --cov-report=html
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest backend/tests/ -m "not property"

# Run only property-based tests
pytest backend/tests/ -m property

# Run only integration tests
pytest backend/tests/ -m integration

# Run only slow tests
pytest backend/tests/ -m slow
```

### Run Specific Test Files

```bash
# Run specific test file
pytest backend/tests/test_fixtures.py -v

# Run specific test class
pytest backend/tests/test_fixtures.py::TestImageGenerationFixtures -v

# Run specific test function
pytest backend/tests/test_fixtures.py::TestImageGenerationFixtures::test_create_test_image_fixture -v
```

### Property-Based Testing Options

```bash
# Run with more examples (default: 100)
pytest backend/tests/ --hypothesis-show-statistics

# Run with specific seed for reproducibility
pytest backend/tests/ --hypothesis-seed=12345

# Show failing examples
pytest backend/tests/ --hypothesis-verbosity=verbose
```

## Writing Tests

### Unit Test Example

```python
import pytest
from PIL import Image

class TestImageProcessor:
    """Unit tests for ImageProcessor."""
    
    def test_upscale_small_image(self, mock_model_manager, sample_image_small):
        """Test upscaling a small image."""
        from app.services.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        result = processor._upscale_ai(sample_image_small)
        
        assert isinstance(result, Image.Image)
        assert result.size == (400, 400)  # 100 * 4
```

### Property-Based Test Example

```python
from hypothesis import given, settings
from tests.test_utils import image_dimensions_strategy

class TestAspectRatioPreservation:
    """Property-based tests for aspect ratio preservation."""
    
    @settings(max_examples=100)
    @given(dimensions=image_dimensions_strategy())
    def test_aspect_ratio_preserved(self, dimensions):
        """
        Feature: ai-image-upscaling
        Property 6: Aspect Ratio Preservation
        
        For any valid image dimensions, after upscaling,
        the aspect ratio should be preserved within tolerance 0.01.
        """
        width, height = dimensions
        
        # Generate test image
        image = generate_test_image(width, height, 'RGB')
        
        # Process through system
        # ... (implementation)
        
        # Verify aspect ratio preserved
        assert verify_aspect_ratio_preserved(
            original_width=width,
            original_height=height,
            final_width=final_width,
            final_height=final_height,
            tolerance=0.01
        )
```

## Test Markers

Tests có thể được mark với các markers sau:

- `@pytest.mark.slow`: Tests chạy chậm (> 1 second)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.property`: Property-based tests

```python
@pytest.mark.slow
def test_large_image_processing():
    """Test processing very large images."""
    pass

@pytest.mark.integration
def test_complete_upscale_flow():
    """Test complete upscale workflow end-to-end."""
    pass

@pytest.mark.property
@given(width=image_width_strategy)
def test_property_example(width):
    """Property-based test example."""
    pass
```

## Configuration

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts =
    -v
    --tb=short
    --strict-markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    property: marks tests as property-based tests
```

### Hypothesis Configuration

Hypothesis được cấu hình với:
- Default max_examples: 100
- Database: `.hypothesis/examples/` (để cache failing examples)
- Profile: default

## Best Practices

### 1. Use Fixtures Over Setup/Teardown

```python
# ❌ Bad
class TestExample:
    def setup_method(self):
        self.image = Image.new('RGB', (100, 100))
    
    def test_something(self):
        assert self.image.size == (100, 100)

# ✅ Good
def test_something(sample_image_small):
    assert sample_image_small.size == (100, 100)
```

### 2. Mock External Dependencies

```python
# ✅ Good - Mock AI model để avoid loading real model
def test_upscale(mock_model_manager, sample_image_small):
    result = mock_model_manager.upscale(sample_image_small)
    assert result.size == (400, 400)
```

### 3. Use Property-Based Tests for Universal Properties

```python
# ✅ Good - Test universal property across all inputs
@given(width=image_width_strategy, height=image_height_strategy)
def test_aspect_ratio_always_preserved(width, height):
    # Test property holds for all valid dimensions
    pass
```

### 4. Use Descriptive Test Names

```python
# ❌ Bad
def test_1():
    pass

# ✅ Good
def test_upscale_preserves_aspect_ratio_for_small_images():
    pass
```

### 5. Tag Property Tests with Feature and Property Number

```python
@given(...)
def test_property_6_aspect_ratio_preservation(...):
    """
    Feature: ai-image-upscaling
    Property 6: Aspect Ratio Preservation
    
    For any valid image, aspect ratio should be preserved.
    """
    pass
```

## Troubleshooting

### Hypothesis Examples Not Reproducing

```bash
# Run with specific seed
pytest --hypothesis-seed=12345
```

### Tests Running Slow

```bash
# Skip slow tests
pytest -m "not slow"

# Reduce hypothesis examples
pytest --hypothesis-max-examples=10
```

### Fixtures Not Found

Ensure `conftest.py` is in the correct location and pytest can discover it:

```bash
pytest --fixtures  # List all available fixtures
```

### Import Errors

Ensure backend directory is in PYTHONPATH:

```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"
pytest backend/tests/
```

## Coverage

Generate coverage report:

```bash
# Run tests with coverage
pytest backend/tests/ --cov=app --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

Target coverage: **>80%** for all modules.

## CI/CD Integration

Tests được chạy automatically trong CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    cd backend
    pytest tests/ -v --cov=app --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./backend/coverage.xml
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [PIL/Pillow Documentation](https://pillow.readthedocs.io/)
