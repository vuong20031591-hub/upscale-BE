# AI Image Upscaling Backend

Backend API service cho AI Image Upscaling sử dụng Real-ESRGAN model.

## Tổng quan

Service này cung cấp REST API để upscale ảnh sử dụng:
- **AI-based upscaling**: Real-ESRGAN model (4x scale factor)
- **Standard upscaling**: LANCZOS resampling algorithm

## Yêu cầu hệ thống

- Python 3.11+
- CUDA-compatible GPU (khuyến nghị cho AI upscaling)
- 2GB+ GPU memory (cho FP16 mode)
- 10GB+ disk space (cho model weights)

## Cài đặt

### 1. Cài đặt dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Cấu hình môi trường

Tạo file `.env` trong thư mục `backend/`:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# CORS
CORS_ORIGINS=http://localhost:3000

# Upload Configuration
MAX_FILE_SIZE=10485760  # 10 MiB
ALLOWED_EXTENSIONS=jpg,jpeg,png
ALLOWED_CONTENT_TYPES=image/jpeg,image/png

# Model Configuration
MODEL_NAME=RealESRGAN_x4plus
MODEL_SCALE=4
MODEL_PATH=weights
TILE_SIZE=0
HALF_PRECISION=true

# Output Configuration
SUPPORTED_RESOLUTIONS=2k,4k
DEFAULT_TARGET_RESOLUTION=2k
OUTPUT_FORMAT=png
OUTPUT_QUALITY=95
```

### 3. Chạy server

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server sẽ chạy tại: http://localhost:8000

API documentation: http://localhost:8000/docs

## Testing

### Cài đặt testing dependencies

Testing dependencies đã được include trong `requirements.txt`:
- `pytest`: Test framework
- `pytest-asyncio`: Async test support
- `httpx`: HTTP client cho API testing
- `hypothesis`: Property-based testing library

### Chạy tests

#### Chạy tất cả tests

```bash
cd backend
pytest
```

#### Chạy tests với coverage report

```bash
pytest --cov=app --cov-report=html
```

Coverage report sẽ được tạo trong thư mục `htmlcov/`.

#### Chạy tests theo module

```bash
# Unit tests cho data models
pytest tests/unit/test_models.py

# Unit tests cho ImageProcessor
pytest tests/unit/test_image_processor.py

# Unit tests cho ModelManager
pytest tests/unit/test_model_manager.py

# Integration tests cho API endpoints
pytest tests/integration/test_api.py

# Property-based tests
pytest tests/property/
```

#### Chạy tests với verbose output

```bash
pytest -v
```

#### Chạy specific test

```bash
pytest tests/unit/test_models.py::test_processed_image_to_bytes
```

### Property-Based Testing

Project sử dụng `hypothesis` library cho property-based testing để verify correctness properties across nhiều inputs.

#### Chạy property tests

```bash
# Chạy tất cả property tests
pytest tests/property/

# Chạy với nhiều iterations hơn (default: 100)
pytest tests/property/ --hypothesis-seed=random
```

**Lưu ý**: Property-based tests có thể chạy lâu hơn unit tests vì chúng generate và test nhiều random inputs.

#### Property tests coverage

Property tests verify các correctness properties sau:
- **Property 1**: File validation rejects invalid inputs
- **Property 2**: RGB conversion idempotence
- **Property 3**: ModelManager singleton uniqueness
- **Property 4**: Model loading idempotence
- **Property 5**: AI upscale output type consistency
- **Property 6**: Aspect ratio preservation
- **Property 7**: Resize scale calculation correctness
- **Property 8**: Fit-within downscaling
- **Property 9**: No additional upscaling
- **Property 10**: Response headers format consistency
- **Property 11**: Overall scale factor calculation formula
- **Property 12**: Invalid resolution fallback
- **Property 13**: Standard upscaling fit-within behavior
- **Property 14**: Response format consistency between methods

### Test Structure

```
backend/tests/
├── unit/                    # Unit tests
│   ├── test_models.py       # Data models tests
│   ├── test_image_processor.py
│   ├── test_model_manager.py
│   └── test_config.py
├── integration/             # Integration tests
│   └── test_api.py          # API endpoint tests
├── property/                # Property-based tests
│   ├── test_validation_properties.py
│   ├── test_processing_properties.py
│   └── test_api_properties.py
└── conftest.py              # Shared fixtures
```

### Test Fixtures

Common fixtures được định nghĩa trong `tests/conftest.py`:
- `test_image`: Generate test PIL Image
- `test_upload_file`: Mock FastAPI UploadFile
- `mock_model_manager`: Mock ModelManager để avoid loading real model
- `client`: FastAPI TestClient cho integration tests

### Debugging Failed Tests

#### Xem detailed error output

```bash
pytest -vv --tb=long
```

#### Chạy test với pdb debugger

```bash
pytest --pdb
```

#### Xem print statements trong tests

```bash
pytest -s
```

### Continuous Integration

Tests được chạy tự động trong CI/CD pipeline. Để pass CI:
- Tất cả tests phải pass
- Code coverage >= 80%
- Không có linting errors

## API Endpoints

### Health Checks

- `GET /health` - Basic health check
- `GET /health/ready` - Readiness check (model loaded status)
- `GET /health/config` - Current configuration

### Image Upscaling

- `POST /upscale/ai` - AI-based upscaling (Real-ESRGAN)
- `POST /upscale/standard` - Standard LANCZOS upscaling
- `GET /upscale/resolutions` - List supported resolutions

### Metrics

- `GET /metrics` - Prometheus metrics

## Architecture

### Layered Architecture

```
app/
├── core/              # Core configuration và exceptions
│   ├── config.py      # Environment configuration
│   └── exceptions.py  # Custom exceptions
├── models/            # Data models
│   └── image.py       # Image-related models
├── services/          # Business logic
│   ├── model_manager.py    # AI model management (Singleton)
│   └── image_processor.py  # Image processing workflow
├── routers/           # API endpoints
│   ├── health.py      # Health check endpoints
│   └── upscale.py     # Upscaling endpoints
├── utils/             # Utilities
│   ├── file_handler.py     # File upload handling
│   └── logging_utils.py    # Structured logging
└── main.py            # FastAPI application
```

### Key Components

#### ModelManager (Singleton)
- Lazy loading: Model chỉ load khi có request đầu tiên
- Auto-download: Tự động download model từ GitHub nếu chưa có
- Caching: Model được cache trong memory cho requests tiếp theo
- FP16 support: Half precision mode để tối ưu GPU memory

#### ImageProcessor
- File validation: Size, type, extension checks
- RGB conversion: Tự động convert mọi ảnh sang RGB mode
- Aspect ratio preservation: Giữ nguyên tỷ lệ ảnh khi resize
- Fit-within strategy: Resize để fit trong target resolution

## Performance

### Typical Processing Times

- Model loading (first request): 3-5 seconds
- AI upscaling (2K image): 5-10 seconds
- Standard upscaling (2K image): <1 second

### Memory Usage

- GPU memory (FP16 mode): ~2GB
- CPU memory: ~1GB

## Troubleshooting

### Model download fails

```bash
# Manual download
cd backend/weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

### CUDA out of memory

Giảm `TILE_SIZE` trong `.env`:

```env
TILE_SIZE=256  # Process image in tiles
```

### Tests fail với "Model not found"

Tests sử dụng mock ModelManager. Nếu integration tests fail:

```bash
# Ensure model file exists
ls -lh backend/weights/RealESRGAN_x4plus.pth
```

## Development

### Code Style

- Follow PEP 8 style guide
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Docstrings: Google style

### Adding New Tests

1. **Unit tests**: Test individual functions/methods
2. **Integration tests**: Test API endpoints end-to-end
3. **Property tests**: Test universal properties across inputs

Example unit test:

```python
def test_validate_upload_file_too_large():
    processor = ImageProcessor()
    file_info = UploadFileInfo(
        filename="large.jpg",
        content_type="image/jpeg",
        size=20_000_000,  # 20MB > 10MB limit
        content=b""
    )
    
    with pytest.raises(ValidationError, match="File too large"):
        processor.validate_upload(file_info)
```

Example property test:

```python
from hypothesis import given, strategies as st

@given(
    width=st.integers(min_value=100, max_value=4000),
    height=st.integers(min_value=100, max_value=4000)
)
def test_aspect_ratio_preserved(width, height):
    """Property 6: Aspect Ratio Preservation"""
    image = generate_test_image(width, height)
    result = processor.process_from_image(image, Resolution.K2, use_ai=False)
    
    input_ratio = width / height
    output_ratio = result.final_width / result.final_height
    
    assert abs(input_ratio - output_ratio) < 0.01
```

## License

[Your License Here]

## Contributors

[Your Team/Contributors Here]
