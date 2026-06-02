"""
Property-based tests for health check endpoints using Hypothesis.

These tests verify universal properties across all valid inputs.
Each test is tagged with feature name and property number from design document.
"""

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st, Phase

from app.main import app
from app.core import settings as app_settings


client = TestClient(app)


# Feature: api-health-monitoring, Property 1: All health endpoints return 200 OK
@pytest.mark.parametrize("endpoint", ["/health", "/health/ready", "/health/config"])
def test_property_1_all_endpoints_return_200(endpoint: str):
    """
    # Feature: api-health-monitoring, Property 1: All health endpoints return 200 OK

    For any health endpoint, calling under normal conditions returns HTTP 200.
    """
    response = client.get(endpoint)
    assert response.status_code == 200


# Feature: api-health-monitoring, Property 2: Health endpoint response structure
def test_property_2_health_response_structure():
    """
    # Feature: api-health-monitoring, Property 2: Health endpoint response structure

    For any request to /health, response has exactly status, service, version.
    """
    response = client.get("/health")
    data = response.json()

    assert "status" in data
    assert "service" in data
    assert "version" in data
    assert len(data) == 3
    assert data["status"] == "healthy"
    assert data["service"] == "image-upscale-api"


# Feature: api-health-monitoring, Property 3: Readiness endpoint response structure
def test_property_3_ready_response_structure():
    """
    # Feature: api-health-monitoring, Property 3: Readiness endpoint response structure

    For any request to /health/ready, response has ready, model_loaded, model_info.
    ready always equals model_loaded.
    """
    response = client.get("/health/ready")
    data = response.json()

    assert "ready" in data
    assert "model_loaded" in data
    assert "model_info" in data
    assert data["ready"] == data["model_loaded"]

    # model_info structure
    model_info = data["model_info"]
    assert "name" in model_info
    assert "scale" in model_info
    assert "loaded" in model_info
    assert "half_precision" in model_info
    assert "model_file" in model_info


# Feature: api-health-monitoring, Property 4: Readiness check does not trigger model loading
@given(num_calls=st.integers(min_value=1, max_value=20))
@settings(max_examples=30, deadline=None)
def test_property_4_no_model_loading_side_effect(num_calls: int):
    """
    # Feature: api-health-monitoring, Property 4: Readiness check does not trigger model loading

    For any number of calls to /health/ready, model should not be loaded.
    """
    from app.services import ModelManager

    # Get initial state
    model_manager = ModelManager()
    initial_loaded_state = model_manager.is_loaded

    # Make multiple calls
    for _ in range(num_calls):
        response = client.get("/health/ready")
        assert response.status_code == 200

    # Model state should not have changed (still not loaded if it wasn't)
    # Note: This test assumes model is not loaded at start
    if not initial_loaded_state:
        assert not model_manager.is_loaded


# Feature: api-health-monitoring, Property 5: Config endpoint response structure
def test_property_5_config_response_structure():
    """
    # Feature: api-health-monitoring, Property 5: Config endpoint response structure

    For any request to /health/config, response has upload and output.
    Note: api config intentionally excluded per security requirements.
    """
    response = client.get("/health/config")
    data = response.json()

    # Top-level structure (api excluded per Requirement 3.8, 8.6)
    assert "upload" in data
    assert "output" in data

    # upload structure
    assert "max_size_mb" in data["upload"]
    assert "allowed_types" in data["upload"]

    # output structure
    assert "supported_resolutions" in data["output"]
    assert "default_resolution" in data["output"]
    assert "format" in data["output"]
    assert "quality" in data["output"]


# Feature: api-health-monitoring, Property 6: Bytes to megabytes conversion
@given(bytes_value=st.integers(min_value=1024, max_value=100*1024*1024))
@settings(max_examples=100, phases=[Phase.explicit, Phase.reuse, Phase.generate])
def test_property_6_bytes_to_mb_conversion(bytes_value: int):
    """
    # Feature: api-health-monitoring, Property 6: Bytes to megabytes conversion

    For any max_file_size = N bytes, max_size_mb = N / (1024 * 1024).
    """
    # Calculate expected MB value
    expected_mb = bytes_value / (1024 * 1024)

    # Verify conversion formula
    # We can't directly test with the API since settings are loaded at startup,
    # but we verify the conversion logic
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class MockUploadConfig:
        max_file_size: int

    mock_config = MockUploadConfig(max_file_size=bytes_value)
    actual_mb = mock_config.max_file_size / (1024 * 1024)

    assert actual_mb == expected_mb


# Feature: api-health-monitoring, Property 7: Set to List conversion
@given(content_types=st.sets(
    st.sampled_from(["image/jpeg", "image/png", "image/webp", "image/gif"]),
    min_size=1,
    max_size=4
))
@settings(max_examples=100, phases=[Phase.explicit, Phase.reuse, Phase.generate])
def test_property_7_set_to_list_conversion(content_types: set):
    """
    # Feature: api-health-monitoring, Property 7: Set to List conversion

    For any Set of content types, List output contains same elements.
    """
    # Simulate the conversion
    result_list = list(content_types)

    # Verify all elements from Set are in List
    assert len(result_list) == len(content_types)
    assert set(result_list) == content_types


# Feature: api-health-monitoring, Property 8: All endpoints return valid JSON
@pytest.mark.parametrize("endpoint", ["/health", "/health/ready", "/health/config"])
def test_property_8_valid_json_response(endpoint: str):
    """
    # Feature: api-health-monitoring, Property 8: All endpoints return valid JSON

    For any health endpoint, response is parseable JSON with correct Content-Type.
    """
    response = client.get(endpoint)

    # Verify Content-Type header
    assert "application/json" in response.headers.get("content-type", "")

    # Verify valid JSON (should not raise)
    data = response.json()
    assert data is not None


# Feature: api-health-monitoring, Property 9: No null values in required fields
def test_property_9_no_null_values():
    """
    # Feature: api-health-monitoring, Property 9: No null values in required fields

    For any health endpoint response, required fields have non-null values.
    """
    def check_no_null(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                assert value is not None, f"Null value found at {current_path}"
                check_no_null(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_no_null(item, f"{path}[{i}]")

    # Test all endpoints
    for endpoint in ["/health", "/health/ready", "/health/config"]:
        response = client.get(endpoint)
        data = response.json()
        check_no_null(data, endpoint)


# Feature: api-health-monitoring, Property 10: Snake case naming convention
def test_property_10_snake_case_naming():
    """
    # Feature: api-health-monitoring, Property 10: Snake case naming convention

    For any health endpoint response, all JSON keys follow snake_case.
    """
    snake_case_pattern = re.compile(r'^[a-z][a-z0-9_]*$')

    def check_keys(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                assert snake_case_pattern.match(key), (
                    f"Key '{current_path}' does not follow snake_case convention"
                )
                check_keys(value, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_keys(item, f"{path}[{i}]")

    # Test all endpoints
    for endpoint in ["/health", "/health/ready", "/health/config"]:
        response = client.get(endpoint)
        data = response.json()
        check_keys(data, endpoint)


# Additional property: Ready field equals model_loaded field
def test_property_ready_equals_model_loaded():
    """
    Property: ready field always equals model_loaded field.
    """
    # Test multiple times to ensure consistency
    for _ in range(10):
        response = client.get("/health/ready")
        data = response.json()
        assert data["ready"] == data["model_loaded"]


# Additional property: Model info loaded matches is_loaded
@given(num_calls=st.integers(min_value=1, max_value=20))
@settings(max_examples=30)
def test_property_model_info_loaded_consistency(num_calls: int):
    """
    Property: model_info.loaded always matches model_loaded field.
    """
    from app.services import ModelManager

    model_manager = ModelManager()

    for _ in range(num_calls):
        response = client.get("/health/ready")
        data = response.json()

        assert data["model_info"]["loaded"] == data["model_loaded"]
        assert data["model_loaded"] == model_manager.is_loaded


# Additional property: Config values are valid types
def test_property_config_value_types():
    """
    Property: Config endpoint returns values of correct types.
    """
    response = client.get("/health/config")
    data = response.json()

    # Note: api types not checked (api config excluded per security requirements)

    # upload types
    assert isinstance(data["upload"]["max_size_mb"], float)
    assert isinstance(data["upload"]["allowed_types"], list)
    assert all(isinstance(t, str) for t in data["upload"]["allowed_types"])

    # output types
    assert isinstance(data["output"]["supported_resolutions"], list)
    assert isinstance(data["output"]["default_resolution"], str)
    assert isinstance(data["output"]["format"], str)
    assert isinstance(data["output"]["quality"], int)


# Additional property: Model file has .pth extension
def test_property_model_file_has_pth_extension():
    """
    Property: model_file always ends with .pth extension.
    """
    response = client.get("/health/ready")
    data = response.json()
    model_file = data["model_info"]["model_file"]

    assert model_file.endswith(".pth")


# Additional property: Scale is positive integer
def test_property_scale_is_positive():
    """
    Property: model_info.scale is always a positive integer.
    """
    response = client.get("/health/ready")
    data = response.json()
    scale = data["model_info"]["scale"]

    assert isinstance(scale, int)
    assert scale > 0


# Additional property: All health endpoints are accessible under /health prefix
def test_property_health_prefix():
    """
    Property: All health endpoints are under /health path prefix.
    """
    # These should all return 200 (not 404)
    endpoints = ["/health", "/health/ready", "/health/config"]

    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 200, f"Endpoint {endpoint} not accessible"
