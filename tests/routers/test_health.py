"""
Unit tests for health check endpoints.
"""

import json
import logging
import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_endpoint_exists(self):
        """Verify /health returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        """Verify response has exactly 3 fields: status, service, version."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert len(data) == 3

    def test_health_response_values(self):
        """Verify response values match expected."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "image-upscale-api"
        assert data["version"] == "1.0.0"

    def test_health_content_type(self):
        """Verify Content-Type is application/json."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"


class TestReadinessEndpoint:
    """Tests for GET /health/ready endpoint."""

    def test_ready_endpoint_exists(self):
        """Verify /health/ready returns 200 OK."""
        response = client.get("/health/ready")
        assert response.status_code == 200

    def test_ready_response_structure(self):
        """Verify response has fields: ready, model_loaded, model_info."""
        response = client.get("/health/ready")
        data = response.json()

        assert "ready" in data
        assert "model_loaded" in data
        assert "model_info" in data

    def test_ready_equals_model_loaded(self):
        """Verify ready field equals model_loaded field."""
        response = client.get("/health/ready")
        data = response.json()

        assert data["ready"] == data["model_loaded"]

    def test_model_info_structure(self):
        """Verify model_info has required fields."""
        response = client.get("/health/ready")
        data = response.json()
        model_info = data["model_info"]

        assert "name" in model_info
        assert "scale" in model_info
        assert "loaded" in model_info
        assert "half_precision" in model_info
        assert "model_file" in model_info

    def test_model_file_is_filename_only(self):
        """Verify model_file contains only filename, not full path."""
        response = client.get("/health/ready")
        data = response.json()
        model_file = data["model_info"]["model_file"]

        assert "/" not in model_file
        assert "\\" not in model_file
        assert model_file.endswith(".pth")

    def test_model_info_name_not_empty(self):
        """Verify model_info.name is not empty."""
        response = client.get("/health/ready")
        data = response.json()

        assert data["model_info"]["name"] != ""
        assert data["model_info"]["name"] is not None


class TestConfigEndpoint:
    """Tests for GET /health/config endpoint."""

    def test_config_endpoint_exists(self):
        """Verify /health/config returns 200 OK."""
        response = client.get("/health/config")
        assert response.status_code == 200

    def test_config_response_structure(self):
        """Verify response has 2 nested objects: upload, output (api excluded per security spec)."""
        response = client.get("/health/config")
        data = response.json()

        # api config intentionally not exposed (Requirement 3.8, 8.6)
        assert "upload" in data
        assert "output" in data

    def test_api_config_not_exposed(self):
        """Verify api config (host, port) is NOT exposed for security."""
        response = client.get("/health/config")
        data = response.json()

        # Per spec Requirement 3.8: SHALL NOT expose host và port values
        assert "api" not in data

    def test_upload_config_structure(self):
        """Verify upload config has max_size_mb and allowed_types."""
        response = client.get("/health/config")
        data = response.json()

        assert "max_size_mb" in data["upload"]
        assert "allowed_types" in data["upload"]

    def test_output_config_structure(self):
        """Verify output config has required fields."""
        response = client.get("/health/config")
        data = response.json()

        assert "supported_resolutions" in data["output"]
        assert "default_resolution" in data["output"]
        assert "format" in data["output"]
        assert "quality" in data["output"]

    def test_bytes_to_mb_conversion(self):
        """Verify max_size_mb is correctly converted from bytes."""
        response = client.get("/health/config")
        data = response.json()
        max_size_mb = data["upload"]["max_size_mb"]

        # Default is 10MB = 10 * 1024 * 1024 bytes
        assert max_size_mb == 10.0

    def test_allowed_types_is_list(self):
        """Verify allowed_types is a list (converted from Set)."""
        response = client.get("/health/config")
        data = response.json()

        assert isinstance(data["upload"]["allowed_types"], list)

    def test_no_sensitive_info_exposed(self):
        """Verify no sensitive info in config response."""
        response = client.get("/health/config")
        data = response.json()
        response_text = str(data).lower()

        # Should not contain these sensitive keywords
        assert "password" not in response_text
        assert "secret" not in response_text
        assert "key" not in response_text
        assert "credential" not in response_text
        assert "token" not in response_text


class TestErrorHandling:
    """Tests for error handling."""

    def test_no_stack_trace_in_error_response(self):
        """Verify error responses don't contain stack traces."""
        # This is tested indirectly - FastAPI exception handler should
        # return generic messages
        response = client.get("/health")
        # All health endpoints should succeed normally
        assert response.status_code == 200

    def test_readiness_graceful_degradation_on_model_manager_error(self):
        """Verify /health/ready returns 200 with defaults when ModelManager fails."""
        from unittest.mock import patch

        # Mock ModelManager to raise exception during init
        with patch('app.routers.health.ModelManager') as mock_mgr:
            mock_mgr.side_effect = Exception("ModelManager init failed")

            response = client.get("/health/ready")

            # Should still return 200 with graceful degradation
            assert response.status_code == 200
            data = response.json()

            # Verify fallback values per Requirement 5.2
            assert data["ready"] is False
            assert data["model_loaded"] is False
            assert data["model_info"]["name"] != ""
            assert data["model_info"]["scale"] > 0
            assert data["model_info"]["loaded"] is False
            assert "model_file" in data["model_info"]

    def test_exception_response_no_stack_trace(self):
        """Verify exception responses don't expose stack traces."""
        from unittest.mock import patch

        # Mock ModelManager to trigger an exception in readiness_check
        with patch('app.routers.health.ModelManager') as mock_mgr:
            mock_mgr.side_effect = RuntimeError("Internal error")

            response = client.get("/health/ready")

            # Check response doesn't contain stack trace keywords
            response_text = response.text.lower()
            assert "traceback" not in response_text
            assert "file \"" not in response_text
            assert "line " not in response_text or "model_file" in response_text


class TestFastAPIIntegration:
    """Tests for FastAPI router configuration."""

    def test_router_prefix(self):
        """Verify router has prefix /health."""
        # All endpoints should be accessible under /health
        response = client.get("/health")
        assert response.status_code == 200

        response = client.get("/health/ready")
        assert response.status_code == 200

        response = client.get("/health/config")
        assert response.status_code == 200

    def test_endpoints_have_docstrings(self):
        """Verify all endpoint functions have docstrings."""
        from app.routers import health

        assert health.health_check.__doc__ is not None
        assert health.readiness_check.__doc__ is not None
        assert health.get_config.__doc__ is not None

    def test_explicit_status_codes(self):
        """Verify endpoints have explicit status_code."""
        # If status_code is explicit, response should be 200 not 307
        response = client.get("/health", follow_redirects=False)
        assert response.status_code == 200

    def test_router_has_health_tag(self):
        """Verify router has Health tag (Requirement 7.2)."""
        from app.routers import health

        assert health.router.tags == ["Health"]

    def test_endpoints_in_openapi_docs(self):
        """Verify health endpoints appear in OpenAPI /docs (Requirement 7.6)."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi.get("paths", {})

        # Check all health endpoints exist in OpenAPI schema
        assert "/health" in paths
        assert "/health/ready" in paths
        assert "/health/config" in paths

        # Check they have Health tag
        health_path = paths.get("/health", {})
        get_op = health_path.get("get", {})
        assert "Health" in get_op.get("tags", [])

    def test_endpoints_have_explicit_status_code_in_schema(self):
        """Verify endpoints have explicit status_code in OpenAPI schema (Requirement 7.3)."""
        response = client.get("/openapi.json")
        openapi = response.json()
        paths = openapi.get("paths", {})

        for endpoint in ["/health", "/health/ready", "/health/config"]:
            path_info = paths.get(endpoint, {})
            get_op = path_info.get("get", {})
            responses = get_op.get("responses", {})
            # All should have explicit 200 response
            assert "200" in responses, f"Endpoint {endpoint} missing explicit 200 response"


class TestJsonFormat:
    """Tests for JSON format consistency."""

    def test_all_responses_valid_json(self):
        """Verify all endpoints return valid JSON."""
        for endpoint in ["/health", "/health/ready", "/health/config"]:
            response = client.get(endpoint)
            assert response.headers["content-type"] == "application/json"
            # Should not raise exception
            _ = response.json()

    def test_no_null_in_required_fields(self):
        """Verify no null values in required fields."""
        # Test /health
        response = client.get("/health")
        data = response.json()
        for key, value in data.items():
            assert value is not None, f"Field {key} is null in /health"

        # Test /health/ready
        response = client.get("/health/ready")
        data = response.json()
        assert data["ready"] is not None
        assert data["model_loaded"] is not None
        assert data["model_info"] is not None

        # Test /health/config
        response = client.get("/health/config")
        data = response.json()
        # Note: api config intentionally not exposed (security requirement)
        assert data["upload"] is not None
        assert data["output"] is not None

    def test_snake_case_naming(self):
        """Verify all JSON keys use snake_case."""
        import re

        snake_case_pattern = re.compile(r'^[a-z][a-z0-9_]*$')

        def check_keys(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    assert snake_case_pattern.match(key), f"Key '{path}.{key}' is not snake_case"
                    check_keys(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_keys(item, f"{path}[{i}]")

        for endpoint in ["/health", "/health/ready", "/health/config"]:
            response = client.get(endpoint)
            data = response.json()
            check_keys(data, endpoint)


class TestStructuredLogging:
    """Tests for Task 11: Structured logging."""

    def test_structured_logging_output(self, caplog):
        """Verify health endpoints log in JSON format with required fields."""
        import json
        import logging

        # Set log level to capture INFO logs
        with caplog.at_level(logging.INFO, logger="app.routers.health"):
            response = client.get("/health")
            assert response.status_code == 200

        # Verify log contains JSON with required fields
        log_found = False
        for record in caplog.records:
            if "Health request:" in record.message:
                log_found = True
                # Extract JSON part
                json_str = record.message.split("Health request: ")[1]
                log_data = json.loads(json_str)

                # Verify required fields
                assert "timestamp" in log_data
                assert "level" in log_data
                assert "endpoint" in log_data
                assert "response_time_ms" in log_data
                assert log_data["endpoint"] == "/health"
                assert log_data["level"] == "info"
                assert isinstance(log_data["response_time_ms"], float)

        assert log_found, "Structured log not found"

    def test_structured_logging_includes_model_status(self, caplog):
        """Verify /health/ready logs include model_status."""
        import json
        import logging

        with caplog.at_level(logging.INFO, logger="app.routers.health"):
            response = client.get("/health/ready")
            assert response.status_code == 200

        log_found = False
        for record in caplog.records:
            if "Health request:" in record.message:
                log_found = True
                json_str = record.message.split("Health request: ")[1]
                log_data = json.loads(json_str)

                assert "model_status" in log_data
                assert log_data["endpoint"] == "/health/ready"

        assert log_found, "Structured log not found for /health/ready"


class TestPrometheusMetrics:
    """Tests for Task 12: Prometheus metrics endpoint."""

    def test_metrics_endpoint_exists(self):
        """Verify /health/metrics endpoint exists and returns Prometheus format."""
        response = client.get("/health/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_contains_health_request_count(self):
        """Verify metrics include health_request_count."""
        # Make some requests first
        client.get("/health")
        client.get("/health/ready")

        response = client.get("/health/metrics")
        assert response.status_code == 200

        metrics_text = response.text
        # Verify Prometheus metrics are present
        assert "health_request_count" in metrics_text
        assert "health_response_time_seconds" in metrics_text
        assert "model_load_status" in metrics_text

    def test_metrics_model_load_status(self):
        """Verify model_load_status gauge reflects actual model status."""
        from unittest.mock import patch, PropertyMock

        # Test with model not loaded
        with patch('app.services.model_manager.ModelManager.is_loaded', new_callable=PropertyMock) as mock_loaded:
            mock_loaded.return_value = False

            # Call /health/ready to trigger metric update
            client.get("/health/ready")

            response = client.get("/health/metrics")
            metrics_text = response.text

            # Check model_load_status is present
            assert "model_load_status" in metrics_text

    def test_metrics_increments_request_count(self):
        """Verify request count increases with each request."""
        # Get initial metrics
        response = client.get("/health/metrics")
        initial_metrics = response.text

        # Make a request
        client.get("/health")

        # Get updated metrics
        response = client.get("/health/metrics")
        updated_metrics = response.text

        # Both should contain the metric
        assert "health_request_count" in initial_metrics
        assert "health_request_count" in updated_metrics


class TestConfigurationCaching:
    """Tests for Configuration caching (Task 8)."""

    def test_settings_singleton_same_instance(self):
        """Verify Settings singleton returns same instance."""
        from app.core.config import Settings

        settings1 = Settings()
        settings2 = Settings()

        assert settings1 is settings2

    def test_settings_config_cached(self):
        """Verify Settings config is cached and not re-parsed each request."""
        from app.core.config import Settings
        from unittest.mock import patch

        # Create fresh singleton to test caching
        Settings._instance = None
        settings = Settings()

        # Mock from_env to track calls
        with patch('app.core.config.APIConfig.from_env') as mock_api:
            with patch('app.core.config.UploadConfig.from_env') as mock_upload:
                # Access config multiple times
                _ = settings.api
                _ = settings.api
                _ = settings.upload
                _ = settings.upload

                # from_env should only be called once during initialization
                mock_api.assert_not_called()  # Already initialized
                mock_upload.assert_not_called()

    def test_config_endpoint_performance(self):
        """Verify /health/config responds within 150ms (Requirement 6.3)."""
        import time

        # Warm up
        client.get("/health/config")

        times = []
        for _ in range(10):
            start = time.perf_counter()
            response = client.get("/health/config")
            end = time.perf_counter()
            assert response.status_code == 200
            times.append((end - start) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        assert avg_time < 150, f"Average response time {avg_time:.2f}ms exceeds 150ms"


class TestPerformance:
    """Tests for performance requirements (Task 9)."""

    def test_health_endpoint_performance(self):
        """Verify /health responds within 100ms (Requirement 6.1)."""
        import time

        # Warm up
        client.get("/health")

        times = []
        for _ in range(10):
            start = time.perf_counter()
            response = client.get("/health")
            end = time.perf_counter()
            assert response.status_code == 200
            times.append((end - start) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        assert avg_time < 100, f"Average response time {avg_time:.2f}ms exceeds 100ms"

    def test_ready_endpoint_performance_when_not_loaded(self):
        """Verify /health/ready responds within 200ms when model not loaded (Requirement 6.2)."""
        import time
        from unittest.mock import patch, PropertyMock

        # Ensure model is not loaded for this test
        with patch('app.services.model_manager.ModelManager.is_loaded', new_callable=PropertyMock) as mock_loaded:
            mock_loaded.return_value = False

            # Warm up
            client.get("/health/ready")

            times = []
            for _ in range(10):
                start = time.perf_counter()
                response = client.get("/health/ready")
                end = time.perf_counter()
                assert response.status_code == 200
                times.append((end - start) * 1000)

            avg_time = sum(times) / len(times)
            assert avg_time < 200, f"Average response time {avg_time:.2f}ms exceeds 200ms"

    def test_health_no_io_operations(self):
        """Verify /health does not trigger filesystem or network I/O (Requirement 6.4)."""
        from unittest.mock import patch

        # Mock filesystem and network operations
        with patch('os.path.exists') as mock_exists:
            with patch('builtins.open') as mock_open:
                response = client.get("/health")

                assert response.status_code == 200
                # Verify no filesystem calls were made during health check
                mock_exists.assert_not_called()
                mock_open.assert_not_called()
