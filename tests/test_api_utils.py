"""Tests for shared.python.api_utils."""

import builtins

import pytest
from requests.exceptions import RequestException

from shared.python.api_utils import (
    create_azure_openai_client,
    create_openai_client,
    download_image,
    make_safe_request,
)


class _FakeResponse:
    def __init__(self, content: bytes = b""):
        self.raised = False
        self.content = content

    def raise_for_status(self):
        self.raised = True


def _block_openai_import(monkeypatch):
    """Make ``from openai import OpenAI`` raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


class TestMakeSafeRequest:
    def test_returns_response_on_success(self, monkeypatch):
        fake = _FakeResponse()

        def fake_request(method, url, timeout, **kwargs):
            assert timeout == 30
            return fake

        monkeypatch.setattr("shared.python.api_utils.requests.request", fake_request)
        result = make_safe_request("https://example.com")
        assert result is fake
        assert fake.raised is True

    def test_retries_then_raises(self, monkeypatch):
        calls = {"count": 0}

        def fake_request(method, url, timeout, **kwargs):
            calls["count"] += 1
            raise RequestException("boom")

        monkeypatch.setattr("shared.python.api_utils.requests.request", fake_request)
        with pytest.raises(RequestException):
            make_safe_request("https://example.com", retries=3)
        assert calls["count"] == 3

    def test_retries_then_succeeds(self, monkeypatch):
        fake = _FakeResponse()
        calls = {"count": 0}

        def fake_request(method, url, timeout, **kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                raise RequestException("transient")
            return fake

        monkeypatch.setattr("shared.python.api_utils.requests.request", fake_request)
        assert make_safe_request("https://example.com", retries=3) is fake
        assert calls["count"] == 3

    def test_forwards_method_and_extra_kwargs(self, monkeypatch):
        fake = _FakeResponse()
        seen = {}

        def fake_request(method, url, timeout, **kwargs):
            seen.update({"method": method, "url": url, "timeout": timeout, **kwargs})
            return fake

        monkeypatch.setattr("shared.python.api_utils.requests.request", fake_request)
        make_safe_request(
            "https://example.com/api",
            method="POST",
            timeout=5,
            json={"a": 1},
            headers={"X-Test": "1"},
        )
        assert seen["method"] == "POST"
        assert seen["url"] == "https://example.com/api"
        assert seen["timeout"] == 5
        assert seen["json"] == {"a": 1}
        assert seen["headers"] == {"X-Test": "1"}

    def test_zero_retries_raises_without_requesting(self, monkeypatch):
        def fake_request(method, url, timeout, **kwargs):
            raise AssertionError("request should not be attempted")

        monkeypatch.setattr("shared.python.api_utils.requests.request", fake_request)
        with pytest.raises(RequestException, match="Request failed"):
            make_safe_request("https://example.com", retries=0)


class TestCreateOpenAIClient:
    def test_missing_key_raises_value_error(self, monkeypatch):
        pytest.importorskip("openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            create_openai_client()

    def test_explicit_key_creates_client(self, monkeypatch):
        openai = pytest.importorskip("openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = create_openai_client(api_key="explicit-key")
        assert isinstance(client, openai.OpenAI)
        assert client.api_key == "explicit-key"

    def test_env_key_creates_client(self, monkeypatch):
        pytest.importorskip("openai")
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        assert create_openai_client().api_key == "env-key"

    def test_missing_package_raises_import_error(self, monkeypatch):
        _block_openai_import(monkeypatch)
        with pytest.raises(ImportError, match="pip install openai"):
            create_openai_client(api_key="key")


class TestCreateAzureOpenAIClient:
    def test_missing_endpoint_raises_value_error(self, monkeypatch):
        pytest.importorskip("openai")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        with pytest.raises(ValueError, match="endpoint"):
            create_azure_openai_client()

    def test_missing_key_raises_value_error(self, monkeypatch):
        pytest.importorskip("openai")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            create_azure_openai_client()

    def test_explicit_arguments_build_v1_base_url(self, monkeypatch):
        pytest.importorskip("openai")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        client = create_azure_openai_client(
            endpoint="https://example.openai.azure.com/",
            api_key="explicit-key",
        )
        assert client.api_key == "explicit-key"
        assert str(client.base_url) == "https://example.openai.azure.com/openai/v1/"

    def test_environment_variables_are_used(self, monkeypatch):
        pytest.importorskip("openai")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
        client = create_azure_openai_client()
        assert client.api_key == "env-key"
        assert str(client.base_url) == "https://env.openai.azure.com/openai/v1/"

    def test_missing_package_raises_import_error(self, monkeypatch):
        _block_openai_import(monkeypatch)
        with pytest.raises(ImportError, match="pip install openai"):
            create_azure_openai_client(endpoint="https://example.com", api_key="key")


class TestDownloadImage:
    def test_writes_content_and_returns_path(self, monkeypatch, tmp_path):
        fake = _FakeResponse(content=b"image-bytes")
        monkeypatch.setattr(
            "shared.python.api_utils.requests.request",
            lambda method, url, timeout, **kwargs: fake,
        )
        target = tmp_path / "nested" / "dir" / "image.png"
        result = download_image("https://example.com/image.png", str(target))
        assert result == str(target)
        assert target.read_bytes() == b"image-bytes"

    def test_saves_into_current_directory(self, monkeypatch, tmp_path):
        fake = _FakeResponse(content=b"bytes")
        monkeypatch.setattr(
            "shared.python.api_utils.requests.request",
            lambda method, url, timeout, **kwargs: fake,
        )
        monkeypatch.chdir(tmp_path)
        assert download_image("https://example.com/i.png", "image.png") == "image.png"
        assert (tmp_path / "image.png").read_bytes() == b"bytes"

    def test_passes_timeout_through(self, monkeypatch, tmp_path):
        seen = {}

        def fake_request(method, url, timeout, **kwargs):
            seen["timeout"] = timeout
            return _FakeResponse(content=b"x")

        monkeypatch.setattr("shared.python.api_utils.requests.request", fake_request)
        download_image("https://example.com/i.png", str(tmp_path / "i.png"), timeout=7)
        assert seen["timeout"] == 7

    def test_request_failure_propagates(self, monkeypatch, tmp_path):
        def fake_request(method, url, timeout, **kwargs):
            raise RequestException("boom")

        monkeypatch.setattr("shared.python.api_utils.requests.request", fake_request)
        target = tmp_path / "missing.png"
        with pytest.raises(RequestException):
            download_image("https://example.com/i.png", str(target))
        assert not target.exists()
