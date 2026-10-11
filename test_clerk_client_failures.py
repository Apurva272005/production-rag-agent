from unittest.mock import Mock

import pytest
import requests

import clerk_api
from clerk_api import (
    ClerkAPIError,
    ClerkConfigurationError,
    check_connection,
    get_user,
    update_public_metadata,
    validate_clerk_secret,
)


def _response(status: int, body: str = '{"errors":[{"message":"test"}]}') -> Mock:
    response = Mock(status_code=status, text=body)
    response.json.return_value = {"id": "user_1"} if status == 200 else {"errors": [{"message": "test"}]}
    return response


def test_clerk_user_success(monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_secret")
    monkeypatch.setattr(clerk_api.requests, "get", Mock(return_value=_response(200)))
    assert get_user("user_1")["id"] == "user_1"


@pytest.mark.parametrize("status", [401, 403])
def test_invalid_secret_is_reported_as_unavailable_for_user_lookup(monkeypatch, status):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_secret")
    monkeypatch.setattr(clerk_api.time, "sleep", lambda _: None)
    monkeypatch.setattr(clerk_api.requests, "get", Mock(return_value=_response(status)))
    with pytest.raises(ClerkAPIError) as error:
        get_user("user_1")
    assert error.value.reason == "auth_service_unavailable"
    assert error.value.status_code == status


def test_user_not_found_is_session_invalid(monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_secret")
    monkeypatch.setattr(clerk_api.requests, "get", Mock(return_value=_response(404)))
    with pytest.raises(ClerkAPIError) as error:
        get_user("deleted_user")
    assert error.value.reason == "session_invalid"


@pytest.mark.parametrize("failure", [requests.Timeout("timeout"), requests.exceptions.SSLError("certificate")])
def test_network_and_ssl_failures_are_unavailable(monkeypatch, failure):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_secret")
    monkeypatch.setattr(clerk_api.time, "sleep", lambda _: None)
    monkeypatch.setattr(clerk_api.requests, "get", Mock(side_effect=failure))
    with pytest.raises(ClerkAPIError) as error:
        get_user("user_1")
    assert error.value.reason == "auth_service_unavailable"


def test_server_error_and_rate_limit_retry_then_fail(monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_secret")
    monkeypatch.setattr(clerk_api.time, "sleep", lambda _: None)
    monkeypatch.setattr(clerk_api.requests, "get", Mock(side_effect=[_response(500), _response(429)]))
    with pytest.raises(ClerkAPIError) as error:
        get_user("user_1")
    assert error.value.reason == "auth_service_unavailable"
    assert error.value.status_code == 429


def test_health_connection_success_and_invalid_secret(monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_secret")
    monkeypatch.setattr(clerk_api.requests, "get", Mock(return_value=_response(200)))
    check_connection()

    monkeypatch.setattr(clerk_api.requests, "get", Mock(return_value=_response(401)))
    with pytest.raises(ClerkAPIError) as error:
        check_connection()
    assert error.value.reason == "invalid_secret"


def test_role_metadata_uses_dedicated_clerk_metadata_endpoint(monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_secret")
    response = _response(200)
    request = Mock(return_value=response)
    monkeypatch.setattr(clerk_api.requests, "patch", request)

    update_public_metadata("user_1", {"role": "student"})

    assert request.call_args.args[0] == "https://api.clerk.com/v1/users/user_1/metadata"
    assert request.call_args.kwargs["json"] == {"public_metadata": {"role": "student"}}


@pytest.mark.parametrize("value", [None, "", "pk_test_wrong", " sk_test_secret", '"sk_test_secret"', "sk test_secret"])
def test_startup_secret_validation_is_strict(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CLERK_SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("CLERK_SECRET_KEY", value)
    with pytest.raises(ClerkConfigurationError):
        validate_clerk_secret()
