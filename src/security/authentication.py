from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "security"

SESSION_TIMEOUT_MINUTES = 60
PBKDF2_ITERATIONS = 600_000
SALT_BYTES = 16
SESSION_TOKEN_BYTES = 32


USERS = {
    "dhun.aadhar@gmail.com": {
        "username": "dhun.aadhar@gmail.com",
        "role": "ANALYST",
        "password_hash": (
            "pbkdf2_sha256$600000$08bcfb7347ea0806339775d594d94b9a$"
            "0c9598f0f565b92a15ac646ed9c7b5112fd2a270855756b11335b0fce682a838"
        ),
    },
}


_SESSIONS: dict[str, dict[str, Any]] = {}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def hash_password(
    password: str,
    salt: bytes | None = None,
    iterations: int = PBKDF2_ITERATIONS,
) -> str:
    """Create a PBKDF2-HMAC-SHA256 password hash."""
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")

    if salt is None:
        salt = secrets.token_bytes(SALT_BYTES)

    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )

    return (
        f"pbkdf2_sha256${iterations}$"
        f"{salt.hex()}${derived_key.hex()}"
    )


def verify_password(
    password: str,
    encoded_hash: str,
) -> bool:
    """Verify a plaintext password against a PBKDF2 password hash."""
    if not password or not encoded_hash:
        return False

    try:
        algorithm, iterations_text, salt_hex, digest_hex = (
            encoded_hash.split("$")
        )

        if algorithm != "pbkdf2_sha256":
            return False

        iterations = int(iterations_text)
        salt = bytes.fromhex(salt_hex)
        expected_digest = bytes.fromhex(digest_hex)

        actual_digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )

        return hmac.compare_digest(
            actual_digest,
            expected_digest,
        )

    except (ValueError, TypeError):
        return False


def authenticate(
    username: str,
    password: str,
) -> dict[str, Any] | None:
    """Authenticate a local user."""
    if not isinstance(username, str):
        return None

    normalized_username = username.strip().lower()
    user = USERS.get(normalized_username)

    if user is None:
        return None

    if not verify_password(
        password,
        user["password_hash"],
    ):
        return None

    return {
        "username": user["username"],
        "role": user["role"],
    }


def create_session(
    username: str,
    role: str,
) -> str:
    """Create an authenticated local session."""
    token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)

    created_at = utc_now()
    expires_at = created_at + timedelta(
        minutes=SESSION_TIMEOUT_MINUTES
    )

    _SESSIONS[token] = {
        "username": username,
        "role": role,
        "created_at": created_at,
        "expires_at": expires_at,
    }

    return token


def get_session(
    token: str | None,
) -> dict[str, Any] | None:
    """Return a valid session or None if missing or expired."""
    if not token:
        return None

    session = _SESSIONS.get(token)

    if session is None:
        return None

    if utc_now() >= session["expires_at"]:
        _SESSIONS.pop(token, None)
        return None

    return {
        "username": session["username"],
        "role": session["role"],
        "created_at": session["created_at"].isoformat(),
        "expires_at": session["expires_at"].isoformat(),
    }


def destroy_session(
    token: str | None,
) -> bool:
    """Invalidate an authenticated session."""
    if not token:
        return False

    return _SESSIONS.pop(token, None) is not None


def cleanup_expired_sessions() -> int:
    """Remove expired sessions."""
    now = utc_now()

    expired_tokens = [
        token
        for token, session in _SESSIONS.items()
        if now >= session["expires_at"]
    ]

    for token in expired_tokens:
        _SESSIONS.pop(token, None)

    return len(expired_tokens)


def get_authenticated_identity(
    token: str | None,
) -> dict[str, Any] | None:
    """Return the identity associated with a valid session."""
    session = get_session(token)

    if session is None:
        return None

    return {
        "username": session["username"],
        "role": session["role"],
    }


def run_authentication_validation() -> dict[str, Any]:
    """Validate authentication and local session management."""
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    valid_identity = authenticate(
        "dhun.aadhar@gmail.com",
        "12345",
    )

    invalid_password = authenticate(
        "dhun.aadhar@gmail.com",
        "incorrect-password",
    )

    unknown_user = authenticate(
        "unknown@example.com",
        "incorrect-password",
    )

    session_token = None

    if valid_identity is not None:
        session_token = create_session(
            valid_identity["username"],
            valid_identity["role"],
        )

    session_identity = get_authenticated_identity(
        session_token,
    )

    logout_result = destroy_session(
        session_token,
    )

    session_after_logout = get_session(
        session_token,
    )

    checks = {
        "valid_credentials_accepted": (
            valid_identity is not None
        ),
        "valid_identity_returned": (
            valid_identity is not None
            and valid_identity["username"]
            == "dhun.aadhar@gmail.com"
        ),
        "valid_role_returned": (
            valid_identity is not None
            and valid_identity["role"] == "ANALYST"
        ),
        "invalid_password_rejected": (
            invalid_password is None
        ),
        "unknown_user_rejected": (
            unknown_user is None
        ),
        "session_created": (
            session_token is not None
        ),
        "session_identity_available": (
            session_identity is not None
            and session_identity.get("role") == "ANALYST"
        ),
        "logout_invalidates_session": (
            logout_result
            and session_after_logout is None
        ),
    }

    result = {
        "status": (
            "PASS"
            if all(checks.values())
            else "FAIL"
        ),
        "timestamp_utc": utc_now_iso(),
        "mode": "offline",
        "authentication": {
            "method": "local_pbkdf2_sha256",
            "iterations": PBKDF2_ITERATIONS,
            "passwords_stored_as_hashes": True,
            "credentials_exposed_to_frontend": False,
        },
        "session": {
            "type": "local_in_memory",
            "timeout_minutes": SESSION_TIMEOUT_MINUTES,
            "random_token": True,
        },
        "test_account": {
            "username": "dhun.aadhar@gmail.com",
            "role": "ANALYST",
        },
        "checks": checks,
    }

    report_path = (
        REPORT_DIR / "authentication_validation.json"
    )

    report_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


if __name__ == "__main__":
    print(
        json.dumps(
            run_authentication_validation(),
            indent=2,
        )
    )