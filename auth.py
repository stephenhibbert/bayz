"""Magic-link authentication using Resend + itsdangerous."""

from __future__ import annotations

import logging
import os

import resend
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

log = logging.getLogger(__name__)

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
ALLOWED_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ALLOWED_EMAILS", "").split(",")
    if e.strip()
}
APP_URL = os.environ.get("APP_URL", "http://localhost:8001")
FROM_EMAIL = os.environ.get("FROM_EMAIL") or os.environ.get("EMAIL_FROM", "Bayz <noreply@bayz.app>")

_serializer = URLSafeTimedSerializer(SECRET_KEY, salt="magic-link")


def generate_token(email: str) -> str:
    return _serializer.dumps(email.lower())


def verify_token(token: str, max_age: int = 3600) -> str | None:
    """Returns the email address if the token is valid, None otherwise."""
    try:
        email = _serializer.loads(token, max_age=max_age)
        return email if email.lower() in ALLOWED_EMAILS else None
    except (SignatureExpired, BadSignature):
        return None


def send_magic_link(email: str) -> bool:
    """Send a magic-link email. Returns True if allowed and sent successfully."""
    if email.lower() not in ALLOWED_EMAILS:
        log.warning("Login attempt from disallowed email: %s", email)
        return False

    token = generate_token(email)
    url = f"{APP_URL}/auth/verify?token={token}"

    resend.api_key = os.environ.get("RESEND_API_KEY", "")
    try:
        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": email,
            "subject": "Your Bayz login link",
            "html": f"""<!DOCTYPE html>
<html>
<body style="background:#0a0a0f; color:#e0e0e0; font-family:monospace; padding:40px;">
  <div style="max-width:480px; margin:0 auto;">
    <h2 style="color:#4ab4e0; margin-bottom:24px;">Bayz</h2>
    <p style="margin-bottom:24px; line-height:1.6;">
      Click the link below to sign in. This link expires in <strong>1 hour</strong>.
    </p>
    <a href="{url}"
       style="display:inline-block; background:#4ab4e0; color:#0a0a0f;
              padding:12px 24px; text-decoration:none; font-weight:bold;
              border-radius:4px; font-size:16px;">
      Sign in to Bayz
    </a>
    <p style="margin-top:32px; font-size:12px; color:#666;">
      If you didn't request this, you can safely ignore it.
    </p>
  </div>
</body>
</html>""",
        })
        log.info("Magic link sent to %s", email)
        return True
    except Exception:
        log.exception("Failed to send magic link to %s", email)
        return False
