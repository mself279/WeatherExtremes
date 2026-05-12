"""Site-wide HTTP Basic Auth for the alpha.

Activates only when both ``BASIC_AUTH_USER`` and ``BASIC_AUTH_PASSWORD`` are
set in the environment / settings. Otherwise the middleware is a no-op so
local development is unaffected. Once activated it gates every URL on the
site, including /admin/ — Django's session auth still runs after, so admins
enter the basic-auth creds first and then their Django superuser creds.

Placed after WhiteNoise in the middleware stack so static assets (CSS, JS)
load without challenge, but every HTML page and form submission requires
the shared credentials.
"""
from __future__ import annotations

import base64
import secrets

from django.conf import settings
from django.http import HttpResponse


class BasicAuthMiddleware:
    """Gate the entire site behind one shared username + password."""

    def __init__(self, get_response):
        self.get_response = get_response
        self.username = getattr(settings, "BASIC_AUTH_USER", "") or ""
        self.password = getattr(settings, "BASIC_AUTH_PASSWORD", "") or ""
        self.realm = getattr(settings, "BASIC_AUTH_REALM", "WeatherExtremes")
        self.enabled = bool(self.username and self.password)

    def __call__(self, request):
        if not self.enabled:
            return self.get_response(request)
        if self._authorized(request):
            return self.get_response(request)
        return self._challenge()

    def _authorized(self, request) -> bool:
        header = request.META.get("HTTP_AUTHORIZATION", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:].strip()).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return False
        user, sep, pwd = decoded.partition(":")
        if not sep:
            return False
        # Constant-time comparison so an attacker can't infer the credentials
        # from response-time differences.
        user_ok = secrets.compare_digest(user, self.username)
        pwd_ok = secrets.compare_digest(pwd, self.password)
        return user_ok and pwd_ok

    def _challenge(self) -> HttpResponse:
        resp = HttpResponse(
            "Authentication required.",
            status=401,
            content_type="text/plain",
        )
        resp["WWW-Authenticate"] = f'Basic realm="{self.realm}", charset="UTF-8"'
        return resp
