"""
Django settings for the WeatherExtremes project.

Local development uses SQLite and DEBUG=True. Production (Railway) sets
DATABASE_URL, DEBUG=False, SECRET_KEY, and serves static files via WhiteNoise.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    # Dev-only fallback. Set SECRET_KEY in the Railway dashboard for prod.
    "django-insecure-change-me-in-production-please-do-not-use-this-key",
)
DEBUG = os.environ.get("DEBUG", "True") == "True"

# ALLOWED_HOSTS: comma-separated env var overrides; always include Railway
# subdomains and local defaults so a fresh deploy works without extra config.
_default_hosts = ["localhost", "127.0.0.1", ".up.railway.app", ".railway.app"]
_env_hosts = [h.strip() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip()]
ALLOWED_HOSTS = _env_hosts or _default_hosts

# CSRF: Django 4+ requires explicit trusted origins for HTTPS POSTs.
CSRF_TRUSTED_ORIGINS = [
    f"https://{h.lstrip('.')}" for h in ALLOWED_HOSTS if "." in h and h != "127.0.0.1"
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "analysis.apps.AnalysisConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # WhiteNoise must come right after SecurityMiddleware.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    # Site-wide basic auth (no-op when BASIC_AUTH_USER / BASIC_AUTH_PASSWORD
    # are not set in the environment).
    "config.middleware.BasicAuthMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# ---------------------------------------------------------------------------
# Basic auth (alpha gate) — site is locked behind one shared username +
# password when both env vars are set. No effect locally if unset.
# ---------------------------------------------------------------------------
BASIC_AUTH_USER = os.environ.get("BASIC_AUTH_USER", "")
BASIC_AUTH_PASSWORD = os.environ.get("BASIC_AUTH_PASSWORD", "")
BASIC_AUTH_REALM = os.environ.get("BASIC_AUTH_REALM", "WeatherExtremes alpha")

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# ---------------------------------------------------------------------------
# Database — DATABASE_URL on Railway, SQLite locally
# ---------------------------------------------------------------------------
if os.environ.get("DATABASE_URL"):
    import dj_database_url  # lazy: only needed in environments that set it
    DATABASES = {
        "default": dj_database_url.config(
            conn_max_age=600,
            conn_health_checks=True,
            ssl_require=False,
        )
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "admin:login"

# ---------------------------------------------------------------------------
# Internationalization
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static & media files
# ---------------------------------------------------------------------------
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

# Django 5 STORAGES syntax; WhiteNoise compresses + hashes static assets.
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

# ---------------------------------------------------------------------------
# Production security (only enabled when DEBUG is off)
# ---------------------------------------------------------------------------
if not DEBUG:
    # Railway/most PaaS terminate TLS at the proxy and forward via X-Forwarded-Proto.
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 60 * 60 * 24 * 30  # 30 days; bump to a year once stable
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = False  # set True only after deciding to preload
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_REFERRER_POLICY = "same-origin"
    X_FRAME_OPTIONS = "DENY"

# ---------------------------------------------------------------------------
# Project-specific
# ---------------------------------------------------------------------------
# Cap on bootstrap iterations to keep synchronous runs responsive. When the
# analysis moves to a background worker this limit can be raised.
ANALYSIS_MAX_BOOTSTRAP_ITER = int(
    os.environ.get("ANALYSIS_MAX_BOOTSTRAP_ITER", "500")
)
ANALYSIS_DEFAULT_BOOTSTRAP_ITER = int(
    os.environ.get("ANALYSIS_DEFAULT_BOOTSTRAP_ITER", "100")
)
