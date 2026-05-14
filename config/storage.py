"""Custom static-files storage.

The default WhiteNoise ``CompressedManifestStaticFilesStorage`` is strict —
any ``{% static %}`` reference to a file that's not in the manifest built at
``collectstatic`` time raises ``ValueError`` at template render time, which
turns into a 500 in production.

For templates with placeholder image slots (the Use Cases page, in
particular) we want graceful degradation: missing files should produce a
plain URL that 404s in the browser so the ``onerror`` JavaScript handler
can swap in a placeholder. ``manifest_strict = False`` does exactly that
while keeping the hash-based cache-busting on files that *do* exist.
"""
from whitenoise.storage import CompressedManifestStaticFilesStorage


class PermissiveManifestStaticFilesStorage(CompressedManifestStaticFilesStorage):
    manifest_strict = False
