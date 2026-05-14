"""Reserved for custom static-files storage classes.

Currently empty — we use ``whitenoise.storage.CompressedStaticFilesStorage``
directly from settings.py. An earlier attempt subclassed the manifest variant
with ``manifest_strict=False`` but that doesn't fully prevent ValueError on
missing files (the fallback path also performs an existence check). Kept this
module as a placeholder for future static-storage customizations.
"""
