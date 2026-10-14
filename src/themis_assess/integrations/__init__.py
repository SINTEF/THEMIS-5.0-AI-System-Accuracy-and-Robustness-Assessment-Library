"""Bridges to optional third-party libraries.

Nothing here is imported by ``core/`` or ``metrics/``, and every module imports its
third-party dependency lazily, so the library installs and runs without any of them.
"""
