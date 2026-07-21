"""Smoke: package imports and exposes a version."""

import mcrtx


def test_version():
    assert isinstance(mcrtx.__version__, str)
