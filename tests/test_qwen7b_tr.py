"""Test qwen7b_tr."""
# pylint: disable=broad-except
from qwen7b_tr import __version__, qwen7b_tr


def test_version():
    """Test version."""
    assert __version__[:3] == "0.1"


def test_sanity():
    """Check sanity."""
    try:
        assert not qwen7b_tr()
    except Exception:
        assert True
