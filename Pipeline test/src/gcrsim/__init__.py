try:
    from ._version import version as __version__
except Exception:
    __version__ = "0.0.0"

from .example_module import greetings, meaning

__all__ = ["greetings", "meaning"]
