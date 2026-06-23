"""Compatibility wrapper for the reorganized project structure.

The implementation now lives in ``src/downscaling/utils.py``. Existing
commands such as ``python PyESD/train.py`` can keep importing ``utils``.
"""

from src.downscaling.utils import *  # noqa: F401,F403

