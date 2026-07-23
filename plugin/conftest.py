"""conftest.py — register context_guard__plugin as an importable package for tests.

Hermes plugins are loaded at runtime under the ``hermes_plugins.<slug>`` namespace
(plugins.py:1832-1868). For testing, we register the same directory as
``context_guard__plugin`` so that ``from context_guard__plugin.X import Y`` works
and relative imports inside plugin modules (e.g. ``from .watermark import Watermark``)
resolve correctly.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_plugin_dir = str(Path(__file__).parent)

pkg_name = "context_guard__plugin"
if pkg_name not in sys.modules:
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [_plugin_dir]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
