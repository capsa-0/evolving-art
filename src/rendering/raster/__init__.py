"""
Raster rendering: Matplotlib-based figure and canvas orchestration.

Modules:
- core: Sampling and axis-based rendering helpers
- io: File and canvas I/O, figure management
"""

from .core import autosize_bounds, sample_shape_rgba, render_to_axes
from .io import render_shape_to_file

__all__ = [
    "autosize_bounds",
    "sample_shape_rgba",
    "render_to_axes",
    "render_shape_to_file",
]
