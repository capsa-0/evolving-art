from __future__ import annotations

import gc
import io
import os
from typing import Iterable, Optional, Tuple, Union

from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from src.core.shapes import Shape

from .core import render_to_axes


def render_shape_to_file(
    shape: Union[Shape, Iterable[Shape]],
    out_path: Optional[str],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    resolution: int = 600,
    figsize: Tuple[float, float] = (6.0, 6.0),
    title: Optional[str] = None,
    draw_edges: bool = False,
    edge_color: str = "black",
    edge_width: float = 1.5,
    interpolation: str = "none",
    show_axes: bool = False,
    show_grid: bool = False,
    frame_only: bool = True,
    dpi: int = 220,
    transparent: bool = False,
    return_image: bool = False,
) -> Optional[Image.Image]:
    """Render Shape instances to disk (PNG) or return an in-memory image."""

    fig = Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    render_to_axes(
        ax,
        shape,
        xlim=xlim,
        ylim=ylim,
        resolution=resolution,
        title=title,
        draw_edges=draw_edges,
        edge_color=edge_color,
        edge_width=edge_width,
        interpolation=interpolation,
        show_axes=show_axes,
        show_grid=show_grid,
        frame_only=frame_only,
    )

    canvas.draw()

    image: Optional[Image.Image] = None
    if return_image:
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=dpi,
            transparent=transparent,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0,
        )
        buffer.seek(0)
        image = Image.open(buffer).convert("RGBA")

    if out_path:
        directory = os.path.dirname(out_path) or "."
        os.makedirs(directory, exist_ok=True)
        fig.savefig(
            out_path,
            format="png",
            dpi=dpi,
            transparent=transparent,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0,
        )

    fig.clear()
    del ax
    del canvas
    del fig
    gc.collect()

    return image
