from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

from shapes import Shape, UnionN


def _as_shape(shape_or_shapes: Union[Shape, Iterable[Shape]]) -> Shape:
    if isinstance(shape_or_shapes, Shape):
        return shape_or_shapes
    shapes = list(shape_or_shapes)
    if not shapes:
        raise ValueError("No shapes provided")
    return UnionN(*shapes)


def autosize_bounds(
    shape: Union[Shape, Iterable[Shape]],
    initial_bounds: Tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0),
    coarse_resolution: int = 100,
    margin: float = 0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Estimate tight x/y limits by coarse sampling of sdf/evaluate.
    Returns (xlim, ylim).
    """
    shape = _as_shape(shape)
    xmin, xmax, ymin, ymax = initial_bounds
    xs = np.linspace(xmin, xmax, coarse_resolution)
    ys = np.linspace(ymin, ymax, coarse_resolution)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dists, _ = shape.evaluate_many(pts)
    inside_mask = dists <= 0.0
    if not np.any(inside_mask):
        return (xmin, xmax), (ymin, ymax)
    inside_pts = pts[inside_mask]
    pxmin, pymin = inside_pts.min(axis=0)
    pxmax, pymax = inside_pts.max(axis=0)
    return (float(pxmin - margin), float(pxmax + margin)), (float(pymin - margin), float(pymax + margin))


def sample_shape_rgba(
    shape: Union[Shape, Iterable[Shape]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a shape (or list of shapes) over a grid, returning X, Y, Z(sdf), RGBA.
    Vectorized across the full grid using Shape.evaluate_many.
    """
    shape = _as_shape(shape)
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dists, colors = shape.evaluate_many(pts)
    Z = dists.reshape(X.shape)
    RGBA = np.zeros((resolution, resolution, 4), dtype=float)
    inside_mask = (Z <= 0.0)
    if colors is None:
        c = np.tile(np.array([0.6, 0.6, 0.6], dtype=float).reshape(1, 3), (pts.shape[0], 1))
    else:
        c = np.clip(np.asarray(colors, dtype=float).reshape(-1, 3), 0.0, 1.0)
    Cimg = c.reshape(resolution, resolution, 3)
    RGBA[:, :, :3] = np.where(inside_mask[:, :, None], Cimg, 0.0)
    RGBA[:, :, 3] = inside_mask.astype(float)
    return X, Y, Z, RGBA


def render_to_axes(
    ax,
    shape: Union[Shape, Iterable[Shape]],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    resolution: int = 512,
    title: Optional[str] = None,
    draw_edges: bool = False,
    edge_color: str = "black",
    edge_width: float = 1.5,
    interpolation: str = "none",
    parallel: bool = True,
    workers: Optional[int] = None,
    show_axes: bool = False,
    show_grid: bool = False,
    frame_only: bool = True,
) -> None:
    shape = _as_shape(shape)
    if xlim is None or ylim is None:
        xlim, ylim = autosize_bounds(shape)
    # Single-process, fully vectorized sampling
    X, Y, Z, RGBA = sample_shape_rgba(shape, xlim, ylim, resolution=resolution)
    ax.imshow(
        RGBA,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        interpolation=interpolation,
        aspect="equal",
    )
    if draw_edges:
        ax.contour(X, Y, Z, levels=[0.0], colors=[edge_color], linewidths=edge_width, antialiased=True)
    if title:
        ax.set_title(title)
    if show_axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(show_grid, alpha=0.2, linestyle="--")
        if frame_only:
            # Keep spines, hide ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    else:
        ax.axis("off")


def render_to_file(
    shape: Union[Shape, Iterable[Shape]],
    out_path: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    resolution: int = 600,
    figsize: Tuple[float, float] = (6.0, 6.0),
    title: Optional[str] = None,
    draw_edges: bool = False,
    edge_color: str = "black",
    edge_width: float = 1.5,
    interpolation: str = "none",
    parallel: bool = True,
    workers: Optional[int] = None,
    show_axes: bool = False,
    show_grid: bool = False,
    frame_only: bool = True,
    dpi: int = 220,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
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
        parallel=parallel,
        workers=workers,
        show_axes=show_axes,
        show_grid=show_grid,
        frame_only=frame_only,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


