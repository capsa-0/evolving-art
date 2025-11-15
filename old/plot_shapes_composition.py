from __future__ import annotations

import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from shapes import (
    UnitSquare,
    UnitDisk,
    Polygon,
)


def plot_shape_colored(ax, shape, xlim, ylim, title: str, resolution=400, edge_color="black"):
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X, dtype=float)
    RGBA = np.zeros((resolution, resolution, 4), dtype=float)
    fallback = np.array([0.6, 0.6, 0.6], dtype=float)
    # Evaluate using shape.evaluate if available, otherwise fallback to sdf only
    has_eval = hasattr(shape, "evaluate")
    for i in range(resolution):
        for j in range(resolution):
            p = np.array([X[i, j], Y[i, j]])
            if has_eval:
                d, c = shape.evaluate(p)
            else:
                d = float(shape.sdf(p))
                c = None
            Z[i, j] = d
            if d <= 0.0:
                col = c if c is not None else fallback
                RGBA[i, j, :3] = np.clip(col, 0.0, 1.0)
                RGBA[i, j, 3] = 1.0
            else:
                RGBA[i, j, :3] = 0.0
                RGBA[i, j, 3] = 0.0
    ax.imshow(
        RGBA,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        interpolation="bilinear",
        aspect="equal",
    )
    ax.contour(X, Y, Z, levels=[0.0], colors=[edge_color], linewidths=1.2, antialiased=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2, linestyle="--")


def add_color(shape, rgb):
    if hasattr(shape, "with_color"):
        return shape.with_color(*rgb)
    return shape


def main():
    os.makedirs("plots", exist_ok=True)

    # Base primitives
    circle = UnitDisk()
    square = UnitSquare()
    tri = Polygon([np.array([0.0, 1.2]), np.array([-1.0, -0.8]), np.array([1.1, -0.7])])

    # Colors for the three transformed instances
    C1 = (0.90, 0.25, 0.25)
    C2 = (0.25, 0.80, 0.35)
    C3 = (0.25, 0.45, 0.95)

    # Three transform functions using the DSL on any Shape
    # T1: scale and rotate and translate left-up
    def T1(shape):
        return shape.scale(1.2, 0.7).rotate(0.35).translate(-1.2, 0.6)

    # T2: anisotropic scale, rotate negative, translate right-up
    def T2(shape):
        return shape.scale(0.8, 1.4).rotate(-0.55).translate(0.9, 0.7)

    # T3: scale and translate downward
    def T3(shape):
        return shape.scale(1.1).rotate(0.1).translate(0.1, -0.7)

    # Compose function for 4th column: (A ∪ B) − C
    def compose(a, b, c):
        return (a | b) | c

    # Set global bounds for consistent grid
    xlim = (-2.6, 2.6)
    ylim = (-2.2, 2.2)

    fig, axs = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)

    # Row 1: Circle
    circ1 = add_color(T1(circle), C1)
    circ2 = add_color(T2(circle), C2)
    circ3 = add_color(T3(circle), C3)
    circ_comp = compose(circ1, circ2, circ3)
    plot_shape_colored(axs[0, 0], circ1, xlim, ylim, "Circle T1")
    plot_shape_colored(axs[0, 1], circ2, xlim, ylim, "Circle T2")
    plot_shape_colored(axs[0, 2], circ3, xlim, ylim, "Circle T3")
    plot_shape_colored(axs[0, 3], circ_comp, xlim, ylim, "T1 ∪ T2 ∪ T3")
    axs[0, 0].set_ylabel("Circle", fontsize=11)

    # Row 2: Square
    sq1 = add_color(T1(square), C1)
    sq2 = add_color(T2(square), C2)
    sq3 = add_color(T3(square), C3)
    sq_comp = compose(sq1, sq2, sq3)
    plot_shape_colored(axs[1, 0], sq1, xlim, ylim, "Square T1")
    plot_shape_colored(axs[1, 1], sq2, xlim, ylim, "Square T2")
    plot_shape_colored(axs[1, 2], sq3, xlim, ylim, "Square T3")
    plot_shape_colored(axs[1, 3], sq_comp, xlim, ylim, "T1 ∪ T2 ∪ T3")
    axs[1, 0].set_ylabel("Square", fontsize=11)

    # Row 3: Triangle
    tr1 = add_color(T1(tri), C1)
    tr2 = add_color(T2(tri), C2)
    tr3 = add_color(T3(tri), C3)
    tr_comp = compose(tr1, tr2, tr3)
    plot_shape_colored(axs[2, 0], tr1, xlim, ylim, "Triangle T1")
    plot_shape_colored(axs[2, 1], tr2, xlim, ylim, "Triangle T2")
    plot_shape_colored(axs[2, 2], tr3, xlim, ylim, "Triangle T3")
    plot_shape_colored(axs[2, 3], tr_comp, xlim, ylim, "T1 ∪ T2 ∪ T3")
    axs[2, 0].set_ylabel("Triangle", fontsize=11)

    # Column titles
    axs[0, 0].set_title("Transform 1")
    axs[0, 1].set_title("Transform 2")
    axs[0, 2].set_title("Transform 3")
    axs[0, 3].set_title("Composition")

    fig.suptitle("Composition grid: three transforms per shape, then T1 ∪ T2 ∪ T3", fontsize=14)
    out_path = "plots/shapes_composition.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


