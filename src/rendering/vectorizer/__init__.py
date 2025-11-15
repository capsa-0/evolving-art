"""
Vectorizer: Geometry-based rendering using Shapely.

Modules:
- geometry: Geometry helpers, safe operations, node processing
- drawing: Matplotlib-based drawing on axes, SVG/PNG saving
"""

from .geometry import (
    fix_geom,
    safe_intersection,
    safe_difference,
    gene_to_shapely,
    process_node,
)

from .drawing import (
    draw_genome_on_axis,
    save_genome_as_svg,
    save_genome_as_png,
    genome_to_png_bytes,
    render_to_file,
    render_population_grid,
)

__all__ = [
    "fix_geom",
    "safe_intersection",
    "safe_difference",
    "gene_to_shapely",
    "process_node",
    "draw_genome_on_axis",
    "save_genome_as_svg",
    "save_genome_as_png",
    "genome_to_png_bytes",
    "render_to_file",
    "render_population_grid",
]
