"""Rendering utilities for composition trees using Matplotlib.

These helpers stay unaware of any UI toolkit details and return raw image data.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle, PathPatch
from matplotlib.path import Path
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry

from src.core.evolution.genome import PrimitiveGene, TransformParams
from src.rendering.vectorizer.geometry import gene_to_shapely


def _hex_to_rgb_01(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color string to RGB tuple in [0,1]."""
    normalized = hex_color.lstrip("#")
    if len(normalized) != 6:
        return 0.5, 0.5, 0.5
    r = int(normalized[0:2], 16) / 255.0
    g = int(normalized[2:4], 16) / 255.0
    b = int(normalized[4:6], 16) / 255.0
    return r, g, b


@dataclass
class TreeNode:
    kind: str
    label: str
    color_rgb: Optional[Tuple[float, float, float]] = None
    children: List["TreeNode"] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    primitive_payload: Optional[dict] = None
    geometry: Optional[BaseGeometry] = None


def _parse_tree(node_obj: dict) -> TreeNode:
    """Parse a composition dictionary into a TreeNode hierarchy."""
    if "primitive" in node_obj:
        prim = node_obj["primitive"]
        kind = prim.get("kind", "primitive")
        color = prim.get("color_rgb", [0.7, 0.7, 0.7])
        try:
            color_tuple = (float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            color_tuple = (0.7, 0.7, 0.7)
        label = kind
        poly_vertices = prim.get("polygon_vertices")
        if kind == "polygon" and poly_vertices is not None:
            try:
                n_vertices = len(poly_vertices)
                label = f"{kind}({n_vertices})"
            except Exception:
                label = kind

        transform_data = prim.get("transform", {})
        transform = TransformParams(
            sx=float(transform_data.get("sx", 1.0)),
            sy=float(transform_data.get("sy", 1.0)),
            theta=float(transform_data.get("theta", 0.0)),
            dx=float(transform_data.get("dx", 0.0)),
            dy=float(transform_data.get("dy", 0.0)),
        )

        vertices_arr = None
        if poly_vertices is not None:
            try:
                vertices_arr = np.asarray(poly_vertices, dtype=float)
            except Exception:
                vertices_arr = None

        gene = PrimitiveGene(
            kind=kind,
            transform=transform,
            color_rgb=np.asarray(color_tuple, dtype=float),
            polygon_vertices=vertices_arr,
        )

        geometry = gene_to_shapely(gene)
        return TreeNode(
            kind="primitive",
            label=label,
            color_rgb=color_tuple,
            children=[],
            primitive_payload=prim,
            geometry=geometry,
        )

    op = node_obj.get("op", "operator")
    children = [_parse_tree(child) for child in node_obj.get("children", [])]
    label = str(op).lower().strip()
    return TreeNode(kind="op", label=label, color_rgb=None, children=children)


def _compute_depth(root: TreeNode) -> int:
    if not root.children:
        return 1
    return 1 + max(_compute_depth(child) for child in root.children)


def _assign_positions(root: TreeNode, y_step: float = 1.6) -> None:
    """Assign x/y positions to each node for plotting."""

    def assign_x(node: TreeNode, depth: int, next_x: float) -> Tuple[float, List[float]]:
        if not node.children:
            node.x = next_x
            node.y = -depth * y_step
            return next_x + 1.0, [node.x]
        x_positions: List[float] = []
        for child in node.children:
            next_x, child_xs = assign_x(child, depth + 1, next_x)
            x_positions.extend(child_xs)
        node.x = sum(x_positions) / len(x_positions)
        node.y = -depth * y_step
        return next_x, [node.x]

    assign_x(root, 0, 0.0)


def _prepare_preview_geometry(geometry: Optional[BaseGeometry], center_x: float, center_y: float, radius: float) -> Optional[BaseGeometry]:
    if geometry is None:
        return None
    try:
        if getattr(geometry, "is_empty", True):
            return None

        minx, miny, maxx, maxy = geometry.bounds
        width = maxx - minx
        height = maxy - miny
        max_extent = max(width, height)
        scale_factor = (radius * 1.5) / max_extent if max_extent > 1e-6 else radius * 0.8

        centroid = geometry.centroid
        normalized = affinity.translate(geometry, xoff=-centroid.x, yoff=-centroid.y)
        scaled = affinity.scale(normalized, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        return affinity.translate(scaled, xoff=center_x, yoff=center_y)
    except Exception as exc:
        print(f"Warning: Could not normalize geometry preview: {exc}")
        return None


def _flatten_polygons(geometry: Optional[BaseGeometry]) -> List[ShapelyPolygon]:
    if geometry is None or getattr(geometry, "is_empty", True):
        return []
    if isinstance(geometry, ShapelyPolygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [poly for poly in geometry.geoms if not poly.is_empty]
    if isinstance(geometry, GeometryCollection):
        polys: List[ShapelyPolygon] = []
        for geom in geometry.geoms:
            polys.extend(_flatten_polygons(geom))
        return polys
    return []


def _polygon_to_path(polygon: ShapelyPolygon) -> Optional[Path]:
    if polygon.is_empty:
        return None

    vertices: List[Tuple[float, float]] = []
    codes: List[int] = []

    def _add_ring(coords) -> None:
        ring = list(coords)
        if len(ring) < 2:
            return
        for idx, coord in enumerate(ring):
            vertices.append((float(coord[0]), float(coord[1])))
            if idx == 0:
                codes.append(int(Path.MOVETO))
            elif idx == len(ring) - 1:
                codes.append(int(Path.CLOSEPOLY))
            else:
                codes.append(int(Path.LINETO))

    _add_ring(polygon.exterior.coords)
    for interior in polygon.interiors:
        _add_ring(interior.coords)

    if not vertices:
        return None

    return Path(np.asarray(vertices, dtype=float), codes)


def _geometry_to_patches(
    geometry: Optional[BaseGeometry],
    color: Tuple[float, float, float],
    clip_circle: Circle,
) -> List[PathPatch]:
    patches: List[PathPatch] = []
    polygons = _flatten_polygons(geometry)
    if not polygons:
        return patches

    face_color = (
        float(np.clip(color[0], 0.0, 1.0)),
        float(np.clip(color[1], 0.0, 1.0)),
        float(np.clip(color[2], 0.0, 1.0)),
    )

    for poly in polygons:
        path = _polygon_to_path(poly)
        if path is None:
            continue
        patch = PathPatch(path, facecolor=face_color, edgecolor="none", lw=0, zorder=3, fill=True)
        patch.set_clip_path(clip_circle)
        patches.append(patch)

    return patches


def _draw_tree(
    root: TreeNode,
    accent_color: str,
    node_radius: float = 0.22,
    font_size: int = 7,
) -> Tuple[Figure, FigureCanvas]:
    """Create a Matplotlib figure representing the composition tree."""
    leaves: List[TreeNode] = []

    def collect_leaves(node: TreeNode) -> None:
        if not node.children:
            leaves.append(node)
        for child in node.children:
            collect_leaves(child)

    collect_leaves(root)

    if leaves:
        x_min = min(leaf.x for leaf in leaves) - 1.0
        x_max = max(leaf.x for leaf in leaves) + 1.0
    else:
        x_min, x_max = -1.0, 1.0

    depth = _compute_depth(root)
    y_min = -depth * 1.6 - 0.6
    y_max = 0.6
    width = x_max - x_min
    height = y_max - y_min
    fig_w = max(4.0, min(18.0, 0.9 * width * 1.2))
    fig_h = max(3.0, min(12.0, 0.9 * height * 0.9))

    fig = Figure(figsize=(fig_w, fig_h))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    fig.patch.set_facecolor("#2d2d2d")
    ax.set_facecolor("#2d2d2d")

                                             
    accent_rgb = _hex_to_rgb_01(accent_color)
    op_edge_color = (0.8, 0.8, 0.8)
    line_color = (0.6, 0.6, 0.6)
    shadow_color = (0, 0, 0, 0.2)
    shadow_offset = 0.03                                      
    node_rounding = 0.1                                      
                                 

    symbol_font = {"family": "DejaVu Sans", "size": font_size + 10, "weight": "bold"}
    symbol_map = {
        "union": "∪",
        "intersection": "∩",
        "difference": "−",
        "complement": "¬",
        "smooth_union": "S∪",
        "smooth_intersection": "S∩",
        "smooth_difference": "S−",
    }

                                                 

                                             
    def draw_node_recursive(node: TreeNode, ax) -> None:
        
                                        
        for i, child in enumerate(node.children):
            
                                                     
                                        
            curve_rad = -0.3 
            
                                                                 
            if child.x < node.x:
                curve_rad = 0.3 
                                                                      
                                                                        
            
                                   

            conn = ConnectionPatch(
                xyA=(node.x, node.y),                     
                xyB=(child.x, child.y),                  
                coordsA="data",
                coordsB="data",
                axesA=ax,
                axesB=ax,
                arrowstyle="-",
                                                                       
                shrinkA=node_radius * 1.8, 
                shrinkB=node_radius * 2.2,
                                               
                connectionstyle=f"arc3,rad={curve_rad}", 
                color=line_color,
                linewidth=1.5,
                zorder=1,
            )
            ax.add_patch(conn)

                                             
        if node.kind == "op":
            box_width = node_radius * 2.2
            box_height = node_radius * 1.8
            
                                     
            shadow_box = FancyBboxPatch(
                (node.x - box_width / 2 + shadow_offset, node.y - box_height / 2 - shadow_offset),
                box_width,
                box_height,
                boxstyle=f"round,pad=0.0,rounding_size={node_rounding}",
                facecolor=shadow_color,
                edgecolor='none',
                linewidth=0,
                zorder=2, 
            )
            ax.add_patch(shadow_box)
            
                                     
            box = FancyBboxPatch(
                (node.x - box_width / 2, node.y - box_height / 2),
                box_width,
                box_height,
                boxstyle=f"round,pad=0.0,rounding_size={node_rounding}",
                facecolor=accent_rgb,
                edgecolor=op_edge_color,
                linewidth=1.5,
                zorder=3, 
            )
            ax.add_patch(box)
            
            display_label = symbol_map.get(node.label, node.label.upper())
            ax.text(
                node.x,
                node.y,
                display_label,
                ha="center",
                va="center",
                fontdict=symbol_font,
                zorder=4,
                color="white",
            )
        
                                              
        else:
            half = node_radius * 2.2

            frame = Circle(
                (node.x, node.y),
                radius=half,
                facecolor="none",
                edgecolor=op_edge_color,
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(frame)

            clip_circle = Circle((node.x, node.y), radius=half, transform=ax.transData)

            try:
                preview_geom = _prepare_preview_geometry(node.geometry, node.x, node.y, half)
                color = node.color_rgb or (0.7, 0.7, 0.7)
                patches = _geometry_to_patches(preview_geom, color, clip_circle)
                                            
                                                                 
                white_bg_circle = Circle(
                    (node.x, node.y),
                    radius=half,
                    facecolor='white',                  
                    edgecolor='none',
                    zorder=1.5                              
                )
                ax.add_patch(white_bg_circle)
                if patches:
                    for patch in patches:
                        ax.add_patch(patch)
                else:
                    raise ValueError("Empty geometry preview")
            except Exception as exc:
                print(f"Warning: Failed to render primitive thumbnail: {exc}")
                ax.text(
                    node.x,
                    node.y,
                    node.label,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    zorder=4,
                    color="white",
                )

                              
        for child in node.children:
            draw_node_recursive(child, ax)
                                               

                                               
    draw_node_recursive(root, ax)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.1)

    return fig, canvas


def render_tree_png_bytes(
    composition_dict: dict,
    dpi: int = 150,
    accent_color: str = "#4e9cff",
) -> bytes:
    """Render a composition dictionary to PNG bytes."""
    if not composition_dict:
        return b""

    root = _parse_tree(composition_dict)
    _assign_positions(root)

    fig, canvas = _draw_tree(root, accent_color=accent_color)
    canvas.draw()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        return buffer.getvalue()
    finally:
        buffer.close()


def save_tree_image(
    composition_dict: dict,
    output_path: str,
    dpi: int = 300,
    accent_color: str = "#4e9cff",
) -> None:
    """Render a composition tree and save it to disk."""
    if not composition_dict:
        return

    root = _parse_tree(composition_dict)
    _assign_positions(root)
    fig, canvas = _draw_tree(root, accent_color=accent_color)
    canvas.draw()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())