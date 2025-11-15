"""
Geometry helpers for vectorizer: geometry fixing, safe operations, and node processing.
"""
from typing import List, Tuple, Any, Optional
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import shapely.ops

GRID_SIZE = 0.001


def fix_geom(geom: Any) -> Any:
    """Conservatively stabilize geometry. Avoid risky GEOS calls; return empty Polygon on failure."""
    if geom is None or getattr(geom, 'is_empty', True):
        return Polygon()

                                                                               
    try:
        if getattr(geom, 'is_valid', True) and geom.geom_type in ['Polygon', 'MultiPolygon'] and not geom.is_empty:
            return geom
    except Exception:
        return Polygon()

                                                                                              
    try:
        repaired = geom.buffer(0)
        if repaired is None or getattr(repaired, 'is_empty', True):
            return Polygon()
        geom = repaired
    except Exception:
        return Polygon()

                                
    try:
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            return geom if not geom.is_empty else Polygon()
        elif geom.geom_type == 'GeometryCollection':
            polys = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
            if not polys:
                return Polygon()
            return shapely.ops.unary_union(polys)
    except Exception:
        return Polygon()

    return Polygon()


def safe_intersection(a: Any, b: Any) -> Any:
    """Safely intersect two geometries."""
    if a.is_empty or b.is_empty:
        return Polygon()

    try:
        res = a.intersection(b)
    except Exception as e:
        try:
            res = fix_geom(a).intersection(fix_geom(b))
        except Exception as e2:
            return Polygon()

    return fix_geom(res)


def safe_difference(a: Any, b: Any) -> Any:
    """Safely subtract geometry b from geometry a."""
    if a.is_empty:
        return Polygon()
    if b.is_empty:
        return a

    try:
        res = a.difference(b)
    except Exception as e:
        try:
            res = fix_geom(a).difference(fix_geom(b))
        except Exception as e2:
            return a

    return fix_geom(res)


def gene_to_shapely(gene: Any) -> Any:
    """Convert a PrimitiveGene to a Shapely geometry."""
    if gene.kind == "disk":
        base = __import__('shapely.geometry', fromlist=['Point']).Point(0, 0).buffer(1.0, resolution=32)
    elif gene.kind == "square":
        base = __import__('shapely.geometry', fromlist=['box']).box(-0.5, -0.5, 0.5, 0.5)
    elif gene.kind == "polygon":
        try:
            base = Polygon(gene.polygon_vertices)
        except Exception:
            return Polygon()
    else:
        raise ValueError(f"Unknown kind {gene.kind}")

    base = fix_geom(base)

    t = gene.transform
    try:
        geom = affinity.scale(base, xfact=t.sx, yfact=t.sy, origin=(0, 0))
        geom = affinity.rotate(geom, t.theta, origin=(0, 0), use_radians=True)
        geom = affinity.translate(geom, xoff=t.dx, yoff=t.dy)
    except Exception:
        return Polygon()

    return fix_geom(geom)


def process_node(node: Any) -> List[Tuple[Any, np.ndarray]]:
    """
    Recursively process a genome node (PrimitiveNode or OpNode) and return
    list of (geometry, color) tuples.
    """
    from src.core.evolution.genome import PrimitiveNode, OpNode
    from src.core.shapes.geometry import ColorAlgebra

    COLOR_ALG = ColorAlgebra()

    if isinstance(node, PrimitiveNode):
        geom = gene_to_shapely(node.gene)
        if geom.is_empty:
            return []
        return [(geom, node.gene.color_rgb)]

    elif isinstance(node, OpNode):
        children_results = []
        for child in node.children:
            res = process_node(child)
            if res:
                children_results.append(res)

        if not children_results:
            return []

        if node.kind == "union":
            current_patches = children_results[0]
            for next_child_list in children_results[1:]:
                for (new_geom, new_col) in next_child_list:
                    current_patches = _shatter_and_blend_with_color_alg(
                        current_patches, new_geom, new_col, blend_mode="or", color_alg=COLOR_ALG
                    )
            return current_patches

        if node.kind == "intersection":
            current_shapes = children_results[0]
            for next_child_list in children_results[1:]:
                new_shapes = []
                for (geom_a, col_a) in current_shapes:
                    for (geom_b, col_b) in next_child_list:
                        inter = safe_intersection(geom_a, geom_b)
                        if not inter.is_empty:
                            new_col = COLOR_ALG.and_color(col_a, col_b)
                            new_shapes.append((inter, new_col))
                current_shapes = new_shapes
            return current_shapes

        if node.kind == "difference":
            current_shapes = children_results[0]
            subtractors = []
            for child_list in children_results[1:]:
                for geom, _ in child_list:
                    if not geom.is_empty:
                        subtractors.append(geom)

            if subtractors:
                try:
                    total_subtractor = shapely.ops.unary_union(subtractors)
                    total_subtractor = fix_geom(total_subtractor)
                except Exception:
                    return current_shapes

                if total_subtractor.is_empty:
                    return current_shapes

                final_shapes = []
                for (geom_a, col_a) in current_shapes:
                    diff = safe_difference(geom_a, total_subtractor)
                    if not diff.is_empty:
                        final_shapes.append((diff, col_a))
                return final_shapes
            else:
                return current_shapes

    return []


def _shatter_and_blend_with_color_alg(
    existing_patches: List[Tuple[Any, np.ndarray]],
    new_geom: Any,
    new_color: np.ndarray,
    blend_mode: str = "or",
    color_alg: Any = None,
) -> List[Tuple[Any, np.ndarray]]:
    """Internal helper: blend new geometry with existing patches."""
    if color_alg is None:
        from src.core.shapes.geometry import ColorAlgebra
        color_alg = ColorAlgebra()

    next_gen_patches = []
    shape_to_add = fix_geom(new_geom)

    if shape_to_add.is_empty:
        return existing_patches

    for exist_geom, exist_color in existing_patches:
        if shape_to_add.is_empty:
            next_gen_patches.append((exist_geom, exist_color))
            continue

        intersection = safe_intersection(exist_geom, shape_to_add)

        if intersection.is_empty:
            next_gen_patches.append((exist_geom, exist_color))
        else:
            exist_non_overlap = safe_difference(exist_geom, shape_to_add)
            if not exist_non_overlap.is_empty:
                next_gen_patches.append((exist_non_overlap, exist_color))

            if blend_mode == "or":
                blended_color = color_alg.or_color(exist_color, new_color)
            elif blend_mode == "and":
                blended_color = color_alg.and_color(exist_color, new_color)
            else:
                blended_color = new_color

            if not intersection.is_empty:
                next_gen_patches.append((intersection, blended_color))

            shape_to_add = safe_difference(shape_to_add, exist_geom)

            if shape_to_add.is_empty:
                pass

    if not shape_to_add.is_empty:
        next_gen_patches.append((shape_to_add, new_color))

    return next_gen_patches
