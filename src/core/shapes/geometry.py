from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import os
import math
import numpy as np


class Shape:
    def sdf(self, point_xy: np.ndarray) -> float:
        raise NotImplementedError

    def contains(self, point_xy: np.ndarray) -> bool:
        dists, _ = self.evaluate_many(np.asarray(point_xy, dtype=float).reshape(1, 2))
        return bool(dists[0] <= 0.0)

    # ---- Composition DSL ----
    def union(self, other: "Shape") -> "UnionN":
        return UnionN(self, other)

    def __or__(self, other: "Shape") -> "UnionN":
        return self.union(other)

    def intersect(self, other: "Shape") -> "IntersectionN":
        return IntersectionN(self, other)

    def __and__(self, other: "Shape") -> "IntersectionN":
        return self.intersect(other)

    def difference(self, other: "Shape") -> "Difference":
        return Difference(self, other)

    def __sub__(self, other: "Shape") -> "Difference":
        return self.difference(other)

    # ---- Transform helpers ----
    def transformed(self, T: "Affine2D") -> "Transformed":
        return Transformed(self, T)

    def translate(self, dx: float, dy: float) -> "Transformed":
        return self.transformed(Affine2D.from_translate(dx, dy))

    def scale(self, sx: float, sy: float | None = None) -> "Transformed":
        return self.transformed(Affine2D.from_scale(sx, sy))

    def rotate(self, theta_radians: float) -> "Transformed":
        return self.transformed(Affine2D.from_rotation(theta_radians))

    # ---- Color / evaluation ----
    def color_at(self, point_xy: np.ndarray) -> Optional[np.ndarray]:
        """
        Optional per-point color in RGB [0,1]. Default: None (no color).
        """
        return None

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Return (sdf, color) at a point. Default uses sdf() and color_at().
        """
        dists, colors = self.evaluate_many(np.asarray(point_xy, dtype=float).reshape(1, 2))
        return float(dists[0]), (None if colors is None else np.asarray(colors[0], dtype=float).reshape(3,))

    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Vectorized evaluation over an array of points with shape (N, 2).
        Subclasses must implement this using NumPy. No scalar fallback.
        """
        raise NotImplementedError("Shapes must implement evaluate_many(points_xy: (N,2))")

    def with_color(self, r: float, g: float, b: float) -> "Colored":
        return Colored(self, np.array([r, g, b], dtype=float))


@dataclass(frozen=True)
class Affine2D:
    """
    2D affine transform x -> A x + t
    """
    A: np.ndarray  # shape (2, 2)
    t: np.ndarray  # shape (2,)

    def __post_init__(self):
        if self.A.shape != (2, 2):
            raise ValueError("A must be 2x2")
        if self.t.shape != (2,):
            raise ValueError("t must be length-2")
        # Precompute inverse for efficient inverse application
        object.__setattr__(self, "_Ainv", np.linalg.inv(self.A))

    def apply(self, point_xy: np.ndarray) -> np.ndarray:
        return self.A @ point_xy + self.t

    def inverse_apply(self, point_xy: np.ndarray) -> np.ndarray:
        return self._Ainv @ (point_xy - self.t)

    # ---- Constructors and composition ----
    @staticmethod
    def identity() -> "Affine2D":
        return Affine2D(A=np.eye(2), t=np.zeros(2))

    @staticmethod
    def from_translate(dx: float, dy: float) -> "Affine2D":
        return Affine2D(A=np.eye(2), t=np.array([dx, dy], dtype=float))

    @staticmethod
    def from_scale(sx: float, sy: float | None = None) -> "Affine2D":
        if sy is None:
            sy = sx
        return Affine2D(A=np.array([[sx, 0.0], [0.0, sy]], dtype=float), t=np.zeros(2))

    @staticmethod
    def from_rotation(theta_radians: float) -> "Affine2D":
        c = math.cos(theta_radians)
        s = math.sin(theta_radians)
        return Affine2D(A=np.array([[c, -s], [s, c]], dtype=float), t=np.zeros(2))

    def then(self, after: "Affine2D") -> "Affine2D":
        """
        First apply self, then apply 'after'.
        y = after.apply(self.apply(x))
        """
        A_new = after.A @ self.A
        t_new = after.A @ self.t + after.t
        return Affine2D(A=A_new, t=t_new)


class Transformed(Shape):
    def __init__(self, shape: Shape, transform: Affine2D):
        self.shape = shape
        self.transform = transform

    def sdf(self, point_xy: np.ndarray) -> float:
        # Pullback: evaluate base shape at inverse-mapped point
        return self.shape.sdf(self.transform.inverse_apply(point_xy))

    def color_at(self, point_xy: np.ndarray) -> Optional[np.ndarray]:
        # Delegate to underlying shape's evaluation for color to preserve composites' colors
        d, c = self.shape.evaluate(self.transform.inverse_apply(point_xy))
        return c

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        # Pullback and fully delegate to shape.evaluate to keep color algebra intact
        pin = self.transform.inverse_apply(point_xy)
        return self.shape.evaluate(pin)

    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        # Vectorized inverse apply: x' = Ainv @ (x - t)
        Ainv = self.transform._Ainv
        t = self.transform.t
        pin = (Ainv @ (pts.T - t[:, None])).T
        return self.shape.evaluate_many(pin)


class UnitSquare(Shape):
    """
    Axis-aligned square centered at origin with side length 1.
    """
    def sdf(self, point_xy: np.ndarray) -> float:
        q = np.abs(point_xy) - 0.5
        outside = np.maximum(q, 0.0)
        # distance to outside + inside max component (if inside)
        return float(np.linalg.norm(outside, ord=2) + min(max(q[0], q[1]), 0.0))

    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        q = np.abs(pts) - 0.5
        outside = np.maximum(q, 0.0)
        outside_norm = np.linalg.norm(outside, axis=1)
        q_max = np.maximum(q[:, 0], q[:, 1])
        inside_term = np.minimum(q_max, 0.0)
        dists = outside_norm + inside_term
        return dists.astype(float, copy=False), None


class UnitDisk(Shape):
    """
    Unit circle centered at origin.
    """
    def sdf(self, point_xy: np.ndarray) -> float:
        return float(np.linalg.norm(point_xy) - 1.0)

    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        dists = np.linalg.norm(pts, axis=1) - 1.0
        return dists.astype(float, copy=False), None

class Polygon(Shape):
    """
    Simple polygon defined by an ordered list of vertices (2D points).
    Works for convex or concave simple polygons. Orientation can be CW or CCW.
    Signed distance is negative inside, positive outside.
    """
    def __init__(self, vertices: list[np.ndarray] | np.ndarray):
        verts = np.array(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            raise ValueError("Polygon requires an array/list of N>=3 vertices of shape (N,2)")
        self.vertices = verts

    def sdf(self, point_xy: np.ndarray) -> float:
        p = np.array(point_xy, dtype=float).reshape(2,)
        verts = self.vertices
        n = verts.shape[0]
        # Distance to edges
        min_dist2 = float("inf")
        # Point-in-polygon (ray casting)
        inside = False
        px, py = p[0], p[1]
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            # Segment distance
            ab = b - a
            ap = p - a
            denom = ab @ ab
            if denom > 0.0:
                t = max(0.0, min(1.0, (ap @ ab) / denom))
            else:
                t = 0.0
            closest = a + t * ab
            d2 = float(np.sum((p - closest) ** 2))
            if d2 < min_dist2:
                min_dist2 = d2
            # Ray casting for inside
            xi, yi = a[0], a[1]
            xj, yj = b[0], b[1]
            intersects = ((yi > py) != (yj > py))
            if intersects:
                # Compute x coordinate of intersection with horizontal ray y=py
                denom_y = (yj - yi) if (yj - yi) != 0.0 else 1e-18
                x_int = (xj - xi) * (py - yi) / denom_y + xi
                if px < x_int:
                    inside = not inside
        dist = math.sqrt(min_dist2)
        return -dist if inside else dist

    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        n_pts = pts.shape[0]
        verts = self.vertices
        n = verts.shape[0]
        # Initialize
        min_dist2 = np.full(n_pts, np.inf, dtype=float)
        inside = np.zeros(n_pts, dtype=bool)
        px = pts[:, 0]
        py = pts[:, 1]
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            ab = b - a  # (2,)
            denom = float(ab @ ab)
            ap = pts - a  # (N,2)
            if denom > 0.0:
                t = np.clip((ap @ ab) / denom, 0.0, 1.0)  # (N,)
            else:
                t = np.zeros(n_pts, dtype=float)
            closest = a + t[:, None] * ab  # (N,2)
            diff = pts - closest
            d2 = np.sum(diff * diff, axis=1)
            min_dist2 = np.minimum(min_dist2, d2)
            # Ray casting toggle
            xi, yi = a[0], a[1]
            xj, yj = b[0], b[1]
            intersects = ((yi > py) != (yj > py))
            denom_y = (yj - yi)
            denom_y = denom_y if denom_y != 0.0 else 1e-18
            x_int = (xj - xi) * (py - yi) / denom_y + xi
            inside ^= (intersects & (px < x_int))
        d = np.sqrt(min_dist2)
        d[inside] *= -1.0
        return d, None

class Colored(Shape):
    """
    Wrapper that assigns a constant RGB color to a shape.
    """
    def __init__(self, shape: Shape, rgb: np.ndarray):
        self.shape = shape
        rgb = np.array(rgb, dtype=float).reshape(3,)
        self.rgb = np.clip(rgb, 0.0, 1.0)

    def sdf(self, point_xy: np.ndarray) -> float:
        return self.shape.sdf(point_xy)

    def color_at(self, point_xy: np.ndarray) -> Optional[np.ndarray]:
        return self.rgb

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        d, _ = self.shape.evaluate_many(np.asarray(point_xy, dtype=float).reshape(1, 2))
        return float(d[0]), self.rgb

    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        # We use the underlying SDF, ignoring its color, to match scalar behavior.
        try:
            dists, _ = self.shape.evaluate_many(pts)
        except Exception:
            dists = np.array([self.shape.sdf(p) for p in pts], dtype=float)
        colors = np.tile(self.rgb.reshape(1, 3), (pts.shape[0], 1))
        return dists, colors


class HalfSpace(Shape):
    """
    { x : n·x + c <= 0 }, with n normalized
    """
    def __init__(self, normal_xy: np.ndarray, c: float):
        n = np.array(normal_xy, dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0.0:
            raise ValueError("HalfSpace normal must be non-zero")
        self.n = n / norm
        self.c = float(c)

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(self.n @ point_xy + self.c)

    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        d = (pts @ self.n) + self.c
        return d.astype(float, copy=False), None


def _to_color(c: Optional[np.ndarray]) -> np.ndarray:
    if c is None:
        return np.zeros(3, dtype=float)
    c = np.array(c, dtype=float).reshape(3,)
    return np.clip(c, 0.0, 1.0)

def _to_color_array(c: Optional[np.ndarray], n: int) -> np.ndarray:
    if c is None:
        return np.zeros((n, 3), dtype=float)
    arr = np.asarray(c, dtype=float)
    if arr.ndim == 1:
        arr = np.tile(arr.reshape(1, 3), (n, 1))
    return np.clip(arr.reshape(n, 3), 0.0, 1.0)


class ColorAlgebra:
    """
    Channel-wise color algebra in [0,1]^3.
    Default uses:
      - OR (union): 1 - (1-a)(1-b)  (probabilistic sum / screen-like)
      - AND (intersection): a * b   (product t-norm)
      - DIFF (a minus b): a * (1 - b)
    """
    @staticmethod
    def or_color(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return 1.0 - (1.0 - a) * (1.0 - b)

    @staticmethod
    def and_color(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    @staticmethod
    def diff_color(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * (1.0 - b)


DEFAULT_COLOR_ALGEBRA = ColorAlgebra()


class UnionN(Shape):
    def __init__(self, *shapes: Shape):
        # flatten nested UnionN
        flat: list[Shape] = []
        for s in shapes:
            if isinstance(s, UnionN):
                flat.extend(s.shapes)
            else:
                flat.append(s)
        if len(flat) == 0:
            raise ValueError("UnionN requires at least one shape")
        self.shapes = flat

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(min(s.sdf(point_xy) for s in self.shapes))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        # SDF: min; Color: OR-combine colors of inside contributors
        ds: list[float] = []
        color_acc = np.zeros(3, dtype=float)
        for s in self.shapes:
            d, c = s.evaluate(point_xy)
            ds.append(d)
            if d <= 0.0:
                color_acc = DEFAULT_COLOR_ALGEBRA.or_color(color_acc, _to_color(c))
        return float(min(ds)), color_acc

    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        n = pts.shape[0]
        if len(self.shapes) == 1:
            return self.shapes[0].evaluate_many(pts)
        dist_stack = []
        color_acc = np.zeros((n, 3), dtype=float)
        for s in self.shapes:
            d, c = s.evaluate_many(pts)
            dist_stack.append(d)
            inside = d <= 0.0
            ci = _to_color_array(c, n)
            combined = DEFAULT_COLOR_ALGEBRA.or_color(color_acc, ci)
            color_acc = np.where(inside[:, None], combined, color_acc)
        dmin = np.min(np.vstack(dist_stack), axis=0)
        return dmin, color_acc


class Union(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(min(self.a.sdf(point_xy), self.b.sdf(point_xy)))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        da, ca = self.a.evaluate(point_xy)
        db, cb = self.b.evaluate(point_xy)
        d = float(min(da, db))
        color = np.zeros(3, dtype=float)
        if da <= 0.0:
            color = DEFAULT_COLOR_ALGEBRA.or_color(color, _to_color(ca))
        if db <= 0.0:
            color = DEFAULT_COLOR_ALGEBRA.or_color(color, _to_color(cb))
        return d, color
    
    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        da, ca = self.a.evaluate_many(pts)
        db, cb = self.b.evaluate_many(pts)
        d = np.minimum(da, db)
        n = pts.shape[0]
        color_a = np.where((da <= 0.0)[:, None], _to_color_array(ca, n), 0.0)
        color_b = np.where((db <= 0.0)[:, None], _to_color_array(cb, n), 0.0)
        color = DEFAULT_COLOR_ALGEBRA.or_color(color_a, color_b)
        return d, color


class IntersectionN(Shape):
    def __init__(self, *shapes: Shape):
        # flatten nested IntersectionN
        flat: list[Shape] = []
        for s in shapes:
            if isinstance(s, IntersectionN):
                flat.extend(s.shapes)
            else:
                flat.append(s)
        if len(flat) == 0:
            raise ValueError("IntersectionN requires at least one shape")
        self.shapes = flat

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(max(s.sdf(point_xy) for s in self.shapes))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        # SDF: max; Color: AND-combine colors of inside contributors
        ds: list[float] = []
        color_acc = np.ones(3, dtype=float)
        any_outside = False
        for s in self.shapes:
            d, c = s.evaluate(point_xy)
            ds.append(d)
            if d <= 0.0:
                color_acc = DEFAULT_COLOR_ALGEBRA.and_color(color_acc, _to_color(c))
            else:
                any_outside = True
        # If any contributor is outside, intersection is outside -> color shouldn't matter
        # but AND with zeros already pushes toward zeros; keep it consistent.
        return float(max(ds)), (np.zeros(3, dtype=float) if any_outside else color_acc)

    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        n = pts.shape[0]
        if len(self.shapes) == 1:
            return self.shapes[0].evaluate_many(pts)
        dist_stack = []
        color_acc = np.ones((n, 3), dtype=float)
        any_outside = np.zeros(n, dtype=bool)
        for s in self.shapes:
            d, c = s.evaluate_many(pts)
            dist_stack.append(d)
            inside = d <= 0.0
            ci = _to_color_array(c, n)
            combined = DEFAULT_COLOR_ALGEBRA.and_color(color_acc, ci)
            color_acc = np.where(inside[:, None], combined, color_acc)
            any_outside |= ~inside
        dmax = np.max(np.vstack(dist_stack), axis=0)
        color = np.where(any_outside[:, None], 0.0, color_acc)
        return dmax, color


class Intersection(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(max(self.a.sdf(point_xy), self.b.sdf(point_xy)))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        da, ca = self.a.evaluate(point_xy)
        db, cb = self.b.evaluate(point_xy)
        d = float(max(da, db))
        if da <= 0.0 and db <= 0.0:
            color = DEFAULT_COLOR_ALGEBRA.and_color(_to_color(ca), _to_color(cb))
        else:
            color = np.zeros(3, dtype=float)
        return d, color
    
    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        da, ca = self.a.evaluate_many(pts)
        db, cb = self.b.evaluate_many(pts)
        d = np.maximum(da, db)
        n = pts.shape[0]
        both_inside = (da <= 0.0) & (db <= 0.0)
        color_a = _to_color_array(ca, n)
        color_b = _to_color_array(cb, n)
        color = DEFAULT_COLOR_ALGEBRA.and_color(color_a, color_b)
        color = np.where(both_inside[:, None], color, 0.0)
        return d, color


class Difference(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(max(self.a.sdf(point_xy), -self.b.sdf(point_xy)))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        da, ca = self.a.evaluate(point_xy)
        db, cb = self.b.evaluate(point_xy)
        d = float(max(da, -db))
        # Color: A AND NOT B  -> a * (1 - b)
        color_a = _to_color(ca) if da <= 0.0 else np.zeros(3, dtype=float)
        color_b = _to_color(cb) if db <= 0.0 else np.zeros(3, dtype=float)
        color = DEFAULT_COLOR_ALGEBRA.diff_color(color_a, color_b)
        return d, color

    def evaluate_many(self, points_xy: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        da, ca = self.a.evaluate_many(pts)
        db, cb = self.b.evaluate_many(pts)
        d = np.maximum(da, -db)
        n = pts.shape[0]
        color_a = np.where((da <= 0.0)[:, None], _to_color_array(ca, n), 0.0)
        color_b = np.where((db <= 0.0)[:, None], _to_color_array(cb, n), 0.0)
        color = DEFAULT_COLOR_ALGEBRA.diff_color(color_a, color_b)
        return d, color


def sample_sdf_grid(
    shape: Shape,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample an SDF on a regular grid suitable for contouring.
    """
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dists, _ = shape.evaluate_many(pts)
    Z = dists.reshape(X.shape)
    return X, Y, Z


