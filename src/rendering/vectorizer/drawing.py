"""Drawing layer for vectorizer: matplotlib-based rendering and file saving."""
import gc
import io
import math
from typing import Any, Optional, Sequence, Tuple, cast

from PIL import Image, ImageDraw, ImageFont
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from .geometry import process_node


def draw_genome_on_axis(ax: Axes, genome: Any) -> None:
    """Draw a genome on a matplotlib axis."""
    from shapely.geometry import box

    shapes_and_colors = process_node(genome.root)

    all_geoms = [s[0] for s in shapes_and_colors if not s[0].is_empty]

    if not all_geoms:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        return

    import shapely.ops

                                                                           
                                                                            
    cleaned = []
    for g in all_geoms:
        try:
            g2 = None
            if not getattr(g, 'is_valid', True):
                try:
                    g2 = g.buffer(0)
                except Exception:
                    g2 = None
            else:
                g2 = g

            if g2 is not None and not g2.is_empty and getattr(g2, 'area', 0) > 0:
                cleaned.append(g2)
        except Exception:
            pass

                                                                     
                                                      
    try:
        if cleaned:
            mins_x = []
            mins_y = []
            maxs_x = []
            maxs_y = []
            for g in cleaned:
                bx0, by0, bx1, by1 = g.bounds
                mins_x.append(bx0)
                mins_y.append(by0)
                maxs_x.append(bx1)
                maxs_y.append(by1)
            minx = min(mins_x)
            miny = min(mins_y)
            maxx = max(maxs_x)
            maxy = max(maxs_y)
        else:
            raise ValueError("no geoms")
    except Exception:
        minx, miny, maxx, maxy = -1, -1, 1, 1

    width = maxx - minx
    height = maxy - miny
    if width < 1e-6:
        width = 1.0
        minx -= 0.5
        maxx += 0.5
    if height < 1e-6:
        height = 1.0
        miny -= 0.5
        maxy += 0.5

    max_dim = max(width, height)
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    margin = 0.1
    half_side = (max_dim / 2) + margin

    clip_xmin = center_x - half_side
    clip_xmax = center_x + half_side
    clip_ymin = center_y - half_side
    clip_ymax = center_y + half_side

    clip_box = box(clip_xmin, clip_ymin, clip_xmax, clip_ymax)

    ax.set_aspect('equal')
    ax.set_xlim(clip_xmin, clip_xmax)
    ax.set_ylim(clip_ymin, clip_ymax)
    ax.axis('off')

    import numpy as np
    from shapely.geometry import Polygon

    for geom, rgb in shapes_and_colors:
        rgb = np.array(rgb).flatten()
        rgba = np.append(rgb[:3], 1.0)

        if hasattr(geom, 'geoms'):
            sub_geoms = geom.geoms
        else:
            sub_geoms = [geom]

        for part in sub_geoms:
            try:
                clipped_part = part.intersection(clip_box)
            except Exception:
                continue
            if clipped_part.is_empty:
                continue

            if hasattr(clipped_part, 'geoms'):
                draw_geoms = clipped_part.geoms
            else:
                draw_geoms = [clipped_part]

            for final_part in draw_geoms:
                if isinstance(final_part, Polygon):
                    try:
                        x, y = final_part.exterior.xy
                        ax.fill(x, y, fc=rgba, ec=rgba, linewidth=0.5, joinstyle='round')

                        for interior in final_part.interiors:
                            xi, yi = interior.xy
                            ax.fill(xi, yi, fc='white', ec=None)
                    except Exception:
                        pass


def save_genome_as_svg(genome: Any, filename: str) -> None:
    """Save a genome as SVG."""
    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    draw_genome_on_axis(ax, genome)

    canvas.draw()

    try:
        fig.savefig(
            filename,
            format='svg',
            bbox_inches='tight',
            transparent=True,
            pad_inches=0
        )
    except Exception as e:
        print(f"Error saving SVG: {e}")
    finally:
        fig.clear()
        del ax
        del canvas
        del fig
        gc.collect()


def save_genome_as_png(
    genome: Any,
    filename: Optional[str] = None,
    resolution: int = 128,
) -> Optional[Image.Image]:
    """Save a genome as PNG or return PIL Image."""
    DPI_CALC = resolution / 3.0

    fig = Figure(figsize=(3, 3))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    draw_genome_on_axis(ax, genome)

    canvas.draw()

    return_image: Optional[Image.Image] = None

    if filename is None:
        buffer = io.BytesIO()
        try:
            fig.savefig(
                buffer,
                format='png',
                dpi=DPI_CALC,
                bbox_inches='tight',
                pad_inches=0,
                transparent=False,
                facecolor='white'
            )
            buffer.seek(0)
            return_image = Image.open(buffer).convert("RGBA")
        except Exception as e:
            print(f"Error saving PNG to buffer: {e}")
            return_image = Image.new('RGBA', (resolution, resolution), 'white')
    else:
        try:
            fig.savefig(
                filename,
                format='png',
                dpi=DPI_CALC,
                bbox_inches='tight',
                pad_inches=0,
                transparent=False,
                facecolor='white'
            )
        except Exception as e:
            print(f"Error saving PNG to file: {e}")

              
    fig.clear()
    del ax
    del canvas
    del fig
    gc.collect()

    return return_image


def genome_to_png_bytes(genome: Any, resolution: int = 128) -> bytes:
    """Return the rendered genome as raw PNG bytes."""

    image = save_genome_as_png(genome, filename=None, resolution=resolution)
    if image is None:
        return b""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_to_file(
    genome: Any,
    out_path: Optional[str],
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    resolution: int = 600,
    figsize: Optional[tuple[float, float]] = None,
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
    format: Optional[str] = None,
    transparent: bool = False,
    return_image: bool = False,
) -> Optional[Image.Image]:
    """Compatibility wrapper that renders using vectorizer helpers."""

    fmt = (format or "png").lower()

    if fmt == "png":
        if return_image:
            return save_genome_as_png(genome, filename=None, resolution=resolution)
        if out_path is None:
            raise ValueError("'out_path' is required when return_image is False.")
        save_genome_as_png(genome, filename=out_path, resolution=resolution)
        return None

    if fmt == "svg":
        if return_image:
            raise ValueError("SVG rendering cannot return an in-memory image.")
        if out_path is None:
            raise ValueError("'out_path' is required for SVG rendering.")
        save_genome_as_svg(genome, filename=out_path)
        return None

    raise ValueError(f"Unsupported format '{fmt}'. Only 'png' and 'svg' are available.")


def render_population_grid(
    population: Sequence[Any],
    out_path: Optional[str],
    cols: int = 4,
    resolution: int = 128,
    figsize_per_cell: Optional[tuple[float, float]] = None,
    draw_edges: bool = False,
    use_vector: bool = True,
    return_image: bool = False,
    cell_padding: int = 24,
) -> Optional[Image.Image]:
    """Render a grid of genomes to a PNG file or return it as an Image."""

    if not population:
        raise ValueError("Population is empty; nothing to render.")

    cols = max(1, cols)
    rows = math.ceil(len(population) / cols)

    cell_images: list[Image.Image] = []
    for genome in population:
        img = save_genome_as_png(genome, filename=None, resolution=resolution)
        if img is None:
            img = Image.new("RGBA", (resolution, resolution), "white")
        else:
            img = img.convert("RGBA")
        cell_images.append(img)

    cell_width, cell_height = cell_images[0].size
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    label_padding = 6
    bbox_measure = getattr(font, "getbbox", None)
    if callable(bbox_measure):
        bbox_zero = cast(Tuple[int, int, int, int], bbox_measure("0"))
        label_height = int((bbox_zero[3] - bbox_zero[1]) + label_padding)
    else:
        metrics_fn = getattr(font, "getmetrics", None)
        if callable(metrics_fn):
            ascent, descent = cast(Tuple[int, int], metrics_fn())
            label_height = int((ascent + descent) + label_padding)
        else:
            label_height = int(18 + label_padding)

    grid_width = int(cols * cell_width + (cols + 1) * cell_padding)
    grid_height = int(rows * (cell_height + label_height) + (rows + 1) * cell_padding)

    grid_image = Image.new("RGBA", (grid_width, grid_height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(grid_image)

    for idx, img in enumerate(cell_images):
        row = idx // cols
        col = idx % cols
        origin_x = int(cell_padding + col * (cell_width + cell_padding))
        origin_y = int(cell_padding + row * (cell_height + label_height + cell_padding))
        grid_image.paste(img, (origin_x, origin_y), mask=img)

        label = str(idx)
        text_bbox_fn = getattr(draw, "textbbox", None)
        if callable(text_bbox_fn):
            text_bbox = cast(Tuple[int, int, int, int], text_bbox_fn((0, 0), label, font=font))
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        else:
            mask = font.getmask(label)
            text_w, text_h = mask.size
        label_x = int(origin_x + (cell_width - text_w) / 2)
        label_y = int(origin_y + cell_height + (label_padding // 2))
        draw.text((label_x, label_y), label, fill=(0, 0, 0, 255), font=font)

    if out_path:
        grid_image.convert("RGB").save(out_path, format="PNG")

    if return_image or not out_path:
        return grid_image
    return None
