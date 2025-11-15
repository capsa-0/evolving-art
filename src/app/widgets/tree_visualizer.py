"""Qt-friendly helpers around rendering composition trees."""

from PySide6.QtGui import QImage, QPixmap

from src.app.theme import VisualConfig
from src.rendering.tree_renderer import render_tree_png_bytes, save_tree_image


def render_tree_to_pixmap(composition_dict: dict, dpi: int = 150) -> QPixmap:
    """Render a composition dictionary into a QPixmap for display."""
    if not composition_dict:
        return QPixmap()

    png_bytes = render_tree_png_bytes(
        composition_dict,
        dpi=dpi,
        accent_color=VisualConfig.color_accent,
    )
    if not png_bytes:
        return QPixmap()

    q_image = QImage()
    q_image.loadFromData(png_bytes)
    return QPixmap.fromImage(q_image)


def save_tree_to_file(composition_dict: dict, out_path: str, dpi: int = 300) -> None:
    """Render and persist a composition tree to disk."""
    if not composition_dict:
        return

    try:
        save_tree_image(
            composition_dict,
            output_path=out_path,
            dpi=dpi,
            accent_color=VisualConfig.color_accent,
        )
    except Exception as exc:
        print(f"Error saving tree: {exc}")
