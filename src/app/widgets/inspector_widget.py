from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton, QApplication
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QColor
import os

from ..widgets.mutation_panel import MutationPanel
from ..widgets import tree_visualizer as TreeViz
from ..theme import VisualConfig
from src.population_manager.backend_adapter import GUI_IMAGE_RES
from src.rendering.vectorizer.drawing import render_population_grid


class InspectorWidget(QFrame):
    """Encapsulates the individual inspector panel together with the mutation panel.

    Provides public methods update_inspector(index), clear_inspector(), set_generation(gen)
    and emits a `status_message` signal to notify the container.
    """

    status_message = Signal(str, int)

    def __init__(self, backend_adapter, parent=None):
        super().__init__(parent)
        self.setFixedWidth(400)
        self.setObjectName("inspector_panel")

        self.backend = backend_adapter
        self.generation = 0

        accent = VisualConfig.color_accent
        accent_hover = QColor(accent).lighter(120).name()
        bg_color = "#161616"
        border_color = "#2d2d2d"
        text_primary = "#f0f0f0"
        text_secondary = "#b0b0b0"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(14)

        stats_group = QGroupBox("Population Stats")
        stats_group.setObjectName("stats_group")
        stats_form = QFormLayout()
        self.lbl_avg_size = QLabel("N/A")
        self.lbl_avg_size.setObjectName("avg_value_label")
        stats_form.addRow("Avg. Composition:", self.lbl_avg_size)
        stats_group.setLayout(stats_form)

        inspector_group = QGroupBox("Individual Inspector")
        inspector_group.setObjectName("inspector_group")
        tree_layout = QVBoxLayout()
        tree_layout.setSpacing(10)

        ind_stats_form = QFormLayout()
        ind_stats_form.setSpacing(6)
        self.lbl_inspector_index = QLabel("N/A")
        self.lbl_inspector_size = QLabel("N/A")
        self.lbl_inspector_index.setObjectName("inspector_value_label")
        self.lbl_inspector_size.setObjectName("inspector_value_label")

        ind_stats_form.addRow("Inspecting Index:", self.lbl_inspector_index)
        ind_stats_form.addRow("Composition:", self.lbl_inspector_size)

        self.lbl_tree_image = QLabel("Select an individual to view its tree.")
        self.lbl_tree_image.setMinimumHeight(300)
        self.lbl_tree_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_tree_image.setStyleSheet(
            "background-color: #121212; border: 1px solid #2a2a2a; border-radius: 12px; "
            "color: #777; margin-top: 12px; padding: 12px;"
        )

        self.btn_save_tree = QPushButton("Save Tree (PNG)")
        self.btn_save_tree.setEnabled(False)
        self.btn_save_tree.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_tree.setStyleSheet(
            f"""
            QPushButton {{
                font-size: 13px;
                font-weight: 600;
                color: {accent};
                background-color: transparent;
                border: 1px solid {accent};
                border-radius: 8px;
                padding: 8px 18px;
                margin-top: 8px;
            }}
            QPushButton:hover {{
                background-color: {accent_hover};
                color: #111;
            }}
            QPushButton:disabled {{
                border-color: #303030;
                color: #555;
            }}
            """
        )
        self.btn_save_tree.clicked.connect(self._handle_save_tree)

        tree_layout.addLayout(ind_stats_form)
        tree_layout.addWidget(self.lbl_tree_image)
        tree_layout.addWidget(self.btn_save_tree)
        inspector_group.setLayout(tree_layout)

        self.mutation_panel = MutationPanel()
        self.mutation_panel.setObjectName("mutation_panel")
                                                 
        self.mutation_panel.parameter_changed.connect(self.backend.update_mutation_config)

        self.btn_save_generation = QPushButton("Save Generation Grid (PNG)")
        self.btn_save_generation.setEnabled(False)
        self.btn_save_generation.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_generation.setStyleSheet(
            f"""
            QPushButton {{
                font-size: 13px;
                font-weight: 600;
                color: {accent};
                background-color: transparent;
                border: 1px solid {accent};
                border-radius: 8px;
                padding: 8px 18px;
                margin: 6px 0;
            }}
            QPushButton:hover {{
                background-color: {accent_hover};
                color: #111;
            }}
            QPushButton:disabled {{
                border-color: #303030;
                color: #555;
            }}
            """
        )
        self.btn_save_generation.clicked.connect(self._handle_save_generation)

        layout.addWidget(stats_group)
        layout.addWidget(inspector_group)
        layout.addWidget(self.btn_save_generation)
        layout.addWidget(self.mutation_panel)
        layout.addStretch()

        self.setStyleSheet(
            f"""
            QFrame#inspector_panel {{
                background-color: {bg_color};
                border-left: 1px solid {border_color};
            }}

            QGroupBox {{
                color: {text_primary};
                font-size: 15px;
                font-weight: 700;
                border: 1px solid {border_color};
                border-radius: 12px;
                margin-top: 12px;
                padding: 14px;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: {text_primary};
                font-size: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}

            QLabel {{
                color: {text_secondary};
            }}

            QLabel#avg_value_label {{
                color: {text_primary};
                font-size: 16px;
                font-weight: 700;
            }}

            QLabel#inspector_value_label {{
                color: {text_primary};
                font-size: 14px;
                font-weight: 600;
            }}
            """
        )

    def set_generation(self, generation: int):
        self.generation = generation
        self.btn_save_generation.setEnabled(bool(self.backend.population))

    def update_inspector(self, index: int):
        """Renders the tree for the given individual index and updates labels."""
        self.status_message.emit(f"Rendering tree for individual #{index}...", 1000)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            comp_dict = self.backend.get_individual_composition_dict(index)
            genome_size_dict = self.backend.get_individual_genome_size(index)

            prims = genome_size_dict.get("primitives", 0)
            ops = genome_size_dict.get("operations", 0)

            self.lbl_inspector_index.setText(f"#{index}")
            self.lbl_inspector_size.setText(f"Primitives: {prims}  Operators: {ops}")

            if comp_dict:
                pixmap = TreeViz.render_tree_to_pixmap(comp_dict)
                scaled_pixmap = pixmap.scaled(
                    self.lbl_tree_image.width() - 10,
                    400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.lbl_tree_image.setPixmap(scaled_pixmap)
                self.btn_save_tree.setEnabled(True)
            else:
                self.lbl_tree_image.setText(f"Error: No composition data for #{index}.")
                self.btn_save_tree.setEnabled(False)

        except Exception as e:
            print(f"Error in InspectorWidget.update_inspector: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            self.status_message.emit(f"Inspecting individual #{index}.", 4000)

    def clear_inspector(self):
        self.lbl_inspector_index.setText("N/A")
        self.lbl_inspector_size.setText("N/A")
        self.lbl_tree_image.setText("Select an individual to view its tree.")
        self.lbl_tree_image.setPixmap(QPixmap())
        self.btn_save_tree.setEnabled(False)

    def _handle_save_tree(self):
                                                   
                                                                          
        text = self.lbl_inspector_index.text()
        if not text.startswith("#"):
            return
        try:
            index = int(text[1:])
        except ValueError:
            return

        self.status_message.emit(f"Saving tree for individual #{index}...", 2000)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            comp_dict = self.backend.get_individual_composition_dict(index)
            if comp_dict:
                folder = self.backend.dirs["saves"]
                filename = f"tree_gen_{self.generation}_ind_{index}_tree.png"
                full_path = os.path.join(folder, filename)
                TreeViz.save_tree_to_file(comp_dict, full_path)
                self.status_message.emit(f"Saved: {os.path.basename(full_path)}", 4000)
        except Exception as e:
            self.status_message.emit(f"Error saving tree: {str(e)}", 4000)
            print(e)
        finally:
            QApplication.restoreOverrideCursor()

    def _handle_save_generation(self):
        if not self.backend.population:
            self.status_message.emit("No population to save.", 4000)
            return

        saves_dir = self.backend.dirs.get("saves") if hasattr(self.backend, "dirs") else None
        if not saves_dir:
            self.status_message.emit("Population save directory is unavailable.", 4000)
            return

        os.makedirs(saves_dir, exist_ok=True)
        filename = f"grid_gen_{self.generation}_grid.png"
        full_path = os.path.join(saves_dir, filename)

        self.status_message.emit("Rendering generation grid...", 1000)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            render_population_grid(self.backend.population, full_path, resolution=GUI_IMAGE_RES, cols=4)
            self.status_message.emit(f"Saved: {os.path.basename(full_path)}", 4000)
        except Exception as e:
            self.status_message.emit(f"Error saving generation grid: {e}", 4000)
            print(e)
        finally:
            QApplication.restoreOverrideCursor()
