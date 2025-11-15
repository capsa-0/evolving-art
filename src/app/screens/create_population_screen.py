import io
import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QPushButton,
    QHBoxLayout,
    QFrame,
    QScrollArea,
    QGridLayout,
    QApplication,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QColor

from src.core.evolution import GAConfig, initialize_population
from src.rendering.vectorizer import save_genome_as_png
from src.app.theme import VisualConfig
from ..widgets.screen_header import ScreenHeader

class CreatePopulationScreen(QWidget):
    """Population creation screen styled consistently with other top-level views."""

    population_created = Signal(dict)
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("create_population_screen")

        accent = VisualConfig.color_accent
        accent_hover = QColor(accent).lighter(125).name()
        bg_color = "#1a1a1a"
        card_bg = "#202020"
        border_color = "#333333"
        text_primary = "#f0f0f0"
        text_secondary = "#b0b0b0"

        outline_button_style = (
            f"""
            QPushButton {{
                font-size: 15px;
                font-weight: 600;
                color: {accent};
                background-color: transparent;
                border: 2px solid {accent};
                border-radius: 8px;
                padding: 10px 22px;
            }}
            QPushButton:hover {{
                background-color: {accent_hover};
                color: #111;
            }}
            """
        )

        secondary_button_style = (
            f"""
            QPushButton {{
                font-size: 13px;
                font-weight: 600;
                color: {accent};
                background-color: transparent;
                border: 1px solid {accent};
                border-radius: 6px;
                padding: 6px 16px;
                min-width: 110px;
            }}
            QPushButton:hover {{
                background-color: {accent_hover};
                color: #111;
            }}
            """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.header = ScreenHeader("Create New Population")
        self.header.btn_back.setText("Back to Populations")
        self.header.back_clicked.connect(self._on_cancel)
        self.header.action_clicked.connect(self.update_preview)
        main_layout.addWidget(self.header)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(60, 40, 60, 40)
        content_layout.setSpacing(40)
        main_layout.addWidget(content, 1)

        form_card = QFrame()
        form_card.setObjectName("form_card")
        form_card.setMinimumWidth(360)
        form_card.setMaximumWidth(420)
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(32, 32, 32, 32)
        form_layout.setSpacing(18)

        section_title = QLabel("Population Settings")
        section_title.setObjectName("section_title")
        form_layout.addWidget(section_title)

        def make_form_label(text: str) -> QLabel:
            label = QLabel(text)
            label.setObjectName("form_label")
            return label

        def add_field_block(label_text: str, widget: QWidget) -> None:
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(6)
            container_layout.addWidget(make_form_label(label_text))
            container_layout.addWidget(widget)
            form_layout.addWidget(container)

        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("My_Experiment_01")
        add_field_block("Name", self.edit_name)

        self.spin_pop = QSpinBox()
        self.spin_pop.setRange(4, 1000)
        self.spin_pop.setValue(18)
        add_field_block("Population size", self.spin_pop)

        self.spin_genes = QSpinBox()
        self.spin_genes.setRange(1, 100)
        self.spin_genes.setValue(4)
        add_field_block("Primitives per genome", self.spin_genes)

        self.edit_seed = QLineEdit()
        self.edit_seed.setText("42")

        seed_row_widget = QWidget()
        seed_row_layout = QHBoxLayout(seed_row_widget)
        seed_row_layout.setContentsMargins(0, 0, 0, 0)
        seed_row_layout.setSpacing(8)
        seed_row_layout.addWidget(self.edit_seed, 1)

        self.btn_random_seed = QPushButton("Random")
        self.btn_random_seed.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_random_seed.setStyleSheet(secondary_button_style)
        self.btn_random_seed.clicked.connect(self._generate_random_seed)
        seed_row_layout.addWidget(self.btn_random_seed)
        add_field_block("Seed", seed_row_widget)


        self.btn_start = QPushButton("Create population")
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setStyleSheet(outline_button_style)
        form_layout.addWidget(self.btn_start, alignment=Qt.AlignmentFlag.AlignLeft)
        form_layout.addStretch()

        content_layout.addWidget(form_card)

        preview_card = QFrame()
        preview_card.setObjectName("preview_card")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(32, 32, 32, 32)
        preview_layout.setSpacing(24)

        preview_header = QLabel("Preview")
        preview_header.setObjectName("section_title")
        preview_layout.addWidget(preview_header)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)

        self.btn_preview = QPushButton("Preview Initial Population")
        self.btn_preview.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_preview.setStyleSheet(outline_button_style)
        self.btn_preview.clicked.connect(self.update_preview)
        controls_layout.addWidget(self.btn_preview, 0, Qt.AlignmentFlag.AlignLeft)
        controls_layout.addStretch()
        preview_layout.addLayout(controls_layout)

        self.preview_title = QLabel()
        self.preview_title.setObjectName("preview_title")
        self.preview_title.setVisible(False)
        preview_layout.addWidget(self.preview_title)

        self.preview_hint = QLabel("Generate a preview to see the initial individuals before starting.")
        self.preview_hint.setObjectName("helper_text")
        self.preview_hint.setWordWrap(True)
        preview_layout.addWidget(self.preview_hint)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setObjectName("preview_scroll")
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
        preview_layout.addWidget(self.preview_scroll, 1)

        self.preview_container = QWidget()
        self.preview_container.setObjectName("preview_container")
        self.preview_grid = QGridLayout(self.preview_container)
        self.preview_grid.setContentsMargins(12, 12, 12, 12)
        self.preview_grid.setSpacing(20)
        self.preview_scroll.setWidget(self.preview_container)

        content_layout.addWidget(preview_card, 1)

        self.btn_start.clicked.connect(self.accept_data)

        self.setStyleSheet(
            f"""
            #create_population_screen {{
                background-color: {bg_color};
            }}

            ScreenHeader {{
                background-color: {bg_color};
                border-bottom: 2px solid {border_color};
            }}

            ScreenHeader #header_title {{
                font-size: 40px;
                font-weight: 800;
                color: {text_primary};
                background-color: transparent;
            }}

            ScreenHeader #back_button,
            ScreenHeader #action_button {{
                font-size: 15px;
                font-weight: bold;
                color: {accent};
                background-color: transparent;
                border: 2px solid {accent};
                border-radius: 8px;
                padding: 10px 22px;
            }}

            ScreenHeader #back_button:hover,
            ScreenHeader #action_button:hover {{
                background-color: {accent_hover};
                color: #111;
            }}

            QFrame#form_card,
            QFrame#preview_card {{
                background-color: {card_bg};
                border: 1px solid {border_color};
                border-radius: 18px;
            }}

            QLabel#section_title {{
                font-size: 24px;
                font-weight: 700;
                color: {text_primary};
                margin-bottom: 4px;
            }}

            QLabel#form_label {{
                font-size: 15px;
                color: {text_secondary};
                margin-bottom: 4px;
            }}

            QLabel#preview_title {{
                font-size: 18px;
                font-weight: 600;
                color: {text_primary};
            }}

            QLabel#helper_text {{
                font-size: 13px;
                color: {text_secondary};
                background-color: transparent;
            }}

            QLineEdit, QSpinBox {{
                background-color: #151515;
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 15px;
                color: {text_primary};
                selection-background-color: {accent};
                selection-color: #111;
            }}

            QScrollArea#preview_scroll {{
                border: none;
                background-color: transparent;
            }}

            QWidget#preview_container {{
                background-color: transparent;
            }}
            """
        )

    def _clear_preview(self) -> None:
        while self.preview_grid.count():
            item = self.preview_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.preview_title.setVisible(False)
        self.preview_hint.setVisible(True)

    def _generate_random_seed(self) -> None:
        random_seed = np.random.randint(0, 1_000_000)
        self.edit_seed.setText(str(random_seed))

    def reset_parameters(self) -> None:
        self.edit_name.blockSignals(True)
        self.spin_pop.blockSignals(True)
        self.spin_genes.blockSignals(True)
        self.edit_seed.blockSignals(True)

        self.edit_name.clear()
        self.spin_pop.setValue(16)
        self.spin_genes.setValue(4)
        self.edit_seed.clear()

        self.edit_name.blockSignals(False)
        self.spin_pop.blockSignals(False)
        self.spin_genes.blockSignals(False)
        self.edit_seed.blockSignals(False)

        self._clear_preview()

    def _on_cancel(self) -> None:
        self.reset_parameters()
        self.cancel_requested.emit()

    def update_preview(self) -> None:
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            self._clear_preview()

            pop_size = self.spin_pop.value()
            genes = self.spin_genes.value()
            seed_text = self.edit_seed.text().strip()
            seed = int(seed_text) if seed_text.isdigit() else None

            cfg = GAConfig(
                population_size=pop_size,
                num_genes=genes,
                random_seed=seed if seed is not None else np.random.randint(0, 1_000_000),
            )
            _, population = initialize_population(cfg)

            image_size = 180
            render_resolution = 400
            spacing = self.preview_grid.spacing()

                                                                 
            available_width = self.preview_scroll.viewport().width()
            margins = self.preview_grid.contentsMargins()
            content_width = available_width - margins.left() - margins.right()

                                                        
            if content_width <= image_size:
                cols = 1
            else:
                                                                                 
                item_total_width = image_size + spacing
                cols = max(1, (content_width + spacing) // item_total_width)

            self.preview_hint.setVisible(False)

            for index, genome in enumerate(population):
                label = QLabel()
                label.setFixedSize(image_size, image_size)
                label.setStyleSheet(
                    "background-color: #151515; border: 1px solid #2d2d2d; border-radius: 10px;"
                )
                label.setScaledContents(True)

                try:
                    pil_image = save_genome_as_png(genome, filename=None, resolution=render_resolution)
                    if pil_image:
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format="PNG")
                        buffer.seek(0)
                        pixmap = QPixmap()
                        pixmap.loadFromData(buffer.read())
                        label.setPixmap(pixmap)
                    else:
                        placeholder = QPixmap(image_size, image_size)
                        placeholder.fill(QColor("darkred"))
                        label.setPixmap(placeholder)
                except Exception as exc:                                         
                    print(f"Error rendering individual {index}: {exc}")
                    placeholder = QPixmap(image_size, image_size)
                    placeholder.fill(QColor("darkred"))
                    label.setPixmap(placeholder)

                row = index // cols
                col = index % cols
                self.preview_grid.addWidget(label, row, col)

            self.preview_scroll.verticalScrollBar().setValue(0)

        except Exception as exc:                                         
            print(f"Error updating preview: {exc}")
        finally:
            QApplication.restoreOverrideCursor()


    def accept_data(self) -> None:
        seed_val = self.edit_seed.text().strip()
        name_val = self.edit_name.text().strip()
        params = {
            "pop_name": name_val if name_val else "Unnamed_Pop",
            "pop_size": self.spin_pop.value(),
            "genes": self.spin_genes.value(),
            "seed": int(seed_val) if seed_val.isdigit() else None,
        }
        self.population_created.emit(params)