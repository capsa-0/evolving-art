"""Clean PopulationCardWidget rewrite ensuring label text is always visible."""

from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont, QCursor, QColor, QPainter, QPen

from src.app.theme import VisualConfig


class ForcedLabel(QLabel):
    """QLabel subclass that forces custom color/font rendering to bypass stylesheet conflicts."""

    def __init__(self, text: str, color: QColor, point_size: int, bold: bool = False, parent=None):
        super().__init__(text, parent)
        self._color = color
        self._font = QFont()
        if point_size > 0:
            self._font.setPointSize(point_size)
        self._font.setBold(bold)
        self.setStyleSheet("background: transparent;")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event):              
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(QPen(self._color))
        painter.setFont(self._font)
        alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(self.rect(), int(alignment), self.text())

class PopulationCardWidget(QFrame):
    load_requested = Signal(str)
    clone_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, metadata: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("population_card")
        self.pop_name = metadata.get('name', 'Unknown')

                
        accent = VisualConfig.color_accent
        bg_card = QColor(45,45,45)
        border = QColor(68,68,68)
        col_primary = QColor(238,238,238)
        col_secondary = QColor(170,170,170)
        col_tertiary = QColor(119,119,119)

        self.setMinimumHeight(110)
        self.setMaximumHeight(140)
                                                                                                                        
        self.setAutoFillBackground(False)

        main = QHBoxLayout(self)
        horizontal_padding = 60                                                              
        main.setContentsMargins(horizontal_padding, 15, horizontal_padding, 15)
        main.setSpacing(25)

        info = QVBoxLayout()
        info.setContentsMargins(0, 0, 0, 0)
        info.setSpacing(4)

        def make(text, name, size, color: QColor, bold=False):
            lbl = ForcedLabel(text, color, size, bold, parent=self)
            lbl.setObjectName(name)
            return lbl

        name_lbl = make(self.pop_name, 'name_label', 18, col_primary, bold=True)

        mean = metadata.get('mean_current_genes', {'primitives': -1, 'operations': -1})
        if isinstance(mean, dict):
            p = mean.get('primitives', -1)
            o = mean.get('operations', -1)
            mean_str = f"Primitives: {p:.1f}, Operators: {o:.1f}" if p >=0 and o>=0 else 'N/A'
        else:
            mean_str = 'N/A'
        meta_text = (
            f"Size: {metadata.get('size','N/A')} | "
            f"Current Gen: {metadata.get('current_gen','N/A')} | "
            f"Mean Composition: {mean_str}"
        )
        meta_lbl = make(meta_text, 'meta_label', 14, col_secondary)
        path_lbl = make(metadata.get('path',''), 'path_label', 12, col_tertiary)

        info.addWidget(name_lbl)
        info.addWidget(meta_lbl)
        info.addWidget(path_lbl)
        main.addLayout(info)
        main.addStretch(1)

                 
        btn_box = QHBoxLayout()
        btn_box.setContentsMargins(0, 0, 0, 0)
        btn_box.setSpacing(10)
                                          
        style_btn = (
            "QPushButton { background: transparent; border-radius: 8px; padding: 8px 14px; font-weight: 600; }"
            f"QPushButton {{ color: {accent}; border: 2px solid {accent}; }}"
            f"QPushButton:hover {{ background: {accent}; color: #111111; }}"
        )
        self.btn_clone = QPushButton('Clone')
        self.btn_delete = QPushButton('Delete')
        self.btn_load = QPushButton('Load')
        for b in (self.btn_clone, self.btn_delete, self.btn_load):
            b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            b.setStyleSheet(style_btn)
            btn_box.addWidget(b)
        main.addLayout(btn_box)

                                                                                                       
        self.setStyleSheet(
            f"#population_card {{ background-color: {bg_card.name()}; border:1px solid {border.name()}; border-radius:8px; }}"
            f"#population_card:hover {{ border-color:{accent}; }}"
        )

                 
        self.btn_load.clicked.connect(lambda: self.load_requested.emit(self.pop_name))
        self.btn_clone.clicked.connect(lambda: self.clone_requested.emit(self.pop_name))
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self.pop_name))
