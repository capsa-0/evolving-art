from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, Qt
from src.app.theme import VisualConfig

class IndividualWidget(QFrame):
    """Card-style widget that shows an individual and related controls.

    Styled with the accent color for visual consistency.
    """
    toggled_selection = Signal(int, bool) 

    def __init__(self, index, pixmap, parent=None):
        super().__init__(parent)
        self.index = index
        self.is_selected = False
        
                        
        self.ACCENT_COLOR = VisualConfig.color_accent
        self.DEFAULT_BORDER = "#3d3d3d"
        self.DEFAULT_TEXT = "#888"
        
        self.setFixedSize(220, 270)
        self.setStyleSheet(f"""
            QFrame {{ 
                background-color: #2d2d2d; 
                border-radius: 8px; 
                border: 2px solid {self.DEFAULT_BORDER}; 
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        self.img_label = QLabel()
        self.img_label.setFixedSize(206, 206)
        self.img_label.setPixmap(pixmap)
        self.img_label.setScaledContents(True)
        self.img_label.setStyleSheet("border: none; border-radius: 4px; background-color: transparent;")
        self.img_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.img_label)
        
        bottom_layout = QHBoxLayout()
        self.lbl_id = QLabel(f"#{index}")
        self.lbl_id.setStyleSheet(f"font-weight: bold; border: none; color: {self.DEFAULT_TEXT}; background-color: transparent;")
        self.lbl_id.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        bottom_layout.addWidget(self.lbl_id)
        bottom_layout.addStretch()
        
                                      
        btn_style = f"""
            QPushButton {{
                font-size: 10px; 
                font-weight: bold;
                color: {self.ACCENT_COLOR}; 
                background-color: transparent; 
                border: 1px solid {self.ACCENT_COLOR}; 
                border-radius: 4px;
                padding: 3px 8px;
            }}
            QPushButton:hover {{
                background-color: {self.ACCENT_COLOR}; 
                color: #111; 
                border: 1px solid {self.ACCENT_COLOR};
            }}
        """
        
        self.btn_png = QPushButton("PNG")
        self.btn_png.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_png.setStyleSheet(btn_style)
        
        self.btn_svg = QPushButton("SVG")
        self.btn_svg.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_svg.setStyleSheet(btn_style)
        
        bottom_layout.addWidget(self.btn_png)
        bottom_layout.addWidget(self.btn_svg)
        layout.addLayout(bottom_layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_select()
            
    def toggle_select(self):
        self.is_selected = not self.is_selected
        
                                                  
        color = self.ACCENT_COLOR if self.is_selected else self.DEFAULT_BORDER
        text_color = self.ACCENT_COLOR if self.is_selected else self.DEFAULT_TEXT
        
        self.setStyleSheet(f"""
            QFrame {{ 
                background-color: #2d2d2d; 
                border-radius: 8px; 
                border: 2px solid {color}; 
            }}
        """)
        self.lbl_id.setStyleSheet(f"""
            color: {text_color}; 
            font-weight: bold; 
            border: none; 
            background-color: transparent;
        """)
        self.toggled_selection.emit(self.index, self.is_selected)