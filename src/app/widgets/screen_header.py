from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal, Qt
                                                           

class ScreenHeader(QWidget):
    """Small reusable header with title, back button and optional action button.
    (Cleaned of styles, ready to be styled by a parent stylesheet)
    """
    back_clicked = Signal()
    action_clicked = Signal()

    def __init__(self, title: str, action_label: str | None = None, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)                                  

        self.title = QLabel(title)
        self.title.setObjectName("header_title")                     
        layout.addWidget(self.title)
        layout.addStretch()

        self.btn_back = QPushButton("Back to Menu")
        self.btn_back.setCursor(Qt.CursorShape.PointingHandCursor)                   
        self.btn_back.setObjectName("back_button")                     
        self.btn_back.clicked.connect(self.back_clicked.emit)
        layout.addWidget(self.btn_back)

        self.action_button = None
        if action_label:
            self.action_button = QPushButton(action_label)
            self.action_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.action_button.setObjectName("action_button")                     
            self.action_button.clicked.connect(self.action_clicked.emit)
            layout.addWidget(self.action_button)