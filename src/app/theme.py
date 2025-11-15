from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt

class VisualConfig:
    color_accent = "#9669ff" 

def set_modern_theme(app):
    """Apply a modern dark 'Fusion' theme to the application."""
    app.setStyle("Fusion")
    palette = QPalette()
    dark_bg = QColor(30, 30, 30)
    light_bg = QColor(45, 45, 45)
    accent = QColor(42, 130, 218)
    text = QColor(230, 230, 230)
    

    palette.setColor(QPalette.ColorRole.Window, dark_bg)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, light_bg)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark_bg)
    palette.setColor(QPalette.ColorRole.Button, light_bg)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)
    
    app.setStyleSheet("""
        QMainWindow { background-color: #1e1e1e; }
        QScrollArea { border: none; background-color: #1e1e1e; }
        QWidget { font-family: 'Segoe UI', sans-serif; font-size: 14px; }
        QDialog { background-color: #252525; }
        QPushButton {
            background-color: #3d3d3d; border: 1px solid #555;
            border-radius: 5px; padding: 8px 15px; color: #eee;
        }
        QPushButton:hover { background-color: #4d4d4d; border-color: #777; }
        QPushButton:pressed { background-color: #9669ff; color: #fff; }
        QLabel { color: #ccc; }
        QLineEdit, QSpinBox { 
            background-color: #151515; color: #eee; 
            border: 1px solid #555; padding: 4px; border-radius: 4px;
        }
        QStatusBar { color: #aaa; background-color: #252525; }
    """
)