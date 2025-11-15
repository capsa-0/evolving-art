from PySide6.QtWidgets import QWidget, QVBoxLayout, QMessageBox, QInputDialog, QLineEdit
from PySide6.QtCore import Signal
from src.app.theme import VisualConfig

from ..widgets.screen_header import ScreenHeader
from ..widgets.populations_list import PopulationsList

class PopulationsScreen(QWidget):
    """Screen that manages the list of populations. Uses ScreenHeader and PopulationsList widgets.
    """

    create_requested = Signal()
    load_requested = Signal(str)
    menu_requested = Signal()

    def __init__(self, backend_adapter, parent=None):
        super().__init__(parent)
        self.backend = backend_adapter

                               
        BG_COLOR = "#1a1a1a"
        PRIMARY_TEXT_COLOR = "#ffffff"
        SECONDARY_TEXT_COLOR = "#aaaaaa"
        TERTIARY_TEXT_COLOR = "#777777"
        BORDER_COLOR = "#333"
        
        ACCENT_COLOR = VisualConfig.color_accent

                                                              
        self.setObjectName("populations_screen")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)                    
        main_layout.setSpacing(0)                                   

                       
        self.header = ScreenHeader("Populations", action_label="Create New Population")
        self.header.back_clicked.connect(self.menu_requested.emit)
        self.header.action_clicked.connect(self.create_requested.emit)
        main_layout.addWidget(self.header)

                                 
        self.pop_list = PopulationsList(self.backend)
        self.pop_list.load_requested.connect(self.load_requested.emit)
        self.pop_list.clone_requested.connect(self.handle_clone_population)
        self.pop_list.delete_requested.connect(self.handle_delete_population)
        main_layout.addWidget(self.pop_list, 1)                            

                                               
                               
        self.setStyleSheet(f"""
            /* --- Root Screen --- */
            #populations_screen {{
                background-color: {BG_COLOR};
            }}

            /* --- Header Widget --- */
            ScreenHeader {{
                background-color: {BG_COLOR};
                border-bottom: 2px solid {BORDER_COLOR}; 
            }}
            
            /* Header Title (targets the objectName) */
            ScreenHeader #header_title {{
                font-size: 80px;
                font-weight: 800;
                color: {PRIMARY_TEXT_COLOR};
                background-color: transparent;
            }}
            
            /* Header Back Button (targets the objectName) */
            ScreenHeader #back_button {{
                font-size: 16px;
                font-weight: bold;
                color: {ACCENT_COLOR};
                background-color: transparent;
                border: 2px solid {ACCENT_COLOR};
                border-radius: 8px;
                padding: 12px 20px;
            }}
            ScreenHeader #back_button:hover {{
                background-color: {ACCENT_COLOR};
                color: #111;
            }}
            
            /* Header Action Button (targets the objectName) */
            ScreenHeader #action_button {{
                font-size: 16px; 
                font-weight: bold;
                color: {ACCENT_COLOR}; 
                background-color: transparent; 
                border: 2px solid {ACCENT_COLOR}; 
                border-radius: 8px;
                padding: 12px 20px;
            }}
            ScreenHeader #action_button:hover {{
                background-color: {ACCENT_COLOR}; 
                color: #111; 
            }}

            /* --- List Widget Container --- */
            PopulationsList {{
                background-color: transparent;
            }}
            
            /* Empty Message Label */
            PopulationsList QLabel {{
                font-size: 18px;
                color: {TERTIARY_TEXT_COLOR};
                background-color: transparent;
                padding: 40px;
                qproperty-alignment: 'AlignCenter';
            }}
            
            /* Scroll Area (reapplies the removed style) */
            QScrollArea {{
                border: none; 
                background-color: transparent;
            }}
            
            /* --- Scrollbar --- */
            QScrollBar:vertical {{
                border: none;
                background: {BG_COLOR}; 
                width: 14px;
                margin: 0px 0px 0px 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {TERTIARY_TEXT_COLOR};
                min-height: 20px;
                border-radius: 7px;
                border: 3px solid {BG_COLOR}; 
            }}
            QScrollBar::handle:vertical:hover {{
                background: {SECONDARY_TEXT_COLOR};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)

    def refresh_population_list(self):
        try:
            self.pop_list.refresh()
        except Exception as e:
            print(f"Error refreshing population list: {e}")
            self.pop_list.show_empty_message(f"Error: {e}")

    def show_empty_message(self, text="No populations found. Click 'Create New Population' to start."):
        self.pop_list.show_empty_message(text)

                                                                
    def handle_delete_population(self, pop_name):
        reply = QMessageBox.warning(
            self,
            "Confirm Deletion",
            f"Are you sure you want to permanently delete the population:\n\n<b>{pop_name}</b>?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.backend.delete_population(pop_name)
                self.refresh_population_list()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete population:\n{e}")

    def handle_clone_population(self, original_name):
        new_name, ok = QInputDialog.getText(
            self,
            "Clone Population",
            f"Enter a new name for the clone of:\n<b>{original_name}</b>",
            QLineEdit.EchoMode.Normal,
            f"{original_name}_clone",
        )

        if ok and new_name:
            if not new_name.strip():
                QMessageBox.warning(self, "Invalid Name", "The population name cannot be empty.")
                return
            try:
                self.backend.clone_population(original_name, new_name)
                self.refresh_population_list()
            except FileExistsError:
                QMessageBox.critical(self, "Error", f"A population with the name '{new_name}' already exists.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not clone population:\n{e}")