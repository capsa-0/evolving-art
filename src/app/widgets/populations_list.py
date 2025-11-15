from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel
from PySide6.QtCore import Signal, Qt

from ..widgets.population_card import PopulationCardWidget


class PopulationsList(QWidget):
    """Scroll list widget that populates PopulationCardWidget items from a backend.
    (Cleaned of all internal styles - will be styled by its parent)
    """

    load_requested = Signal(str)
    clone_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, backend_adapter, parent=None):
        super().__init__(parent)
        self.backend = backend_adapter

        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 60)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
                            
                                                                                              

        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 0, 0)

        self.list_layout.setSpacing(15)
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.list_container)
        layout.addWidget(self.scroll_area)

    def refresh(self):
                              
        for i in reversed(range(self.list_layout.count())):
            w = self.list_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        populations = self.backend.list_populations()
        if not populations:
            self.show_empty_message()
            return

        for pop_meta in populations:
            card = PopulationCardWidget(pop_meta)
            card.load_requested.connect(self.load_requested.emit)
            card.clone_requested.connect(self.clone_requested.emit)
            card.delete_requested.connect(self.delete_requested.emit)
            self.list_layout.addWidget(card)

    def show_empty_message(self, text="No populations found. Click 'Create New Population' to start."):
                                    
        for i in reversed(range(self.list_layout.count())):
            w = self.list_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
                
        label = QLabel(text)
                                                                        
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.list_layout.addWidget(label)