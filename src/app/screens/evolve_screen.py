import os
import sys 
import psutil 
from functools import partial
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QScrollArea, QGridLayout, QFrame, QMessageBox, QApplication,
    QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QColor

from src.app.theme import VisualConfig
from ..widgets.individual_widget import IndividualWidget
from ..widgets.inspector_widget import InspectorWidget

class EvolveScreen(QWidget):
    """
    Main evolution screen with multi-selection support and an inspector panel.
    """
    new_population_requested = Signal()
    status_message = Signal(str, int) 

    def __init__(self, backend_adapter, parent=None):
        super().__init__(parent)
        
        self.process = psutil.Process(os.getpid())

        self.backend = backend_adapter
        self.generation = 0
        self.pop_name = ""
        self.selected_indices = set()
        self.current_selected_index = None

        accent = VisualConfig.color_accent
        accent_hover = QColor(accent).lighter(125).name()
        bg_color = "#1a1a1a"
        panel_bg = "#191919"
        border_color = "#2d2d2d"
        text_primary = "#f0f0f0"
        text_secondary = "#b0b0b0"

        primary_button_style = (
            f"""
            QPushButton {{
                font-size: 15px;
                font-weight: 700;
                color: #111;
                background-color: {accent};
                border: none;
                border-radius: 10px;
                padding: 10px 26px;
                min-width: 200px;
            }}
            QPushButton:hover {{
                background-color: {accent_hover};
            }}
            QPushButton:disabled {{
                background-color: #333;
                color: #666;
            }}
            """
        )

        outline_button_style = (
            f"""
            QPushButton {{
                font-size: 14px;
                font-weight: 600;
                color: {accent};
                background-color: transparent;
                border: 1px solid {accent};
                border-radius: 9px;
                padding: 9px 20px;
                min-width: 170px;
            }}
            QPushButton:hover {{
                background-color: {accent_hover};
                color: #111;
            }}
            QPushButton:disabled {{
                border-color: #3a3a3a;
                color: #555;
            }}
            """
        )

        self.setObjectName("evolve_screen")

                         
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        left_panel = QFrame()
        left_panel.setObjectName("left_panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        header = QFrame()
        header.setObjectName("evolve_header")
        header.setFixedHeight(76)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        self.lbl_pop_name = QLabel("Population")
        self.lbl_pop_name.setObjectName("population_label")

        self.lbl_gen = QLabel("Gen: 0")
        self.lbl_gen.setObjectName("generation_label")

        header_layout.addWidget(self.lbl_pop_name)
        header_layout.addWidget(self.lbl_gen)
        header_layout.addStretch()

        btn_reset = QPushButton("Back to Menu")
        btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset.setStyleSheet(outline_button_style)
        btn_reset.clicked.connect(self.new_population_requested.emit)
        header_layout.addWidget(btn_reset)

        self.btn_prev_gen = QPushButton("◀ Previous Gen")
        self.btn_prev_gen.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_prev_gen.clicked.connect(self.go_previous_generation)
        self.btn_prev_gen.setStyleSheet(outline_button_style)
        self.btn_prev_gen.setMinimumHeight(44)
        header_layout.addWidget(self.btn_prev_gen)

        self.btn_evolve = QPushButton("Evolve Next Gen ▶")
        self.btn_evolve.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_evolve.clicked.connect(self.evolve_step)
        self.btn_evolve.setStyleSheet(primary_button_style)
        self.btn_evolve.setMinimumHeight(44)
        header_layout.addWidget(self.btn_evolve)
        left_layout.addWidget(header)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("population_scroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none;")
        
        self.grid_container = QWidget()
        self.grid_container.setObjectName("grid_container")
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(15)
        self.grid_layout.setContentsMargins(30, 30, 30, 30)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
                                            
        self.loading_label = QLabel("Generating next generation...")
        self.loading_label.setObjectName("loading_label")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.hide()
        self.grid_layout.addWidget(self.loading_label, 0, 0)
        
        self.scroll_area.setWidget(self.grid_container)
        left_layout.addWidget(self.scroll_area)

                                                                                    
        self.inspector_panel = InspectorWidget(self.backend)
                                                                      
        self.inspector_panel.status_message.connect(self.status_message.emit)

        main_layout.addWidget(left_panel, stretch=3)
        main_layout.addWidget(self.inspector_panel, stretch=1)

        self.setStyleSheet(
            f"""
            #evolve_screen {{
                background-color: {bg_color};
            }}

            QFrame#left_panel {{
                background-color: {panel_bg};
            }}

            QFrame#evolve_header {{
                background-color: {bg_color};
                border-bottom: 1px solid {border_color};
            }}

            QLabel#population_label {{
                font-size: 16px;
                font-weight: 600;
                color: {text_secondary};
                text-transform: uppercase;
                letter-spacing: 1px;
            }}

            QLabel#generation_label {{
                font-size: 24px;
                font-weight: 800;
                color: {text_primary};
                margin-left: 12px;
            }}

            QWidget#grid_container {{
                background-color: transparent;
            }}

            QLabel#loading_label {{
                font-size: 24px;
                font-weight: 700;
                color: {text_secondary};
                background-color: #202020;
                border: 1px dashed {border_color};
                border-radius: 12px;
                padding: 60px;
            }}

            QScrollArea#population_scroll {{
                background-color: transparent;
            }}
            QScrollArea#population_scroll QWidget {{
                background-color: transparent;
            }}

            QLabel {{
                color: {text_secondary};
            }}
            """
        )

    def set_population_data(self, pop_name, generation):
        self.pop_name = pop_name
        self.generation = generation

        self.lbl_pop_name.setText(self.pop_name)
        self.lbl_gen.setText(f"Gen: {self.generation}")
        self.inspector_panel.set_generation(self.generation)

                                  
        self.render_population_grid()                             
        self.update_population_stats()
        self.inspector_panel.clear_inspector()

                                                              
        try:
            self.inspector_panel.mutation_panel.update_ui(self.backend.get_mutation_config())
        except Exception:
                                                                       
            pass

        self.btn_evolve.setEnabled(True)
        self.btn_prev_gen.setEnabled(self.generation > 0)

    def update_population_stats(self):
        try:
            avg_size_dict = self.backend.get_average_genome_size()
            prims = avg_size_dict.get("primitives", 0.0)
            ops = avg_size_dict.get("operations", 0.0)
            
                                                              
            try:
                self.inspector_panel.lbl_avg_size.setText(f"P: {prims:.1f} | O: {ops:.1f}")
            except Exception:
                pass
        except Exception as e:
            print(f"Error calculating average size: {e}")
            try:
                self.inspector_panel.lbl_avg_size.setText("Error")
            except Exception:
                pass

                               
    def render_population_grid(self):
        """
        Clear the grid and redraw it by asking the backend to render all previews in parallel.
        """
                              
        self.loading_label.hide()

                                               
        widgets_to_delete = []
        for i in reversed(range(self.grid_layout.count())): 
            item = self.grid_layout.itemAt(i)
            if item:
                w = item.widget()
                if w and w != self.loading_label:                              
                    widgets_to_delete.append(w)

        if widgets_to_delete:
            for w in widgets_to_delete:
                w.setParent(None)
                w.deleteLater() 

        self.selected_indices.clear()
        self.current_selected_index = None
        
                                       
        pop = self.backend.population
        if not pop:
            return

                                                                   
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
                                                                    
        image_payloads = self.backend.render_all_previews_parallel() 

        QApplication.restoreOverrideCursor()

                                              
        parent = self.parentWidget()
        if parent:
            available_width = parent.width() - self.inspector_panel.width()
        else:
            available_width = self.width() - self.inspector_panel.width()
        margins = self.grid_layout.contentsMargins()
        available_width -= (margins.left() + margins.right())
        item_width_with_spacing = 220 + 15
        cols = max(1, available_width // item_width_with_spacing)

                                                 
        for i, payload in enumerate(image_payloads):
            pixmap = QPixmap()
            if not payload:
                pixmap = QPixmap(200, 200)
                pixmap.fill(QColor("darkred"))
            else:
                if not pixmap.loadFromData(payload):
                    pixmap = QPixmap(200, 200)
                    pixmap.fill(QColor("darkred"))

            scaled_pixmap = pixmap.scaled(
                200,
                200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            
            widget = IndividualWidget(i, scaled_pixmap)
            
            widget.toggled_selection.connect(self.handle_selection)
            widget.btn_png.clicked.connect(partial(self.save_individual_immediate, i, "png"))
            widget.btn_svg.clicked.connect(partial(self.save_individual_immediate, i, "svg"))
            
            self.grid_layout.addWidget(widget, i // cols, i % cols)
        
                                 
        
        

    def _show_loading_message(self):
        """Clear the grid and show loading message."""
                                                
        widgets_to_delete = []
        for i in reversed(range(self.grid_layout.count())): 
            item = self.grid_layout.itemAt(i)
            if item:
                w = item.widget()
                if w and w != self.loading_label:
                    widgets_to_delete.append(w)

        for w in widgets_to_delete:
            w.setParent(None)
            w.deleteLater()
        
                              
        self.loading_label.show()
        self.loading_label.raise_()                  
        QApplication.processEvents()                   

    def handle_selection(self, index, is_selected):
        if is_selected:
            self.selected_indices.add(index)
        else:
            self.selected_indices.discard(index)
            
        if is_selected:
            self.current_selected_index = index
            self.inspector_panel.update_inspector(index)
        else:
            if self.current_selected_index == index:
                self.inspector_panel.clear_inspector()
                self.current_selected_index = None

                                                                                          
            
    def save_individual_immediate(self, index, fmt):
        try:
            path = self.backend.save_hq_individual(index, self.generation, fmt)
            self.status_message.emit(f"Saved: {os.path.basename(path)}", 4000)
        except Exception as e:
            self.status_message.emit(f"Error saving: {str(e)}", 4000)
            print(e)

    def go_previous_generation(self):
        """
        Load the previous generation from disk.
        """
        if self.generation <= 0:
            return
        
                                     
        self.btn_prev_gen.setEnabled(False)
        self.btn_evolve.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            
                                      
            self.backend.load_generation(self.generation - 1)
            self.generation = self.backend.generation
            
                       
            self.lbl_gen.setText(f"Gen: {self.generation}")
            self.inspector_panel.set_generation(self.generation)
            self.render_population_grid()
            self.update_population_stats()
            self.inspector_panel.clear_inspector()
            
            self.scroll_area.verticalScrollBar().setValue(0)
            self.status_message.emit(f"Loaded generation {self.generation}", 3000)
            
        except FileNotFoundError as e:
            self.status_message.emit(f"Generation file not found: {e}", 5000)
            print(f"Error loading generation: {e}")
        except Exception as e:
            self.status_message.emit(f"Error loading generation: {e}", 5000)
            print(f"Error loading generation: {e}")
        finally:
            QApplication.restoreOverrideCursor()
                               
            self.btn_evolve.setEnabled(True)
            self.btn_prev_gen.setEnabled(self.generation > 0)

    def evolve_step(self):
        """
        Main function that evolves the population to the next generation.
        """
                                                                         
        if not self.selected_indices and self.current_selected_index is not None:
            self.selected_indices.add(self.current_selected_index)
        if not self.selected_indices:
            QMessageBox.warning(self, "Selection Required", "Select at least one individual.")
            return
        
                                             
        self._show_loading_message()
        
                                                                                
        self.btn_evolve.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.backend.evolve(self.selected_indices)
            
            self.generation += 1
            self.lbl_gen.setText(f"Gen: {self.generation}")
            self.inspector_panel.set_generation(self.generation)
            
            self.backend.save_generation_state(self.generation)

            self.render_population_grid()                               
            self.update_population_stats()
            self.inspector_panel.clear_inspector()
            
            self.scroll_area.verticalScrollBar().setValue(0)
            self.status_message.emit(f"Generation {self.generation} created and saved.", 4000)
                                                                 
            self.btn_prev_gen.setEnabled(self.generation > 0)
        except Exception as e:
                                                                 
            self.status_message.emit(f"Error during evolution: {e}", 6000)
            print(f"Error during evolution: {e}")
        finally:
            QApplication.restoreOverrideCursor()
                                                                          
            self.btn_evolve.setEnabled(True)