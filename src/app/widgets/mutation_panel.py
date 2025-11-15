import sys
from functools import partial
from typing import Any
from PySide6.QtWidgets import (
    QGroupBox, QFormLayout, QSlider, QLabel, QHBoxLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

                                              
                                                    
try:
    from src.core.evolution.genome import MutationConfig
except ImportError:
    try:
        from evolution import MutationConfig                        
    except ImportError:
        class _MutationConfigFallback:                                           
            """Fallback placeholder when evolution module is unavailable."""

            pass

        MutationConfig = _MutationConfigFallback                            

from ..theme import VisualConfig

class MutationPanel(QGroupBox):
    """QGroupBox that houses sliders for all evolution mutation parameters.

    Emits a 'parameter_changed' signal whenever a value changes.
    """
    
                                                           
    parameter_changed = Signal(str, float)

                                                  
    PARAM_DEFINITIONS = {
                                            
        "gene_mutation_prob": ("Gene Mut %", 0.0, 1.0),
        "add_primitive_prob": ("Add Prim %", 0.0, 1.0),
    "remove_primitive_prob": ("Remove Prim %", 0.0, 1.0),
        "operator_mutation_prob": ("Op Mut %", 0.0, 1.0),
        "translate_sigma": ("Trans Sigma", 0.0, 0.5),
        "rotate_sigma": ("Rotate Sigma", 0.0, 0.5),
        "scale_sigma": ("Scale Sigma", 0.0, 0.5),
        "color_sigma": ("Color Sigma", 0.0, 0.5),
    }

    def __init__(self, parent=None):
        super().__init__("Evolution Parameters", parent)
        self.setObjectName("mutation_panel")

        accent = VisualConfig.color_accent
        accent_hover = QColor(accent).lighter(120).name()
        groove_color = QColor(accent).darker(220).name()
        background_color = "#202020"

        self.param_widgets = {}

        evo_form = QFormLayout(self)
        evo_form.setSpacing(10)

        for param_name, (label_text, min_val, max_val) in self.PARAM_DEFINITIONS.items():
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setStyleSheet(
                f"""
                QSlider::groove:horizontal {{
                    height: 6px;
                    background: {background_color};
                    border-radius: 4px;
                }}
                QSlider::sub-page:horizontal {{
                    background: {accent};
                    border-radius: 4px;
                }}
                QSlider::add-page:horizontal {{
                    background: {groove_color};
                    border-radius: 4px;
                }}
                QSlider::handle:horizontal {{
                    background: {accent};
                    border: 2px solid #111;
                    width: 16px;
                    margin: -6px 0;
                    border-radius: 9px;
                }}
                QSlider::handle:horizontal:hover {{
                    background: {accent_hover};
                }}
                """
            )
            
                                                    
            value_label = QLabel("N/A")
            value_label.setFixedWidth(40)                          
            value_label.setStyleSheet(f"color: {accent}; font-weight: 600;")

                                                          
            slider.valueChanged.connect(
                partial(self._on_slider_updated, param_name, slider, value_label, min_val, max_val)
            )
            
                                    
            self.param_widgets[param_name] = (slider, value_label)
            
                                                          
            slider_container = QHBoxLayout()
            slider_container.setContentsMargins(0, 0, 0, 0)
            slider_container.addWidget(slider)
            slider_container.addWidget(value_label)
            
                                                   
            evo_form.addRow(label_text, slider_container)

    def _on_slider_updated(self, param_name: str, slider_widget: QSlider, label_widget: QLabel, min_val: float, max_val: float, int_value: int):
        """Internal slot triggered when a slider moves."""
                                                            
        float_value = min_val + (max_val - min_val) * (int_value / 100.0)
        
                                            
        label_widget.setText(f"{float_value:.2f}")
        
                                                            
        self.parameter_changed.emit(param_name, float_value)

    def update_ui(self, config_obj: Any) -> None:
        """Public method that refreshes sliders and labels from a config object."""
        if config_obj is None:
            return
            
        try:
            for param_name, (slider, value_label) in self.param_widgets.items():
                
                                                   
                (_, min_val, max_val) = self.PARAM_DEFINITIONS[param_name]
                
                                                                 
                new_val = getattr(config_obj, param_name)
                
                                           
                value_label.setText(f"{new_val:.2f}")
                
                                                                         
                int_val = int(((new_val - min_val) / (max_val - min_val)) * 100)
                
                                                                      
                                                                       
                slider.blockSignals(True)
                slider.setValue(int_val)
                slider.blockSignals(False)
                
        except Exception as e:
            print(f"Error updating MutationPanel UI: {e}", file=sys.stderr)