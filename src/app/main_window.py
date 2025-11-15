import sys
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QApplication
from PySide6.QtCore import Qt

from src.population_manager.backend_adapter import BackendAdapter

from .screens.menu_screen import MenuScreen
from .screens.create_population_screen import CreatePopulationScreen
from .screens.evolve_screen import EvolveScreen
from .screens.populations_screen import PopulationsScreen

class MainWindow(QMainWindow):
    """Main application window acting as a navigator (QStackedWidget)."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Evolving Art - Interactive")
        
        self.backend = BackendAdapter()
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
                                     
        self.menu_screen = MenuScreen()
        self.create_screen = CreatePopulationScreen()
        self.evolve_screen = EvolveScreen(self.backend)
        self.populations_screen = PopulationsScreen(self.backend)                    
        
                                          
        self.stack.addWidget(self.menu_screen)                  
        self.stack.addWidget(self.populations_screen)                   
        self.stack.addWidget(self.create_screen)                
        self.stack.addWidget(self.evolve_screen)                
        
                                                
        
                                    
        self.menu_screen.populations_requested.connect(self.go_to_populations_screen)
        self.menu_screen.quit_requested.connect(self.close)
        
                                                      
        self.populations_screen.menu_requested.connect(self.go_to_menu_screen)
        self.populations_screen.create_requested.connect(self.go_to_create_screen)
        self.populations_screen.load_requested.connect(self.load_and_go_to_evolve)

                                                          
        self.create_screen.cancel_requested.connect(self.go_to_populations_screen)
        self.create_screen.population_created.connect(self.create_and_go_to_evolve)
        
                        
        self.evolve_screen.new_population_requested.connect(self.go_to_menu_screen)
        
                                  
        self.status_bar = self.statusBar()
        self.evolve_screen.status_message.connect(self.show_status)
        
                                         
        self.go_to_menu_screen()

    def go_to_menu_screen(self):
        self.stack.setCurrentIndex(0)

    def go_to_populations_screen(self):
        """Navigate to the population list screen and refresh it."""
        self.populations_screen.refresh_population_list()
        self.stack.setCurrentIndex(1)

    def go_to_create_screen(self):
        self.stack.setCurrentIndex(2)

    def create_and_go_to_evolve(self, params):
        """Initialize the backend with provided parameters and open the evolve screen."""
        self.show_status(f"Initializing population '{params['pop_name']}'...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.backend.initialize(
                params['pop_size'], params['genes'], 
                params['seed'], pop_name=params['pop_name']
            )
            self.backend.save_generation_state(0)                
            
            self.evolve_screen.set_population_data(
                pop_name=params['pop_name'],
                generation=0
            )
            
            self.stack.setCurrentIndex(3)              
            self.show_status(f"Population '{params['pop_name']}' initialized.")
            
        except Exception as e:
            self.show_status(f"Error initializing: {e}")
            print(f"Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def load_and_go_to_evolve(self, pop_name):
        """Load an existing population into the backend and show the evolve screen."""
        self.show_status(f"Loading population '{pop_name}'...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
                                           
            loaded_data = self.backend.load_population(pop_name)
            
                                                    
            self.evolve_screen.set_population_data(
                pop_name=loaded_data['pop_name'],
                generation=loaded_data['generation']
            )
            
                                                   
            self.stack.setCurrentIndex(3)
            self.show_status(f"Population '{pop_name}' loaded.")
            
        except Exception as e:
            self.show_status(f"Error loading: {e}")
            print(f"Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def show_status(self, message, timeout=4000):
        self.status_bar.showMessage(message, timeout)