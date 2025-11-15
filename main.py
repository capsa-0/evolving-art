import sys
import os
import multiprocessing  # keep available at module level but don't import GEOS-using modules yet


if __name__ == "__main__":

    # Ensure we set a safe start method before importing modules that
    # initialize GEOS or other C extensions. Using 'spawn' prevents
    # the child processes from inheriting a potentially unstable
    # native library state created by the main process.
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # start method was already set earlier (possibly by tests or embedding);
        # that's fine — proceed.
        pass

    # Support frozen executables (PyInstaller)
    multiprocessing.freeze_support()

    # Ensure isolated rendering defaults can be controlled centrally
    # Default to sequential + isolated worker to avoid native crashes;
    # allow user to override via environment.
    os.environ.setdefault("EVOART_RENDER_SEQUENTIAL", "1")
    os.environ.setdefault("EVOART_RENDER_ISOLATE", "1")

    # Reduce allocator contention and C-extension thread interactions globally
    # (NumPy/BLAS/OpenMP). Users can override if needed.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Import GUI and app modules after start method is set
    from PySide6.QtWidgets import QApplication
    from src.app.main_window import MainWindow
    from src.app.theme import set_modern_theme

    app = QApplication(sys.argv)

    # Apply modern theme
    set_modern_theme(app)

    # Create and show main window
    window = MainWindow()
    window.showFullScreen()

    # Run the application
    sys.exit(app.exec())