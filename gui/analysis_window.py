import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QProgressBar,
                             QLabel, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWebEngineWidgets import QWebEngineView
from plotly.offline import plot
import plotly.graph_objects as go
from typing import Dict, Any


class AnalysisWindow(QMainWindow):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.setup_ui()
        self.plot_widgets = []  # To keep references to plot widgets
        self.show()

    def setup_ui(self):
        """Initialize the main UI components"""
        self.setWindowTitle("Human Movement Analysis")
        self.setGeometry(100, 100, 1400, 900)  # Larger window for better visualization

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.layout = QVBoxLayout(main_widget)

        # Control panel
        self.setup_control_panel()

        # Visualization tabs
        self.setup_visualization_tabs()

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def setup_control_panel(self):
        """Set up the control panel with buttons and progress bar"""
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # Upload button
        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.clicked.connect(self.open_file_dialog)
        self.upload_btn.setFixedWidth(200)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(400)

        # Analysis button
        self.analyze_btn = QPushButton("Analyze Movements")
        self.analyze_btn.clicked.connect(self.analyzer.analyze_movements)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setFixedWidth(200)

        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addStretch()

        self.layout.addWidget(control_panel)

    def setup_visualization_tabs(self):
        """Set up the tab widget for different visualizations"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)

        # Add placeholder tab
        placeholder = QWidget()
        self.tab_widget.addTab(placeholder, "Waiting for data...")

        self.layout.addWidget(self.tab_widget)

    def open_file_dialog(self):
        """Open file dialog to select video and start processing"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
            options=options
        )

        if file_name:
            self.prepare_for_processing()
            self.analyzer.process_video_file(file_name)

    def prepare_for_processing(self):
        """Prepare UI for video processing"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.upload_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.clear_visualizations()
        self.status_bar.showMessage("Processing video...")

    def clear_visualizations(self):
        """Clear all existing visualizations"""
        while self.tab_widget.count():
            self.tab_widget.removeTab(0)
        self.plot_widgets = []

    def update_progress(self, current: int, total: int):
        """Update progress bar during processing"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"Processing: {current}/{total} frames ({current / total:.1%})")

    def update_visualizations(self, data: Dict[str, Any]):
        """Update all visualizations with analysis results"""
        try:
            self.clear_visualizations()

            # Enable analyze button (though analysis is already done)
            self.analyze_btn.setEnabled(True)

            # Add visualizations to tabs
            self.add_cluster_visualization(data)
            self.add_timeline_visualization(data)
            self.add_heatmap_visualizations(data)

            self.status_bar.showMessage("Analysis complete!")

        except Exception as e:
            self.show_error(f"Visualization error: {str(e)}")

    def add_cluster_visualization(self, data: Dict[str, Any]):
        """Add cluster visualization tab"""
        from utils.visualization import show_cluster_visualization

        fig = show_cluster_visualization(
            features=data['features'],
            cluster_labels=data['clusters']
        )

        self.add_plotly_tab(fig, "Cluster Visualization")

    def add_timeline_visualization(self, data: Dict[str, Any]):
        """Add timeline visualization tab"""
        from utils.visualization import show_cluster_timeline

        fig = show_cluster_timeline(
            timestamps=data['timestamps'],
            cluster_labels=data['clusters'],
            magnitudes=np.array([np.mean(m) for m in data['magnitudes']])
        )

        self.add_plotly_tab(fig, "Movement Timeline")

    def add_heatmap_visualizations(self, data: Dict[str, Any]):
        """Add heatmap visualization tabs"""
        from utils.visualization import show_motion_heatmaps

        if 'first_frame' not in data or data['first_frame'] is None:
            return

        heatmap_figs = show_motion_heatmaps(
            magnitudes=data['magnitudes'],
            cluster_labels=data['clusters'],
            original_frame=data['first_frame']
        )

        for i, fig in enumerate(heatmap_figs):
            self.add_plotly_tab(fig, f"Cluster {i} Heatmap")

    def add_plotly_tab(self, figure: go.Figure, title: str):
        """Add a new tab with a Plotly figure"""
        plot_html = plot(figure, output_type='div', include_plotlyjs='cdn')

        web_view = QWebEngineView()
        web_view.setHtml(plot_html)

        self.tab_widget.addTab(web_view, title)
        self.plot_widgets.append(web_view)

    def show_error(self, message: str):
        """Show error message in status bar and message box"""
        self.status_bar.showMessage(f"Error: {message}")
        QMessageBox.critical(self, "Error", message)

        # Reset UI state
        self.progress_bar.setVisible(False)
        self.upload_btn.setEnabled(True)
        self.analyze_btn.setEnabled(False)

    def show(self):
        """Show the window and ensure QApplication exists"""
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
            sys.exit(self.app.exec_())
        super().show()