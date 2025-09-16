"""
Main application module for Optical Flow Cluster Analyzer.
This module contains the main window and application logic.
"""

import os

from twisted.conch.scripts.tkconch import frame

os.environ["OMP_NUM_THREADS"] = "1"  # Fix KMeans warning

import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QComboBox, QSpinBox, QFileDialog, QGroupBox,
    QCheckBox, QMessageBox, QSlider, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QScrollArea, QSizePolicy, QDoubleSpinBox, QDialog, QDialogButtonBox,
    QProgressBar, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QFont, QPainter, QColor, QPen
import cv2

from .analysis.movement_quality import MovementQualityAnalyzer
from .analysis.fluid_motion import FluidMotionAnalyzer
from .vision.detector import HumanDetector
from .vision.optical_flow import OpticalFlowProcessor
from .ui.dialogs import ValidationResultsDialog, LabelFormatHelpDialog
from .utils.visuals import create_heatmap, draw_motion_trails, draw_cluster_centroids
from .utils.metrics import calculate_hopkins_statistic
from .io.reporting import export_analysis_report

from collections import deque




class OpticalFlowClusterAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Variables
        self.cap = None
        self.timer = QTimer()
        self.frame_count = 0
        self.optical_flow_data = []
        self.cluster_history = []
        self.is_webcam = False
        self.is_playing = False
        self.analysis_data = pd.DataFrame()
        self.prev_gray = None
        self.prev_keypoints = None
        self.flow_accumulator = None
        self.hopkins_history = []
        self.clusterability_scores = []
        self.feature_points = None
        self.worker = None
        self.motion_trails = deque(maxlen=30)
        self.heatmap_data = None
        self.cluster_centroids = {}

        # Initialize analyzers
        self.fluid_analyzer = FluidMotionAnalyzer()
        self.movement_analyzer = MovementQualityAnalyzer()
        self.human_detector = HumanDetector()
        self.optical_flow_processor = OpticalFlowProcessor()

        # UI setup
        self.setWindowTitle("Optical Flow Cluster Analyzer")
        self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(100, 100, 1800, 1000)
        self.setup_ui()
        self.apply_styles()

        # Signals
        self.timer.timeout.connect(self.update_frame)
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def setup_ui(self):
        """Setup the main application UI"""
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel with controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel)

        # Right panel with tabs
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_left_panel(self):
        """Create the left control panel"""
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_label = QLabel("Optical Flow Cluster Analyzer")
        title_label.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        left_layout.addWidget(title_label)

        # Source group
        source_group = self.create_source_group()
        left_layout.addWidget(source_group)

        # Analysis settings group
        analysis_group = self.create_analysis_group()
        left_layout.addWidget(analysis_group)

        # Visualization group
        viz_group = self.create_visualization_group()
        left_layout.addWidget(viz_group)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(350)

        return left_panel

    def create_source_group(self):
        """Create video source control group"""
        group = QGroupBox("Video Source")
        layout = QVBoxLayout()

        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.webcam_btn.clicked.connect(self.start_webcam)
        layout.addWidget(self.webcam_btn)

        self.stop_webcam_btn = QPushButton("Stop Camera")
        self.stop_webcam_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        self.stop_webcam_btn.setEnabled(False)
        self.stop_webcam_btn.clicked.connect(self.stop_webcam)
        layout.addWidget(self.stop_webcam_btn)

        self.load_btn = QPushButton("Load Video")
        self.load_btn.setStyleSheet("background-color: #2ecc71; color: white;")
        self.load_btn.clicked.connect(self.load_video)
        layout.addWidget(self.load_btn)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        self.play_btn.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_btn)

        group.setLayout(layout)
        return group

    def create_analysis_group(self):
        """Create analysis settings group"""
        group = QGroupBox("Analysis Settings")
        layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["K-Means", "DBSCAN", "Hierarchical", "OPTICS"])
        layout.addWidget(QLabel("Clustering Method:"))
        layout.addWidget(self.method_combo)

        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.1, 5.0)
        self.epsilon_spin.setSingleStep(0.1)
        self.epsilon_spin.setValue(0.5)
        layout.addWidget(QLabel("Epsilon (for DBSCAN/OPTICS):"))
        layout.addWidget(self.epsilon_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 50)
        self.min_samples_spin.setValue(5)
        layout.addWidget(QLabel("Min Samples (for DBSCAN/OPTICS):"))
        layout.addWidget(self.min_samples_spin)

        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.addItems(["Nano (fastest)", "Small", "Medium", "Large (most accurate)"])
        self.yolo_model_combo.currentIndexChanged.connect(self.change_yolo_model)
        layout.addWidget(QLabel("YOLO Model Size:"))
        layout.addWidget(self.yolo_model_combo)

        self.analyze_btn = QPushButton("Analyze Optical Flow")
        self.analyze_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        self.analyze_btn.clicked.connect(self.analyze_optical_flow)
        layout.addWidget(self.analyze_btn)

        self.export_btn = QPushButton("Export Report")
        self.export_btn.setStyleSheet("background-color: #f39c12; color: white;")
        self.export_btn.clicked.connect(self.export_report)
        layout.addWidget(self.export_btn)

        group.setLayout(layout)
        return group

    def create_visualization_group(self):
        """Create visualization options group"""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout()

        self.flow_check = QCheckBox("Show Optical Flow")
        self.flow_check.setChecked(True)
        layout.addWidget(self.flow_check)

        self.cluster_check = QCheckBox("Show Clusters")
        self.cluster_check.setChecked(True)
        layout.addWidget(self.cluster_check)

        self.hopkins_check = QCheckBox("Show Hopkins Statistic")
        layout.addWidget(self.hopkins_check)

        self.trails_check = QCheckBox("Show Motion Trails")
        self.trails_check.setChecked(True)
        layout.addWidget(self.trails_check)

        self.heatmap_check = QCheckBox("Show Heatmap")
        self.heatmap_check.setChecked(False)
        layout.addWidget(self.heatmap_check)

        self.centroids_check = QCheckBox("Show Cluster Centroids")
        self.centroids_check.setChecked(False)
        layout.addWidget(self.centroids_check)

        group.setLayout(layout)
        return group

    def create_right_panel(self):
        """Create the right panel with tabs"""
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

        # Add tabs
        from .ui.tabs import (
            create_video_tab, create_dashboard_tab, create_cluster_analysis_tab,
            create_hopkins_analysis_tab, create_fluid_analysis_tab,
            create_movement_quality_tab, create_origin_analysis_tab
        )

        self.tab_widget.addTab(create_video_tab(self), "Optical Flow Analysis")
        self.tab_widget.addTab(create_dashboard_tab(self), "Dashboard")
        self.tab_widget.addTab(create_cluster_analysis_tab(self), "Cluster Analysis")
        self.tab_widget.addTab(create_hopkins_analysis_tab(self), "Clusterability Analysis")
        self.tab_widget.addTab(create_fluid_analysis_tab(self), "Fluid Motion Analysis")
        self.tab_widget.addTab(create_movement_quality_tab(self), "Movement Quality Analysis")
        self.tab_widget.addTab(create_origin_analysis_tab(self), "Origin Analysis")

        right_layout.addWidget(self.tab_widget)
        right_panel.setLayout(right_layout)

        return right_panel

    def apply_styles(self):
        """Apply styles to the application"""
        self.setStyleSheet('''
            QMainWindow { background-color: #f7f7fa; }
            QWidget { background-color: #f7f7fa; color: #222; }
            QGroupBox { border: 1px solid #d1d5db; margin-top: 10px; }
            QGroupBox:title { color: #1976d2; subcontrol-origin: margin; left: 10px; }
            QLabel { color: #222; font-size: 14px; }
            QPushButton { background-color: #e3eafc; color: #1976d2; border-radius: 6px; padding: 6px; font-weight: bold; }
            QPushButton:disabled { background-color: #eee; color: #aaa; }
            QComboBox, QSpinBox, QSlider, QCheckBox, QTabWidget, QTableWidget, QTextEdit {
                background-color: #fff; color: #222; border: 1px solid #d1d5db;
            }
            QTabBar::tab:selected { background: #e3eafc; color: #1976d2; }
            QTabBar::tab:!selected { background: #f7f7fa; color: #222; }
            QTableWidget QHeaderView::section { background-color: #e3eafc; color: #1976d2; }
        ''')

    # Main application methods will be implemented here
    # (start_webcam, stop_webcam, load_video, toggle_playback, update_frame, etc.)

    def start_webcam(self):
        """Start webcam capture"""
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                self.cap = None
                self.status_bar.showMessage("Webcam error: Not available or in use.")
                QMessageBox.critical(self, "Error", "Could not open webcam! Make sure it is not in use.")
                return
            self.is_webcam = True
            self.webcam_btn.setEnabled(False)
            self.stop_webcam_btn.setEnabled(True)
            self.timer.start(30)
            self.status_bar.showMessage("Webcam active - Press 'Analyze'")
        except Exception as e:
            self.status_bar.showMessage(f"Webcam error: {e}")
            QMessageBox.critical(self, "Error", f"Webcam error: {e}")
            if self.cap:
                self.cap.release()
            self.is_webcam = False
            self.webcam_btn.setEnabled(True)
            self.stop_webcam_btn.setEnabled(False)

    def stop_webcam(self):
        """Stop webcam capture"""
        try:
            if self.timer.isActive():
                self.timer.stop()
            if self.cap:
                self.cap.release()
            self.is_webcam = False
            self.webcam_btn.setEnabled(True)
            self.stop_webcam_btn.setEnabled(False)
            self.status_bar.showMessage("Webcam stopped")
        except Exception as e:
            self.status_bar.showMessage(f"Stop webcam error: {e}")
            QMessageBox.critical(self, "Error", f"Stop webcam error: {e}")

    def load_video(self):
        """Load video file"""
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
            options=options
        )
        if file_name:
            if self.is_webcam:
                self.stop_webcam()
            self.cap = cv2.VideoCapture(file_name)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open video file!")
                return
            self.frame_count = 0
            self.is_playing = True
            self.timer.start(30)
            self.status_bar.showMessage(f"Loaded: {os.path.basename(file_name)}")

    def toggle_playback(self):
        """Toggle video playback"""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶ Play")
            self.play_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        else:
            if self.cap and self.cap.isOpened():
                self.timer.start(30)
                self.is_playing = True
                self.play_btn.setText("⏸ Pause")
                self.play_btn.setStyleSheet("background-color: #e74c3c; color: white;")

    def update_frame(self):
        """Update frame processing"""
        try:
            if not self.cap or not self.cap.isOpened():
                self.status_bar.showMessage("Camera not available.")
                return

            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self.is_webcam:
                    self.timer.stop()
                    self.is_playing = False
                    self.play_btn.setText("▶ Play")
                    self.play_btn.setStyleSheet("background-color: #9b59b6; color: white;")
                    self.status_bar.showMessage("Video ended or frame not available.")
                return

            # Process the frame
            processed_frame, flow_data = self.process_frame(frame)

            # Display the frame
            self.display_frame(processed_frame)

            # Update analysis data
            self.update_analysis_data(flow_data)

            self.frame_count += 1

        except Exception as e:
            self.status_bar.showMessage(f"Frame error: {e}")
            QMessageBox.critical(self, "Error", f"Frame error: {e}")
            if self.timer.isActive():
                self.timer.stop()
            if self.cap:
                self.cap.release()
            self.is_webcam = False
            self.is_playing = False
            self.webcam_btn.setText("Start Webcam")
            self.webcam_btn.setStyleSheet("background-color: #3498db; color: white;")

    def process_frame(self, frame):
        """Process a single frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect humans
        human_rects = self.human_detector.detect(frame)

        # Process optical flow
        processed_frame, flow_data = self.optical_flow_processor.process(
            frame, gray, human_rects, self.prev_gray, self.feature_points,
            self.trails_check.isChecked(), self.heatmap_check.isChecked(),
            self.centroids_check.isChecked(), self.motion_trails, self.cluster_centroids
        )

        # Update previous frame data
        self.prev_gray = gray.copy()
        if hasattr(self.optical_flow_processor, 'feature_points'):
            self.feature_points = self.optical_flow_processor.feature_points

        return processed_frame, flow_data

    def display_frame(self, frame):
        """Display the processed frame"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(
            frame.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def show_proof_frame(self):
        """Show the proof frame for selected motion pattern"""
        selected = self.motion_origins_table.currentRow()
        if selected >= 0:
            sig_item = self.motion_origins_table.item(selected, 0)
            if sig_item:
                sig = sig_item.text()
                # Find the full signature that starts with this prefix
                full_sig = next((k for k in self.fluid_analyzer.motion_origins.keys()
                                 if k.startswith(sig.split("...")[0])), None)
                if full_sig:
                    proof_frame = self.fluid_analyzer.motion_origins[full_sig]['proof_frame']

                    # Convert to QImage and display
                    proof_frame = cv2.cvtColor(proof_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = proof_frame.shape
                    bytes_per_line = ch * w
                    q_img = QImage(proof_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.proof_frame_label.setPixmap(pixmap.scaled(
                        self.proof_frame_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    ))


    def update_analysis_data(self, flow_data):
        """Update analysis data with new frame information"""
        if flow_data is not None and len(flow_data) > 0:
            # Calculate Hopkins statistic
            hopkins_stat = calculate_hopkins_statistic(flow_data)
            self.hopkins_history.append({
                'frame': self.frame_count,
                'hopkins': hopkins_stat
            })

            # Store frame data
            frame_data = {
                'frame': self.frame_count,
                'time': self.frame_count / 30,
                'flow_points': flow_data,
                'hopkins': hopkins_stat
            }
            self.optical_flow_data.append(frame_data)

            # Analyze fluid motion
            is_fluid, _ = self.fluid_analyzer.analyze_frame(
                frame_data,
                self.frame_count,
                frame.copy() if hasattr(self, 'frame') else None
            )

            # Update dashboard
            if self.frame_count % 5 == 0:
                self.update_dashboard_plots(flow_data)
                self.update_metrics_display()
                self.update_fluid_analysis()

                # Predict movement quality if model is trained
                if hasattr(self.movement_analyzer, 'trained_model') and self.movement_analyzer.trained_model:
                    self.predict_current_movement_quality()

    def change_yolo_model(self, index):
        """Change YOLO model size"""
        model_sizes = ['n', 's', 'm', 'l']
        if 0 <= index < len(model_sizes):
            self.human_detector.change_model(model_sizes[index])
            self.status_bar.showMessage(f"Switched to YOLOv8{model_sizes[index].upper()} model")

    def analyze_optical_flow(self):
        """Analyze optical flow data"""
        if not hasattr(self, 'optical_flow_data') or len(self.optical_flow_data) < 5:
            QMessageBox.warning(self, "Warning", "Not enough data for analysis!")
            return

        method = self.method_combo.currentText()
        eps = self.epsilon_spin.value()
        min_samples = self.min_samples_spin.value()

        self.status_bar.showMessage("Running analysis in background...")
        self.analyze_btn.setEnabled(False)

        # Start analysis in background thread
        from .analysis.worker import OpticalFlowAnalyzerWorker
        self.worker = OpticalFlowAnalyzerWorker(self.optical_flow_data, method, eps, min_samples)
        self.worker.analysis_finished.connect(self.on_analysis_finished)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self):
        """Handle worker thread finished"""
        self.analyze_btn.setEnabled(True)

    def on_analysis_finished(self, clustering_results, X):
        """Handle analysis finished signal"""
        try:
            # Process clustering results
            self.process_clustering_results(clustering_results, X)

            # Update UI with results
            self.update_cluster_visualization(X, clustering_results)
            self.update_hopkins_analysis()
            self.generate_cluster_summary()
            self.update_metrics_display()

            self.status_bar.showMessage("Analysis completed")
        except Exception as e:
            self.status_bar.showMessage(f"Analysis failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")

    def export_report(self):
        """Export analysis report"""
        if not self.optical_flow_data:
            QMessageBox.warning(self, "Warning", "No analysis data to export!")
            return

        # Use the reporting module to export data
        export_analysis_report(self)

    # Additional helper methods will be implemented here
    # (update_dashboard_plots, update_metrics_display, etc.)


def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    window = OpticalFlowClusterAnalyzer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()