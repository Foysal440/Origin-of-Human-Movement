"""
Tab creation functions for Optical Flow Cluster Analyzer.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QPushButton
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def create_video_tab(parent):
    """Create the video analysis tab"""
    video_tab = QWidget()
    video_layout = QVBoxLayout()

    parent.video_label = QLabel()
    parent.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    parent.video_label.setMinimumSize(1200, 800)
    parent.video_label.setScaledContents(True)
    parent.video_label.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding
    )
    video_layout.addWidget(parent.video_label)
    video_tab.setLayout(video_layout)

    return video_tab


def create_dashboard_tab(parent):
    """Create the dashboard tab"""
    dashboard_tab = QWidget()
    dashboard_layout = QVBoxLayout()

    # Metrics group
    metrics_group = QGroupBox("Flow Metrics")
    metrics_layout = QHBoxLayout()

    parent.metrics = {
        'total_flow': QLabel("0.00"),
        'avg_magnitude': QLabel("0.00"),
        'hopkins_stat': QLabel("0.00"),
        'clusters': QLabel("0")
    }

    for name, label in parent.metrics.items():
        group = QGroupBox(name.replace('_', ' ').title())
        group_layout = QVBoxLayout()
        label.setFont(QFont('Arial', 16, QFont.Weight.Bold))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group_layout.addWidget(label)
        group.setLayout(group_layout)
        metrics_layout.addWidget(group)

    metrics_group.setLayout(metrics_layout)
    dashboard_layout.addWidget(metrics_group)

    # Dashboard plots
    parent.dashboard_figure, (parent.flow_ax, parent.cluster_ax) = plt.subplots(1, 2, figsize=(12, 4))
    parent.dashboard_canvas = FigureCanvas(parent.dashboard_figure)
    dashboard_layout.addWidget(parent.dashboard_canvas)

    # Data table
    parent.data_table = QTableWidget()
    parent.data_table.setColumnCount(5)
    parent.data_table.setHorizontalHeaderLabels([
        "Frame", "Flow Points", "Avg Magnitude", "Hopkins", "Clusters"
    ])
    dashboard_layout.addWidget(parent.data_table)

    dashboard_tab.setLayout(dashboard_layout)
    return dashboard_tab


def create_cluster_analysis_tab(parent):
    """Create the cluster analysis tab"""
    cluster_tab = QWidget()
    cluster_layout = QVBoxLayout()

    parent.cluster_summary = QTextEdit()
    parent.cluster_summary.setReadOnly(True)
    cluster_layout.addWidget(parent.cluster_summary)

    parent.cluster_figure, parent.cluster_axes = plt.subplots(1, 2, figsize=(12, 4))
    parent.cluster_canvas = FigureCanvas(parent.cluster_figure)
    cluster_layout.addWidget(parent.cluster_canvas)

    parent.cluster_table = QTableWidget()
    parent.cluster_table.setColumnCount(5)
    parent.cluster_table.setHorizontalHeaderLabels([
        "Method", "Clusters", "Silhouette", "DB Index", "CH Score"
    ])
    cluster_layout.addWidget(parent.cluster_table)

    cluster_tab.setLayout(cluster_layout)
    return cluster_tab


def create_hopkins_analysis_tab(parent):
    """Create the Hopkins analysis tab"""
    hopkins_tab = QWidget()
    hopkins_layout = QVBoxLayout()

    parent.hopkins_figure, parent.hopkins_ax = plt.subplots(figsize=(10, 4))
    parent.hopkins_canvas = FigureCanvas(parent.hopkins_figure)
    hopkins_layout.addWidget(parent.hopkins_canvas)

    parent.hopkins_table = QTableWidget()
    parent.hopkins_table.setColumnCount(3)
    parent.hopkins_table.setHorizontalHeaderLabels([
        "Frame", "Hopkins", "Interpretation"
    ])
    hopkins_layout.addWidget(parent.hopkins_table)

    hopkins_tab.setLayout(hopkins_layout)
    return hopkins_tab


def create_fluid_analysis_tab(parent):
    """Create the fluid motion analysis tab"""
    fluid_tab = QWidget()
    fluid_layout = QVBoxLayout()

    # Fluid metrics
    parent.fluid_metrics = {
        'fluid_percent': QLabel("0%"),
        'non_fluid_percent': QLabel("0%"),
        'num_patterns': QLabel("0"),
        'avg_hopkins_fluid': QLabel("0.00"),
        'avg_hopkins_non_fluid': QLabel("0.00")
    }

    metrics_group = QGroupBox("Fluid Motion Metrics")
    metrics_layout = QHBoxLayout()

    for name, label in parent.fluid_metrics.items():
        group = QGroupBox(name.replace('_', ' ').title())
        group_layout = QVBoxLayout()
        label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group_layout.addWidget(label)
        group.setLayout(group_layout)
        metrics_layout.addWidget(group)

    metrics_group.setLayout(metrics_layout)
    fluid_layout.addWidget(metrics_group)

    # Motion origins table
    parent.motion_origins_table = QTableWidget()
    parent.motion_origins_table.setColumnCount(5)
    parent.motion_origins_table.setHorizontalHeaderLabels([
        "Pattern ID", "Start Frame", "Type", "Hopkins", "Avg Magnitude"
    ])
    fluid_layout.addWidget(QLabel("Motion Pattern Origins:"))
    fluid_layout.addWidget(parent.motion_origins_table)

    # Proof frame display
    parent.proof_frame_label = QLabel()
    parent.proof_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    parent.proof_frame_label.setMinimumSize(400, 300)
    fluid_layout.addWidget(QLabel("Proof Frame:"))
    fluid_layout.addWidget(parent.proof_frame_label)

    # Connect table selection to show proof frames
    parent.motion_origins_table.itemSelectionChanged.connect(parent.show_proof_frame)

    fluid_tab.setLayout(fluid_layout)
    return fluid_tab


def create_movement_quality_tab(parent):
    """Create the movement quality analysis tab"""
    movement_tab = QWidget()
    movement_layout = QVBoxLayout()

    # Training section
    training_group = QGroupBox("Movement Quality Analysis")
    training_layout = QVBoxLayout()

    # Auto analysis button
    parent.auto_analyze_btn = QPushButton("Analyze Movement Quality")
    parent.auto_analyze_btn.clicked.connect(parent.auto_analyze_movement_quality)
    training_layout.addWidget(parent.auto_analyze_btn)

    # Load ground truth button
    parent.load_labels_btn = QPushButton("Load Ground Truth Labels")
    parent.load_labels_btn.clicked.connect(parent.load_ground_truth_labels)
    training_layout.addWidget(parent.load_labels_btn)

    # Validate button
    parent.validate_btn = QPushButton("Validate Against Ground Truth")
    parent.validate_btn.clicked.connect(parent.validate_against_ground_truth)
    parent.validate_btn.setEnabled(False)
    training_layout.addWidget(parent.validate_btn)

    # Format help button
    parent.format_help_btn = QPushButton("Label Format Help")
    parent.format_help_btn.clicked.connect(parent.show_format_help)
    training_layout.addWidget(parent.format_help_btn)

    parent.model_status_label = QLabel("Status: Ready to analyze")
    training_layout.addWidget(parent.model_status_label)

    # Label statistics display
    parent.label_stats_label = QLabel("No labels loaded")
    parent.label_stats_label.setWordWrap(True)
    training_layout.addWidget(parent.label_stats_label)

    # Quality distribution display
    parent.quality_dist_label = QLabel("No analysis performed yet")
    parent.quality_dist_label.setWordWrap(True)
    training_layout.addWidget(parent.quality_dist_label)

    training_group.setLayout(training_layout)
    movement_layout.addWidget(training_group)

    # Prediction section
    prediction_group = QGroupBox("Movement Quality Prediction")
    prediction_layout = QVBoxLayout()

    parent.prediction_label = QLabel("Current Prediction: N/A")
    parent.prediction_label.setFont(QFont('Arial', 14, QFont.Weight.Bold))
    prediction_layout.addWidget(parent.prediction_label)

    parent.confidence_label = QLabel("Confidence: N/A")
    prediction_layout.addWidget(parent.confidence_label)

    # Origin detection
    parent.origin_label = QLabel("Movement Origin: N/A")
    prediction_layout.addWidget(parent.origin_label)

    # Feature importance visualization
    parent.feature_importance_figure, parent.feature_importance_ax = plt.subplots(figsize=(10, 6))
    parent.feature_importance_canvas = FigureCanvas(parent.feature_importance_figure)
    prediction_layout.addWidget(parent.feature_importance_canvas)

    prediction_group.setLayout(prediction_layout)
    movement_layout.addWidget(prediction_group)

    # Statistical testing section
    stats_group = QGroupBox("Analysis Summary")
    stats_layout = QVBoxLayout()

    parent.stats_results = QTextEdit()
    parent.stats_results.setReadOnly(True)
    stats_layout.addWidget(parent.stats_results)

    stats_group.setLayout(stats_layout)
    movement_layout.addWidget(stats_group)

    movement_tab.setLayout(movement_layout)
    return movement_tab


def create_origin_analysis_tab(parent):
    """Create the origin of movement analysis tab"""
    origin_tab = QWidget()
    origin_layout = QVBoxLayout()

    # Origin metrics
    parent.origin_metrics = {
        'expansion': QLabel("0%"),
        'contraction': QLabel("0%"),
        'rotation_cw': QLabel("0%"),
        'rotation_ccw': QLabel("0%"),
        'dominant_origin': QLabel("None")
    }

    metrics_group = QGroupBox("Movement Origin Metrics")
    metrics_layout = QHBoxLayout()

    for name, label in parent.origin_metrics.items():
        group = QGroupBox(name.replace('_', ' ').title())
        group_layout = QVBoxLayout()
        label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group_layout.addWidget(label)
        group.setLayout(group_layout)
        metrics_layout.addWidget(group)

    metrics_group.setLayout(metrics_layout)
    origin_layout.addWidget(metrics_group)

    # Origin visualization
    parent.origin_figure, parent.origin_ax = plt.subplots(figsize=(10, 6))
    parent.origin_canvas = FigureCanvas(parent.origin_figure)
    origin_layout.addWidget(parent.origin_canvas)

    # Origin history table
    parent.origin_table = QTableWidget()
    parent.origin_table.setColumnCount(4)
    parent.origin_table.setHorizontalHeaderLabels([
        "Frame", "Origin Type", "Confidence", "Body Region"
    ])
    origin_layout.addWidget(parent.origin_table)

    origin_tab.setLayout(origin_layout)
    return origin_tab