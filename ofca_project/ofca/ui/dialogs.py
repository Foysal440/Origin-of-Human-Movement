"""
Dialog classes for Optical Flow Cluster Analyzer.
"""

import os
import json
import csv
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QDialogButtonBox, QTabWidget, QGroupBox, QFileDialog,
    QMessageBox, QWidget
)
from PyQt6.QtCore import Qt

class ValidationResultsDialog(QDialog):
    """Dialog to show validation results between automatic and ground truth labels"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Validation Results")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def set_results(self, results):
        self.results_text.setText(results)


class LabelFormatHelpDialog(QDialog):
    """Dialog to show label format help and create sample files"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label Format Help")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Tab widget for different formats
        tab_widget = QTabWidget()

        # JSON format tab
        json_tab = QWidget()
        json_layout = QVBoxLayout()
        json_help = QTextEdit()
        json_help.setReadOnly(True)
        json_help.setText(
            "JSON Label Format:\n\n"
            "Frame-specific format:\n"
            "[\n"
            '  {"frame": 1, "quality_label": "good"},\n'
            '  {"frame": 2, "quality_label": "poor"},\n'
            '  {"frame": 5, "quality_label": "excellent"}\n'
            "]\n\n"
            "Range-based format:\n"
            "[\n"
            '  {"frame_start": 10, "frame_end": 25, "quality_label": "good"},\n'
            '  {"frame_start": 26, "frame_end": 40, "quality_label": "poor"}\n'
            "]\n\n"
            "Mixed format is also supported."
        )
        json_layout.addWidget(json_help)
        json_tab.setLayout(json_layout)
        tab_widget.addTab(json_tab, "JSON")

        # CSV format tab
        csv_tab = QWidget()
        csv_layout = QVBoxLayout()
        csv_help = QTextEdit()
        csv_help.setReadOnly(True)
        csv_help.setText(
            "CSV Label Format:\n\n"
            "Frame-specific format:\n"
            "frame,quality_label\n"
            "1,good\n"
            "2,poor\n"
            "5,excellent\n\n"
            "Range-based format:\n"
            "frame_start,frame_end,quality_label\n"
            "10,25,good\n"
            "26,40,poor\n\n"
            "Mixed format is also supported. The application will auto-detect the format."
        )
        csv_layout.addWidget(csv_help)
        csv_tab.setLayout(csv_layout)
        tab_widget.addTab(csv_tab, "CSV")

        # TXT format tab
        txt_tab = QWidget()
        txt_layout = QVBoxLayout()
        txt_help = QTextEdit()
        txt_help.setReadOnly(True)
        txt_help.setText(
            "TXT Label Format:\n\n"
            "Frame-specific format:\n"
            "1 good\n"
            "2 poor\n"
            "5 excellent\n\n"
            "Range-based format:\n"
            "10-25 good\n"
            "26-40 poor\n\n"
            "Mixed format is also supported. Use spaces or tabs as separators."
        )
        txt_layout.addWidget(txt_help)
        txt_tab.setLayout(txt_layout)
        tab_widget.addTab(txt_tab, "TXT")

        layout.addWidget(tab_widget)

        # Create sample buttons
        sample_group = QGroupBox("Create Sample Files")
        sample_layout = QHBoxLayout()

        self.json_sample_btn = QPushButton("Create JSON Sample")
        self.json_sample_btn.clicked.connect(self.create_json_sample)
        sample_layout.addWidget(self.json_sample_btn)

        self.csv_sample_btn = QPushButton("Create CSV Sample")
        self.csv_sample_btn.clicked.connect(self.create_csv_sample)
        sample_layout.addWidget(self.csv_sample_btn)

        self.txt_sample_btn = QPushButton("Create TXT Sample")
        self.txt_sample_btn.clicked.connect(self.create_txt_sample)
        sample_layout.addWidget(self.txt_sample_btn)

        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def create_json_sample(self):
        """Create a sample JSON label file"""
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save JSON Sample", "sample_labels.json",
            "JSON Files (*.json)", options=options
        )

        if file_name:
            if not file_name.endswith('.json'):
                file_name += '.json'

            sample_data = [
                {"frame": 1, "quality_label": "good"},
                {"frame": 2, "quality_label": "poor"},
                {"frame": 5, "quality_label": "excellent"},
                {"frame_start": 10, "frame_end": 25, "quality_label": "good"},
                {"frame_start": 26, "frame_end": 40, "quality_label": "poor"}
            ]

            try:
                with open(file_name, 'w') as f:
                    json.dump(sample_data, f, indent=2)
                QMessageBox.information(self, "Success", f"Sample JSON file created:\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create sample: {str(e)}")

    def create_csv_sample(self):
        """Create a sample CSV label file"""
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save CSV Sample", "sample_labels.csv",
            "CSV Files (*.csv)", options=options
        )

        if file_name:
            if not file_name.endswith('.csv'):
                file_name += '.csv'

            try:
                with open(file_name, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write frame-specific examples
                    writer.writerow(["frame", "quality_label"])
                    writer.writerow([1, "good"])
                    writer.writerow([2, "poor"])
                    writer.writerow([5, "excellent"])

                    # Write range-based examples
                    writer.writerow(["frame_start", "frame_end", "quality_label"])
                    writer.writerow([10, 25, "good"])
                    writer.writerow([26, 40, "poor"])

                QMessageBox.information(self, "Success", f"Sample CSV file created:\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create sample: {str(e)}")

    def create_txt_sample(self):
        """Create a sample TXT label file"""
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save TXT Sample", "sample_labels.txt",
            "Text Files (*.txt)", options=options
        )

        if file_name:
            if not file_name.endswith('.txt'):
                file_name += '.txt'

            try:
                with open(file_name, 'w') as f:
                    # Write frame-specific examples
                    f.write("1 good\n")
                    f.write("2 poor\n")
                    f.write("5 excellent\n")

                    # Write range-based examples
                    f.write("10-25 good\n")
                    f.write("26-40 poor\n")

                QMessageBox.information(self, "Success", f"Sample TXT file created:\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create sample: {str(e)}")