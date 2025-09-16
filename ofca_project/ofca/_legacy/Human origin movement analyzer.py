# Untouched backup of original file. Keeping it for reference if further need

import os

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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
import random
from ultralytics import YOLO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
import json
import csv
import seaborn as sns
from scipy import stats
from collections import deque


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


class MovementQualityAnalyzer:
    def __init__(self):
        self.trained_model = None
        self.feature_importances = {}
        self.prediction_history = {}
        self.quality_categories = ['poor', 'average', 'good', 'excellent']
        self.auto_label_thresholds = {
            'hopkins': [0.3, 0.5, 0.7],
            'magnitude': [5, 15, 25],
            'consistency': [0.7, 1.0, 1.3]
        }
        self.ground_truth_labels = {}
        self.validation_results = {}

    def extract_features(self, optical_flow_data, hopkins_stat):
        """Extract features for movement quality classification"""
        if len(optical_flow_data) == 0:
            return None

        features = {}

        # Basic flow statistics
        magnitudes = np.sqrt(optical_flow_data[:, 2] ** 2 + optical_flow_data[:, 3] ** 2)
        directions = np.arctan2(optical_flow_data[:, 3], optical_flow_data[:, 2])

        # Use safe calculations to avoid empty array warnings
        features['avg_magnitude'] = np.mean(magnitudes) if len(magnitudes) > 0 else 0
        features['std_magnitude'] = np.std(magnitudes) if len(magnitudes) > 0 else 0
        features['max_magnitude'] = np.max(magnitudes) if len(magnitudes) > 0 else 0

        # Direction consistency
        features['direction_std'] = np.std(directions) if len(directions) > 0 else 0
        features['direction_entropy'] = self.calculate_entropy(directions) if len(directions) > 0 else 0

        # Spatial distribution
        features['spatial_std_x'] = np.std(optical_flow_data[:, 0]) if len(optical_flow_data) > 0 else 0
        features['spatial_std_y'] = np.std(optical_flow_data[:, 1]) if len(optical_flow_data) > 0 else 0

        # Clusterability metrics
        features['hopkins_statistic'] = hopkins_stat

        # Add more features as needed
        if len(optical_flow_data) > 1 and (np.max(optical_flow_data[:, 0]) - np.min(optical_flow_data[:, 0])) > 0:
            features['flow_density'] = len(optical_flow_data) / (
                    np.max(optical_flow_data[:, 0]) - np.min(optical_flow_data[:, 0]))
        else:
            features['flow_density'] = 0

        return features

    def calculate_entropy(self, directions, bins=10):
        """Calculate entropy of direction distribution"""
        hist, _ = np.histogram(directions, bins=bins, range=(-np.pi, np.pi))
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Remove zeros for log calculation
        return -np.sum(prob * np.log(prob))

    def auto_label_movement(self, features):
        """Automatically label movement quality based on feature thresholds"""
        if not features:
            return "unknown"

        # Calculate quality score based on thresholds
        quality_score = 0

        # Hopkins statistic contribution (higher is better)
        hopkins = features.get('hopkins_statistic', 0.5)
        if hopkins > self.auto_label_thresholds['hopkins'][2]:
            quality_score += 3
        elif hopkins > self.auto_label_thresholds['hopkins'][1]:
            quality_score += 2
        elif hopkins > self.auto_label_thresholds['hopkins'][0]:
            quality_score += 1

        # Magnitude contribution (moderate is best)
        magnitude = features.get('avg_magnitude', 0)
        if 10 <= magnitude <= 20:  # Optimal range
            quality_score += 3
        elif 5 <= magnitude < 10 or 20 < magnitude <= 30:
            quality_score += 2
        elif magnitude > 30:  # Too high - could indicate jerky movement
            quality_score += 0
        else:
            quality_score += 1

        # Direction consistency (lower std is better)
        direction_std = features.get('direction_std', 0)
        if direction_std < 0.7:
            quality_score += 3
        elif direction_std < 1.0:
            quality_score += 2
        elif direction_std < 1.3:
            quality_score += 1

        # Normalize score to quality category
        max_score = 9  # 3 features * 3 points max each
        normalized_score = quality_score / max_score

        if normalized_score >= 0.75:
            return "excellent"
        elif normalized_score >= 0.5:
            return "good"
        elif normalized_score >= 0.25:
            return "average"
        else:
            return "poor"

    def train_model(self, features, labels):
        """Train a classifier for movement quality"""
        if len(features) < 2 or len(set(labels)) < 2:
            print("Not enough data or labels for training")
            return None, 0

        # Convert features to array
        feature_names = list(features[0].keys())
        X = np.array([[f[name] for name in feature_names] for f in features])
        y = np.array(labels)

        # Check if we have at least 2 classes for classification
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Only one class found: {unique_classes}. Need at least 2 classes for classification.")
            return None, 0

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Store feature importances
        self.feature_importances = dict(zip(feature_names, clf.feature_importances_))
        self.trained_model = clf

        return clf, accuracy

    def predict_movement_quality(self, features):
        """Predict movement quality using trained model"""
        if self.trained_model is None:
            # Fall back to auto-labeling if no model is trained
            auto_label = self.auto_label_movement(features)
            return auto_label, 0.7  # Medium confidence for auto-labeling

        feature_names = list(features.keys())
        X = np.array([[features[name] for name in feature_names]])

        prediction = self.trained_model.predict(X)[0]
        probability = np.max(self.trained_model.predict_proba(X))

        return prediction, probability

    def load_labels(self, file_path):
        """Load labels from various file formats with auto-detection"""
        labels = []
        label_stats = {
            'total_labels': 0,
            'frame_specific': 0,
            'range_based': 0,
            'unique_labels': set(),
            'frame_coverage': 0
        }

        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)

                for item in data:
                    if 'frame' in item:
                        # Frame-specific format
                        labels.append({
                            'frame': int(item['frame']),
                            'quality_label': item['quality_label']
                        })
                        label_stats['frame_specific'] += 1
                    elif 'frame_start' in item and 'frame_end' in item:
                        # Range-based format
                        labels.append({
                            'frame_start': int(item['frame_start']),
                            'frame_end': int(item['frame_end']),
                            'quality_label': item['quality_label']
                        })
                        label_stats['range_based'] += 1
                    label_stats['unique_labels'].add(item['quality_label'])

            elif file_path.endswith('.csv'):
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)

                    # Auto-detect format
                    first_row = next(reader, None)
                    if first_row:
                        f.seek(0)  # Reset to beginning
                        reader = csv.DictReader(f)  # Recreate reader with headers

                        if 'frame' in first_row and 'quality_label' in first_row:
                            # Frame-specific format
                            for row in reader:
                                labels.append({
                                    'frame': int(row['frame']),
                                    'quality_label': row['quality_label']
                                })
                                label_stats['frame_specific'] += 1
                                label_stats['unique_labels'].add(row['quality_label'])

                        elif 'frame_start' in first_row and 'frame_end' in first_row and 'quality_label' in first_row:
                            # Range-based format
                            for row in reader:
                                labels.append({
                                    'frame_start': int(row['frame_start']),
                                    'frame_end': int(row['frame_end']),
                                    'quality_label': row['quality_label']
                                })
                                label_stats['range_based'] += 1
                                label_stats['unique_labels'].add(row['quality_label'])

            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue

                        # Try to detect format
                        if '-' in line and not line.startswith('-'):  # Range format
                            parts = line.split()
                            if len(parts) >= 2:
                                frame_range = parts[0]
                                label = ' '.join(parts[1:])

                                if '-' in frame_range:
                                    start, end = frame_range.split('-')
                                    labels.append({
                                        'frame_start': int(start),
                                        'frame_end': int(end),
                                        'quality_label': label
                                    })
                                    label_stats['range_based'] += 1
                                    label_stats['unique_labels'].add(label)
                        else:  # Frame-specific format
                            parts = line.split()
                            if len(parts) >= 2:
                                labels.append({
                                    'frame': int(parts[0]),
                                    'quality_label': ' '.join(parts[1:])
                                })
                                label_stats['frame_specific'] += 1
                                label_stats['unique_labels'].add(' '.join(parts[1:]))

            label_stats['total_labels'] = len(labels)
            label_stats['unique_labels'] = list(label_stats['unique_labels'])

            return labels, label_stats

        except Exception as e:
            raise Exception(f"Error loading labels from {file_path}: {str(e)}")

    def validate_against_ground_truth(self, frame_predictions):
        """Validate automatic predictions against ground truth labels"""
        if not self.ground_truth_labels:
            return "No ground truth labels available for validation"

        # Match predictions with ground truth
        matched_data = []
        for frame_num, pred_data in frame_predictions.items():
            gt_label = self.find_ground_truth_label(frame_num)
            if gt_label:
                matched_data.append({
                    'frame': frame_num,
                    'prediction': pred_data['prediction'],
                    'ground_truth': gt_label,
                    'confidence': pred_data['confidence']
                })

        if not matched_data:
            return "No matching frames found between predictions and ground truth"

        # Prepare data for analysis
        y_true = [d['ground_truth'] for d in matched_data]
        y_pred = [d['prediction'] for d in matched_data]

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        # Statistical tests
        # Chi-square test for independence
        try:
            chi2, p_chi2, _, _ = chi2_contingency(cm)
            chi2_result = f"Chi-square test: χ²={chi2:.3f}, p={p_chi2:.4f}"
        except:
            chi2_result = "Chi-square test: Could not compute"

        # T-test for confidence scores between correct and incorrect predictions
        correct_conf = [d['confidence'] for d in matched_data if d['prediction'] == d['ground_truth']]
        incorrect_conf = [d['confidence'] for d in matched_data if d['prediction'] != d['ground_truth']]

        if correct_conf and incorrect_conf:
            t_stat, p_ttest = ttest_ind(correct_conf, incorrect_conf)
            ttest_result = f"T-test: t={t_stat:.3f}, p={p_ttest:.4f}"
        else:
            ttest_result = "T-test: Could not compute (need both correct and incorrect predictions)"

        # ANOVA for confidence across quality categories
        conf_by_category = {}
        for d in matched_data:
            if d['ground_truth'] not in conf_by_category:
                conf_by_category[d['ground_truth']] = []
            conf_by_category[d['ground_truth']].append(d['confidence'])

        if len(conf_by_category) >= 2:
            anova_groups = [conf_by_category[cat] for cat in conf_by_category]
            f_stat, p_anova = f_oneway(*anova_groups)
            anova_result = f"ANOVA: F={f_stat:.3f}, p={p_anova:.4f}"
        else:
            anova_result = "ANOVA: Could not compute (need at least 2 categories)"

        # Cross-validation if we have enough data
        cv_scores = []
        if len(matched_data) >= 10:
            try:
                feature_names = list(matched_data[0]['features'].keys()) if 'features' in matched_data[0] else []
                if feature_names:
                    X = np.array([[d['features'][name] for name in feature_names] for d in matched_data])
                    y = np.array(y_true)

                    clf = RandomForestClassifier(n_estimators=50, random_state=42)
                    cv_scores = cross_val_score(clf, X, y, cv=min(5, len(X)))
                    cv_result = f"Cross-validation: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}"
                else:
                    cv_result = "Cross-validation: No features available"
            except:
                cv_result = "Cross-validation: Error during computation"
        else:
            cv_result = "Cross-validation: Not enough data (need at least 10 samples)"

        # Compile results
        results = f"VALIDATION RESULTS\n"
        results += "=" * 50 + "\n\n"
        results += f"Matched frames: {len(matched_data)}\n"
        results += f"Accuracy: {accuracy:.3f}\n\n"
        results += f"Classification Report:\n{report}\n"
        results += f"Confusion Matrix:\n{cm}\n\n"
        results += "Statistical Tests:\n"
        results += f"{chi2_result}\n"
        results += f"{ttest_result}\n"
        results += f"{anova_result}\n"
        results += f"{cv_result}\n\n"

        # Store results for later access
        self.validation_results = {
            'matched_data': matched_data,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'chi2_test': (chi2, p_chi2) if 'chi2' in locals() else None,
            't_test': (t_stat, p_ttest) if 't_stat' in locals() else None,
            'anova_test': (f_stat, p_anova) if 'f_stat' in locals() else None,
            'cv_scores': cv_scores
        }

        return results

    def find_ground_truth_label(self, frame_num):
        """Find the ground truth label for a specific frame"""
        for label_data in self.ground_truth_labels:
            if 'frame' in label_data:
                # Frame-specific format
                if label_data['frame'] == frame_num:
                    return label_data['quality_label']
            elif 'frame_start' in label_data and 'frame_end' in label_data:
                # Range-based format
                if label_data['frame_start'] <= frame_num <= label_data['frame_end']:
                    return label_data['quality_label']
        return None


class FluidMotionAnalyzer:
    def __init__(self):
        self.fluid_threshold = 0.7  # Hopkins threshold for fluid motion
        self.motion_origins = {}  # Track where motion patterns started
        self.fluid_frames = []
        self.non_fluid_frames = []
        self.proof_frames = {}  # Store sample frames for each motion pattern
        self.motion_trails = deque(maxlen=30)  # Store motion trails for visualization

    def analyze_frame(self, frame_data, frame_number, frame_image):
        """Analyze if motion is fluid based on Hopkins statistic"""
        if len(frame_data['flow_points']) < 10:  # Not enough points
            return False, None

        hopkins = frame_data.get('hopkins', 0.5)
        is_fluid = hopkins > self.fluid_threshold

        # Track motion origins
        motion_signature = self._create_motion_signature(frame_data)
        if motion_signature not in self.motion_origins:
            self.motion_origins[motion_signature] = {
                'start_frame': frame_number,
                'proof_frame': frame_image.copy(),
                'type': 'fluid' if is_fluid else 'non-fluid',
                'hopkins': hopkins,
                'avg_magnitude': np.mean(
                    np.sqrt(frame_data['flow_points'][:, 2] ** 2 + frame_data['flow_points'][:, 3] ** 2))
                if len(frame_data['flow_points']) > 0 else 0
            }

        if is_fluid:
            self.fluid_frames.append(frame_number)
        else:
            self.non_fluid_frames.append(frame_number)

        # Add to motion trails
        self.motion_trails.append({
            'frame': frame_number,
            'flow_points': frame_data['flow_points'],
            'is_fluid': is_fluid
        })

        return is_fluid, motion_signature

    def _create_motion_signature(self, frame_data):
        """Create a signature for the motion pattern"""
        if len(frame_data['flow_points']) == 0:
            return "0" * 10

        # Use direction histogram as signature
        directions = np.arctan2(frame_data['flow_points'][:, 3], frame_data['flow_points'][:, 2])
        hist, _ = np.histogram(directions, bins=10, range=(-np.pi, np.pi))
        return ",".join(map(str, hist))

    def get_fluid_summary(self):
        """Generate summary statistics about fluid motion"""
        total_frames = len(self.fluid_frames) + len(self.non_fluid_frames)
        if total_frames == 0:
            return {}

        # Use safe calculations to avoid warnings
        fluid_hopkins = [o['hopkins'] for o in self.motion_origins.values() if o['type'] == 'fluid']
        non_fluid_hopkins = [o['hopkins'] for o in self.motion_origins.values() if o['type'] == 'non-fluid']

        avg_hopkins_fluid = np.mean(fluid_hopkins) if fluid_hopkins else 0
        avg_hopkins_non_fluid = np.mean(non_fluid_hopkins) if non_fluid_hopkins else 0

        return {
            'fluid_percentage': len(self.fluid_frames) / total_frames * 100,
            'non_fluid_percentage': len(self.non_fluid_frames) / total_frames * 100,
            'num_motion_patterns': len(self.motion_origins),
            'avg_hopkins_fluid': avg_hopkins_fluid,
            'avg_hopkins_non_fluid': avg_hopkins_non_fluid,
            'motion_origins': self.motion_origins
        }


class OpticalFlowAnalyzerWorker(QThread):
    analysis_finished = pyqtSignal(dict, np.ndarray)

    def __init__(self, optical_flow_data, method, eps, min_samples):
        super().__init__()
        self.optical_flow_data = optical_flow_data
        self.method = method
        self.eps = eps
        self.min_samples = min_samples

    def run(self):
        try:
            all_flow_data = np.vstack(
                [frame['flow_points'] for frame in self.optical_flow_data if 'flow_points' in frame])

            if len(all_flow_data) > 5000:
                indices = np.random.choice(len(all_flow_data), size=5000, replace=False)
                sampled_flow_data = all_flow_data[indices]
            else:
                sampled_flow_data = all_flow_data

            scaler = StandardScaler()
            X = scaler.fit_transform(sampled_flow_data)

            algorithms = {
                'K-Means': self.run_kmeans,
                'DBSCAN': self.run_dbscan,
                'Hierarchical': self.run_hierarchical,
                'OPTICS': self.run_optics
            }

            clustering_results = {}
            for name, algorithm in algorithms.items():
                try:
                    clustering_results[name] = algorithm(X)
                except Exception as e:
                    print(f"Error in {name}: {str(e)}")
                    clustering_results[name] = np.zeros(len(X))

            self.analysis_finished.emit(clustering_results, X)

        except Exception as e:
            print(f"[Worker] Error: {str(e)}")

    def run_kmeans(self, X):
        max_k = min(10, len(X) - 1)
        best_k = 2
        best_score = -1

        if max_k < 2:
            return np.zeros(len(X))

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        return kmeans.fit_predict(X)

    def run_dbscan(self, X):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return dbscan.fit_predict(X)

    def run_hierarchical(self, X):
        max_k = min(10, len(X) - 1)
        best_k = 2
        best_score = -1

        if max_k < 2:
            return np.zeros(len(X))

        for k in range(2, max_k + 1):
            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        agg = AgglomerativeClustering(n_clusters=best_k)
        return agg.fit_predict(X)

    def run_optics(self, X):
        optics = OPTICS(min_samples=self.min_samples, xi=0.05)
        return optics.fit_predict(X)


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
        self.motion_trails = deque(maxlen=30)  # Store motion trails for visualization
        self.heatmap_data = None  # Store heatmap data
        self.cluster_centroids = {}  # Store cluster centroids over time

        # Fluid motion analysis
        self.fluid_analyzer = FluidMotionAnalyzer()

        # Movement quality analysis
        self.movement_analyzer = MovementQualityAnalyzer()

        # Human detection
        self.yolo_model_size = 'n'  # Default to nano model
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("YOLOv8n initialized for person detection")
        except Exception as e:
            print(f"Error initializing YOLOv8n: {e}")
            self.yolo_model = None

        # UI setup
        self.setWindowTitle("Optical Flow Cluster Analyzer")
        self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(100, 100, 1800, 1000)
        self.setup_ui()

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

        # Signals
        self.timer.timeout.connect(self.update_frame)
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(5, 5, 5, 5)

        title_label = QLabel("Optical Flow Cluster Analyzer")
        title_label.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        left_layout.addWidget(title_label)

        source_group = QGroupBox("Video Source")
        source_layout = QVBoxLayout()

        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.setStyleSheet("background-color: #3498db; color: white;")
        self.webcam_btn.clicked.connect(self.start_webcam)
        source_layout.addWidget(self.webcam_btn)

        self.stop_webcam_btn = QPushButton("Stop Camera")
        self.stop_webcam_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        self.stop_webcam_btn.setEnabled(False)
        self.stop_webcam_btn.clicked.connect(self.stop_webcam)
        source_layout.addWidget(self.stop_webcam_btn)

        self.load_btn = QPushButton("Load Video")
        self.load_btn.setStyleSheet("background-color: #2ecc71; color: white;")
        self.load_btn.clicked.connect(self.load_video)
        source_layout.addWidget(self.load_btn)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        self.play_btn.clicked.connect(self.toggle_playback)
        source_layout.addWidget(self.play_btn)

        source_group.setLayout(source_layout)
        left_layout.addWidget(source_group)

        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["K-Means", "DBSCAN", "Hierarchical", "OPTICS"])
        analysis_layout.addWidget(QLabel("Clustering Method:"))
        analysis_layout.addWidget(self.method_combo)

        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.1, 5.0)
        self.epsilon_spin.setSingleStep(0.1)
        self.epsilon_spin.setValue(0.5)
        analysis_layout.addWidget(QLabel("Epsilon (for DBSCAN/OPTICS):"))
        analysis_layout.addWidget(self.epsilon_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 50)
        self.min_samples_spin.setValue(5)
        analysis_layout.addWidget(QLabel("Min Samples (for DBSCAN/OPTICS):"))
        analysis_layout.addWidget(self.min_samples_spin)

        # YOLO model selection
        self.yolo_model_combo = QComboBox()
        self.yolo_model_combo.addItems(["Nano (fastest)", "Small", "Medium", "Large (most accurate)"])
        self.yolo_model_combo.currentIndexChanged.connect(self.change_yolo_model)
        analysis_layout.addWidget(QLabel("YOLO Model Size:"))
        analysis_layout.addWidget(self.yolo_model_combo)

        self.analyze_btn = QPushButton("Analyze Optical Flow")
        self.analyze_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        self.analyze_btn.clicked.connect(self.analyze_optical_flow)
        analysis_layout.addWidget(self.analyze_btn)

        self.export_btn = QPushButton("Export Report")
        self.export_btn.setStyleSheet("background-color: #f39c12; color: white;")
        self.export_btn.clicked.connect(self.export_report)
        analysis_layout.addWidget(self.export_btn)

        analysis_group.setLayout(analysis_layout)
        left_layout.addWidget(analysis_group)

        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()

        self.flow_check = QCheckBox("Show Optical Flow")
        self.flow_check.setChecked(True)
        viz_layout.addWidget(self.flow_check)

        self.cluster_check = QCheckBox("Show Clusters")
        self.cluster_check.setChecked(True)
        viz_layout.addWidget(self.cluster_check)

        self.hopkins_check = QCheckBox("Show Hopkins Statistic")
        viz_layout.addWidget(self.hopkins_check)

        self.trails_check = QCheckBox("Show Motion Trails")
        self.trails_check.setChecked(True)
        viz_layout.addWidget(self.trails_check)

        self.heatmap_check = QCheckBox("Show Heatmap")
        self.heatmap_check.setChecked(False)
        viz_layout.addWidget(self.heatmap_check)

        self.centroids_check = QCheckBox("Show Cluster Centroids")
        self.centroids_check.setChecked(False)
        viz_layout.addWidget(self.centroids_check)

        viz_group.setLayout(viz_layout)
        left_layout.addWidget(viz_group)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(350)

        # Right panel setup
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

        # Video Tab
        video_tab = QWidget()
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(1200, 800)
        self.video_label.setScaledContents(True)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.video_label)
        video_tab.setLayout(video_layout)
        self.tab_widget.addTab(video_tab, "Optical Flow Analysis")

        # Dashboard Tab
        dashboard_tab = QWidget()
        dashboard_layout = QVBoxLayout()

        metrics_group = QGroupBox("Flow Metrics")
        metrics_layout = QHBoxLayout()
        self.metrics = {
            'total_flow': QLabel("0.00"),
            'avg_magnitude': QLabel("0.00"),
            'hopkins_stat': QLabel("0.00"),
            'clusters': QLabel("0")
        }
        for name, label in self.metrics.items():
            group = QGroupBox(name.replace('_', ' ').title())
            group_layout = QVBoxLayout()
            label.setFont(QFont('Arial', 16, QFont.Weight.Bold))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            group_layout.addWidget(label)
            group.setLayout(group_layout)
            metrics_layout.addWidget(group)
        metrics_group.setLayout(metrics_layout)
        dashboard_layout.addWidget(metrics_group)

        # Initialize dashboard plots
        self.dashboard_figure, (self.flow_ax, self.cluster_ax) = plt.subplots(1, 2, figsize=(12, 4))
        self.dashboard_canvas = FigureCanvas(self.dashboard_figure)
        dashboard_layout.addWidget(self.dashboard_canvas)

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(5)
        self.data_table.setHorizontalHeaderLabels([
            "Frame", "Flow Points", "Avg Magnitude", "Hopkins", "Clusters"
        ])
        dashboard_layout.addWidget(self.data_table)

        dashboard_tab.setLayout(dashboard_layout)
        self.tab_widget.addTab(dashboard_tab, "Dashboard")

        # Cluster Analysis Tab
        cluster_tab = QWidget()
        cluster_layout = QVBoxLayout()

        self.cluster_summary = QTextEdit()
        self.cluster_summary.setReadOnly(True)
        cluster_layout.addWidget(self.cluster_summary)

        self.cluster_figure, self.cluster_axes = plt.subplots(1, 2, figsize=(12, 4))
        self.cluster_canvas = FigureCanvas(self.cluster_figure)
        cluster_layout.addWidget(self.cluster_canvas)

        self.cluster_table = QTableWidget()
        self.cluster_table.setColumnCount(5)
        self.cluster_table.setHorizontalHeaderLabels(["Method", "Clusters", "Silhouette", "DB Index", "CH Score"])
        cluster_layout.addWidget(self.cluster_table)

        cluster_tab.setLayout(cluster_layout)
        self.tab_widget.addTab(cluster_tab, "Cluster Analysis")

        # Hopkins Analysis Tab
        hopkins_tab = QWidget()
        hopkins_layout = QVBoxLayout()

        self.hopkins_figure, self.hopkins_ax = plt.subplots(figsize=(10, 4))
        self.hopkins_canvas = FigureCanvas(self.hopkins_figure)
        hopkins_layout.addWidget(self.hopkins_canvas)

        self.hopkins_table = QTableWidget()
        self.hopkins_table.setColumnCount(3)
        self.hopkins_table.setHorizontalHeaderLabels(["Frame", "Hopkins", "Interpretation"])
        hopkins_layout.addWidget(self.hopkins_table)

        hopkins_tab.setLayout(hopkins_layout)
        self.tab_widget.addTab(hopkins_tab, "Clusterability Analysis")

        # Fluid Motion Analysis Tab
        self.setup_fluid_analysis_tab()

        # Movement Quality Analysis Tab
        self.setup_movement_quality_tab()

        # Origin Analysis Tab
        self.setup_origin_analysis_tab()

        right_layout.addWidget(self.tab_widget)
        right_panel.setLayout(right_layout)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def setup_fluid_analysis_tab(self):
        """Setup the new fluid motion analysis tab"""
        fluid_tab = QWidget()
        fluid_layout = QVBoxLayout()

        # Summary metrics
        self.fluid_metrics = {
            'fluid_percent': QLabel("0%"),
            'non_fluid_percent': QLabel("0%"),
            'num_patterns': QLabel("0"),
            'avg_hopkins_fluid': QLabel("0.00"),
            'avg_hopkins_non_fluid': QLabel("0.00")
        }

        metrics_group = QGroupBox("Fluid Motion Metrics")
        metrics_layout = QHBoxLayout()
        for name, label in self.fluid_metrics.items():
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
        self.motion_origins_table = QTableWidget()
        self.motion_origins_table.setColumnCount(5)
        self.motion_origins_table.setHorizontalHeaderLabels([
            "Pattern ID", "Start Frame", "Type", "Hopkins", "Avg Magnitude"
        ])
        fluid_layout.addWidget(QLabel("Motion Pattern Origins:"))
        fluid_layout.addWidget(self.motion_origins_table)

        # Proof frame display
        self.proof_frame_label = QLabel()
        self.proof_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.proof_frame_label.setMinimumSize(400, 300)
        fluid_layout.addWidget(QLabel("Proof Frame:"))
        fluid_layout.addWidget(self.proof_frame_label)

        # Connect table selection to show proof frames
        self.motion_origins_table.itemSelectionChanged.connect(self.show_proof_frame)

        fluid_tab.setLayout(fluid_layout)
        self.tab_widget.addTab(fluid_tab, "Fluid Motion Analysis")

    def setup_movement_quality_tab(self):
        """Setup the movement quality analysis tab"""
        movement_tab = QWidget()
        movement_layout = QVBoxLayout()

        # Training section
        training_group = QGroupBox("Movement Quality Analysis")
        training_layout = QVBoxLayout()

        # Auto analysis button
        self.auto_analyze_btn = QPushButton("Analyze Movement Quality")
        self.auto_analyze_btn.clicked.connect(self.auto_analyze_movement_quality)
        training_layout.addWidget(self.auto_analyze_btn)

        # Load ground truth button
        self.load_labels_btn = QPushButton("Load Ground Truth Labels")
        self.load_labels_btn.clicked.connect(self.load_ground_truth_labels)
        training_layout.addWidget(self.load_labels_btn)

        # Validate button
        self.validate_btn = QPushButton("Validate Against Ground Truth")
        self.validate_btn.clicked.connect(self.validate_against_ground_truth)
        self.validate_btn.setEnabled(False)
        training_layout.addWidget(self.validate_btn)

        # Format help button
        self.format_help_btn = QPushButton("Label Format Help")
        self.format_help_btn.clicked.connect(self.show_format_help)
        training_layout.addWidget(self.format_help_btn)

        self.model_status_label = QLabel("Status: Ready to analyze")
        training_layout.addWidget(self.model_status_label)

        # Label statistics display
        self.label_stats_label = QLabel("No labels loaded")
        self.label_stats_label.setWordWrap(True)
        training_layout.addWidget(self.label_stats_label)

        # Quality distribution display
        self.quality_dist_label = QLabel("No analysis performed yet")
        self.quality_dist_label.setWordWrap(True)
        training_layout.addWidget(self.quality_dist_label)

        training_group.setLayout(training_layout)
        movement_layout.addWidget(training_group)

        # Prediction section
        prediction_group = QGroupBox("Movement Quality Prediction")
        prediction_layout = QVBoxLayout()

        self.prediction_label = QLabel("Current Prediction: N/A")
        self.prediction_label.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        prediction_layout.addWidget(self.prediction_label)

        self.confidence_label = QLabel("Confidence: N/A")
        prediction_layout.addWidget(self.confidence_label)

        # Origin detection
        self.origin_label = QLabel("Movement Origin: N/A")
        prediction_layout.addWidget(self.origin_label)

        # Feature importance visualization
        self.feature_importance_figure, self.feature_importance_ax = plt.subplots(figsize=(10, 6))
        self.feature_importance_canvas = FigureCanvas(self.feature_importance_figure)
        prediction_layout.addWidget(self.feature_importance_canvas)

        prediction_group.setLayout(prediction_layout)
        movement_layout.addWidget(prediction_group)

        # Statistical testing section
        stats_group = QGroupBox("Analysis Summary")
        stats_layout = QVBoxLayout()

        self.stats_results = QTextEdit()
        self.stats_results.setReadOnly(True)
        stats_layout.addWidget(self.stats_results)

        stats_group.setLayout(stats_layout)
        movement_layout.addWidget(stats_group)

        movement_tab.setLayout(movement_layout)
        self.tab_widget.addTab(movement_tab, "Movement Quality Analysis")

    def setup_origin_analysis_tab(self):
        """Setup the origin of movement analysis tab"""
        origin_tab = QWidget()
        origin_layout = QVBoxLayout()

        # Origin metrics
        self.origin_metrics = {
            'expansion': QLabel("0%"),
            'contraction': QLabel("0%"),
            'rotation_cw': QLabel("0%"),
            'rotation_ccw': QLabel("0%"),
            'dominant_origin': QLabel("None")
        }

        metrics_group = QGroupBox("Movement Origin Metrics")
        metrics_layout = QHBoxLayout()
        for name, label in self.origin_metrics.items():
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
        self.origin_figure, self.origin_ax = plt.subplots(figsize=(10, 6))
        self.origin_canvas = FigureCanvas(self.origin_figure)
        origin_layout.addWidget(self.origin_canvas)

        # Origin history table
        self.origin_table = QTableWidget()
        self.origin_table.setColumnCount(4)
        self.origin_table.setHorizontalHeaderLabels(["Frame", "Origin Type", "Confidence", "Body Region"])
        origin_layout.addWidget(self.origin_table)

        origin_tab.setLayout(origin_layout)
        self.tab_widget.addTab(origin_tab, "Origin Analysis")

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

    def update_fluid_analysis(self):
        """Update the fluid motion analysis tab with latest results"""
        summary = self.fluid_analyzer.get_fluid_summary()
        if not summary:
            return

        # Update metrics
        self.fluid_metrics['fluid_percent'].setText(f"{summary['fluid_percentage']:.1f}%")
        self.fluid_metrics['non_fluid_percent'].setText(f"{summary['non_fluid_percentage']:.1f}%")
        self.fluid_metrics['num_patterns'].setText(str(summary['num_motion_patterns']))

        avg_fluid = summary.get('avg_hopkins_fluid', 0)
        avg_non_fluid = summary.get('avg_hopkins_non_fluid', 0)

        self.fluid_metrics['avg_hopkins_fluid'].setText(f"{avg_fluid:.3f}" if avg_fluid else "N/A")
        self.fluid_metrics['avg_hopkins_non_fluid'].setText(f"{avg_non_fluid:.3f}" if avg_non_fluid else "N/A")

        # Update motion origins table
        self.motion_origins_table.setRowCount(len(summary['motion_origins']))
        for i, (sig, data) in enumerate(summary['motion_origins'].items()):
            self.motion_origins_table.setItem(i, 0, QTableWidgetItem(sig[:15] + "..."))
            self.motion_origins_table.setItem(i, 1, QTableWidgetItem(str(data['start_frame'])))
            self.motion_origins_table.setItem(i, 2, QTableWidgetItem(data['type']))
            self.motion_origins_table.setItem(i, 3, QTableWidgetItem(f"{data['hopkins']:.3f}"))
            self.motion_origins_table.setItem(i, 4, QTableWidgetItem(f"{data['avg_magnitude']:.2f}"))

    def load_ground_truth_labels(self):
        """Load ground truth labels for movement quality validation"""
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Ground Truth Labels", "",
            "JSON Files (*.json);;CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
            options=options
        )

        if file_name:
            try:
                # Use the load_labels method that supports multiple formats
                self.movement_analyzer.ground_truth_labels, stats = self.movement_analyzer.load_labels(file_name)

                # Update UI with statistics
                stats_text = f"Loaded {stats['total_labels']} labels\n"
                stats_text += f"Frame-specific: {stats['frame_specific']}\n"
                stats_text += f"Range-based: {stats['range_based']}\n"
                stats_text += f"Unique labels: {', '.join(stats['unique_labels'])}"

                self.label_stats_label.setText(stats_text)
                self.validate_btn.setEnabled(True)
                self.status_bar.showMessage(
                    f"Loaded {len(self.movement_analyzer.ground_truth_labels)} ground truth labels from {os.path.basename(file_name)}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load labels: {str(e)}")

    def validate_against_ground_truth(self):
        """Validate automatic predictions against ground truth labels"""
        if not self.movement_analyzer.prediction_history:
            QMessageBox.warning(self, "Warning", "No predictions available for validation!")
            return

        results = self.movement_analyzer.validate_against_ground_truth(self.movement_analyzer.prediction_history)

        # Show results in a dialog
        dialog = ValidationResultsDialog(self)
        dialog.set_results(results)
        dialog.exec()

    def auto_analyze_movement_quality(self):
        """Automatically analyze movement quality using collected data"""
        if not self.optical_flow_data or len(self.optical_flow_data) < 10:
            QMessageBox.warning(self, "Warning", "Not enough data for analysis! Collect more frames first.")
            return

        # Extract features from all frames
        features = []
        for frame_data in self.optical_flow_data:
            feature_dict = self.movement_analyzer.extract_features(
                frame_data['flow_points'],
                frame_data.get('hopkins', 0.5)
            )
            if feature_dict:
                features.append(feature_dict)

        if len(features) < 5:
            QMessageBox.warning(self, "Warning", "Not enough valid features for analysis!")
            return

        # Auto-label movements
        labels = []
        for feature in features:
            label = self.movement_analyzer.auto_label_movement(feature)
            labels.append(label)

        # Train the model with auto-labeled data
        model, accuracy = self.movement_analyzer.train_model(features, labels)

        if model:
            self.model_status_label.setText(f"Model trained - Accuracy: {accuracy:.3f}")
            self.update_feature_importance_plot()
            self.update_quality_distribution(labels)
            self.generate_movement_summary(features, labels)
            self.status_bar.showMessage("Movement quality analysis completed")

            # Predict for the current frame
            self.predict_current_movement_quality()

    def update_quality_distribution(self, labels):
        """Update the quality distribution display"""
        from collections import Counter
        counts = Counter(labels)
        total = len(labels)

        dist_text = "Quality Distribution:\n"
        for quality in ['excellent', 'good', 'average', 'poor']:
            count = counts.get(quality, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            dist_text += f"{quality.title()}: {count} ({percentage:.1f}%)\n"

        self.quality_dist_label.setText(dist_text)

    def update_feature_importance_plot(self):
        """Update the feature importance visualization"""
        self.feature_importance_ax.clear()

        if self.movement_analyzer.feature_importances:
            features = list(self.movement_analyzer.feature_importances.keys())
            importances = list(self.movement_analyzer.feature_importances.values())

            # Sort by importance
            sorted_idx = np.argsort(importances)
            features = [features[i] for i in sorted_idx]
            importances = [importances[i] for i in sorted_idx]

            self.feature_importance_ax.barh(features, importances)
            self.feature_importance_ax.set_title('Feature Importances')
            self.feature_importance_ax.set_xlabel('Importance')

            self.feature_importance_figure.tight_layout()
            self.feature_importance_canvas.draw()

    def predict_current_movement_quality(self):
        """Predict movement quality for the current frame"""
        if not self.optical_flow_data:
            return

        # Get the latest frame data
        latest_frame = self.optical_flow_data[-1]
        features = self.movement_analyzer.extract_features(
            latest_frame['flow_points'],
            latest_frame.get('hopkins', 0.5)
        )

        if features:
            prediction, confidence = self.movement_analyzer.predict_movement_quality(features)

            # Store prediction
            frame_num = latest_frame['frame']
            self.movement_analyzer.prediction_history[frame_num] = {
                'prediction': prediction,
                'confidence': confidence,
                'features': features
            }

            # Update UI
            self.prediction_label.setText(f"Current Prediction: {prediction}")
            self.confidence_label.setText(f"Confidence: {confidence:.3f}")

            # Also detect movement origin
            if len(latest_frame['flow_points']) > 0:
                origin, origin_confidence, body_region = self.detect_movement_origin(latest_frame['flow_points'])
                self.origin_label.setText(f"Movement Origin: {origin} ({origin_confidence:.3f})")

                # Update origin analysis
                self.update_origin_analysis(frame_num, origin, origin_confidence, body_region)

    def detect_movement_origin(self, optical_flow_data):
        """Detect the origin of movement based on optical flow patterns"""
        if len(optical_flow_data) == 0:
            return "Unknown", 0.0, "Unknown"

        # Calculate centroid of movement
        points = optical_flow_data[:, :2]  # x,y coordinates
        vectors = optical_flow_data[:, 2:]  # flow vectors

        if len(points) == 0:
            return "Unknown", 0.0, "Unknown"

        centroid = np.mean(points, axis=0)

        # Calculate divergence (to detect expansion/contraction)
        divergence = np.mean(vectors[:, 0] * (points[:, 0] - centroid[0]) +
                             vectors[:, 1] * (points[:, 1] - centroid[1]))

        # Calculate curl (to detect rotational movement)
        curl = np.mean(vectors[:, 1] * (points[:, 0] - centroid[0]) -
                       vectors[:, 0] * (points[:, 1] - centroid[1]))

        # Determine origin type based on patterns
        origin_confidence = max(abs(divergence), abs(curl))

        # Determine body region based on point distribution
        if len(points) > 0:
            # Simple heuristic: if most points are in upper half, it's upper body
            upper_body_points = np.sum(points[:, 1] < centroid[1])
            body_region = "Upper Body" if upper_body_points > len(points) / 2 else "Lower Body"
        else:
            body_region = "Unknown"

        if abs(divergence) > abs(curl):
            if divergence > 0:
                return "Expansion", origin_confidence, body_region
            else:
                return "Contraction", origin_confidence, body_region
        else:
            if curl > 0:
                return "Clockwise Rotation", origin_confidence, body_region
            else:
                return "Counter-clockwise Rotation", origin_confidence, body_region

    def update_origin_analysis(self, frame_num, origin, confidence, body_region):
        """Update the origin analysis tab with new data"""
        # Store origin data in the current frame
        if self.optical_flow_data:
            self.optical_flow_data[-1]['origin'] = origin
            self.optical_flow_data[-1]['origin_confidence'] = confidence
            self.optical_flow_data[-1]['body_region'] = body_region

        # Update metrics
        origin_counts = {'expansion': 0, 'contraction': 0, 'rotation_cw': 0, 'rotation_ccw': 0}

        # Count all origin types in history
        for frame_data in self.optical_flow_data:
            if 'origin' in frame_data:
                origin_type = frame_data['origin']
                if "Expansion" in origin_type:
                    origin_counts['expansion'] += 1
                elif "Contraction" in origin_type:
                    origin_counts['contraction'] += 1
                elif "Clockwise" in origin_type:
                    origin_counts['rotation_cw'] += 1
                elif "Counter-clockwise" in origin_type:
                    origin_counts['rotation_ccw'] += 1

        total = sum(origin_counts.values())
        if total > 0:
            for origin_type in origin_counts:
                percentage = (origin_counts[origin_type] / total) * 100
                self.origin_metrics[origin_type].setText(f"{percentage:.1f}%")

            # Find dominant origin
            dominant = max(origin_counts, key=origin_counts.get)
            self.origin_metrics['dominant_origin'].setText(dominant.replace('_', ' ').title())

        # Add to origin table
        row = self.origin_table.rowCount()
        self.origin_table.insertRow(row)
        self.origin_table.setItem(row, 0, QTableWidgetItem(str(frame_num)))
        self.origin_table.setItem(row, 1, QTableWidgetItem(origin))
        self.origin_table.setItem(row, 2, QTableWidgetItem(f"{confidence:.3f}"))
        self.origin_table.setItem(row, 3, QTableWidgetItem(body_region))

        # Update origin visualization
        self.update_origin_visualization()

    def update_origin_visualization(self):
        """Update the origin visualization plot"""
        self.origin_ax.clear()

        # Count origin types
        origin_counts = {'Expansion': 0, 'Contraction': 0, 'Clockwise Rotation': 0, 'Counter-clockwise Rotation': 0}

        for frame_data in self.optical_flow_data:
            if 'origin' in frame_data:
                origin = frame_data['origin']
                if origin == "Expansion":
                    origin_counts['Expansion'] += 1
                elif origin == "Contraction":
                    origin_counts['Contraction'] += 1
                elif origin == "Clockwise Rotation":
                    origin_counts['Clockwise Rotation'] += 1
                elif origin == "Counter-clockwise Rotation":
                    origin_counts['Counter-clockwise Rotation'] += 1

        # Create bar chart
        origins = list(origin_counts.keys())
        counts = list(origin_counts.values())

        # Only plot if we have data
        if sum(counts) > 0:
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            bars = self.origin_ax.bar(origins, counts, color=colors)
            self.origin_ax.set_title('Movement Origin Distribution')
            self.origin_ax.set_ylabel('Count')

            # Add value labels on top of bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                self.origin_ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                    f'{count}', ha='center', va='bottom')

            # Rotate x labels for better readability
            plt.setp(self.origin_ax.get_xticklabels(), rotation=45, ha='right')
        else:
            self.origin_ax.text(0.5, 0.5, 'No origin data available\nAnalyze movement first',
                                ha='center', va='center', transform=self.origin_ax.transAxes)
            self.origin_ax.set_title('Movement Origin Distribution')

        self.origin_figure.tight_layout()
        self.origin_canvas.draw()

    def generate_movement_summary(self, features, labels):
        """Generate a summary of movement quality analysis"""
        if not features or not labels:
            return

        summary_text = "=== Movement Quality Analysis Summary ===\n\n"

        # Count quality categories
        from collections import Counter
        quality_counts = Counter(labels)
        total = len(labels)

        summary_text += "Quality Distribution:\n"
        for quality in ['excellent', 'good', 'average', 'poor']:
            count = quality_counts.get(quality, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            summary_text += f"{quality.title()}: {count} frames ({percentage:.1f}%)\n"

        summary_text += "\nKey Indicators:\n"

        # Calculate average values for each quality category
        quality_features = {q: [] for q in ['excellent', 'good', 'average', 'poor']}
        for i, label in enumerate(labels):
            if label in quality_features:
                quality_features[label].append(features[i])

        for quality in ['excellent', 'good', 'average', 'poor']:
            if quality_features[quality]:
                # Use safe calculations to avoid warnings
                hopkins_vals = [f['hopkins_statistic'] for f in quality_features[quality] if 'hopkins_statistic' in f]
                magnitude_vals = [f['avg_magnitude'] for f in quality_features[quality] if 'avg_magnitude' in f]
                consistency_vals = [f['direction_std'] for f in quality_features[quality] if 'direction_std' in f]

                avg_hopkins = np.mean(hopkins_vals) if hopkins_vals else 0
                avg_magnitude = np.mean(magnitude_vals) if magnitude_vals else 0
                avg_consistency = np.mean(consistency_vals) if consistency_vals else 0

                summary_text += f"\n{quality.title()} movement:\n"
                summary_text += f"  - Avg Hopkins: {avg_hopkins:.3f}\n"
                summary_text += f"  - Avg Magnitude: {avg_magnitude:.3f}\n"
                summary_text += f"  - Avg Direction Consistency: {avg_consistency:.3f}\n"

        summary_text += "\nRecommendations:\n"
        if quality_counts.get('poor', 0) / total > 0.3:
            summary_text += "- Significant portion of poor quality movement detected\n"
            summary_text += "- Focus on improving movement consistency and flow\n"

        if quality_counts.get('excellent', 0) / total > 0.4:
            summary_text += "- Excellent movement quality detected in many frames\n"
            summary_text += "- Maintain current movement patterns\n"

        self.stats_results.setText(summary_text)

    def change_yolo_model(self, index):
        """Change the YOLO model based on user selection"""
        model_sizes = ['n', 's', 'm', 'l']
        if 0 <= index < len(model_sizes):
            self.yolo_model_size = model_sizes[index]
            try:
                self.yolo_model = YOLO(f'yolov8{self.yolo_model_size}.pt')
                self.status_bar.showMessage(f"Switched to YOLOv8{self.yolo_model_size.upper()} model")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load YOLOv8{self.yolo_model_size.upper()} model: {e}")
                # Fall back to nano model
                self.yolo_model_combo.setCurrentIndex(0)
                self.yolo_model_size = 'n'
                try:
                    self.yolo_model = YOLO('yolov8n.pt')
                except:
                    self.yolo_model = None

    def start_webcam(self):
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

    def detect_humans(self, frame):
        """Detect humans in the frame using YOLOv8"""
        if self.yolo_model is None or frame is None:
            return []
        results = self.yolo_model(frame, verbose=False)
        boxes = []
        for result in results:
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                # Class 0 is 'person' in COCO
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, box[:4])
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    def update_frame(self):
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

            # Detect humans
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            human_rects = self.detect_humans(frame)

            # Process frame with optical flow only on human regions
            processed_frame, flow_data = self.process_frame_with_optical_flow(frame, gray, human_rects)

            h, w, ch = processed_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(
                processed_frame.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

            self.frame_count += 1
            if flow_data is not None and len(flow_data) > 0:
                # Calculate Hopkins statistic
                hopkins_stat = self.calculate_hopkins_statistic(flow_data)
                self.hopkins_history.append({
                    'frame': self.frame_count,
                    'hopkins': hopkins_stat
                })

                # Analyze fluid motion
                frame_data = {
                    'frame': self.frame_count,
                    'time': self.frame_count / 30,
                    'flow_points': flow_data,
                    'hopkins': hopkins_stat
                }
                self.optical_flow_data.append(frame_data)

                # Perform fluid motion analysis
                is_fluid, _ = self.fluid_analyzer.analyze_frame(
                    frame_data,
                    self.frame_count,
                    frame.copy()
                )

                # Update dashboard plots
                self.update_dashboard_plots(flow_data)

                if self.frame_count % 5 == 0:
                    self.update_metrics_display()
                    self.update_fluid_analysis()

                    # Predict movement quality if model is trained
                    if hasattr(self, 'movement_analyzer') and self.movement_analyzer.trained_model:
                        self.predict_current_movement_quality()

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

    def update_dashboard_plots(self, flow_data):
        try:
            self.flow_ax.clear()
            self.cluster_ax.clear()

            if len(flow_data) > 0:
                magnitudes = np.sqrt(flow_data[:, 2] ** 2 + flow_data[:, 3] ** 2)
                self.flow_ax.hist(magnitudes, bins=20, color='blue', alpha=0.7)
                self.flow_ax.set_title('Flow Magnitude Distribution')
                self.flow_ax.set_xlabel('Magnitude')
                self.flow_ax.set_ylabel('Count')

                directions = np.arctan2(flow_data[:, 3], flow_data[:, 2])
                self.cluster_ax.hist(directions, bins=20, color='green', alpha=0.7)
                self.cluster_ax.set_title('Flow Direction Distribution')
                self.cluster_ax.set_xlabel('Direction (radians)')
                self.cluster_ax.set_ylabel('Count')

            self.dashboard_figure.tight_layout()
            self.dashboard_canvas.draw()

        except Exception as e:
            print(f"Error updating dashboard plots: {e}")

    def process_frame_with_optical_flow(self, frame, gray, human_rects):
        # Initialize feature points only in human regions
        if self.prev_gray is None or self.frame_count % 30 == 0:
            self.prev_gray = gray
            # Create mask for human regions
            mask = np.zeros_like(gray)
            for (x, y, w, h) in human_rects:
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            # Get features only in human regions
            self.feature_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7,
                mask=mask
            )
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None

        if self.feature_points is None or len(self.feature_points) == 0:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None

        feature_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.feature_points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        good_new = feature_points[status == 1]
        good_old = self.feature_points[status == 1]

        flow_vectors = good_new - good_old
        flow_data = np.hstack((good_new, flow_vectors))

        self.prev_gray = gray.copy()
        self.feature_points = good_new.reshape(-1, 1, 2)

        vis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes around detected humans
        for (x, y, w, h) in human_rects:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw motion trails if enabled
        if self.trails_check.isChecked() and len(self.motion_trails) > 0:
            vis = self.draw_motion_trails(vis)

        # Draw heatmap if enabled
        if self.heatmap_check.isChecked():
            vis = self.draw_heatmap(vis, flow_data)

        # Draw cluster centroids if enabled
        if self.centroids_check.isChecked() and len(flow_data) > 0:
            vis = self.draw_cluster_centroids(vis, flow_data)

        if self.flow_check.isChecked():
            for new, old, vec in zip(good_new, good_old, flow_vectors):
                a, b = new.ravel()
                c, d = old.ravel()
                vec_x, vec_y = vec.ravel()
                vis = cv2.line(vis, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                vis = cv2.arrowedLine(vis, (int(a), int(b)),
                                      (int(a + vec_x * 5), int(b + vec_y * 5)),
                                      (0, 0, 255), 2)

        return vis, flow_data

    def draw_motion_trails(self, frame):
        """Draw motion trails on the frame"""
        # Create a copy of the frame to draw on
        result = frame.copy()

        # Define colors for trails (from recent to older)
        colors = [
            (0, 255, 0),  # Green (most recent)
            (0, 200, 0),  #
            (0, 150, 0),  #
            (0, 100, 0),  #
            (0, 50, 0)  # Dark green (oldest)
        ]

        # Draw trails for each point in recent frames
        for i, trail_frame in enumerate(self.motion_trails):
            if i >= len(colors):
                break

            color = colors[i]
            for point in trail_frame['flow_points']:
                x, y = int(point[0]), int(point[1])
                cv2.circle(result, (x, y), 2, color, -1)

        return result

    def draw_heatmap(self, frame, flow_data):
        """Draw a heatmap of movement density"""
        if len(flow_data) == 0:
            return frame

        # Create a heatmap based on movement magnitude
        h, w = frame.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        for point in flow_data:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < w and 0 <= y < h:
                magnitude = np.sqrt(point[2] ** 2 + point[3] ** 2)
                heatmap[y, x] += magnitude

        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        # Apply color map
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Blend with original frame
        alpha = 0.5
        result = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)

        return result

    def draw_cluster_centroids(self, frame, flow_data):
        """Draw cluster centroids and their movement over time"""
        if len(flow_data) == 0:
            return frame

        # Perform quick clustering
        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=5)
            points = flow_data[:, :2]
            labels = kmeans.fit_predict(points)
            centroids = kmeans.cluster_centers_

            # Store centroids for tracking
            frame_id = self.frame_count
            self.cluster_centroids[frame_id] = centroids

            # Draw centroids
            for i, centroid in enumerate(centroids):
                x, y = int(centroid[0]), int(centroid[1])
                cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)
                cv2.putText(frame, f"C{i}", (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Draw centroid trails (last 10 frames)
            recent_frames = sorted(self.cluster_centroids.keys())[-10:]
            for i, frame_key in enumerate(recent_frames):
                if frame_key in self.cluster_centroids:
                    alpha = i / len(recent_frames)  # Fade out older frames
                    color = (255, int(255 * alpha), 0)

                    centroids = self.cluster_centroids[frame_key]
                    for centroid in centroids:
                        x, y = int(centroid[0]), int(centroid[1])
                        cv2.circle(frame, (x, y), 3, color, -1)

        except Exception as e:
            print(f"Error drawing centroids: {e}")

        return frame

    def calculate_hopkins_statistic(self, data, sample_size=None):
        if sample_size is None:
            sample_size = min(50, len(data) // 2)

        if len(data) < 2:
            return 0.5

        X = data[:, :2]
        random_indices = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X[random_indices]

        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        X_uniform = np.column_stack([
            np.random.uniform(mins[0], maxs[0], size=sample_size),
            np.random.uniform(mins[1], maxs[1], size=sample_size)
        ])

        nbrs = NearestNeighbors(n_neighbors=2).fit(X)

        distances_data, _ = nbrs.kneighbors(X_sample)
        u_distances = distances_data[:, 1]

        distances_uniform, _ = nbrs.kneighbors(X_uniform)
        w_distances = distances_uniform[:, 0]

        numerator = np.sum(w_distances)
        denominator = np.sum(u_distances) + np.sum(w_distances)

        if denominator == 0:
            return 0.5

        hopkins_stat = numerator / denominator
        return hopkins_stat

    def analyze_optical_flow(self):
        if not hasattr(self, 'optical_flow_data') or len(self.optical_flow_data) < 5:
            QMessageBox.warning(self, "Warning", "Not enough data for analysis!")
            return

        method = self.method_combo.currentText()
        eps = self.epsilon_spin.value()
        min_samples = self.min_samples_spin.value()

        self.status_bar.showMessage("Running analysis in background...")
        self.analyze_btn.setEnabled(False)

        self.cluster_history = []
        self.clusterability_scores = []

        self.worker = OpticalFlowAnalyzerWorker(self.optical_flow_data, method, eps, min_samples)
        self.worker.analysis_finished.connect(self.on_analysis_finished)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self):
        self.analyze_btn.setEnabled(True)

    def on_analysis_finished(self, clustering_results, X):
        try:
            metrics = []
            valid_methods = []
            for name, labels in clustering_results.items():
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                try:
                    if n_clusters_found > 1:
                        silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else 0
                        db_index = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else float('inf')
                        ch_score = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else 0
                        metrics.append({
                            'Method': name,
                            'Clusters': n_clusters_found,
                            'Silhouette': f"{silhouette:.3f}",
                            'DB Index': f"{db_index:.3f}",
                            'CH Score': f"{ch_score:.3f}"
                        })
                        valid_methods.append(name)
                    else:
                        metrics.append({
                            'Method': name,
                            'Clusters': n_clusters_found,
                            'Silhouette': "N/A",
                            'DB Index': "N/A",
                            'CH Score': "N/A"
                        })
                except Exception as e:
                    print(f"[Main] Error calculating metrics for {name}: {str(e)}")
                    metrics.append({
                        'Method': name,
                        'Clusters': n_clusters_found,
                        'Silhouette': "Error",
                        'DB Index': "Error",
                        'CH Score': "Error"
                    })
            self.cluster_history.append({
                'metrics': metrics,
                'data': X,
                'results': clustering_results
            })
            self.update_cluster_metrics_table(metrics)
            selected_method = self.method_combo.currentText()
            self.update_cluster_visualization(X, clustering_results, valid_methods, selected_method=selected_method)
            self.update_hopkins_analysis()
            self.generate_cluster_summary(metrics)
            self.update_metrics_display()
            print("[Main] Analysis completed successfully")
            self.status_bar.showMessage("Analysis completed")
        except Exception as e:
            print(f"[Main] Analysis failed: {str(e)}")
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            self.status_bar.showMessage("Analysis failed")

    def update_cluster_metrics_table(self, metrics):
        self.cluster_table.setRowCount(len(metrics))
        for i, metric in enumerate(metrics):
            self.cluster_table.setItem(i, 0, QTableWidgetItem(metric['Method']))
            self.cluster_table.setItem(i, 1, QTableWidgetItem(str(metric['Clusters'])))
            self.cluster_table.setItem(i, 2, QTableWidgetItem(metric['Silhouette']))
            self.cluster_table.setItem(i, 3, QTableWidgetItem(metric['DB Index']))
            self.cluster_table.setItem(i, 4, QTableWidgetItem(metric.get('CH Score', 'N/A')))

    def update_cluster_visualization(self, X, clustering_results, valid_methods, selected_method=None):
        try:
            if not hasattr(self, 'cluster_axes') or len(self.cluster_axes) < 2:
                print("Cluster axes not properly initialized")
                return

            self.cluster_axes[0].clear()
            self.cluster_axes[1].clear()

            # Plot best clustering on left
            if valid_methods:
                best_method = None
                best_score = -1
                for metric in self.get_current_metrics():
                    if metric['Silhouette'] not in ["N/A", "Error"]:
                        score = float(metric['Silhouette'])
                        if score > best_score:
                            best_score = score
                            best_method = metric['Method']
                if best_method and best_method in clustering_results:
                    labels = clustering_results[best_method]
                    unique_labels = set(labels)
                    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
                    for k, col in zip(unique_labels, colors):
                        if k == -1:
                            col = (0.5, 0.5, 0.5, 0.3)
                        class_member_mask = (labels == k)
                        xy = X[class_member_mask]
                        if len(xy) > 0:
                            self.cluster_axes[0].scatter(
                                xy[:, 0], xy[:, 1],
                                color=[col],
                                label=f'Cluster {k}' if k != -1 else 'Noise',
                                alpha=0.6
                            )
                    self.cluster_axes[0].set_title(f'Best Clustering: {best_method}')
                    self.cluster_axes[0].legend()

            # Plot selected method on right
            if selected_method and selected_method in clustering_results:
                labels = clustering_results[selected_method]
                unique_labels = set(labels)
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        col = (0.5, 0.5, 0.5, 0.3)
                    class_member_mask = (labels == k)
                    xy = X[class_member_mask]
                    if len(xy) > 0:
                        self.cluster_axes[1].scatter(
                            xy[:, 0], xy[:, 1],
                            color=[col],
                            label=f'Cluster {k}' if k != -1 else 'Noise',
                            alpha=0.6
                        )
                self.cluster_axes[1].set_title(f'Selected: {selected_method}')
                self.cluster_axes[1].legend()
            else:
                self.cluster_axes[1].set_title('No result for selected method')

            self.cluster_figure.tight_layout()
            self.cluster_canvas.draw()

        except Exception as e:
            print(f"Error updating cluster visualization: {e}")

    def get_current_metrics(self):
        if not self.cluster_history:
            return []
        return self.cluster_history[-1]['metrics']

    def generate_cluster_summary(self, metrics):
        summary = ["=== Clustering Analysis Summary ==="]
        summary.append(f"Total frames analyzed: {len(self.optical_flow_data)}")
        summary.append(f"Total flow points: {sum(len(f['flow_points']) for f in self.optical_flow_data)}")

        valid_metrics = [m for m in metrics if m['Silhouette'] not in ["N/A", "Error"]]
        if valid_metrics:
            best_metric = max(valid_metrics, key=lambda x: float(x['Silhouette']))
            summary.append(f"\nBest clustering method: {best_metric['Method']}")
            summary.append(f" - Silhouette score: {best_metric['Silhouette']}")
            summary.append(f" - Number of clusters: {best_metric['Clusters']}")
            summary.append(f" - Davies-Bouldin index: {best_metric['DB Index']}")
            if 'CH Score' in best_metric:
                summary.append(f" - Calinski-Harabasz score: {best_metric['CH Score']}")

        if self.hopkins_history:
            avg_hopkins = np.mean([h['hopkins'] for h in self.hopkins_history])
            summary.append(f"\nAverage Hopkins statistic: {avg_hopkins:.3f}")

            if avg_hopkins > 0.75:
                summary.append("The data shows strong clustering tendency.")
            elif avg_hopkins > 0.5:
                summary.append("The data shows some clustering tendency.")
            elif avg_hopkins > 0.3:
                summary.append("The data appears mostly random.")
            else:
                summary.append("The data appears uniformly distributed.")

        # Add fluid motion summary
        fluid_summary = self.fluid_analyzer.get_fluid_summary()
        if fluid_summary:
            summary.append("\n=== Fluid Motion Analysis ===")
            summary.append(f"Fluid frames: {fluid_summary['fluid_percentage']:.1f}%")
            summary.append(f"Non-fluid frames: {fluid_summary['non_fluid_percentage']:.1f}%")
            summary.append(f"Unique motion patterns: {fluid_summary['num_motion_patterns']}")
            if 'avg_hopkins_fluid' in fluid_summary:
                summary.append(f"Avg Hopkins (fluid): {fluid_summary['avg_hopkins_fluid']:.3f}")
            if 'avg_hopkins_non_fluid' in fluid_summary:
                summary.append(f"Avg Hopkins (non-fluid): {fluid_summary['avg_hopkins_non_fluid']:.3f}")

        summary.append("\n=== Recommendations ===")
        if valid_metrics:
            best_method = best_metric['Method']
            if best_method in ["DBSCAN", "OPTICS"]:
                summary.append("- Your data may contain noise or varying density clusters.")
                summary.append(
                    f"- Current parameters: eps={self.epsilon_spin.value()}, min_samples={self.min_samples_spin.value()}")
            elif best_method == "K-Means":
                summary.append("- Your data appears to have well-separated, spherical clusters.")
                summary.append(f"- Optimal number of clusters found: {best_metric['Clusters']}")
            elif best_method == "Hierarchical":
                summary.append("- Your data may have hierarchical cluster structure.")
                summary.append(f"- Optimal number of clusters found: {best_metric['Clusters']}")

        self.cluster_summary.setText('\n'.join(summary))

    def update_hopkins_analysis(self):
        try:
            if not self.hopkins_history:
                print("No Hopkins history data available")
                return

            if not hasattr(self, 'hopkins_ax'):
                print("Hopkins axis not initialized")
                return

            frames = [h['frame'] for h in self.hopkins_history]
            hopkins_values = [h['hopkins'] for h in self.hopkins_history]

            interpretations = []
            for val in hopkins_values:
                if val > 0.75:
                    interpretations.append("Highly Clusterable")
                elif val > 0.5:
                    interpretations.append("Clusterable")
                elif val > 0.3:
                    interpretations.append("Random")
                else:
                    interpretations.append("Uniform")

            self.hopkins_table.setRowCount(len(self.hopkins_history))
            for i, (frame, val, interp) in enumerate(zip(frames, hopkins_values, interpretations)):
                self.hopkins_table.setItem(i, 0, QTableWidgetItem(str(frame)))
                self.hopkins_table.setItem(i, 1, QTableWidgetItem(f"{val:.3f}"))
                self.hopkins_table.setItem(i, 2, QTableWidgetItem(interp))

            self.hopkins_ax.clear()

            if len(frames) > 0 and len(hopkins_values) > 0:
                self.hopkins_ax.plot(frames, hopkins_values, 'b-', label='Hopkins Statistic')
                self.hopkins_ax.axhline(0.5, color='r', linestyle='--', label='Random')
                self.hopkins_ax.axhline(0.75, color='g', linestyle=':', label='Clusterable Threshold')

                self.hopkins_ax.fill_between(frames, 0.75, 1.0, color='green', alpha=0.1, label='Highly Clusterable')
                self.hopkins_ax.fill_between(frames, 0.5, 0.75, color='yellow', alpha=0.1, label='Clusterable')
                self.hopkins_ax.fill_between(frames, 0.3, 0.5, color='orange', alpha=0.1, label='Random')
                self.hopkins_ax.fill_between(frames, 0.0, 0.3, color='red', alpha=0.1, label='Uniform')

                self.hopkins_ax.set_title('Clusterability Over Time (Hopkins Statistic)')
                self.hopkins_ax.set_xlabel('Frame')
                self.hopkins_ax.set_ylabel('Hopkins Statistic')
                self.hopkins_ax.set_ylim(0, 1)
                self.hopkins_ax.legend(loc='upper right')

                self.hopkins_figure.tight_layout()
                self.hopkins_canvas.draw()

        except Exception as e:
            print(f"Error updating Hopkins analysis: {e}")

    def update_metrics_display(self):
        if not self.optical_flow_data:
            return

        magnitudes = []
        total_clusters = set()
        for frame in self.optical_flow_data:
            flow_points = frame['flow_points']
            if len(flow_points) > 0:
                magnitudes.extend(np.sqrt(flow_points[:, 2] ** 2 + flow_points[:, 3] ** 2))
                if 'cluster_labels' not in frame or len(frame['cluster_labels']) != len(flow_points):
                    try:
                        n_clusters = 2 if len(flow_points) > 5 else 1
                        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
                        labels = kmeans.fit_predict(flow_points[:, :2])
                        frame['cluster_labels'] = labels
                    except Exception:
                        frame['cluster_labels'] = [0] * len(flow_points)
                total_clusters.update(frame['cluster_labels'])
            else:
                frame['cluster_labels'] = []

        avg_magnitude = np.mean(magnitudes) if magnitudes else 0
        total_flow = np.sum(magnitudes) if magnitudes else 0
        unique_clusters = len(total_clusters) if total_clusters else 0

        latest_hopkins = self.hopkins_history[-1]['hopkins'] if self.hopkins_history else 0.5

        self.metrics['total_flow'].setText(f"{total_flow:.2f}")
        self.metrics['avg_magnitude'].setText(f"{avg_magnitude:.2f}")
        self.metrics['hopkins_stat'].setText(f"{latest_hopkins:.3f}")
        self.metrics['clusters'].setText(str(unique_clusters))

        # Update the data table with per-frame statistics
        self.data_table.setRowCount(len(self.optical_flow_data))
        for i, frame in enumerate(self.optical_flow_data):
            n_points = len(frame['flow_points'])
            if n_points > 0:
                frame_magnitudes = np.sqrt(frame['flow_points'][:, 2] ** 2 + frame['flow_points'][:, 3] ** 2)
                avg_frame_magnitude = np.mean(frame_magnitudes)
            else:
                avg_frame_magnitude = 0

            frame_hopkins = next(
                (h['hopkins'] for h in self.hopkins_history if h['frame'] == frame['frame']),
                0.5
            )

            frame_clusters = len(set(frame['cluster_labels'])) if n_points > 0 else 0

            self.data_table.setItem(i, 0, QTableWidgetItem(str(frame['frame'])))
            self.data_table.setItem(i, 1, QTableWidgetItem(str(n_points)))
            self.data_table.setItem(i, 2, QTableWidgetItem(f"{avg_frame_magnitude:.2f}"))
            self.data_table.setItem(i, 3, QTableWidgetItem(f"{frame_hopkins:.3f}"))
            self.data_table.setItem(i, 4, QTableWidgetItem(str(frame_clusters)))

    def export_report(self):
        if not self.optical_flow_data:
            QMessageBox.warning(self, "Warning", "No analysis data to export!")
            return

        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Report", "",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )

        if not file_name:
            return

        # Determine format based on selected filter or file extension
        if selected_filter == "CSV Files (*.csv)" or file_name.lower().endswith('.csv'):
            export_format = 'csv'
        else:
            # Default to Excel
            if not file_name.lower().endswith('.xlsx'):
                file_name += '.xlsx'
            export_format = 'excel'

        # Create a dictionary to hold all our data
        report_data = {}

        # 1. Frame-by-frame optical flow metrics
        frame_data = []
        for frame in self.optical_flow_data:
            frame_data.append({
                'frame': frame['frame'],
                'time': frame['time'],
                'flow_points': len(frame['flow_points']),
                'avg_magnitude': np.mean(np.sqrt(frame['flow_points'][:, 2] ** 2 + frame['flow_points'][:, 3] ** 2))
                if len(frame['flow_points']) > 0 else 0,
                'hopkins': next((h['hopkins'] for h in self.hopkins_history
                                 if h['frame'] == frame['frame']), 0.5),
                'clusters': len(set(frame.get('cluster_labels', [])))
            })
        report_data['frame_metrics'] = pd.DataFrame(frame_data)

        # 2. Cluster metrics from the cluster table
        cluster_metrics = []
        for row in range(self.cluster_table.rowCount()):
            cluster_metrics.append({
                'method': self.cluster_table.item(row, 0).text(),
                'clusters': self.cluster_table.item(row, 1).text(),
                'silhouette': self.cluster_table.item(row, 2).text(),
                'db_index': self.cluster_table.item(row, 3).text(),
                'ch_score': self.cluster_table.item(row, 4).text()
            })
        report_data['cluster_metrics'] = pd.DataFrame(cluster_metrics)

        # 3. Hopkins statistics from the hopkins table
        hopkins_data = []
        for row in range(self.hopkins_table.rowCount()):
            hopkins_data.append({
                'frame': self.hopkins_table.item(row, 0).text(),
                'hopkins': self.hopkins_table.item(row, 1).text(),
                'interpretation': self.hopkins_table.item(row, 2).text()
            })
        report_data['hopkins_analysis'] = pd.DataFrame(hopkins_data)

        # 4. Current metrics display
        report_data['current_metrics'] = pd.DataFrame({
            'metric': ['Total Flow', 'Average Magnitude', 'Hopkins Statistic', 'Clusters'],
            'value': [
                self.metrics['total_flow'].text(),
                self.metrics['avg_magnitude'].text(),
                self.metrics['hopkins_stat'].text(),
                self.metrics['clusters'].text()
            ]
        })

        # 5. Cluster analysis summary
        summary_text = self.cluster_summary.toPlainText()
        summary_sections = []
        current_section = {'title': '', 'content': ''}

        for line in summary_text.split('\n'):
            if line.startswith('===') and line.endswith('==='):
                if current_section['title']:
                    summary_sections.append(current_section)
                current_section = {'title': line.strip('= '), 'content': ''}
            else:
                current_section['content'] += line + '\n'

        if current_section['title']:
            summary_sections.append(current_section)

        report_data['cluster_summary'] = pd.DataFrame(summary_sections)

        # 6. Fluid motion analysis data
        fluid_summary = self.fluid_analyzer.get_fluid_summary()
        if fluid_summary:
            # Convert motion origins to DataFrame
            origins_data = []
            for sig, data in fluid_summary['motion_origins'].items():
                origins_data.append({
                    'pattern_signature': sig[:50],  # Truncate long signatures
                    'start_frame': data['start_frame'],
                    'type': data['type'],
                    'hopkins': data['hopkins'],
                    'avg_magnitude': data['avg_magnitude']
                })

            report_data['fluid_analysis'] = pd.DataFrame({
                'metric': ['Fluid Frames %', 'Non-Fluid Frames %', 'Unique Motion Patterns'],
                'value': [
                    f"{fluid_summary['fluid_percentage']:.1f}%",
                    f"{fluid_summary['non_fluid_percentage']:.1f}%",
                    fluid_summary['num_motion_patterns']
                ]
            })

            report_data['motion_origins'] = pd.DataFrame(origins_data)

        # 7. Movement quality analysis data
        if hasattr(self, 'movement_analyzer') and self.movement_analyzer.prediction_history:
            movement_data = []
            for frame_num, pred_data in self.movement_analyzer.prediction_history.items():
                movement_data.append({
                    'frame': frame_num,
                    'prediction': pred_data['prediction'],
                    'confidence': pred_data['confidence']
                })
            report_data['movement_quality'] = pd.DataFrame(movement_data)

        # 8. Origin analysis data
        origin_data = []
        for row in range(self.origin_table.rowCount()):
            origin_data.append({
                'frame': self.origin_table.item(row, 0).text(),
                'origin_type': self.origin_table.item(row, 1).text(),
                'confidence': self.origin_table.item(row, 2).text(),
                'body_region': self.origin_table.item(row, 3).text()
            })
        if origin_data:
            report_data['origin_analysis'] = pd.DataFrame(origin_data)

        try:
            if export_format == 'excel':
                # Write all data to a single Excel file with multiple sheets
                with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                    for sheet_name, df in report_data.items():
                        # Clean sheet name to be Excel-compatible
                        clean_sheet_name = sheet_name[:31].replace(':', '_').replace('\\', '_').replace('/', '_')
                        df.to_excel(writer, sheet_name=clean_sheet_name, index=False)

                self.status_bar.showMessage(f"Full report saved: {os.path.basename(file_name)}")
                QMessageBox.information(self, "Success", "Excel report exported successfully with all analysis data!")
            else:
                # For CSV, we'll create multiple files with suffixes
                base_name = os.path.splitext(file_name)[0]
                for name, df in report_data.items():
                    csv_file = f"{base_name}_{name}.csv"
                    df.to_csv(csv_file, index=False)

                self.status_bar.showMessage(f"CSV reports saved with prefix: {os.path.basename(base_name)}")
                QMessageBox.information(self, "Success",
                                        f"Multiple CSV files exported with prefix:\n{os.path.basename(base_name)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export report: {str(e)}")
            self.status_bar.showMessage("Report export failed")

    def show_format_help(self):
        """Show the label format help dialog"""
        dialog = LabelFormatHelpDialog(self)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpticalFlowClusterAnalyzer()
    window.show()
    sys.exit(app.exec())