import numpy as np
import json
import csv
from collections import deque, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import ttest_ind, f_oneway, chi2_contingency


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