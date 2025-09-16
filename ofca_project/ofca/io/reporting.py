"""
Reporting module for Optical Flow Cluster Analyzer.
Handles export and generation of analysis reports in various formats.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import json
import csv


class ReportGenerator:
    """Generates comprehensive analysis reports in various formats"""

    def __init__(self, parent_app):
        """
        Initialize the report generator.

        Args:
            parent_app: The main application instance
        """
        self.parent = parent_app

    def export_analysis_report(self):
        """Export a comprehensive analysis report"""
        if not hasattr(self.parent, 'optical_flow_data') or len(self.parent.optical_flow_data) == 0:
            QMessageBox.warning(self.parent, "Warning", "No analysis data to export!")
            return False

        # Get export format from user
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self.parent, "Save Analysis Report", "",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;JSON Files (*.json);;HTML Files (*.html);;All Files (*)"
        )

        if not file_name:
            return False

        # Determine format based on selected filter or file extension
        if selected_filter == "CSV Files (*.csv)" or file_name.lower().endswith('.csv'):
            return self.export_csv_report(file_name)
        elif selected_filter == "JSON Files (*.json)" or file_name.lower().endswith('.json'):
            return self.export_json_report(file_name)
        elif selected_filter == "HTML Files (*.html)" or file_name.lower().endswith('.html'):
            return self.export_html_report(file_name)
        else:
            # Default to Excel
            if not file_name.lower().endswith('.xlsx'):
                file_name += '.xlsx'
            return self.export_excel_report(file_name)

    def export_excel_report(self, file_path):
        """Export report to Excel format"""
        try:
            # Create a dictionary to hold all our data
            report_data = self._prepare_report_data()

            # Write all data to a single Excel file with multiple sheets
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name, df in report_data.items():
                    # Clean sheet name to be Excel-compatible
                    clean_sheet_name = self._clean_sheet_name(sheet_name)
                    df.to_excel(writer, sheet_name=clean_sheet_name, index=False)

            self.parent.status_bar.showMessage(f"Excel report saved: {os.path.basename(file_path)}")
            QMessageBox.information(self.parent, "Success",
                                    "Excel report exported successfully with all analysis data!")
            return True

        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to export Excel report: {str(e)}")
            self.parent.status_bar.showMessage("Report export failed")
            return False

    def export_csv_report(self, file_path):
        """Export report to multiple CSV files"""
        try:
            base_name = os.path.splitext(file_path)[0]
            report_data = self._prepare_report_data()

            for name, df in report_data.items():
                csv_file = f"{base_name}_{name}.csv"
                df.to_csv(csv_file, index=False)

            self.parent.status_bar.showMessage(f"CSV reports saved with prefix: {os.path.basename(base_name)}")
            QMessageBox.information(self.parent, "Success",
                                    f"Multiple CSV files exported with prefix:\n{os.path.basename(base_name)}")
            return True

        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to export CSV reports: {str(e)}")
            return False

    def export_json_report(self, file_path):
        """Export report to JSON format"""
        try:
            report_data = self._prepare_comprehensive_data()

            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=self._json_serializer)

            self.parent.status_bar.showMessage(f"JSON report saved: {os.path.basename(file_path)}")
            QMessageBox.information(self.parent, "Success", "JSON report exported successfully!")
            return True

        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to export JSON report: {str(e)}")
            return False

    def export_html_report(self, file_path):
        """Export report to HTML format"""
        try:
            report_data = self._prepare_comprehensive_data()
            html_content = self._generate_html_report(report_data)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.parent.status_bar.showMessage(f"HTML report saved: {os.path.basename(file_path)}")
            QMessageBox.information(self.parent, "Success", "HTML report exported successfully!")
            return True

        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to export HTML report: {str(e)}")
            return False

    def _prepare_report_data(self):
        """Prepare all report data as DataFrames"""
        report_data = {}

        # 1. Frame-by-frame optical flow metrics
        frame_data = []
        for frame in self.parent.optical_flow_data:
            frame_info = {
                'frame': frame['frame'],
                'time': frame['time'],
                'flow_points': len(frame['flow_points']),
                'avg_magnitude': 0,
                'hopkins': 0.5,
                'clusters': 0
            }

            if len(frame['flow_points']) > 0:
                magnitudes = np.sqrt(frame['flow_points'][:, 2] ** 2 + frame['flow_points'][:, 3] ** 2)
                frame_info['avg_magnitude'] = np.mean(magnitudes)

                # Get cluster count
                if 'cluster_labels' in frame:
                    frame_info['clusters'] = len(set(frame['cluster_labels']))

            # Get Hopkins statistic
            hopkins_entry = next((h for h in self.parent.hopkins_history
                                  if h['frame'] == frame['frame']), None)
            if hopkins_entry:
                frame_info['hopkins'] = hopkins_entry['hopkins']

            frame_data.append(frame_info)

        report_data['frame_metrics'] = pd.DataFrame(frame_data)

        # 2. Cluster metrics
        cluster_metrics = []
        if hasattr(self.parent, 'cluster_table') and self.parent.cluster_table.rowCount() > 0:
            for row in range(self.parent.cluster_table.rowCount()):
                cluster_metrics.append({
                    'method': self.parent.cluster_table.item(row, 0).text(),
                    'clusters': self.parent.cluster_table.item(row, 1).text(),
                    'silhouette': self.parent.cluster_table.item(row, 2).text(),
                    'db_index': self.parent.cluster_table.item(row, 3).text(),
                    'ch_score': self.parent.cluster_table.item(row, 4).text()
                })

        report_data['cluster_metrics'] = pd.DataFrame(cluster_metrics)

        # 3. Hopkins statistics
        hopkins_data = []
        for h in self.parent.hopkins_history:
            interpretation = self._interpret_hopkins(h['hopkins'])
            hopkins_data.append({
                'frame': h['frame'],
                'hopkins': h['hopkins'],
                'interpretation': interpretation
            })

        report_data['hopkins_analysis'] = pd.DataFrame(hopkins_data)

        # 4. Current metrics
        current_metrics = []
        if hasattr(self.parent, 'metrics'):
            for name, label in self.parent.metrics.items():
                current_metrics.append({
                    'metric': name.replace('_', ' ').title(),
                    'value': label.text()
                })

        report_data['current_metrics'] = pd.DataFrame(current_metrics)

        # 5. Fluid motion analysis
        fluid_summary = self.parent.fluid_analyzer.get_fluid_summary() if hasattr(self.parent, 'fluid_analyzer') else {}
        if fluid_summary:
            report_data['fluid_analysis'] = pd.DataFrame({
                'metric': ['Fluid Frames %', 'Non-Fluid Frames %', 'Unique Motion Patterns'],
                'value': [
                    f"{fluid_summary.get('fluid_percentage', 0):.1f}%",
                    f"{fluid_summary.get('non_fluid_percentage', 0):.1f}%",
                    fluid_summary.get('num_motion_patterns', 0)
                ]
            })

            # Motion origins
            origins_data = []
            for sig, data in fluid_summary.get('motion_origins', {}).items():
                origins_data.append({
                    'pattern_signature': sig[:50],
                    'start_frame': data.get('start_frame', 0),
                    'type': data.get('type', 'unknown'),
                    'hopkins': data.get('hopkins', 0),
                    'avg_magnitude': data.get('avg_magnitude', 0)
                })

            report_data['motion_origins'] = pd.DataFrame(origins_data)

        # 6. Movement quality predictions
        movement_data = []
        if hasattr(self.parent, 'movement_analyzer') and hasattr(self.parent.movement_analyzer, 'prediction_history'):
            for frame_num, pred_data in self.parent.movement_analyzer.prediction_history.items():
                movement_data.append({
                    'frame': frame_num,
                    'prediction': pred_data.get('prediction', 'unknown'),
                    'confidence': pred_data.get('confidence', 0)
                })

        report_data['movement_quality'] = pd.DataFrame(movement_data)

        # 7. Origin analysis
        origin_data = []
        if hasattr(self.parent, 'origin_table') and self.parent.origin_table.rowCount() > 0:
            for row in range(self.parent.origin_table.rowCount()):
                origin_data.append({
                    'frame': self.parent.origin_table.item(row, 0).text(),
                    'origin_type': self.parent.origin_table.item(row, 1).text(),
                    'confidence': self.parent.origin_table.item(row, 2).text(),
                    'body_region': self.parent.origin_table.item(row, 3).text()
                })

        report_data['origin_analysis'] = pd.DataFrame(origin_data)

        return report_data

    def _prepare_comprehensive_data(self):
        """Prepare comprehensive data for JSON/HTML export"""
        comprehensive_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_frames': len(self.parent.optical_flow_data),
                'analysis_duration': self.parent.optical_flow_data[-1]['time'] if self.parent.optical_flow_data else 0,
                'software_version': 'OFCA 1.0.0'
            },
            'frame_metrics': [],
            'cluster_analysis': {},
            'hopkins_analysis': [],
            'fluid_motion_analysis': {},
            'movement_quality_analysis': [],
            'origin_analysis': []
        }

        # Frame metrics
        for frame in self.parent.optical_flow_data:
            frame_data = {
                'frame': frame['frame'],
                'time': frame['time'],
                'flow_points': len(frame['flow_points']),
                'avg_magnitude': 0,
                'hopkins': 0.5
            }

            if len(frame['flow_points']) > 0:
                magnitudes = np.sqrt(frame['flow_points'][:, 2] ** 2 + frame['flow_points'][:, 3] ** 2)
                frame_data['avg_magnitude'] = float(np.mean(magnitudes))

            # Get Hopkins statistic
            hopkins_entry = next((h for h in self.parent.hopkins_history
                                  if h['frame'] == frame['frame']), None)
            if hopkins_entry:
                frame_data['hopkins'] = hopkins_entry['hopkins']
                frame_data['hopkins_interpretation'] = self._interpret_hopkins(hopkins_entry['hopkins'])

            comprehensive_data['frame_metrics'].append(frame_data)

        # Cluster analysis
        if hasattr(self.parent, 'cluster_table') and self.parent.cluster_table.rowCount() > 0:
            comprehensive_data['cluster_analysis']['methods'] = []
            for row in range(self.parent.cluster_table.rowCount()):
                comprehensive_data['cluster_analysis']['methods'].append({
                    'method': self.parent.cluster_table.item(row, 0).text(),
                    'clusters': self.parent.cluster_table.item(row, 1).text(),
                    'silhouette': self.parent.cluster_table.item(row, 2).text(),
                    'db_index': self.parent.cluster_table.item(row, 3).text(),
                    'ch_score': self.parent.cluster_table.item(row, 4).text()
                })

        # Hopkins analysis
        for h in self.parent.hopkins_history:
            comprehensive_data['hopkins_analysis'].append({
                'frame': h['frame'],
                'hopkins': h['hopkins'],
                'interpretation': self._interpret_hopkins(h['hopkins'])
            })

        # Fluid motion analysis
        fluid_summary = self.parent.fluid_analyzer.get_fluid_summary() if hasattr(self.parent, 'fluid_analyzer') else {}
        if fluid_summary:
            comprehensive_data['fluid_motion_analysis'] = {
                'fluid_percentage': fluid_summary.get('fluid_percentage', 0),
                'non_fluid_percentage': fluid_summary.get('non_fluid_percentage', 0),
                'num_motion_patterns': fluid_summary.get('num_motion_patterns', 0),
                'motion_origins': fluid_summary.get('motion_origins', {})
            }

        # Movement quality analysis
        if hasattr(self.parent, 'movement_analyzer') and hasattr(self.parent.movement_analyzer, 'prediction_history'):
            for frame_num, pred_data in self.parent.movement_analyzer.prediction_history.items():
                comprehensive_data['movement_quality_analysis'].append({
                    'frame': frame_num,
                    'prediction': pred_data.get('prediction', 'unknown'),
                    'confidence': pred_data.get('confidence', 0),
                    'features': pred_data.get('features', {})
                })

        # Origin analysis
        if hasattr(self.parent, 'origin_table') and self.parent.origin_table.rowCount() > 0:
            for row in range(self.parent.origin_table.rowCount()):
                comprehensive_data['origin_analysis'].append({
                    'frame': int(self.parent.origin_table.item(row, 0).text()),
                    'origin_type': self.parent.origin_table.item(row, 1).text(),
                    'confidence': float(self.parent.origin_table.item(row, 2).text()),
                    'body_region': self.parent.origin_table.item(row, 3).text()
                })

        return comprehensive_data

    def _generate_html_report(self, data):
        """Generate a comprehensive HTML report"""
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>OFCA Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f7f7fa; }}
                .header {{ background-color: #1976d2; color: white; padding: 20px; border-radius: 8px; }}
                .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-card {{ background-color: #e3eafc; padding: 15px; margin: 10px; border-radius: 6px; display: inline-block; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #1976d2; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 6px; }}
                .warning {{ background-color: #ffecb3; padding: 15px; border-radius: 6px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Optical Flow Cluster Analyzer Report</h1>
                <p>Generated on: {data['metadata']['export_date']}</p>
                <p>Total Frames Analyzed: {data['metadata']['total_frames']}</p>
                <p>Analysis Duration: {data['metadata']['analysis_duration']:.2f} seconds</p>
            </div>

            <div class="section">
                <h2>Frame Metrics Summary</h2>
                <div class="metric-card">
                    <h3>Total Flow Points</h3>
                    <p>{sum(f['flow_points'] for f in data['frame_metrics']):,}</p>
                </div>
                <div class="metric-card">
                    <h3>Average Magnitude</h3>
                    <p>{np.mean([f['avg_magnitude'] for f in data['frame_metrics']]):.3f}</p>
                </div>
                <div class="metric-card">
                    <h3>Average Hopkins</h3>
                    <p>{np.mean([f['hopkins'] for f in data['frame_metrics']]):.3f}</p>
                </div>
            </div>

            <div class="section">
                <h2>Cluster Analysis</h2>
                {"".join(self._generate_cluster_html(data['cluster_analysis']))}
            </div>

            <div class="section">
                <h2>Fluid Motion Analysis</h2>
                {self._generate_fluid_motion_html(data['fluid_motion_analysis'])}
            </div>

            <div class="section">
                <h2>Movement Quality Analysis</h2>
                {self._generate_movement_quality_html(data['movement_quality_analysis'])}
            </div>

            <div class="section">
                <h2>Origin Analysis</h2>
                {self._generate_origin_html(data['origin_analysis'])}
            </div>
        </body>
        </html>
        """

        return html_template

    def _generate_cluster_html(self, cluster_data):
        """Generate HTML for cluster analysis section"""
        if not cluster_data.get('methods'):
            return "<p>No cluster analysis data available.</p>"

        html = "<table><tr><th>Method</th><th>Clusters</th><th>Silhouette</th><th>DB Index</th><th>CH Score</th></tr>"
        for method in cluster_data['methods']:
            html += f"<tr><td>{method['method']}</td><td>{method['clusters']}</td><td>{method['silhouette']}</td><td>{method['db_index']}</td><td>{method['ch_score']}</td></tr>"
        html += "</table>"

        return html

    def _generate_fluid_motion_html(self, fluid_data):
        """Generate HTML for fluid motion analysis section"""
        if not fluid_data:
            return "<p>No fluid motion analysis data available.</p>"

        html = f"""
        <div class="metric-card">
            <h3>Fluid Frames</h3>
            <p>{fluid_data.get('fluid_percentage', 0):.1f}%</p>
        </div>
        <div class="metric-card">
            <h3>Non-Fluid Frames</h3>
            <p>{fluid_data.get('non_fluid_percentage', 0):.1f}%</p>
        </div>
        <div class="metric-card">
            <h3>Motion Patterns</h3>
            <p>{fluid_data.get('num_motion_patterns', 0)}</p>
        </div>
        """

        return html

    def _generate_movement_quality_html(self, quality_data):
        """Generate HTML for movement quality analysis section"""
        if not quality_data:
            return "<p>No movement quality analysis data available.</p>"

        # Count predictions by quality
        from collections import Counter
        predictions = Counter([q['prediction'] for q in quality_data])
        total = len(quality_data)

        html = "<h3>Quality Distribution</h3><table><tr><th>Quality</th><th>Count</th><th>Percentage</th></tr>"
        for quality, count in predictions.items():
            percentage = (count / total) * 100 if total > 0 else 0
            html += f"<tr><td>{quality.title()}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        html += "</table>"

        return html

    def _generate_origin_html(self, origin_data):
        """Generate HTML for origin analysis section"""
        if not origin_data:
            return "<p>No origin analysis data available.</p>"

        from collections import Counter
        origins = Counter([o['origin_type'] for o in origin_data])
        total = len(origin_data)

        html = "<h3>Origin Distribution</h3><table><tr><th>Origin Type</th><th>Count</th><th>Percentage</th></tr>"
        for origin, count in origins.items():
            percentage = (count / total) * 100 if total > 0 else 0
            html += f"<tr><td>{origin}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        html += "</table>"

        return html

    def _interpret_hopkins(self, hopkins_value):
        """Interpret Hopkins statistic value"""
        if hopkins_value > 0.75:
            return "Highly Clusterable"
        elif hopkins_value > 0.5:
            return "Clusterable"
        elif hopkins_value > 0.3:
            return "Random"
        else:
            return "Uniform"

    def _clean_sheet_name(self, name):
        """Clean sheet name for Excel compatibility"""
        # Excel sheet names max 31 characters, no special characters
        cleaned = name[:31]
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c in ' _-')
        return cleaned

    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return str(obj)


def export_analysis_report(parent_app):
    """
    Convenience function to export analysis report.

    Args:
        parent_app: The main application instance

    Returns:
        bool: True if export was successful, False otherwise
    """
    generator = ReportGenerator(parent_app)
    return generator.export_analysis_report()