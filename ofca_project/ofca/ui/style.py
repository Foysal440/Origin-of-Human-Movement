"""
Styling utilities for Optical Flow Cluster Analyzer.
"""

from PyQt6.QtGui import QColor, QPalette


def apply_style(widget):
    """
    Apply the application style to a widget.

    Args:
        widget: The widget to apply the style to
    """
    style = '''
        QMainWindow { 
            background-color: #f7f7fa; 
            font-family: "Segoe UI", Arial, sans-serif;
        }
        QWidget { 
            background-color: #f7f7fa; 
            color: #222; 
            font-size: 14px;
        }
        QGroupBox { 
            border: 1px solid #d1d5db; 
            border-radius: 6px;
            margin-top: 10px; 
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title { 
            color: #1976d2; 
            subcontrol-origin: margin;
            left: 10px; 
            padding: 0 5px;
        }
        QLabel { 
            color: #222; 
            font-size: 14px; 
        }
        QPushButton { 
            background-color: #e3eafc; 
            color: #1976d2; 
            border-radius: 6px; 
            padding: 8px 12px; 
            font-weight: bold;
            border: 1px solid #d1d5db;
        }
        QPushButton:hover {
            background-color: #d1e0fc;
        }
        QPushButton:pressed {
            background-color: #bfd6fc;
        }
        QPushButton:disabled { 
            background-color: #eee; 
            color: #aaa; 
        }
        QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QTabWidget, QTableWidget, QTextEdit {
            background-color: #fff; 
            color: #222; 
            border: 1px solid #d1d5db;
            border-radius: 4px;
            padding: 4px;
        }
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #1976d2;
        }
        QTabBar::tab:selected { 
            background: #e3eafc; 
            color: #1976d2; 
            border-bottom: 2px solid #1976d2;
        }
        QTabBar::tab:!selected { 
            background: #f7f7fa; 
            color: #222; 
        }
        QTabBar::tab {
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTableWidget {
            gridline-color: #d1d5db;
            alternate-background-color: #f8f9fa;
        }
        QTableWidget QHeaderView::section { 
            background-color: #e3eafc; 
            color: #1976d2; 
            padding: 6px;
            border: none;
            font-weight: bold;
        }
        QTableWidget::item:selected {
            background-color: #1976d2;
            color: white;
        }
        QTextEdit {
            border: 1px solid #d1d5db;
            border-radius: 4px;
            padding: 8px;
        }
        QProgressBar {
            border: 1px solid #d1d5db;
            border-radius: 4px;
            text-align: center;
            background-color: #fff;
        }
        QProgressBar::chunk {
            background-color: #1976d2;
            border-radius: 3px;
        }
    '''
    widget.setStyleSheet(style)

    # Set a modern palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(247, 247, 250))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(34, 34, 34))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(248, 249, 250))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(34, 34, 34))
    palette.setColor(QPalette.ColorRole.Text, QColor(34, 34, 34))
    palette.setColor(QPalette.ColorRole.Button, QColor(227, 234, 252))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(25, 118, 210))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(25, 118, 210))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    widget.setPalette(palette)


def create_quality_palette():
    """
    Create a color palette for movement quality categories.

    Returns:
        dict: Color mapping for quality categories
    """
    return {
        'excellent': '#2ecc71',  # Green
        'good': '#3498db',  # Blue
        'average': '#f39c12',  # Orange
        'poor': '#e74c3c',  # Red
        'unknown': '#95a5a6'  # Gray
    }


def create_fluid_palette():
    """
    Create a color palette for fluid motion categories.

    Returns:
        dict: Color mapping for fluid motion categories
    """
    return {
        'fluid': '#27ae60',  # Dark green
        'non-fluid': '#e74c3c',  # Red
        'transition': '#f39c12'  # Orange
    }


def create_cluster_palette():
    """
    Create a color palette for cluster visualization.

    Returns:
        list: List of colors for clusters
    """
    return [
        '#e74c3c',  # Red
        '#3498db',  # Blue
        '#2ecc71',  # Green
        '#f39c12',  # Orange
        '#9b59b6',  # Purple
        '#1abc9c',  # Teal
        '#34495e',  # Dark blue
        '#e67e22',  # Carrot
        '#16a085',  # Green sea
        '#d35400'  # Pumpkin
    ]


def get_contrasting_text_color(background_color):
    """
    Get a contrasting text color for a given background color.

    Args:
        background_color (str): Hex color code

    Returns:
        str: 'black' or 'white' depending on background brightness
    """
    # Convert hex to RGB
    r = int(background_color[1:3], 16)
    g = int(background_color[3:5], 16)
    b = int(background_color[5:7], 16)

    # Calculate luminance (perceived brightness)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    # Return black for light colors, white for dark colors
    return 'black' if luminance > 0.5 else 'white'


def create_metric_card_style(metric_name, value):
    """
    Create styled HTML for a metric card.

    Args:
        metric_name (str): Name of the metric
        value: Value of the metric

    Returns:
        str: HTML string for the metric card
    """
    colors = {
        'total_flow': '#3498db',
        'avg_magnitude': '#2ecc71',
        'hopkins_stat': '#9b59b6',
        'clusters': '#f39c12'
    }

    color = colors.get(metric_name, '#95a5a6')
    text_color = get_contrasting_text_color(color)

    return f'''
        <div style="
            background-color: {color};
            color: {text_color};
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            margin: 5px;
            min-width: 120px;
        ">
            <div style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">
                {metric_name.replace('_', ' ').title()}
            </div>
            <div style="font-size: 24px; font-weight: bold;">
                {value}
            </div>
        </div>
    '''