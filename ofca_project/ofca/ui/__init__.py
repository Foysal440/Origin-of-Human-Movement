"""
UI module for Optical Flow Cluster Analyzer.
Contains dialogs, tabs, and styling components.
"""

from .dialogs import ValidationResultsDialog, LabelFormatHelpDialog
from .tabs import (
    create_video_tab, create_dashboard_tab, create_cluster_analysis_tab,
    create_hopkins_analysis_tab, create_fluid_analysis_tab,
    create_movement_quality_tab, create_origin_analysis_tab
)
from .style import apply_style, create_quality_palette, create_fluid_palette

__all__ = [
    'ValidationResultsDialog',
    'LabelFormatHelpDialog',
    'create_video_tab',
    'create_dashboard_tab',
    'create_cluster_analysis_tab',
    'create_hopkins_analysis_tab',
    'create_fluid_analysis_tab',
    'create_movement_quality_tab',
    'create_origin_analysis_tab',
    'apply_style',
    'create_quality_palette',
    'create_fluid_palette'
]