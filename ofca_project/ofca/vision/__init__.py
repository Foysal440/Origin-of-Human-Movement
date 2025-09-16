"""
Vision module for Optical Flow Cluster Analyzer.
Contains components for human detection and optical flow processing.
"""

from .detector import HumanDetector
from .optical_flow import OpticalFlowProcessor

__all__ = ['HumanDetector', 'OpticalFlowProcessor']