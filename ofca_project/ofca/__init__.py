"""
Optical Flow Cluster Analyzer (OFCA)
A comprehensive tool for analyzing human movement quality through optical flow clustering.
"""

__version__ = "1.0.0"
__author__ = "Abdullah Al Foysal"
__email__ = "niloyhasanfoysal440@.com"

# Import main components for easier access
from .app import OpticalFlowClusterAnalyzer
from .analysis.movement_quality import MovementQualityAnalyzer
from .analysis.fluid_motion import FluidMotionAnalyzer
from .vision.detector import HumanDetector
from .vision.optical_flow import OpticalFlowProcessor

# Package-level initialization
def initialize():
    """Initialize the OFCA package and check dependencies"""
    import sys
    import os

    # Add package directory to path
    package_dir = os.path.dirname(os.path.abspath(__file__))
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

    print(f"OFCA {__version__} initialized successfully")

    # Check for critical dependencies
    try:
        import cv2
        import numpy as np
        import PyQt6
        from sklearn.cluster import KMeans
        print("✓ All critical dependencies are available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

    return True

# Run initialization when package is imported
if initialize():
    print("OFCA is ready to use")
else:
    print("OFCA initialization completed with warnings")