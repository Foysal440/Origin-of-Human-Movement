import sys
import os

# Set environment variable to fix KMeans warning
os.environ["OMP_NUM_THREADS"] = "1"

from PyQt6.QtWidgets import QApplication
from ofca.app import OpticalFlowClusterAnalyzer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpticalFlowClusterAnalyzer()
    window.show()
    sys.exit(app.exec())