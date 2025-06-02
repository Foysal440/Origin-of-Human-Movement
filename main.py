import sys
import os
import warnings
from analyzer import HumanMovementAnalyzer
from PyQt5.QtWidgets import QApplication


def main():
    # Environment setup
    os.environ["QT_QUICK_BACKEND"] = "software"
    warnings.filterwarnings('ignore')

    # Create QApplication instance
    app = QApplication(sys.argv)

    try:
        print("Initializing analyzer with GUI...")
        analyzer = HumanMovementAnalyzer(with_gui=True)

        # Start the event loop
        sys.exit(app.exec_())

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()