class Config:
    """Central configuration for human movement analysis system"""

    # Visualization settings
    COLOR_PALETTE = "husl"
    MAX_COLORS = 10
    CLUSTER_COLORS = [
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 0, 0),  # Red
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (0, 100, 0)  # Dark Green
    ]

    PLOT_STYLE = "seaborn"  # Matplotlib style
    FIGURE_SIZE = (12, 8)  # Default figure size

    # Analysis parameters
    DEFAULT_CLUSTERS = 4
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 8
    DEFAULT_SENSITIVITY = 0.7  # For movement detection
    DEFAULT_ALGORITHM = "Auto-Detect"  # Options: ["K-Means", "DBSCAN", "HDBSCAN", "Auto-Detect"]

    # Optical Flow parameters
    OPTICAL_FLOW_PARAMS = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0
    }

    # Buffer sizes
    FRAME_BUFFER_SIZE = 30  # Number of frames to keep in memory
    FEATURE_BUFFER_SIZE = 100  # Number of feature vectors to buffer
    CLUSTER_HISTORY_SIZE = 200  # Number of historical cluster assignments

    # Webcam settings
    DEFAULT_FPS = 30
    RESIZE_WIDTH = 640
    RESIZE_HEIGHT = 480
    WEBCAM_SOURCE = 0  # Default camera index

    # File handling
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
    MAX_VIDEO_SIZE_MB = 100  # Maximum allowed video file size

    # Performance settings
    USE_GPU = False  # Enable GPU acceleration if available
    PARALLEL_PROCESSING = True  # Enable multiprocessing

    # Visualization parameters
    DASHBOARD_WIDTH = 1400
    DASHBOARD_HEIGHT = 900
    HEATMAP_OPACITY = 0.7
    SCATTER_MARKER_SIZE = 8
    ANIMATION_DURATION = 500  # ms
    DEFAULT_PLOT_TEMPLATE = "plotly_white"
    GRID_SIZE = 5  # For feature extraction
    LIVE_GRID_SIZE = 3  # For real-time processing

    # Visualization control
    SHOW_STANDALONE_VISUALIZATIONS = True  # Show plots in browser when GUI disabled
    OPTICAL_FLOW_PARAMS = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    }
    GUI_UPDATE_INTERVAL = 10  # Frames between GUI progress updates