from collections import deque
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, List
from config import Config
from utils.feature_extraction import extract_features
from utils.clustering import perform_clustering
from utils.visualization import (
    show_cluster_visualization,
    show_cluster_timeline,
    show_motion_heatmaps
)


class HumanMovementAnalyzer:
    def __init__(self, with_gui: bool = False):
        """Initialize analyzer with enhanced features

        Args:
            with_gui: Whether to initialize GUI components
        """
        self.with_gui = with_gui
        self._initialize_attributes()

        if with_gui:
            self._initialize_gui()

    def _initialize_attributes(self) -> None:
        """Initialize analysis attributes with type hints"""
        self.features: Optional[np.ndarray] = None
        self.cluster_results: Dict = {}
        self.optimal_k: int = Config.DEFAULT_CLUSTERS
        self.current_mode: str = "video_upload"
        self.is_running: bool = False
        self.video_processing: bool = False
        self.progress: int = 0

        # Data storage
        self.magnitudes: List[np.ndarray] = []
        self.angles: List[np.ndarray] = []
        self.video_fps: float = Config.DEFAULT_FPS
        self.frame_timestamps: List[float] = []
        self.first_frame: Optional[np.ndarray] = None

        # Buffers
        self.frame_buffer: deque = deque(maxlen=Config.FRAME_BUFFER_SIZE)
        self.feature_buffer: deque = deque(maxlen=Config.FEATURE_BUFFER_SIZE)
        self.cluster_history: deque = deque(maxlen=Config.CLUSTER_HISTORY_SIZE)

        # Models
        self.scaler: StandardScaler = StandardScaler()
        self.kmeans: MiniBatchKMeans = MiniBatchKMeans(
            n_clusters=Config.DEFAULT_CLUSTERS,
            batch_size=100,
            random_state=42
        )

    def _initialize_gui(self) -> None:
        """Initialize GUI components with error handling"""
        try:
            from gui.analysis_window import AnalysisWindow
            self.gui = AnalysisWindow(analyzer=self)
        except ImportError as e:
            raise ImportError(f"GUI components not available: {str(e)}")

    def show_gui(self) -> None:
        """Show the GUI window if available"""
        if self.with_gui and hasattr(self, 'gui'):
            self.gui.show()

    def process_video_file(self, video_path: str) -> None:
        """Enhanced video processing with progress tracking

        Args:
            video_path: Path to video file to analyze
        """
        self.video_processing = True
        self.progress = 0
        cap = None

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or Config.DEFAULT_FPS
            self.video_fps = fps

            # Initialize containers
            flows: List[Tuple[np.ndarray, np.ndarray]] = []
            self.magnitudes = []
            self.angles = []
            self.frame_timestamps = []

            # Process first frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Could not read first frame")

            self.first_frame = cv2.resize(frame,
                                          (Config.RESIZE_WIDTH, Config.RESIZE_HEIGHT))
            prev_gray = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)

            # Frame processing loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame = cv2.resize(frame,
                                           (Config.RESIZE_WIDTH, Config.RESIZE_HEIGHT))
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                # Compute optical flow with Farneback method
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, current_gray, None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )

                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # Store motion data
                flows.append((magnitude, angle))
                self.magnitudes.append(magnitude)
                self.angles.append(angle)
                self.frame_timestamps.append(self.progress / fps)

                # Update for next frame
                prev_gray = current_gray
                self.progress += 1

                # Update GUI progress if needed
                if self.with_gui and self.progress % 10 == 0:  # Update every 10 frames
                    self.gui.update_progress(self.progress, total_frames)

            # Analyze movements
            self.features = extract_features(flows)
            self.analyze_movements()

        except Exception as e:
            error_msg = f"Video processing error: {str(e)}"
            print(error_msg)
            if self.with_gui:
                self.gui.show_error(error_msg)
        finally:
            if cap and cap.isOpened():
                cap.release()
            self.video_processing = False

    def analyze_movements(self) -> None:
        """Enhanced movement analysis with visualization"""
        try:
            # Perform clustering
            self.cluster_results, self.scaler = perform_clustering(
                self.features,
                algorithm=Config.DEFAULT_ALGORITHM,
                n_clusters=Config.DEFAULT_CLUSTERS
            )

            # Update GUI with results
            if self.with_gui:
                self._update_visualizations()

            # Show standalone visualizations if not using GUI
            elif Config.SHOW_STANDALONE_VISUALIZATIONS:
                self._show_standalone_visualizations()

        except Exception as e:
            error_msg = f"Movement analysis error: {str(e)}"
            print(error_msg)
            if self.with_gui:
                self.gui.show_error(error_msg)

    def _update_visualizations(self) -> None:
        """Update all GUI visualizations"""
        visualization_data = {
            'features': self.features,
            'magnitudes': self.magnitudes,
            'angles': self.angles,
            'timestamps': self.frame_timestamps,
            'clusters': self.cluster_results.get('KMeans'),
            'metrics': self.cluster_results.get('metrics'),
            'first_frame': self.first_frame
        }
        self.gui.update_visualizations(visualization_data)

    def _show_standalone_visualizations(self) -> None:
        """Show visualizations in browser when not using GUI"""
        cluster_labels = self.cluster_results.get('KMeans')
        if cluster_labels is None:
            return

        # Show cluster visualization
        show_cluster_visualization(
            features=self.features,
            cluster_labels=cluster_labels
        ).show()

        # Show timeline
        show_cluster_timeline(
            timestamps=self.frame_timestamps,
            cluster_labels=cluster_labels,
            magnitudes=np.array([np.mean(m) for m in self.magnitudes])
        ).show()

        # Show heatmaps if we have a reference frame
        if self.first_frame is not None:
            heatmap_figs = show_motion_heatmaps(
                magnitudes=self.magnitudes,
                cluster_labels=cluster_labels,
                original_frame=self.first_frame
            )
            for fig in heatmap_figs:
                fig.show()

    def get_analysis_summary(self) -> Dict:
        """Get summary of analysis results"""
        cluster_labels = self.cluster_results.get('KMeans', [])
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_dist = dict(zip(unique_labels, counts))

        return {
            'num_frames': len(self.frame_timestamps),
            'duration': self.frame_timestamps[-1] if self.frame_timestamps else 0,
            'optimal_clusters': self.optimal_k,
            'cluster_distribution': cluster_dist,
            'metrics': self.cluster_results.get('metrics', {})
        }