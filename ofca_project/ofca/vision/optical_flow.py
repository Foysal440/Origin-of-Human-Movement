"""
Optical flow processing module using Lucas-Kanade method.
"""

import cv2
import numpy as np
from collections import deque
from ..utils.visuals import create_heatmap, draw_motion_trails, draw_cluster_centroids


class OpticalFlowProcessor:
    """Processes optical flow using Lucas-Kanade method"""

    def __init__(self, max_corners=200, quality_level=0.01, min_distance=10,
                 win_size=(15, 15), max_level=2, lk_params=None):
        """
        Initialize the optical flow processor.

        Args:
            max_corners: Maximum number of corners to detect
            quality_level: Parameter characterizing the minimal accepted quality of corners
            min_distance: Minimum possible Euclidean distance between corners
            win_size: Size of the search window at each pyramid level
            max_level: 0-based maximal pyramid level number
            lk_params: Additional parameters for Lucas-Kanade method
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.win_size = win_size
        self.max_level = max_level

        # Default Lucas-Kanade parameters
        if lk_params is None:
            self.lk_params = dict(
                winSize=win_size,
                maxLevel=max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
        else:
            self.lk_params = lk_params

        self.prev_gray = None
        self.feature_points = None
        self.flow_accumulator = None

    def process(self, frame, gray_frame, human_rects, prev_gray=None, prev_features=None,
                show_trails=False, show_heatmap=False, show_centroids=False,
                motion_trails=None, cluster_centroids=None):
        """
        Process a frame to calculate optical flow.

        Args:
            frame: Input frame (BGR)
            gray_frame: Grayscale version of the frame
            human_rects: List of human bounding boxes to focus on
            prev_gray: Previous grayscale frame (optional)
            prev_features: Previous feature points (optional)
            show_trails: Whether to show motion trails
            show_heatmap: Whether to show heatmap
            show_centroids: Whether to show cluster centroids
            motion_trails: Motion trails data structure
            cluster_centroids: Cluster centroids data structure

        Returns:
            tuple: (processed_frame, flow_data)
        """
        # Use provided previous frame data or internal state
        current_prev_gray = prev_gray if prev_gray is not None else self.prev_gray
        current_prev_features = prev_features if prev_features is not None else self.feature_points

        # Initialize feature points if needed
        if current_prev_gray is None or self.should_reinitialize_features():
            current_prev_gray = gray_frame
            current_prev_features = self.initialize_features(gray_frame, human_rects)

        # Calculate optical flow
        flow_data = self.calculate_optical_flow(current_prev_gray, gray_frame, current_prev_features)

        # Update internal state
        self.prev_gray = gray_frame
        self.feature_points = flow_data['good_new'].reshape(-1, 1, 2) if len(flow_data['good_new']) > 0 else None

        # Process visualization
        processed_frame = self.visualize_flow(
            frame, flow_data, human_rects, show_trails, show_heatmap,
            show_centroids, motion_trails, cluster_centroids
        )

        return processed_frame, flow_data['flow_points']

    def should_reinitialize_features(self, frame_interval=30):
        """
        Determine if feature points should be reinitialized.

        Args:
            frame_interval: How often to reinitialize features

        Returns:
            bool: True if features should be reinitialized
        """
        return self.feature_points is None or len(self.feature_points) == 0

    def initialize_features(self, gray_frame, human_rects):
        """
        Initialize feature points for tracking.

        Args:
            gray_frame: Grayscale frame
            human_rects: List of human bounding boxes

        Returns:
            array: Feature points to track
        """
        # Create mask for human regions
        mask = np.zeros_like(gray_frame)
        for (x, y, w, h) in human_rects:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # Detect features only in human regions
        features = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7,
            mask=mask
        )

        return features

    def calculate_optical_flow(self, prev_gray, curr_gray, prev_features):
        """
        Calculate optical flow between two frames.

        Args:
            prev_gray: Previous grayscale frame
            curr_gray: Current grayscale frame
            prev_features: Feature points from previous frame

        Returns:
            dict: Optical flow data
        """
        if prev_features is None or len(prev_features) == 0:
            return {
                'good_old': np.array([]),
                'good_new': np.array([]),
                'flow_vectors': np.array([]),
                'flow_points': np.array([]),
                'status': np.array([])
            }

        # Calculate optical flow
        features, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_features, None, **self.lk_params
        )

        # Filter good points
        good_new = features[status == 1]
        good_old = prev_features[status == 1]

        # Calculate flow vectors
        flow_vectors = good_new - good_old

        # Combine points with flow vectors
        flow_points = np.hstack((good_new, flow_vectors)) if len(good_new) > 0 else np.array([])

        return {
            'good_old': good_old,
            'good_new': good_new,
            'flow_vectors': flow_vectors,
            'flow_points': flow_points,
            'status': status
        }

    def visualize_flow(self, frame, flow_data, human_rects, show_trails,
                       show_heatmap, show_centroids, motion_trails, cluster_centroids):
        """
        Visualize optical flow results.

        Args:
            frame: Original frame
            flow_data: Optical flow data
            human_rects: Human bounding boxes
            show_trails: Whether to show motion trails
            show_heatmap: Whether to show heatmap
            show_centroids: Whether to show cluster centroids
            motion_trails: Motion trails data
            cluster_centroids: Cluster centroids data

        Returns:
            Processed frame with visualizations
        """
        # Convert to RGB for visualization
        vis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw human bounding boxes
        for (x, y, w, h) in human_rects:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw motion trails if enabled
        if show_trails and motion_trails is not None:
            vis = draw_motion_trails(vis, motion_trails)

        # Draw optical flow vectors
        if len(flow_data['good_new']) > 0 and len(flow_data['good_old']) > 0:
            for new, old, vec in zip(flow_data['good_new'], flow_data['good_old'], flow_data['flow_vectors']):
                a, b = new.ravel()
                c, d = old.ravel()
                vec_x, vec_y = vec.ravel()

                # Draw flow line
                vis = cv2.line(vis, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                # Draw direction arrow
                vis = cv2.arrowedLine(vis, (int(a), int(b)),
                                      (int(a + vec_x * 5), int(b + vec_y * 5)),
                                      (0, 0, 255), 2)

        # Draw heatmap if enabled
        if show_heatmap and len(flow_data['flow_points']) > 0:
            vis = create_heatmap(vis, flow_data['flow_points'])

        # Draw cluster centroids if enabled
        if show_centroids and len(flow_data['flow_points']) > 0 and cluster_centroids is not None:
            vis = draw_cluster_centroids(vis, flow_data['flow_points'], cluster_centroids, self.frame_count)

        return vis

    def reset(self):
        """Reset the optical flow processor state"""
        self.prev_gray = None
        self.feature_points = None
        self.flow_accumulator = None

    def get_flow_statistics(self, flow_points):
        """
        Calculate statistics from optical flow data.

        Args:
            flow_points: Optical flow points with vectors

        Returns:
            dict: Flow statistics
        """
        if len(flow_points) == 0:
            return {
                'num_points': 0,
                'avg_magnitude': 0,
                'max_magnitude': 0,
                'std_magnitude': 0,
                'avg_direction': 0,
                'std_direction': 0
            }

        # Extract vectors
        vectors = flow_points[:, 2:]

        # Calculate magnitudes
        magnitudes = np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)

        # Calculate directions (in radians)
        directions = np.arctan2(vectors[:, 1], vectors[:, 0])

        # Wrap directions to avoid circular mean issues
        directions = np.arctan2(np.sin(directions), np.cos(directions))

        return {
            'num_points': len(flow_points),
            'avg_magnitude': np.mean(magnitudes),
            'max_magnitude': np.max(magnitudes),
            'std_magnitude': np.std(magnitudes),
            'avg_direction': np.mean(directions),
            'std_direction': np.std(directions)
        }





