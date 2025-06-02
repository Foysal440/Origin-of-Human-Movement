import numpy as np
import pandas as pd
import cv2
from numba import njit  # For performance optimization
from config import Config
from typing import List, Tuple, Union


class FeatureExtractionError(Exception):
    """Custom exception for feature extraction errors"""
    pass


def validate_optical_flow(magnitude: np.ndarray, angle: np.ndarray) -> None:
    """Validate optical flow inputs"""
    if magnitude.shape != angle.shape:
        raise FeatureExtractionError("Magnitude and angle arrays must have same shape")
    if magnitude.size == 0:
        raise FeatureExtractionError("Input arrays cannot be empty")
    if not (np.all(magnitude >= 0)):
        raise FeatureExtractionError("Magnitude values must be non-negative")


@njit(fastmath=True)
def fast_circular_mean(angles: np.ndarray) -> float:
    """Numba-optimized circular mean calculation"""
    sin_sum = 0.0
    cos_sum = 0.0
    for angle in angles.flatten():
        sin_sum += np.sin(angle)
        cos_sum += np.cos(angle)
    return np.arctan2(sin_sum, cos_sum)


@njit(fastmath=True)
def fast_circular_std(angles: np.ndarray) -> float:
    """Numba-optimized circular standard deviation"""
    sin_sum = 0.0
    cos_sum = 0.0
    count = 0
    for angle in angles.flatten():
        sin_sum += np.sin(angle)
        cos_sum += np.cos(angle)
        count += 1
    R = np.sqrt(sin_sum ** 2 + cos_sum ** 2) / max(1, count)
    return np.sqrt(-2 * np.log(max(1e-10, R)))  # Avoid log(0)


def extract_features(flows: List[Tuple[np.ndarray, np.ndarray]],
                     grid_size: int = Config.GRID_SIZE) -> pd.DataFrame:
    """
    Enhanced optical flow feature extraction with error handling and optimization

    Args:
        flows: List of (magnitude, angle) tuples from optical flow
        grid_size: Number of grid divisions along each axis

    Returns:
        DataFrame containing extracted features with column names
    """
    try:
        if not flows:
            raise FeatureExtractionError("Empty flow input")

        features = []
        grid_features_count = grid_size * grid_size * 5  # 5 features per grid cell
        base_feature_count = 10  # 10 base features

        # Pre-allocate array for better performance
        feature_array = np.zeros((len(flows), base_feature_count + grid_features_count))

        for idx, (mag, ang) in enumerate(flows):
            validate_optical_flow(mag, ang)

            # Basic magnitude statistics
            mag_mean = np.mean(mag)
            mag_std = np.std(mag)
            mag_max = np.max(mag)
            mag_min = np.min(mag)

            # Circular statistics
            ang_mean = fast_circular_mean(ang)
            ang_std = fast_circular_std(ang)

            # Motion vector components
            x_flow = np.mean(mag * np.cos(ang))
            y_flow = np.mean(mag * np.sin(ang))
            flow_magnitude = np.sqrt(x_flow ** 2 + y_flow ** 2)
            flow_direction = np.arctan2(y_flow, x_flow)

            # Grid-based features
            grid_features = extract_grid_features(mag, ang, grid_size)

            # Combine all features
            feature_array[idx] = np.concatenate([
                [mag_mean, mag_std, mag_max, mag_min,
                 ang_mean, ang_std,
                 x_flow, y_flow, flow_magnitude, flow_direction],
                grid_features
            ])

        # Create column names
        columns = [
            'mag_mean', 'mag_std', 'mag_max', 'mag_min',
            'ang_mean', 'ang_std',
            'x_flow', 'y_flow', 'flow_magnitude', 'flow_direction'
        ]

        # Add grid feature names
        for i in range(grid_size):
            for j in range(grid_size):
                prefix = f'grid_{i}_{j}_'
                columns.extend([
                    prefix + 'mag_mean',
                    prefix + 'mag_std',
                    prefix + 'mag_max',
                    prefix + 'ang_sin',
                    prefix + 'ang_cos'
                ])

        return pd.DataFrame(feature_array, columns=columns)

    except Exception as e:
        raise FeatureExtractionError(f"Feature extraction failed: {str(e)}")


def extract_live_features(magnitude: np.ndarray,
                          angle: np.ndarray,
                          grid_size: int = Config.LIVE_GRID_SIZE) -> np.ndarray:
    """
    Optimized feature extraction for real-time processing

    Args:
        magnitude: Optical flow magnitude array
        angle: Optical flow angle array
        grid_size: Number of grid divisions for live processing

    Returns:
        Numpy array of extracted features
    """
    try:
        validate_optical_flow(magnitude, angle)

        # Basic statistics
        mag_mean = np.mean(magnitude)
        mag_std = np.std(magnitude)
        mag_max = np.max(magnitude)

        # Circular statistics
        ang_mean = fast_circular_mean(angle)
        ang_std = fast_circular_std(angle)

        # Motion vector components
        x_flow = np.mean(magnitude * np.cos(angle))
        y_flow = np.mean(magnitude * np.sin(angle))
        flow_direction = np.arctan2(y_flow, x_flow)

        # Grid-based features
        grid_features = extract_grid_features(magnitude, angle, grid_size, live=True)

        return np.concatenate([
            [mag_mean, mag_std, mag_max,
             ang_mean, ang_std,
             x_flow, y_flow, flow_direction],
            grid_features
        ])

    except Exception as e:
        raise FeatureExtractionError(f"Live feature extraction failed: {str(e)}")


@njit(fastmath=True)
def extract_grid_features(magnitude: np.ndarray,
                          angle: np.ndarray,
                          grid_size: int,
                          live: bool = False) -> np.ndarray:
    """
    Extract grid-based features with Numba optimization

    Args:
        magnitude: Optical flow magnitude array
        angle: Optical flow angle array
        grid_size: Number of grid divisions
        live: Whether to use reduced features for live processing

    Returns:
        Numpy array of grid features
    """
    h, w = magnitude.shape
    features_per_cell = 4 if live else 5
    features = np.zeros(grid_size * grid_size * features_per_cell)

    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * h // grid_size, (i + 1) * h // grid_size
            x_start, x_end = j * w // grid_size, (j + 1) * w // grid_size

            grid_mag = magnitude[y_start:y_end, x_start:x_end]
            grid_ang = angle[y_start:y_end, x_start:x_end]

            # Grid magnitude features
            features[idx] = np.mean(grid_mag)
            features[idx + 1] = np.std(grid_mag)

            if not live:
                features[idx + 2] = np.max(grid_mag)
                next_idx = 3
            else:
                next_idx = 2

            # Grid direction features
            sin_sum = 0.0
            cos_sum = 0.0
            count = 0
            for y in range(grid_ang.shape[0]):
                for x in range(grid_ang.shape[1]):
                    a = grid_ang[y, x]
                    sin_sum += np.sin(a)
                    cos_sum += np.cos(a)
                    count += 1

            features[idx + next_idx] = sin_sum / max(1, count)
            features[idx + next_idx + 1] = cos_sum / max(1, count)

            idx += features_per_cell

    return features