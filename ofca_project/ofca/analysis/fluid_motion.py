import numpy as np
from collections import deque


class FluidMotionAnalyzer:
    def __init__(self):
        self.fluid_threshold = 0.7  # Hopkins threshold for fluid motion
        self.motion_origins = {}  # Track where motion patterns started
        self.fluid_frames = []
        self.non_fluid_frames = []
        self.proof_frames = {}  # Store sample frames for each motion pattern
        self.motion_trails = deque(maxlen=30)  # Store motion trails for visualization

    def analyze_frame(self, frame_data, frame_number, frame_image):
        """Analyze if motion is fluid based on Hopkins statistic"""
        if len(frame_data['flow_points']) < 10:  # Not enough points
            return False, None

        hopkins = frame_data.get('hopkins', 0.5)
        is_fluid = hopkins > self.fluid_threshold

        # Track motion origins
        motion_signature = self._create_motion_signature(frame_data)
        if motion_signature not in self.motion_origins:
            self.motion_origins[motion_signature] = {
                'start_frame': frame_number,
                'proof_frame': frame_image.copy(),
                'type': 'fluid' if is_fluid else 'non-fluid',
                'hopkins': hopkins,
                'avg_magnitude': np.mean(
                    np.sqrt(frame_data['flow_points'][:, 2] ** 2 + frame_data['flow_points'][:, 3] ** 2))
                if len(frame_data['flow_points']) > 0 else 0
            }

        if is_fluid:
            self.fluid_frames.append(frame_number)
        else:
            self.non_fluid_frames.append(frame_number)

        # Add to motion trails
        self.motion_trails.append({
            'frame': frame_number,
            'flow_points': frame_data['flow_points'],
            'is_fluid': is_fluid
        })

        return is_fluid, motion_signature

    def _create_motion_signature(self, frame_data):
        """Create a signature for the motion pattern"""
        if len(frame_data['flow_points']) == 0:
            return "0" * 10

        # Use direction histogram as signature
        directions = np.arctan2(frame_data['flow_points'][:, 3], frame_data['flow_points'][:, 2])
        hist, _ = np.histogram(directions, bins=10, range=(-np.pi, np.pi))
        return ",".join(map(str, hist))

    def get_fluid_summary(self):
        """Generate summary statistics about fluid motion"""
        total_frames = len(self.fluid_frames) + len(self.non_fluid_frames)
        if total_frames == 0:
            return {}

        # Use safe calculations to avoid warnings
        fluid_hopkins = [o['hopkins'] for o in self.motion_origins.values() if o['type'] == 'fluid']
        non_fluid_hopkins = [o['hopkins'] for o in self.motion_origins.values() if o['type'] == 'non-fluid']

        avg_hopkins_fluid = np.mean(fluid_hopkins) if fluid_hopkins else 0
        avg_hopkins_non_fluid = np.mean(non_fluid_hopkins) if non_fluid_hopkins else 0

        return {
            'fluid_percentage': len(self.fluid_frames) / total_frames * 100,
            'non_fluid_percentage': len(self.non_fluid_frames) / total_frames * 100,
            'num_motion_patterns': len(self.motion_origins),
            'avg_hopkins_fluid': avg_hopkins_fluid,
            'avg_hopkins_non_fluid': avg_hopkins_non_fluid,
            'motion_origins': self.motion_origins
        }