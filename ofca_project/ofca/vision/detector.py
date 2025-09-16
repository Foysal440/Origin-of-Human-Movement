"""
Human detection module using YOLOv8 for person detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class HumanDetector:
    """Detects humans in frames using YOLOv8"""

    def __init__(self, model_size='n'):
        """
        Initialize the human detector.

        Args:
            model_size (str): YOLO model size ('n', 's', 'm', 'l')
        """
        self.model_size = model_size
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize the YOLO model with the specified size"""
        try:
            self.model = YOLO(f'yolov8{self.model_size}.pt')
            print(f"YOLOv8{self.model_size.upper()} initialized for person detection")
        except Exception as e:
            print(f"Error initializing YOLOv8{self.model_size.upper()}: {e}")
            # Try to fall back to nano model
            try:
                self.model = YOLO('yolov8n.pt')
                self.model_size = 'n'
                print("Fell back to YOLOv8n model")
            except:
                self.model = None
                print("Could not initialize any YOLO model")

    def change_model(self, model_size):
        """
        Change the YOLO model size.

        Args:
            model_size (str): New model size ('n', 's', 'm', 'l')
        """
        if model_size != self.model_size:
            self.model_size = model_size
            self.initialize_model()

    def detect(self, frame, confidence_threshold=0.5):
        """
        Detect humans in a frame.

        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Minimum confidence for detection

        Returns:
            list: List of bounding boxes in (x, y, w, h) format
        """
        if self.model is None or frame is None:
            return []

        try:
            # Run inference
            results = self.model(frame, verbose=False, conf=confidence_threshold)

            boxes = []
            for result in results:
                for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                          result.boxes.cls.cpu().numpy(),
                                          result.boxes.conf.cpu().numpy()):
                    # Class 0 is 'person' in COCO dataset
                    if int(cls) == 0 and conf >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box[:4])
                        boxes.append((x1, y1, x2 - x1, y2 - y1))

            return boxes

        except Exception as e:
            print(f"Error during human detection: {e}")
            return []

    def detect_with_details(self, frame, confidence_threshold=0.5):
        """
        Detect humans with additional details.

        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Minimum confidence for detection

        Returns:
            dict: Detection results with boxes, confidences, and class IDs
        """
        if self.model is None or frame is None:
            return {'boxes': [], 'confidences': [], 'class_ids': []}

        try:
            # Run inference
            results = self.model(frame, verbose=False, conf=confidence_threshold)

            boxes = []
            confidences = []
            class_ids = []

            for result in results:
                for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                          result.boxes.cls.cpu().numpy(),
                                          result.boxes.conf.cpu().numpy()):
                    # Class 0 is 'person' in COCO dataset
                    if int(cls) == 0 and conf >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box[:4])
                        boxes.append((x1, y1, x2 - x1, y2 - y1))
                        confidences.append(float(conf))
                        class_ids.append(int(cls))

            return {
                'boxes': boxes,
                'confidences': confidences,
                'class_ids': class_ids
            }

        except Exception as e:
            print(f"Error during human detection: {e}")
            return {'boxes': [], 'confidences': [], 'class_ids': []}

    def draw_detections(self, frame, boxes, color=(255, 0, 0), thickness=2):
        """
        Draw detection boxes on a frame.

        Args:
            frame: Input frame
            boxes: List of bounding boxes in (x, y, w, h) format
            color: Box color (BGR)
            thickness: Box line thickness

        Returns:
            Frame with drawn boxes
        """
        result = frame.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        return result

    def get_model_info(self):
        """
        Get information about the current model.

        Returns:
            dict: Model information
        """
        if self.model is None:
            return {'size': 'none', 'initialized': False}

        return {
            'size': self.model_size,
            'initialized': True,
            'parameters': sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, 'parameters') else 0
        }