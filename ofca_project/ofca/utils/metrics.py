import numpy as np
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO

def calculate_hopkins_statistic(data, sample_size=None):
    if sample_size is None:
        sample_size = min(50, len(data) // 2)

    if len(data) < 2:
        return 0.5

    X = data[:, :2]
    random_indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[random_indices]

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    X_uniform = np.column_stack([
        np.random.uniform(mins[0], maxs[0], size=sample_size),
        np.random.uniform(mins[1], maxs[1], size=sample_size)
    ])

    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    distances_data, _ = nbrs.kneighbors(X_sample)
    u_distances = distances_data[:, 1]

    distances_uniform, _ = nbrs.kneighbors(X_uniform)
    w_distances = distances_uniform[:, 0]

    numerator = np.sum(w_distances)
    denominator = np.sum(u_distances) + np.sum(w_distances)

    if denominator == 0:
        return 0.5

    hopkins_stat = numerator / denominator
    return hopkins_stat

class HumanDetector:
    def __init__(self, model_size='n'):
        self.model_size = model_size
        try:
            self.model = YOLO(f'yolov8{model_size}.pt')
            print(f"YOLOv8{model_size.upper()} initialized for person detection")
        except Exception as e:
            print(f"Error initializing YOLOv8{model_size.upper()}: {e}")
            self.model = None

    def detect(self, frame):
        if self.model is None or frame is None:
            return []

        results = self.model(frame, verbose=False)
        boxes = []
        for result in results:
            for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                # Class 0 is 'person' in COCO
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, box[:4])
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    def change_model(self, model_size):
        self.model_size = model_size
        try:
            self.model = YOLO(f'yolov8{model_size}.pt')
            print(f"Switched to YOLOv8{model_size.upper()} model")
            return True
        except Exception as e:
            print(f"Failed to load YOLOv8{model_size.upper()} model: {e}")
            return False
hopkins_statistic = calculate_hopkins_statistic
