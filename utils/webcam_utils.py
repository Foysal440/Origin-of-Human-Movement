import cv2


def capture_frame():
    """Capture a frame from webcam using OpenCV"""
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not capture frame from webcam")

    return frame


def compute_optical_flow(prev_frame, current_frame):
    """Compute optical flow between frames"""
    if prev_frame is None or current_frame is None:
        return None, None, None

    # Convert to grayscale if needed
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
        current_gray = current_frame

    # Resize frames for consistent processing
    prev_gray = cv2.resize(prev_gray, (640, 480))
    current_gray = cv2.resize(current_gray, (640, 480))

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude, angle, flow