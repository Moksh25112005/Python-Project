import cv2
import numpy as np
from ultralytics import YOLO

class FaceDetector:
    def __init__(self):
        # Use standard YOLOv8n model
        self.model = YOLO("yolov8n.pt")
        
    def detect_faces(self, frame):
        # Run YOLO detection
        results = self.model(frame, stream=True)
        
        # Process results - filter for person class (class 0)
        faces = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Only consider person detections (class 0)
                if int(box.cls) == 0:  # Person class
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Calculate face region (upper portion of person detection)
                    face_height = int((y2 - y1) * 0.4)  # Take upper 40% as face
                    face_y2 = y1 + face_height
                    
                    # Extract face ROI
                    face = frame[y1:face_y2, x1:x2]
                    
                    # Only add if face region is not empty
                    if face.size > 0:
                        faces.append((face, (x1, y1, x2, y2)))
                        
        return faces
