import os
import cv2
import numpy as np
from PIL import Image

class FaceRecognizer:
    def __init__(self, known_faces_dir="images/known"):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
    def load_known_faces(self):
        # Load known faces from directory
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Get name from filename (remove extension)
                name = os.path.splitext(filename)[0]
                
                # Load image
                image_path = os.path.join(self.known_faces_dir, filename)
                image = cv2.imread(image_path)
                # Convert to RGB (OpenCV uses BGR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Compute a simple face encoding (using average color as a very simple feature)
                # In a real application, you would use a proper face recognition algorithm
                encoding = np.mean(image_rgb, axis=(0, 1))
                
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)
                
    def recognize_face(self, face_image):
        # Convert to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Compute simple encoding for the detected face
        face_encoding = np.mean(face_rgb, axis=(0, 1))
        
        # Compare with known faces (using Euclidean distance)
        if len(self.known_face_encodings) == 0:
            return "Unknown"
            
        distances = [np.linalg.norm(face_encoding - enc) for enc in self.known_face_encodings]
        best_match_idx = np.argmin(distances)
        
        # If the best match is too far, consider it unknown
        if distances[best_match_idx] < 50:  # Threshold value
            return self.known_face_names[best_match_idx]
        else:
            return "Unknown"
