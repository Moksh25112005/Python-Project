import os
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox

class FaceRegistrar:
    def __init__(self, known_faces_dir="images/known"):
        self.known_faces_dir = known_faces_dir
        self.ensure_directory_exists()
        
    def ensure_directory_exists(self):
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            
    def register_new_face(self, root):
        # Ask for name
        name = simpledialog.askstring("Register New Face", "Enter name:", parent=root)
        
        if not name:
            messagebox.showinfo("Cancelled", "Registration cancelled")
            return False
            
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return False
            
        messagebox.showinfo("Ready", "Camera will open. Look at the camera and press SPACE to capture or ESC to cancel.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display frame
            cv2.imshow("Register Face - Press SPACE to capture, ESC to cancel", frame)
            
            # Wait for key press
            key = cv2.waitKey(1)
            
            # ESC pressed - cancel
            if key == 27:
                cv2.destroyAllWindows()
                cap.release()
                return False
                
            # SPACE pressed - capture
            if key == 32:
                # Save image
                image_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
                cv2.imwrite(image_path, frame)
                cv2.destroyAllWindows()
                cap.release()
                messagebox.showinfo("Success", f"Face for {name} registered successfully!")
                return True
                
        cv2.destroyAllWindows()
        cap.release()
        return False
