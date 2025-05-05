import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
import os
from datetime import datetime
from ultralytics.nn.tasks import DetectionModel
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer  
from attendance import AttendanceManager
from register_face import FaceRegistrar
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
torch._C._WEIGHTS_ONLY_LOAD = False

from ultralytics import YOLO
YOLO("yolov8n.pt")

model = torch.load("yolov8n.pt", weights_only=False)

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        root.title("Face Recognition Attendance Scanner")
        root.geometry("900x600")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.attendance_manager = AttendanceManager()
        self.face_registrar = FaceRegistrar()
        
        # Video capture variables
        self.cap = None
        self.is_scanning = False
        self.scan_thread = None
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        # Create main frames
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        content_frame = ttk.Frame(self.root, padding=10)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control buttons
        ttk.Label(control_frame, text="Controls", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.scan_button = ttk.Button(control_frame, text="Start Scanning", command=self.toggle_scanning)
        self.scan_button.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Register New Face", command=self.register_face).pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="View Attendance", command=self.view_attendance).pack(fill=tk.X, pady=5)
        
        # Create notebook for content
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab for camera feed
        self.camera_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_tab, text="Camera Feed")
        
        self.camera_label = ttk.Label(self.camera_tab)
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Tab for attendance report
        self.report_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.report_tab, text="Attendance Report")
        
        self.report_tree = ttk.Treeview(self.report_tab, columns=("Name", "Time", "Date"), show="headings")
        self.report_tree.heading("Name", text="Name")
        self.report_tree.heading("Time", text="Time")
        self.report_tree.heading("Date", text="Date")
        self.report_tree.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def toggle_scanning(self):
        if self.is_scanning:
            self.stop_scanning()
        else:
            self.start_scanning()
            
    def start_scanning(self):
        # Switch to camera tab
        self.notebook.select(0)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
            
        self.is_scanning = True
        self.scan_button.config(text="Stop Scanning")
        self.status_var.set("Scanning for faces...")
        
        # Start scanning in a separate thread
        self.scan_thread = threading.Thread(target=self.scan_faces)
        self.scan_thread.daemon = True
        self.scan_thread.start()
        
    def stop_scanning(self):
        self.is_scanning = False
        self.scan_button.config(text="Start Scanning")
        self.status_var.set("Ready")
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
            
    def scan_faces(self):
        # Load YOLO model if we don't have it yet
        if not hasattr(self.face_detector, 'model') or self.face_detector.model is None:
            self.status_var.set("Loading YOLO model...")
            # This will handle the initial YOLO model download if needed
            try:
                from ultralytics import YOLO
                self.face_detector.model = YOLO("yolov8n-face.pt")
            except Exception as e:
                self.status_var.set(f"Error loading model: {str(e)}")
                self.stop_scanning()
                return
        
        while self.is_scanning:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Detect faces
            try:
                faces = self.face_detector.detect_faces(frame)
                
                # Process each detected face
                for face_img, (x1, y1, x2, y2) in faces:
                    # Skip tiny faces (likely false positives)
                    if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                        continue
                        
                    # Recognize face
                    name = self.face_recognizer.recognize_face(face_img)
                    
                    # Mark attendance if recognized
                    if name != "Unknown":
                        marked = self.attendance_manager.mark_attendance(name)
                        status = "✓" if marked else ""
                    else:
                        status = ""
                        
                    # Draw rectangle and name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} {status}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            except Exception as e:
                self.status_var.set(f"Error in face detection: {str(e)}")
                
            # Convert to format suitable for tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb_frame, (640, 480))
            img = np.array(img)
            img = tk.PhotoImage(data=cv2.imencode('.ppm', img)[1].tobytes())
            
            # Update label with the new image
            self.camera_label.config(image=img)
            self.camera_label.image = img  # Keep a reference
            
            # Sleep to reduce CPU usage
            time.sleep(0.03)
            
        self.camera_label.config(image=None)
        
    def register_face(self):
        # Stop scanning if active
        was_scanning = self.is_scanning
        if was_scanning:
            self.stop_scanning()
            
        # Register new face
        success = self.face_registrar.register_new_face(self.root)
        
        # Reload face recognizer if a new face was added
        if success:
            self.face_recognizer = FaceRecognizer()
            
        # Resume scanning if it was active
        if was_scanning:
            self.start_scanning()
            
    def view_attendance(self):
        # Switch to attendance tab
        self.notebook.select(1)
        
        # Clear existing data
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)
            
        # Get today's attendance
        attendance_list = self.attendance_manager.get_attendance_report()
        
        # Populate treeview
        for idx, (name, time, date) in enumerate(attendance_list):
            self.report_tree.insert("", tk.END, values=(name, time, date))
            
        if not attendance_list:
            self.status_var.set("No attendance records for today")
        else:
            self.status_var.set(f"Showing {len(attendance_list)} attendance records")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("images/known", exist_ok=True)
    os.makedirs("attendance_records", exist_ok=True)
    
    # Start application
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
