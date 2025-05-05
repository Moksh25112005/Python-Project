from ultralytics import YOLO

# Download standard YOLOv8n model (this one definitely exists)
model = YOLO("yolov8n.pt")
print("Model downloaded successfully!")
