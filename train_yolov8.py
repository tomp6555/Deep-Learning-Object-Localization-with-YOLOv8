from ultralytics import YOLO

# Load a pre-trained model YOLOv8n mode (n for nano, smallest and fastest)
# You can choose other models like 'yolov8s.pt'(small), 'yolo8m.pt'(medium), etc.
model = YOLO('yolov8n.pt')

#Train the model. Adjust epochs and image size according to task's needs
results = model.train(data='dataset.yaml', epochs=50, imgsz=640)

#Trained model usually saved automatically in runs/detect/trainx/weights/best.pt
