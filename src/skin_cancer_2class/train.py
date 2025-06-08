from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train the model on the dataset for 100 epochs
results = model.train(data="skin-cancer/data.yaml", epochs=10, imgsz=640)