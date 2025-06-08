from ultralytics import YOLO

# Load a YOLOv8n custom model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference with arguments
results = model.predict(source="skin-cancer/test/images/1000_jpg.rf.be94157957df75acecedeb2425941f66.jpg", save=True, imgsz=640, conf=0.5)

results[0].show()