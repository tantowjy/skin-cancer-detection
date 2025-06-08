from ultralytics import YOLO

# Load a YOLO11n custom model
model = YOLO("runs/detect/train/weights/best.pt")

# Run inference with arguments
results = model.predict(source="skin-cancer/test/images/ISIC_0024324_jpg.rf.38306d06443038cc5a75c4b37feba36e.jpg", save=True, imgsz=640, conf=0.5)

results[0].show()