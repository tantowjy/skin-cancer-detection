from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the dataset for 100 epochs
results = model.train(data="skin-cancer/data.yaml", epochs=5, imgsz=640)