import tkinter as tk
from collections import Counter
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load YOLO custom model
model = YOLO("model/skin_cancer_7class.pt")

# Inference function
def run_inference(image_path):
    results = model(image_path)
    result_image = results[0].plot()

    # Get class names from detections
    boxes = results[0].boxes
    class_ids = boxes.cls.tolist() if boxes is not None else []
    class_names = [model.names[int(cls_id)] for cls_id in class_ids]
    
    # Count occurrences
    name_counts = Counter(class_names)
    
    return result_image, name_counts

# Load and show image on canvas
def show_image(img_cv, canvas):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_resized = img_pil.resize((640, 640))
    img_tk = ImageTk.PhotoImage(img_resized)
    canvas.img = img_tk  # prevent garbage collection
    canvas.create_image(0, 0, anchor="nw", image=img_tk)

# Open image and run detection
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Show original image
    original_img = cv2.imread(file_path)
    show_image(original_img, left_canvas)

    # Run detection and show result
    detected_img, name_counts = run_inference(file_path)
    show_image(detected_img, right_canvas)

    # Format detection summary
    if name_counts:
        formatted_names = ", ".join(f"{count} {name}" for name, count in name_counts.items())
    else:
        formatted_names = "No objects detected"

    # Update label
    counter_label.config(text=f"Detected: {formatted_names}")

# GUI window
root = tk.Tk()
root.title("Skin Cancer Object Detection")

# Layout: side by side canvas
left_canvas = tk.Canvas(root, width=640, height=640, bg='gray')
left_canvas.grid(row=0, column=0, padx=10, pady=10)
tk.Label(root, text="Input Image").grid(row=1, column=0)

right_canvas = tk.Canvas(root, width=640, height=640, bg='gray')
right_canvas.grid(row=0, column=1, padx=10, pady=10)
tk.Label(root, text="Detected Output").grid(row=1, column=1)

# Detection summary label
counter_label = tk.Label(root, text="Detected: None", font=("Arial", 14))
counter_label.grid(row=2, column=0, columnspan=2)

# Button to open image
btn = tk.Button(root, text="Upload Image", command=open_image)
btn.grid(row=3, column=0, columnspan=2, pady=20)

# Start GUI
root.mainloop()