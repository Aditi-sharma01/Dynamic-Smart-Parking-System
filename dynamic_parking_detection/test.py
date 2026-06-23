from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(
    "runs/detect/dynamic_parking/yolov8n_v1/weights/best.pt"
)

# Input image
image_path = "test2.jpg"

# Predict
results = model.predict(
    source=image_path,
    conf=0.25,
    save=False
)

# Get plotted image
annotated_frame = results[0].plot()

# Save output
output_path = "output.jpg"
cv2.imwrite(output_path, annotated_frame)

print(f"Output saved as {output_path}")