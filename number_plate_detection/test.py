from ultralytics import YOLO
import cv2

# Load trained number plate model
model = YOLO(
    r"runs/detect/license_plate_detection/yolov8s_v1-2/weights/best.pt"
)

# Input image
image_path = "test2.jpg"

# Run prediction
results = model.predict(
    source=image_path,
    conf=0.25,
    save=False
)

# Draw bounding boxes
annotated_image = results[0].plot()

# Save output image
output_path = "number_plate_output.jpg"
cv2.imwrite(output_path, annotated_image)

print(f"Output saved as {output_path}")

# Display image
cv2.imshow("Number Plate Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()