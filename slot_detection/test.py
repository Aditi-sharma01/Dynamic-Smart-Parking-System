from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(
    r"runs/detect/parking_detection/yolov8s_finetuned_v1/weights/best.pt"
)

# Correct class mapping
class_names = {
    0: "open_slot",
    1: "occupied_slot"
}

# Input image
image_path = "test6.jpg"

# Read image
img = cv2.imread(image_path)

# Run prediction
results = model.predict(
    source=image_path,
    conf=0.80,
    save=False
)

# Draw boxes manually
for box in results[0].boxes:

    cls = int(box.cls[0])
    conf = float(box.conf[0])

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    label = f"{class_names[cls]} {conf:.2f}"

    # Green for open, Red for occupied
    if cls == 0:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        img,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

# Save output image
output_path = "slot_detection_output.jpg"
cv2.imwrite(output_path, img)

print(f"\nOutput saved as: {output_path}")

# Display image
cv2.imshow("Slot Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()