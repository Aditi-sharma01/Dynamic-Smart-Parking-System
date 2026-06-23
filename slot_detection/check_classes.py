from ultralytics import YOLO

model = YOLO(
    r"runs/detect/parking_detection/yolov8s_8000-3/weights/best.pt"
)

print(model.names)

model.names[0] = "open_slot"
model.names[1] = "occupied_slot"

print(model.names)