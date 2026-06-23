from ultralytics import YOLO

def main():

    model = YOLO(
        "runs/detect/license_plate_detection/yolov8s_v1-2/weights/best.pt"
    )

    metrics = model.val(
        data="data.yaml",
        workers=0,
        plots=True
    )

    print(metrics)

if __name__ == "__main__":
    main()