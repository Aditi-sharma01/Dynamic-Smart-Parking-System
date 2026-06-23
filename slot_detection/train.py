from ultralytics import YOLO
import torch

def main():
    print("=" * 50)
    print("PARKING SLOT DETECTION FINE-TUNING")
    print("=" * 50)

    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(r"runs/detect/parking_detection/yolov8s_finetuned_v1/weights/best.pt")

    model.train(
        data="parking_dataset/data.yaml",

        epochs=30,          # start with 50 more epochs
        imgsz=640,
        batch=4,

        device=0,
        workers=2,

        cache=True,

        optimizer="AdamW",

        lr0=0.00005,         # lower LR for fine-tuning

        augment=True,
        degrees=0,
        translate=0.02,
        scale=0.2,
        fliplr=0.5,
        mosaic=0,

        patience=20,

        project="parking_detection",
        name="yolov8s_finetuned_v2"
    )

if __name__ == "__main__":
    main()