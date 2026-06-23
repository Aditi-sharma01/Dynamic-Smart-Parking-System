from ultralytics import YOLO
import torch

def main():

    print("=" * 50)
    print("LICENSE PLATE DETECTION FINE-TUNING")
    print("=" * 50)

    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected!")
        return

    # Load best model from previous training
    model = YOLO(
        "runs/detect/license_plate_detection/yolov8s_v1-2/weights/best.pt"
    )

    model.train(
        data="data.yaml",

        epochs=30,

        imgsz=640,

        batch=8,

        device=0,

        workers=2,

        cache=False,

        optimizer="AdamW",

        # Lower learning rate for fine-tuning
        lr0=0.00005,

        patience=15,

        # Mild augmentations only
        augment=True,

        degrees=0,
        translate=0.02,
        scale=0.1,

        fliplr=0.0,
        flipud=0.0,

        # Disable mosaic during fine-tuning
        mosaic=0,

        project="license_plate_detection",

        name="yolov8s_finetuned_v1"
    )

    print("\nFine-tuning Finished!")

if __name__ == "__main__":
    main()