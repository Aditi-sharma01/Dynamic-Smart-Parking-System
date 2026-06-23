from ultralytics import YOLO
import torch

def main():

    print("=" * 50)
    print("DYNAMIC PARKING FINE-TUNING")
    print("=" * 50)

    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected!")
        return

    # Load previously trained best model
    model = YOLO(
        r"runs/detect/dynamic_parking/yolov8n_v1/weights/best.pt"
    )

    model.train(
        data="data.yaml",

        # Fine-tuning settings
        epochs=30,
        patience=15,

        imgsz=640,
        batch=8,

        device=0,
        workers=2,

        cache=True,

        optimizer="AdamW",

        # Very low learning rate
        lr0=0.00005,

        # Light augmentation only
        augment=True,

        degrees=0,
        translate=0.02,
        scale=0.1,

        fliplr=0.5,
        flipud=0.0,

        # Disable mosaic during fine-tuning
        mosaic=0,

        project="dynamic_parking",

        name="yolov8n_finetuned_v1"
    )

    print("\nFine-tuning Finished!")

if __name__ == "__main__":
    main()