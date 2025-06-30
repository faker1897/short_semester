from ultralytics import YOLO


def main():
    model = YOLO("yolo11n-seg.pt")

    model.train(
        data="./dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        project="./run/yolov11n-seg",
        name="flood_seg_model"
    )

if __name__ == "__main__":
    main()
