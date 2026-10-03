from ultralytics import YOLO
import cv2

def main():
    model = YOLO('yolo26n.pt')

    result = model.train(data='data.yaml', epochs=100, imgsz=640, batch=8, workers=2, project='./models')

    print(result)


if __name__ == "__main__":
    main()
