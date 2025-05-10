from ultralytics import YOLO
import os



if __name__ == '__main__':
    folder = os.path.dirname(__file__)
    model = YOLO("best.pt")
    metrics = model.val(data=os.path.join(folder, 'dataset/dataset/dataset.yaml'))
    metrics.data