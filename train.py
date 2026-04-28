import cv2
from datetime import datetime; current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
import torch
def train():
    from ultralytics import YOLO
    model = YOLO("yolov13.yaml")
    t1 = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args = {
        'data': './dataset.yaml',
        'epochs': 300,
        'imgsz': 640,
        'batch': 8,
        'lr0' : 1e-4,
        'amp' : True,
        'device': '0',
        'workers': 4,
        'optimizer': 'AdamW',
        'name': f'train_{t1}',
        'save_period': 50,
        'cache' : 'ram',
        'patience' : 10,
        'val': True,
        'amp' : True
    }
    results = model.train(**args)
    t2 = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"Mission accomplished! Current time: {t2}. Best model save at: {results.save_dir}")
if __name__ == '__main__':
    train()
