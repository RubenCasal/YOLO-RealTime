import torch
import os
from ultralytics import  YOLO

model_path = "yolov9m.pt"

model = YOLO(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

	
results=model.train(data="./Fisheye Face Detection dataset/data.yaml", epochs=50, imgsz=600, batch=2, device=device)



