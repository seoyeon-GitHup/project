from ultralytics import YOLO

model = YOLO("yolov8n.pt") # YOLOv8 기본 모델 사용

# 사용자 데이터셋으로 학습 (data.yaml 파일 필요)
model.train(data=r"C:/git/task3/cars_detection/data/data.yaml", epochs=50, imgsz=320, batch=8)