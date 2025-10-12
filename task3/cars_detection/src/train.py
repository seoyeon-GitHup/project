#from ultralytics import YOLO

#model = YOLO("yolov8n.pt") # YOLOv8 기본 모델 사용

# 사용자 데이터셋으로 학습 (data.yaml 파일 필요)
#model.train(data=r"C:/git/task3/cars_detection/data/data.yaml", epochs=50, imgsz=320, batch=8)



# 개선된 하이퍼파라미터로 재학습
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(
    data=r"C:/git/task3/cars_detection/data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="train15"  # 새 실험 이름 지정 (기존과 구분 가능) 
)