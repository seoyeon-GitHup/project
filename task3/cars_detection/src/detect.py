import cv2
from ultralytics import YOLO
import os

# 학습된 YOLO 모델 경로 (train14 기준)
model_path = r"C:\git\task3\cars_detection\runs\detect\train14\weights\best.pt"
model = YOLO(model_path)

# 테스트할 이미지 경로 (절대경로 사용)
image_path = r"C:\git\task3\cars_detection\data\test\images\00dea1edf14f09ab_jpg.rf.3f17c8790a68659d03b1939a59ccda80.jpg"

# 파일 존재 여부 확인
if not os.path.exists(image_path):
    raise FileNotFoundError(f"{image_path} 파일이 존재하지 않습니다!")

# 이미지 읽기
image = cv2.imread(image_path)

# 객체 탐지 실행
results = model(image)

# 탐지된 객체 시각화
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌표값 변환
        label = result.names[int(box.cls[0])]   # 클래스 라벨
        confidence = float(box.conf[0])         # 신뢰도
        # 객체 경계 상자 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 결과 출력
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()