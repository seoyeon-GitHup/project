import os
import cv2
from ultralytics import YOLO

# === 테스트 환경 설정 ===
MODEL_PATH = r"C:\git\task3\cars_detection\runs\detect\train14\weights\best.pt"
IMAGE_PATH = r"C:\git\task3\cars_detection\data\test\images\00dea1edf14f09ab_jpg.rf.3f17c8790a68659d03b1939a59ccda80.jpg"


# === test_model_load(): YOLOv8 학습 가중치 정상 로딩 여부 ===
def test_model_load():
    """YOLOv8 모델 가중치가 정상적으로 로드되는지 확인"""
    model = YOLO(MODEL_PATH)
    assert model is not None, "YOLO 모델 로드 실패"


# === test_inference(): 입력 이미지에 대한 객체 탐지 결과 반환 확인 ===
def test_inference():
    """입력 이미지에 대한 객체 탐지 결과가 정상적으로 반환되는지 확인"""
    assert os.path.exists(IMAGE_PATH), "테스트 이미지가 존재하지 않습니다!"
    model = YOLO(MODEL_PATH)
    image = cv2.imread(IMAGE_PATH)
    results = model(image)
    assert len(results) > 0, "탐지 결과가 없습니다!"
    assert hasattr(results[0], "boxes"), "탐지 결과에 boxes 속성이 없습니다!"


# === test_visualization(): OpenCV 시각화 실행 시 결과 이미지 생성 여부 ===
def test_visualization():
    """OpenCV 시각화 코드 실행 후 결과 이미지가 정상적으로 생성되는지 확인"""
    model = YOLO(MODEL_PATH)
    image = cv2.imread(IMAGE_PATH)
    results = model(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 파일 저장 (임시)
    output_path = "test_output.jpg"
    cv2.imwrite(output_path, image)
    assert os.path.exists(output_path), "결과 이미지가 저장되지 않았습니다!"