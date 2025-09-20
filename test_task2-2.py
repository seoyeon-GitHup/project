import cv2
import numpy as np
import os

def test_depth_map_shape():
    # 테스트용 임시 이미지 생성
    test_img = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite('test_sample.jpg', test_img)

    image = cv2.imread('test_sample.jpg')
    assert image is not None, "이미지 로드 실패"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # 깊이 맵 shape 검증
    assert depth_map.shape == image.shape, "깊이 맵 shape 오류"

    os.remove('test_sample.jpg')