import cv2
import numpy as np
import os

def test_point_cloud_shape():
    # 테스트용 임시 이미지 생성
    test_img = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.imwrite('test_sample.jpg', test_img)

    image = cv2.imread('test_sample.jpg')
    assert image is not None, "이미지 로드 실패"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    h, w = depth_map.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)
    points_3d = np.dstack((X, Y, Z))

    # 포인트 클라우드 shape 검증
    assert points_3d.shape == (h, w, 3), "포인트 클라우드 shape 오류"

    os.remove('test_sample.jpg')