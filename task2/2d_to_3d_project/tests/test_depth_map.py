# tests/test_depth_map.py

import cv2
import pytest
import os
import sys
import numpy as np

# 프로젝트 루트 디렉토리를 경로에 추가하여 src 모듈을 찾을 수 있도록 함
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.algorithms.depth_map import generate_depth_map

# 테스트에 사용할 가짜 이미지 파일을 생성하는 fixture
@pytest.fixture(scope="session")
def sample_image_path(tmp_path_factory):
    # 테스트 전용 임시 디렉토리에 더미 이미지 파일 생성
    temp_dir = tmp_path_factory.mktemp("data_raw")
    temp_path = temp_dir / "sample.jpg"
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(temp_path), dummy_image)
    return str(temp_path)

def test_generate_depth_map_returns_three_values(sample_image_path):
    """generate_depth_map 함수가 세 개의 값을 반환하는지 테스트"""
    image, depth_map, gray = generate_depth_map(sample_image_path)
    assert image is not None
    assert depth_map is not None
    assert gray is not None

def test_generated_depth_map_has_correct_shape(sample_image_path):
    """생성된 뎁스 맵의 형태가 원본 이미지와 일치하는지 테스트"""
    image, depth_map, gray = generate_depth_map(sample_image_path)
    assert image.shape[:2] == depth_map.shape[:2]
    assert gray.shape == (image.shape[0], image.shape[1])

def test_generate_depth_map_raises_file_not_found():
    """존재하지 않는 파일 경로에 대해 FileNotFoundError가 발생하는지 테스트"""
    with pytest.raises(FileNotFoundError):
        generate_depth_map("non_existent_path.jpg")