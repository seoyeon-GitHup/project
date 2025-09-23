# tests/test_point_cloud.py

import pytest
import numpy as np
import os
import sys

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
# 이는 tests 폴더에서 src 폴더의 모듈을 임포트하기 위함입니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.algorithms.point_cloud import depth_to_point_cloud

def test_depth_to_point_cloud_returns_correct_shape():
    """
    2D 뎁스 맵이 3D 포인트 클라우드로 변환될 때
    올바른 형태(H, W, 3)를 반환하는지 테스트합니다.
    """
    # 2D 뎁스 맵을 나타내는 더미(dummy) 데이터 생성
    dummy_depth_map = np.random.rand(480, 640).astype(np.float32)
    
    # 함수 호출
    point_cloud = depth_to_point_cloud(dummy_depth_map)
    
    # 반환된 포인트 클라우드의 형태가 (480, 640, 3)인지 확인
    assert point_cloud.shape == (480, 640, 3), "포인트 클라우드의 형태가 예상과 다릅니다."

def test_depth_to_point_cloud_has_float_type():
    """
    반환된 포인트 클라우드의 데이터 타입이 부동 소수점(float)인지 테스트합니다.
    """
    # 더미 데이터 생성
    dummy_depth_map = np.zeros((100, 100), dtype=np.uint8)
    
    # 함수 호출
    point_cloud = depth_to_point_cloud(dummy_depth_map)
    
    # 반환된 배열의 데이터 타입이 np.float32인지 확인
    assert point_cloud.dtype == np.float32, "포인트 클라우드의 데이터 타입이 float32가 아닙니다."

def test_depth_to_point_cloud_handles_empty_input():
    """
    빈 배열 입력에 대해 빈 배열이 반환되는지 테스트합니다.
    """
    # 빈 배열을 입력으로 전달
    result = depth_to_point_cloud(np.array([]))
    
    # 반환된 결과가 빈 배열이고, 형태가 올바른지 확인
    assert result.size == 0
    assert result.shape == (0,), "빈 배열 반환 형태가 올바르지 않습니다."