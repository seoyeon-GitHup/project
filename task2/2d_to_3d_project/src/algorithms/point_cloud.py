import numpy as np

def depth_to_point_cloud(depth_map):
    # 빈 배열일 경우 바로 반환하여 ValueError 방지
    if depth_map.size == 0:
        return np.array([])
    
    h, w = depth_map.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    # X, Y 좌표를 float32로 명시적 변환
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    Z = depth_map.astype(np.float32)
    
    # np.dstack이 모든 배열을 동일한 타입으로 병합합니다.
    points_3d = np.dstack((X, Y, Z))
    return points_3d