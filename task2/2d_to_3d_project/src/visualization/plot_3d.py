import os
import sys
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 경로 설정
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from algorithms.depth_map import generate_depth_map
    from algorithms.point_cloud import depth_to_point_cloud

def plot_point_cloud(points_3d, step=5):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    pts = points_3d[::step, ::step, :]
    ax.scatter(pts[:, :, 0].flatten(), pts[:, :, 1].flatten(), pts[:, :, 2].flatten(),
               c=pts[:, :, 2].flatten(), cmap='jet', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_path = os.path.join(BASE_DIR, "data", "raw", "sample.jpg")
    
    # generate_depth_map이 반환하는 세 가지 값을 정확히 받음
    image, depth_map, gray_image = generate_depth_map(img_path)
    
    # gray_image를 depth_to_point_cloud에 전달
    if gray_image is not None:
        points_3d = depth_to_point_cloud(gray_image)
        plot_point_cloud(points_3d)
        print("3D 시각화 완료")
    else:
        print("이미지 또는 뎁스 맵을 생성할 수 없어 3D 시각화를 건너뜁니다.")