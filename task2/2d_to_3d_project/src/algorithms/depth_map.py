import cv2
import numpy as np
import os

def generate_depth_map(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"{image_path} 파일을 찾을 수 없습니다.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return image, depth_map, gray

def save_depth_map(depth_map, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, depth_map)

def visualize_depth_map(image, depth_map):
    if len(depth_map.shape) == 2:
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((image, depth_map))
    cv2.imshow('Original | Depth Map', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_path = os.path.join(BASE_DIR, "data", "raw", "sample.jpg")
    
    image, depth_map, gray = generate_depth_map(img_path)
    visualize_depth_map(image, depth_map)
    save_depth_map(depth_map, os.path.join(BASE_DIR, "experiments", "results", "sample_depth.jpg"))