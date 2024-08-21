import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from scipy.spatial import distance

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])

def is_in_center(bbox, img_width, img_height, center_fraction=0.5):
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    center_x_min = img_width * (1 - center_fraction) / 2
    center_x_max = img_width * (1 + center_fraction) / 2
    center_y_min = img_height * (1 - center_fraction) / 2
    center_y_max = img_height * (1 + center_fraction) / 2

    return center_x_min <= x_center <= center_x_max and center_y_min <= y_center <= center_y_max

def get_dominant_color(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = image[y1:y2, x1:x2]
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    pixels = np.float32(cropped_img.reshape(-1, 3))
    n_colors = 1
    _, labels, palette = cv2.kmeans(pixels, n_colors, None,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                                    attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = palette[0].astype(int)
    return dominant_color

def detect_and_show_similar_color_objects(image_path, hex_color, model, center_fraction=0.5):
    target_color = hex_to_rgb(hex_color)

    results = model(image_path)

    img = results[0].orig_img
    img_height, img_width, _ = img.shape

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    closest_color = None
    closest_bbox = None
    min_dist = float('inf')

    for bbox in results[0].boxes.xyxy:
        if is_in_center(bbox, img_width, img_height, center_fraction):
            dominant_color = get_dominant_color(img, bbox)
            dist = distance.euclidean(dominant_color, target_color)
            if dist < min_dist:
                min_dist = dist
                closest_color = dominant_color
                closest_bbox = bbox

    if closest_bbox is not None:
        rect = plt.Rectangle(
            (closest_bbox[0], closest_bbox[1]),
            closest_bbox[2] - closest_bbox[0],
            closest_bbox[3] - closest_bbox[1],
            fill=False,
            color='red',
            linewidth=2
        )
        ax.add_patch(rect)

        # 박스 위치 및 크기 정보 표기
        bbox_width = closest_bbox[2] - closest_bbox[0]
        bbox_height = closest_bbox[3] - closest_bbox[1]
        bbox_info = f"Pos: ({closest_bbox[0]:.1f}, {closest_bbox[1]:.1f}) Size: ({bbox_width:.1f}, {bbox_height:.1f})"
        ax.text(closest_bbox[0], closest_bbox[1] - 10, bbox_info, color='red', fontsize=10, weight='bold')

        print(f"Closest color found: {closest_color} with distance {min_dist}")
        print(bbox_info)

    plt.show()

def process_images_in_folder(folder_path, hex_color, model_path='yolov8n.pt', center_fraction=0.5):
    model = YOLO(model_path)
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_path}...")
        detect_and_show_similar_color_objects(image_path, hex_color, model, center_fraction)

# 사용 예시
hex_color = '#DAC5D6'
folder_path = 'InputImage'  # 이미지들이 저장된 폴더 경로
process_images_in_folder(folder_path, hex_color)
