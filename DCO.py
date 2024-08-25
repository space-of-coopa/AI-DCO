import os
import cv2
import numpy as np
from ultralytics import YOLO

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

def save_yolo_label(output_file_path, class_id, bbox, img_width, img_height):
    # 바운딩 박스 좌표를 YOLO 포맷으로 변환
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2 / img_width
    center_y = (y_min + y_max) / 2 / img_height
    bbox_width = (x_max - x_min) / img_width
    bbox_height = (y_max - y_min) / img_height

    # YOLO 포맷으로 라벨 정보 작성
    with open(output_file_path, 'w') as f:
        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

def show_image_with_bbox(image, bbox, image_path, class_id, img_width, img_height, output_dir="bbox"):
    x_min, y_min, x_max, y_max = map(int, bbox)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)  # 빨간색, 두께 4
    info_text = f"Class: {class_id}, Pos: ({x_min}, {y_min}), Size: ({bbox_width}, {bbox_height})"
    text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
    cv2.putText(image, info_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # 노란색, 두께 2
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_bbox.png")
    cv2.imwrite(output_file, image)
    print(f"Image with bbox saved as {output_file}")

def detect_and_save_labels(image_path, hex_color, model, output_dir, img_width, img_height, class_id=3, center_fraction=0.5):
    target_color = hex_to_rgb(hex_color)
    results = model(image_path)
    img = results[0].orig_img

    closest_bbox = None
    min_dist = float('inf')

    for bbox in results[0].boxes.xyxy:
        if is_in_center(bbox, img_width, img_height, center_fraction):
            dominant_color = get_dominant_color(img, bbox)
            dist = np.linalg.norm(dominant_color - target_color)
            if dist < min_dist:
                min_dist = dist
                closest_bbox = bbox

    if closest_bbox is not None:
        output_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_file_path = os.path.join(output_dir, f"{output_filename}.txt")
        save_yolo_label(output_file_path, class_id, closest_bbox, img_width, img_height)
        print(f"Saved label for {image_path} as {output_file_path}")
        show_image_with_bbox(img, closest_bbox, image_path, class_id, img_width, img_height, output_dir="bbox")

        return True  # 라벨이 생성됨

    return False  # 라벨이 생성되지 않음

def process_images_in_folder(folder_path, hex_color, output_dir, model_path='yolov8n.pt', class_id=3, img_width=4000, img_height=3000, center_fraction=0.5):
    model = YOLO(model_path)
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

    os.makedirs(output_dir, exist_ok=True)

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_path}...")
        label_created = detect_and_save_labels(image_path, hex_color, model, output_dir, img_width, img_height, class_id, center_fraction)

        if not label_created:
            os.remove(image_path)  # 라벨이 없으면 이미지 삭제
            print(f"Deleted image {image_path} because no label was created.")

    print("All images processed.")

hex_color = '#DAC5D6'
folder_path = 'InputImage'
output_dir = 'OutputLabels'

process_images_in_folder(folder_path, hex_color, output_dir, class_id=3, img_width=4000, img_height=3000)
