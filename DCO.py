import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

def hex_to_rgb(hex_color):
    """16진수 색상 코드를 RGB로 변환합니다."""
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])

def draw_and_save_box(image_path, boxes, output_path):
    """이미지에 바운딩 박스를 그려서 저장합니다."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image file '{image_path}' not found.")
        return

    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        top_left = (int(x1), int(y1))
        bottom_right = (int(x2), int(y2))

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"Image saved with boxes at: {output_path}")

def save_yolo_format(image_path, boxes, output_txt_path):
    """YOLOv8 포맷으로 바운딩 박스 정보를 텍스트 파일에 저장합니다."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image file '{image_path}' not found.")
        return

    img_height, img_width = image.shape[:2]
    with open(output_txt_path, 'w') as f:
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Assuming class_id = 0 for all boxes; modify if class_id is available
            class_id = 0
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"YOLO format annotations saved at: {output_txt_path}")

def is_in_center(bbox, img_width, img_height, center_fraction=0.5):
    """바운딩 박스가 이미지 중심에 있는지 확인합니다."""
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    center_x_start = img_width * (1 - center_fraction) / 2
    center_x_end = img_width * (1 + center_fraction) / 2
    center_y_start = img_height * (1 - center_fraction) / 2
    center_y_end = img_height * (1 + center_fraction) / 2

    return center_x_start <= x_center <= center_x_end and center_y_start <= y_center <= center_y_end

def process_image(image_path, model, output_image_path, output_txt_path):
    """이미지를 처리하고 바운딩 박스를 그려 저장하며 YOLO 포맷으로 주석을 저장합니다."""
    try:
        results = model.predict(image_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # 이미지 크기 가져오기
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Image file '{image_path}' not found.")
            return
        img_height, img_width = image.shape[:2]

        # 디버깅: 모델이 인식한 바운딩 박스 출력
        print(f"Detected boxes for '{image_path}': {boxes}")

        # 중심에 있는 바운딩 박스만 선택
        center_boxes = [box for box in boxes if is_in_center(box, img_width, img_height)]
        print(f"Boxes in center for '{image_path}': {center_boxes}")

        if len(center_boxes) > 0:
            # 바운딩 박스 그리기 및 저장
            draw_and_save_box(image_path, center_boxes, output_image_path)
            # YOLO 포맷 주석 저장
            save_yolo_format(image_path, center_boxes, output_txt_path)
        else:
            # 빈 이미지와 라벨 파일 삭제
            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(output_image_path):
                os.remove(output_image_path)
            if os.path.exists(output_txt_path):
                os.remove(output_txt_path)
            print(f"Deleted empty files for '{image_path}'.")

    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")

if __name__ == '__main__':
    input_folder = './InputImage'
    output_folder = './output_images'
    os.makedirs(output_folder, exist_ok=True)

    # YOLO 모델 로드
    model = YOLO('yolov8n.pt')  # 더 가벼운 모델로 바꿀 수 있음

    # GPU 가속 설정
    model.to('mps' if torch.backends.mps.is_available() else 'cpu')

    # 모든 이미지 처리
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for image_file in os.listdir(input_folder):
            if image_file.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(input_folder, image_file)
                output_image_path = os.path.join(output_folder, f"boxed_{image_file}")
                output_txt_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
                executor.submit(process_image, image_path, model, output_image_path, output_txt_path)
