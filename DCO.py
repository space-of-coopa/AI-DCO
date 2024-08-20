from ultralytics import YOLO

def is_in_center(bbox, img_width, img_height, center_fraction=0.5):
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    center_x_min = img_width * (1 - center_fraction) / 2
    center_x_max = img_width * (1 + center_fraction) / 2
    center_y_min = img_height * (1 - center_fraction) / 2
    center_y_max = img_height * (1 + center_fraction) / 2

    return center_x_min <= x_center <= center_x_max and center_y_min <= y_center <= center_y_max

def detect_center_objects(image_path, model_path='yolov8n.pt', center_fraction=0.5):
    model = YOLO(model_path)
    results = model(image_path)

    img = results[0].orig_img
    img_height, img_width, _ = img.shape

    center_objects = []
    for bbox in results[0].boxes.xyxy:  # 바운딩 박스 좌표
        if is_in_center(bbox, img_width, img_height, center_fraction):
            center_objects.append(bbox)

    return center_objects

# 사용 예시
center_objects = detect_center_objects('./InputImage/InputImage.jpg', center_fraction=0.5)
print("Central objects:", center_objects)
