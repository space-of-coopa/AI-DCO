# YOLOv8 객체 탐지 및 어노테이션 도구
본 프로젝트는 Ultralytics YOLOv8 모델을 활용한 객체 탐지 파이프라인으로, 이미지 내 중심 영역에 위치한 객체를 자동으로 식별하고 시각화하며 YOLO 형식의 주석 파일을 생성합니다. 추가적으로 빈 검출 결과에 대한 자동 정리 기능을 제공합니다.

##  주요 기능
- 병렬 이미지 처리: ThreadPoolExecutor를 활용한 멀티스레드 처리
- GPU 가속 지원: MPS(Metal Performance Shaders) 및 CUDA 자동 감지
- 중심 객체 필터링: 이미지 중앙 50% 영역 내 객체만 선별
- 자동 파일 관리: 빈 검출 결과에 대한 이미지/주석 파일 자동 삭제
- 다양한 형식 지원: JPG, PNG, JPEG 확장자 호환

## 전제 조건
- Python 3.8 이상
- PyTorch 1.8 이상
- Ultralytics YOLOv8 라이브러리
- OpenCV 4.5 이상

```bash
pip install torch ultralytics opencv-python numpy
```

## 설치 방법
1. 저장소 복제:
```bash
git clone https://github.com/your-repository/yolov8-object-detection.git
cd yolov8-object-detection
```
2. 가상 환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate
```
3. 종속성 설치:
``` bash
pip install -r requirements.txt
```

## 사용 방법
1. 입력 이미지를 InputImage 폴더에 배치
2. 메인 스크립트 실행:
```bash
python main.py
```
3. 처리 결과 확인:
```bash
output_images/
├── boxed_image1.jpg  # 박스 표시 이미지
└── image1.txt        # YOLO 형식 주석
```
실행 매개변수 조정 (선택 사항):
- 모델 경로: model = YOLO('yolov8s.pt') (더 정확한 모델 사용 시)
- 중심 영역 비율: center_fraction=0.7 (70% 중심 영역 적용 시)

## 기능 커스터마이징
모델 변경
```python
model = YOLO('yolov8x.pt')
```
중심 영역 조정
```python
def is_in_center(..., center_fraction=0.7):  # 70% 중심 영역
```
시각화 설정
```python
cv2.rectangle(..., color=(255, 0, 0), thickness=3)
```








