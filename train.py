from ultralytics import YOLO

# Load a pre-trained model (recommended for faster results)
model = YOLO('yolo11n.pt') # Or 'yolov11s.pt', etc.

# Train on your custom dataset
# 'data' points to your data.yaml, 'epochs' for training duration, 'device' for GPU/CPU/MPS
results = model.train(
    data='True Battlebots.v1i.yolov8-obb\data.yaml', 
    epochs=5, 
    imgsz=640, 
    device='cpu' # or 'mps' for Apple Silicon, 'cpu'
)
