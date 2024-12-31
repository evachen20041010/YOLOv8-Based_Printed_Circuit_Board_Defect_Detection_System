from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8x.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
# 用程式碼執行時，workers 要改成0
results = model.train(data = "./datasets/coco8_1/data.yaml", workers = 0, epochs = 100, imgsz = 640)