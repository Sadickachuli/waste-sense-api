from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")
print(model)