from ultralytics import YOLO
from PIL import Image

# Load the model
model = YOLO("best.pt")

# Test on an image
image = Image.open("download.jpeg")  # Replace with the path to your test image
results = model(image, conf=0.02)  # Run inference with a confidence threshold of 0.25

# Print results
for box in results[0].boxes:
    print({
        "class": model.names[int(box.cls)],  # Class name
        "confidence": float(box.conf),      # Confidence score
        "bbox": box.xyxy.tolist()          # Bounding box coordinates
    })

# Save the annotated image
results[0].save("image3.jpg")  # Access the first result and save the annotated image
print("Annotated image saved as 'annotated_image.jpg'")