from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")  # Ensure this file is in the root folder or adjust path

# Average weights in kg
average_weights = {
    "Aluminium foil": 0.01,
    "Battery": 0.2,
    "Aluminium blister pack": 0.05,
    "Carded blister pack": 0.03,
    "Other plastic bottle": 0.5,
    "Clear plastic bottle": 0.5,
    "Glass bottle": 1.0,
    "Plastic bottle cap": 0.02,
    "Metal bottle cap": 0.03,
    "Broken glass": 0.8,
    "Food Can": 0.3,
    "Aerosol": 0.4,
    "Drink can": 0.35,
    "Toilet tube": 0.05,
    "Other carton": 0.2,
    "Egg carton": 0.15,
    "Drink carton": 0.25,
    "Corrugated carton": 0.5,
    "Meal carton": 0.3,
    "Pizza box": 0.4,
    "Paper cup": 0.05,
    "Disposable plastic cup": 0.03,
    "Foam cup": 0.02,
    "Glass cup": 0.8,
    "Other plastic cup": 0.03,
    "Food waste": 1.0,
    "Glass jar": 1.2,
    "Plastic lid": 0.02,
    "Metal lid": 0.03,
    "Other plastic": 0.1,
    "Magazine paper": 0.2,
    "Tissues": 0.01,
    "Wrapping paper": 0.05,
    "Normal paper": 0.1,
    "Paper bag": 0.15,
    "Plastified paper bag": 0.2,
    "Plastic film": 0.1,
    "Six pack rings": 0.05,
    "Garbage bag": 0.2,
    "Other plastic wrapper": 0.05,
    "Single-use carrier bag": 0.03,
    "Polypropylene bag": 0.1,
    "Crisp packet": 0.02,
    "Spread tub": 0.3,
    "Tupperware": 0.5,
    "Disposable food container": 0.2,
    "Foam food container": 0.1,
    "Other plastic container": 0.3,
    "Plastic glooves": 0.02,
    "Plastic utensils": 0.03,
    "Pop tab": 0.01,
    "Rope & strings": 0.2,
    "Scrap metal": 1.5,
    "Shoe": 0.8,
    "Squeezable tube": 0.1,
    "Plastic straw": 0.02,
    "Paper straw": 0.01,
    "Styrofoam piece": 0.05,
    "Unlabeled litter": 0.1,
    "Cigarette": 0.01
}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and open image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run inference
    results = model(image, conf=0.02)
    result = results[0]

    # Detections and weight
    detections = []
    total_weight = 0.0
    for box in result.boxes:
        class_name = model.names[int(box.cls)]
        confidence = float(box.conf)
        bbox = box.xyxy.tolist()

        detections.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": bbox
        })

        if class_name in average_weights:
            total_weight += average_weights[class_name]

    # Get annotated image as a NumPy array
    annotated_image = result.plot()  # Returns np.ndarray (BGR)

    # Convert to RGB for PIL compatibility
    annotated_image_rgb = Image.fromarray(annotated_image[:, :, ::-1])  # BGR to RGB

    # Save to buffer
    buf = io.BytesIO()
    annotated_image_rgb.save(buf, format='JPEG')
    buf.seek(0)
    annotated_image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {
        "detections": detections,
        "total_weight": total_weight,
        "annotated_image": annotated_image_base64
    }