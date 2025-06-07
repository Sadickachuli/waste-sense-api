from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Ensure the best.pt file is in the same directory

# Define average weights for each class (in kilograms)
average_weights = {
    "Aluminium foil": 0.01,  # 10 grams
    "Battery": 0.2,          # 200 grams
    "Aluminium blister pack": 0.05,  # 50 grams
    "Carded blister pack": 0.03,     # 30 grams
    "Other plastic bottle": 0.5,     # 500 grams
    "Clear plastic bottle": 0.5,     # 500 grams
    "Glass bottle": 1.0,             # 1 kg
    "Plastic bottle cap": 0.02,      # 20 grams
    "Metal bottle cap": 0.03,        # 30 grams
    "Broken glass": 0.8,             # 800 grams
    "Food Can": 0.3,                 # 300 grams
    "Aerosol": 0.4,                  # 400 grams
    "Drink can": 0.35,               # 350 grams
    "Toilet tube": 0.05,             # 50 grams
    "Other carton": 0.2,             # 200 grams
    "Egg carton": 0.15,              # 150 grams
    "Drink carton": 0.25,            # 250 grams
    "Corrugated carton": 0.5,        # 500 grams
    "Meal carton": 0.3,              # 300 grams
    "Pizza box": 0.4,                # 400 grams
    "Paper cup": 0.05,               # 50 grams
    "Disposable plastic cup": 0.03,  # 30 grams
    "Foam cup": 0.02,                # 20 grams
    "Glass cup": 0.8,                # 800 grams
    "Other plastic cup": 0.03,       # 30 grams
    "Food waste": 1.0,               # 1 kg
    "Glass jar": 1.2,                # 1.2 kg
    "Plastic lid": 0.02,             # 20 grams
    "Metal lid": 0.03,               # 30 grams
    "Other plastic": 0.1,            # 100 grams
    "Magazine paper": 0.2,           # 200 grams
    "Tissues": 0.01,                 # 10 grams
    "Wrapping paper": 0.05,          # 50 grams
    "Normal paper": 0.1,             # 100 grams
    "Paper bag": 0.15,               # 150 grams
    "Plastified paper bag": 0.2,     # 200 grams
    "Plastic film": 0.1,             # 100 grams
    "Six pack rings": 0.05,          # 50 grams
    "Garbage bag": 0.2,              # 200 grams
    "Other plastic wrapper": 0.05,   # 50 grams
    "Single-use carrier bag": 0.03,  # 30 grams
    "Polypropylene bag": 0.1,        # 100 grams
    "Crisp packet": 0.02,            # 20 grams
    "Spread tub": 0.3,               # 300 grams
    "Tupperware": 0.5,               # 500 grams
    "Disposable food container": 0.2,  # 200 grams
    "Foam food container": 0.1,      # 100 grams
    "Other plastic container": 0.3,  # 300 grams
    "Plastic glooves": 0.02,         # 20 grams
    "Plastic utensils": 0.03,        # 30 grams
    "Pop tab": 0.01,                 # 10 grams
    "Rope & strings": 0.2,           # 200 grams
    "Scrap metal": 1.5,              # 1.5 kg
    "Shoe": 0.8,                     # 800 grams
    "Squeezable tube": 0.1,          # 100 grams
    "Plastic straw": 0.02,           # 20 grams
    "Paper straw": 0.01,             # 10 grams
    "Styrofoam piece": 0.05,         # 50 grams
    "Unlabeled litter": 0.1,         # 100 grams
    "Cigarette": 0.01                # 10 grams
}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Run inference with a low confidence threshold
    results = model(image, conf=0.02)  # Set confidence threshold to 0.02

    # Parse results
    detections = []
    total_weight = 0  # Initialize total weight

    for box in results[0].boxes:
        class_name = model.names[int(box.cls)]  # Class name
        confidence = float(box.conf)           # Confidence score
        bbox = box.xyxy.tolist()               # Bounding box coordinates

        # Add detection to the list
        detections.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": bbox
        })

        # Add the weight of the detected object (if class is in average_weights)
        if class_name in average_weights:
            total_weight += average_weights[class_name]
        else:
            print(f"Warning: No weight defined for class '{class_name}'")

    return {
        "detections": detections,
        "total_weight": total_weight  # Return the total weight in kilograms
    }