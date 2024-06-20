import os
import cv2
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from ultralytics import YOLO
from typing import List
from io import BytesIO
import numpy as np
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8x.pt')  # Adjust the model path if necessary

@app.post("/detect/")
async def detect_objects(files: List[UploadFile] = File(...)):
    detection_results = []

    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {file.filename}")
        
        # Rotate the image 90 degrees to the left (counter-clockwise)
        image_rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Perform object detection
        results = model(image_rotated, imgsz=1280)
        
        # Initialize a counter for the number of people
        person_count = 0
        
        # Iterate through the detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = box.cls[0]
                
                # Check if the detected object is a person (class id for 'person' is 0 in COCO dataset)
                if class_id == 0:
                    person_count += 1
        
        # Append the detection result to the list
        detection_results.append({"image": file.filename, "count": person_count})
    
    return JSONResponse(content=detection_results)

# Serve custom static files for the UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint to serve custom HTML form
@app.get("/")
async def main():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
