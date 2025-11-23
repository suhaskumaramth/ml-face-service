from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import numpy as np

# --------- LOAD YOLOv8 FACE MODEL ONCE ----------
MODEL_PATH = "yolov8n-face.pt"  # this file should be in the same folder
model = YOLO(MODEL_PATH)

app = FastAPI(title="YOLOv8 Face People Counter")

# Allow CORS (so Supabase / Lovable frontend can call this service)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Box(BaseModel):
    x: int
    y: int
    width: int
    height: int
    confidence: float


class DetectionResponse(BaseModel):
    count: int
    boxes: list[Box]
    image_width: int
    image_height: int


def run_detection(image_bytes: bytes, conf: float, iou: float) -> DetectionResponse:
    # Load image
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    # Run YOLOv8 face detection
    results = model(img, conf=conf, iou=iou, verbose=False)[0]

    boxes = []
    # results.boxes.xyxy: [x1, y1, x2, y2]
    for box_tensor, conf_tensor in zip(results.boxes.xyxy, results.boxes.conf):
        x1, y1, x2, y2 = box_tensor.tolist()
        boxes.append(
            Box(
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1),
                confidence=float(conf_tensor),
            )
        )

    return DetectionResponse(
        count=len(boxes),
        boxes=boxes,
        image_width=w,
        image_height=h,
    )


@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.3),
    iou: float = Form(0.4),
):
    image_bytes = await file.read()
    return run_detection(image_bytes, confidence, iou)


@app.post("/detect/frame", response_model=DetectionResponse)
async def detect_frame(
    file: UploadFile = File(...),
    confidence: float = Form(0.3),
    iou: float = Form(0.4),
):
    image_bytes = await file.read()
    return run_detection(image_bytes, confidence, iou)


@app.get("/")
async def root():
    return {"message": "YOLOv8 face-based people counter is running"}
