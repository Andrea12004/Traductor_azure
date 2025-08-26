import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forzar CPU en Render

from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from app.evaluar_procesador import SignLanguageProcessor

app = FastAPI()

# Permitir peticiones desde cualquier origen (ajusta en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = SignLanguageProcessor()

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Imagen inválida"}
    return processor.process_frame(frame)

@app.post("/process_base64/")
async def process_base64(data: dict = Body(...)):
    try:
        image_bytes = base64.b64decode(data["image"])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Imagen inválida"}
        return processor.process_frame(frame)
    except Exception as e:
        return {"error": str(e)}
