import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Forzar CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Descargar modelos al iniciar
logger.info("Iniciando aplicación...")
try:
    from utils.download_models import download_models
    logger.info("Descargando modelos...")
    if download_models():
        logger.info("Modelos descargados correctamente")
    else:
        logger.error(" Error descargando modelos")
except Exception as e:
    logger.error(f"Error en descarga: {e}")

from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from app.evaluar_procesador import SignLanguageProcessor

app = FastAPI(
    title="Traductor Lenguaje de Señas",
    description="API para reconocimiento de lenguaje de señas en tiempo real",
    version="1.0.0"
)

# CORS para React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar procesador
logger.info(" Inicializando procesador...")
processor = SignLanguageProcessor()
logger.info("Procesador listo")

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": " API de Lenguaje de Señas funcionando",
        "endpoints": ["/process_frame", "/process_base64", "/health", "/docs"],
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": processor.model_static is not None and processor.model_dynamic is not None
    }

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Imagen inválida"}
        
        result = processor.process_frame(frame)
        return result
        
    except Exception as e:
        logger.error(f"Error procesando frame: {e}")
        return {"error": str(e)}

@app.post("/process_base64/")
async def process_base64(data: dict = Body(...)):
    try:
        if "image" not in data:
            return {"error": "Falta el campo 'image' en el JSON"}
        
        # Limpiar el string base64
        image_data = data["image"]
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "No se pudo decodificar la imagen"}
        
        result = processor.process_frame(frame)
        return result
        
    except Exception as e:
        logger.error(f"Error procesando base64: {e}")
        return {"error": str(e)}