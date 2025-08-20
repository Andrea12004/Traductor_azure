from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import uuid
from typing import Optional
import logging

# Importar tu lógica
from app.evaluar_procesador import process_frame_simple

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Traductor LSC API", version="1.0.0")

# CORS para React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para sesiones simples
sessions = {}

# Modelo para recibir frames
class FrameData(BaseModel):
    frame_base64: str

# Modelo para respuestas
class WordResponse(BaseModel):
    success: bool
    word: Optional[str] = None
    confidence: Optional[float] = None
    type: Optional[str] = None  # "static" o "dynamic"
    message: str

@app.get("/")
async def root():
    return {"message": "API Traductor LSC funcionando"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/start_session")
async def start_session():
    """Crear nueva sesión"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"frames_count": 0}
    
    logger.info(f"Nueva sesión: {session_id}")
    return {
        "success": True,
        "session_id": session_id,
        "message": "Sesión iniciada"
    }

@app.post("/process_frame/{session_id}")
async def process_frame_endpoint(session_id: str, frame_data: FrameData) -> WordResponse:
    """Procesar frame de React Native"""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    try:
        # Decodificar frame de base64
        frame_bytes = base64.b64decode(frame_data.frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return WordResponse(
                success=False,
                message="Frame inválido"
            )
        
        # AQUÍ LLAMAS A TU FUNCIÓN ORIGINAL
        result = process_frame_simple(frame)
        
        sessions[session_id]["frames_count"] += 1
        
        return WordResponse(
            success=True,
            word=result.get("word"),
            confidence=result.get("confidence"),
            type=result.get("type"),
            message=result.get("message", "Frame procesado")
        )
        
    except Exception as e:
        logger.error(f"Error procesando frame: {e}")
        return WordResponse(
            success=False,
            message=f"Error: {str(e)}"
        )

@app.post("/clear_session/{session_id}")
async def clear_session(session_id: str):
    """Limpiar sesión"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    # Aquí puedes llamar una función para limpiar el estado si es necesario
    return {"success": True, "message": "Sesión limpiada"}

@app.delete("/end_session/{session_id}")
async def end_session(session_id: str):
    """Terminar sesión"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    del sessions[session_id]
    return {"success": True, "message": "Sesión terminada"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)