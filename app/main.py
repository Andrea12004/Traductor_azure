# app/main.py - Versión optimizada para React Native
import os
import asyncio
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager
import time

# Tu lógica de procesamiento
from app.evaluar_procesador import SignLanguageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== MODELOS PYDANTIC PARA VALIDACIÓN ==========
class FrameRequest(BaseModel):
    client_id: str = "mobile_client"
    frame: str  # Base64 encoded
    timestamp: float = None
    
class ClearRequest(BaseModel):
    client_id: str = "mobile_client"

class ConfigRequest(BaseModel):
    client_id: str = "mobile_client"
    static_threshold: float = 0.8
    dynamic_threshold: float = 0.7
    movement_threshold: float = 0.015

# ========== GESTIÓN DE PROCESADORES ==========
class ProcessorManager:
    def __init__(self):
        self.processors = {}
        self.last_activity = {}
    
    def get_processor(self, client_id: str) -> SignLanguageProcessor:
        """Obtiene o crea un procesador para el cliente"""
        if client_id not in self.processors:
            logger.info(f"Creando nuevo procesador para cliente: {client_id}")
            self.processors[client_id] = SignLanguageProcessor()
        
        self.last_activity[client_id] = time.time()
        return self.processors[client_id]
    
    def cleanup_inactive(self, max_idle_seconds: int = 1800):  # 30 minutos
        """Limpia procesadores inactivos para liberar memoria"""
        current_time = time.time()
        to_remove = []
        
        for client_id, last_time in self.last_activity.items():
            if current_time - last_time > max_idle_seconds:
                to_remove.append(client_id)
        
        for client_id in to_remove:
            if client_id in self.processors:
                del self.processors[client_id]
            if client_id in self.last_activity:
                del self.last_activity[client_id]
            logger.info(f"Procesador eliminado por inactividad: {client_id}")

# Instancia global del gestor
processor_manager = ProcessorManager()

# ========== LIMPIEZA PERIÓDICA ==========
async def cleanup_task():
    """Tarea en segundo plano para limpiar procesadores inactivos"""
    while True:
        try:
            processor_manager.cleanup_inactive()
            await asyncio.sleep(300)  # Ejecutar cada 5 minutos
        except Exception as e:
            logger.error(f"Error en cleanup_task: {e}")
            await asyncio.sleep(60)

# ========== CONFIGURACIÓN DE FASTAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Iniciando servidor...")
    # Iniciar tarea de limpieza
    cleanup_task_handle = asyncio.create_task(cleanup_task())
    yield
    # Shutdown
    logger.info("🛑 Cerrando servidor...")
    cleanup_task_handle.cancel()

app = FastAPI(
    title="Traductor LSC API para React Native",
    version="4.0.0",
    description="API optimizada para aplicaciones móviles",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== ENDPOINTS PRINCIPALES ==========

@app.get("/")
async def root():
    """Endpoint de salud básico"""
    return {
        "message": "🎯 Traductor LSC API para React Native", 
        "status": "running",
        "version": "4.0.0",
        "active_processors": len(processor_manager.processors)
    }

@app.get("/health")
async def health():
    """Endpoint de salud detallado"""
    return {
        "status": "healthy", 
        "active_processors": len(processor_manager.processors),
        "uptime": time.time(),
        "memory_usage": f"{len(processor_manager.processors)} procesadores activos"
    }

@app.post("/process")
async def process_frame(request: FrameRequest, background_tasks: BackgroundTasks):
    """
    Endpoint principal para procesar frames desde React Native
    Optimizado para alta frecuencia de requests
    """
    try:
        start_time = time.time()
        
        # Obtener procesador
        processor = processor_manager.get_processor(request.client_id)
        
        # Validar frame
        if not request.frame:
            raise HTTPException(status_code=400, detail="Frame base64 requerido")
        
        # Decodificar frame de manera eficiente
        try:
            # Remover prefix si existe (data:image/jpeg;base64,)
            if "," in request.frame:
                frame_data = request.frame.split(",")[1]
            else:
                frame_data = request.frame
                
            frame_bytes = base64.b64decode(frame_data)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise HTTPException(status_code=400, detail="Frame inválido")
                
        except Exception as e:
            logger.error(f"Error decodificando frame: {e}")
            raise HTTPException(status_code=400, detail=f"Error decodificando frame: {str(e)}")
        
        # Procesar frame
        try:
            result = processor.process_frame(frame)
            status = processor.get_full_status()
            
            # Agregar limpieza en segundo plano cada cierto tiempo
            if len(processor_manager.processors) > 10:
                background_tasks.add_task(processor_manager.cleanup_inactive, 300)  # 5 min
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "success": True,
                "result": result,
                "status": status,
                "sentence": processor.get_sentence(),
                "timestamp": request.timestamp or time.time(),
                "processing_time_ms": round(processing_time, 2),
                "client_id": request.client_id
            }
            
        except Exception as e:
            logger.error(f"Error procesando frame para {request.client_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error procesando: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error general en /process: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/clear")
async def clear_sentence(request: ClearRequest):
    """Limpiar oración del procesador"""
    try:
        processor = processor_manager.get_processor(request.client_id)
        processor.clear_sentence()
        
        return {
            "success": True, 
            "message": "Oración limpiada",
            "sentence": [],
            "client_id": request.client_id
        }
        
    except Exception as e:
        logger.error(f"Error en /clear: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{client_id}")
async def get_status(client_id: str):
    """Obtener estado completo del procesador"""
    try:
        if client_id in processor_manager.processors:
            processor = processor_manager.processors[client_id]
            return {
                "success": True,
                "status": processor.get_full_status(),
                "sentence": processor.get_sentence(),
                "client_id": client_id
            }
        else:
            return {
                "success": False,
                "error": "Cliente no encontrado",
                "client_id": client_id
            }
            
    except Exception as e:
        logger.error(f"Error en /status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentence/{client_id}")
async def get_sentence(client_id: str):
    """Obtener solo la oración actual (endpoint ligero)"""
    try:
        if client_id in processor_manager.processors:
            processor = processor_manager.processors[client_id]
            return {
                "success": True,
                "sentence": processor.get_sentence(),
                "client_id": client_id
            }
        else:
            return {
                "success": True,
                "sentence": [],
                "client_id": client_id
            }
            
    except Exception as e:
        logger.error(f"Error en /sentence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
async def update_config(request: ConfigRequest):
    """Actualizar configuración del procesador (futuro uso)"""
    try:
        # Por ahora solo confirmar recepción
        # En futuras versiones se puede usar para ajustar thresholds
        return {
            "success": True,
            "message": "Configuración recibida",
            "config": request.dict(),
            "client_id": request.client_id
        }
        
    except Exception as e:
        logger.error(f"Error en /config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/processor/{client_id}")
async def delete_processor(client_id: str):
    """Eliminar procesador específico para liberar memoria"""
    try:
        if client_id in processor_manager.processors:
            del processor_manager.processors[client_id]
        if client_id in processor_manager.last_activity:
            del processor_manager.last_activity[client_id]
            
        return {
            "success": True,
            "message": f"Procesador {client_id} eliminado",
            "client_id": client_id
        }
        
    except Exception as e:
        logger.error(f"Error en /delete_processor: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== ENDPOINTS DE ADMINISTRACIÓN ==========

@app.get("/admin/processors")
async def list_processors():
    """Listar procesadores activos (para debugging)"""
    return {
        "active_processors": list(processor_manager.processors.keys()),
        "count": len(processor_manager.processors),
        "last_activity": {
            client_id: time.time() - last_time 
            for client_id, last_time in processor_manager.last_activity.items()
        }
    }

@app.post("/admin/cleanup")
async def force_cleanup():
    """Forzar limpieza de procesadores inactivos"""
    try:
        initial_count = len(processor_manager.processors)
        processor_manager.cleanup_inactive(300)  # 5 minutos
        final_count = len(processor_manager.processors)
        
        return {
            "success": True,
            "message": f"Limpieza completada. {initial_count - final_count} procesadores eliminados",
            "before": initial_count,
            "after": final_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== MANEJO DE ERRORES GLOBALES ==========

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "success": False,
        "error": "Endpoint no encontrado",
        "detail": "Revisa la documentación en /docs"
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Error interno del servidor: {exc}")
    return {
        "success": False,
        "error": "Error interno del servidor",
        "detail": "Contacta al administrador"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Puerto para desarrollo local o producción
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Iniciando servidor optimizado para React Native en puerto {port}")
    logger.info("📡 Endpoints principales:")
    logger.info("   POST /process - Procesar frame (PRINCIPAL)")
    logger.info("   POST /clear - Limpiar oración")
    logger.info("   GET /sentence/{client_id} - Obtener oración")
    logger.info("   GET /status/{client_id} - Estado completo")
    logger.info("   GET /docs - Documentación automática")
    
    # Configuración optimizada para React Native
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        # Configuración de timeouts optimizada para móviles
        timeout_keep_alive=60,
        timeout_graceful_shutdown=30,
        # Performance
        access_log=False,
        loop="asyncio",
        http="h11",  # Más estable que httptools en algunos casos
        # Límites ajustados para móviles
        limit_max_requests=50000,
        limit_concurrency=100
    )