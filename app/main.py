import os
import asyncio
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import signal
import sys

# Tu lógica de procesamiento
from app.evaluar_procesador import SignLanguageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Traductor LSC WebSocket API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== CONFIGURACIÓN ESPECÍFICA PARA RENDER ==========

# Variables globales para manejo de conexiones
active_processors = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}
        self.processors: dict = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        try:
            await websocket.accept()
            self.active_connections[client_id] = websocket
            
            # Usar procesador global para evitar recrear modelos
            if client_id not in active_processors:
                active_processors[client_id] = SignLanguageProcessor()
            self.processors[client_id] = active_processors[client_id]
            
            logger.info(f"Cliente {client_id} conectado")
            
            # Enviar mensaje de bienvenida
            await self.send_personal_message({
                "type": "connected",
                "message": "Conexión establecida correctamente",
                "timestamp": asyncio.get_event_loop().time()
            }, client_id)
            
        except Exception as e:
            logger.error(f"Error conectando cliente {client_id}: {e}")
            raise

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            try:
                del self.active_connections[client_id]
            except:
                pass
        if client_id in self.processors:
            try:
                del self.processors[client_id]
            except:
                pass
        # No eliminar de active_processors para reusar
        logger.info(f"Cliente {client_id} desconectado")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
                return True
            except Exception as e:
                logger.error(f"Error enviando mensaje a {client_id}: {e}")
                self.disconnect(client_id)
                return False
        return False

manager = ConnectionManager()

# ========== ENDPOINTS HTTP (RENDER FUNCIONA MEJOR CON HTTP) ==========

@app.get("/")
async def root():
    return {
        "message": "🎯 Traductor LSC API funcionando", 
        "status": "running",
        "version": "3.0.0",
        "connections": len(manager.active_connections),
        "processors": len(active_processors)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "connections": len(manager.active_connections),
        "processors": len(active_processors)
    }

# ========== ENDPOINT HTTP PRINCIPAL (MÁS CONFIABLE QUE WEBSOCKET EN RENDER) ==========
@app.post("/process")
async def process_frame_http(request_data: dict):
    """Endpoint HTTP principal para procesar frames - MÁS CONFIABLE EN RENDER"""
    try:
        client_id = request_data.get("client_id", "mobile_client")
        frame_data = request_data.get("frame")
        
        if not frame_data:
            return {"success": False, "error": "Se requiere 'frame' en base64"}
        
        # Crear o reusar procesador
        if client_id not in active_processors:
            active_processors[client_id] = SignLanguageProcessor()
        
        processor = active_processors[client_id]
        
        # Decodificar frame
        try:
            frame_bytes = base64.b64decode(frame_data)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"success": False, "error": f"Error decodificando frame: {str(e)}"}
        
        if frame is None:
            return {"success": False, "error": "Frame inválido"}
        
        # Procesar frame
        try:
            result = processor.process_frame(frame)
            status = processor.get_full_status()
            
            return {
                "success": True,
                "result": result,
                "status": status,
                "sentence": processor.get_sentence(),
                "timestamp": request_data.get("timestamp", asyncio.get_event_loop().time())
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error procesando: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Error general en /process: {e}")
        return {"success": False, "error": f"Error general: {str(e)}"}

@app.post("/clear")
async def clear_sentence_http(request_data: dict):
    """Limpiar oración vía HTTP"""
    try:
        client_id = request_data.get("client_id", "mobile_client")
        
        if client_id in active_processors:
            active_processors[client_id].clear_sentence()
            
        return {
            "success": True, 
            "message": "Oración limpiada",
            "sentence": []
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/status/{client_id}")
async def get_status_http(client_id: str):
    """Obtener estado vía HTTP"""
    try:
        if client_id in active_processors:
            processor = active_processors[client_id]
            return {
                "success": True,
                "status": processor.get_full_status(),
                "sentence": processor.get_sentence()
            }
        else:
            return {
                "success": False,
                "error": "Cliente no encontrado"
            }
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# ========== WEBSOCKET (COMO RESPALDO) ==========
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        # Configurar timeout más corto para Render
        while True:
            try:
                # Timeout de 25 segundos (menos que el límite de Render)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=25.0)
                message = json.loads(data)
                
                message_type = message.get("type")
                
                if message_type == "ping":
                    # Responder inmediatamente
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": message.get("timestamp", asyncio.get_event_loop().time())
                    }, client_id)
                
                elif message_type == "frame":
                    frame_data = message.get("data")
                    if frame_data and client_id in manager.processors:
                        try:
                            # Decodificar y procesar
                            frame_bytes = base64.b64decode(frame_data)
                            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                processor = manager.processors[client_id]
                                result = processor.process_frame(frame)
                                
                                await manager.send_personal_message({
                                    "type": "prediction",
                                    "data": result,
                                    "timestamp": message.get("timestamp", asyncio.get_event_loop().time())
                                }, client_id)
                            
                        except Exception as e:
                            await manager.send_personal_message({
                                "type": "error",
                                "message": f"Error procesando frame: {str(e)}"
                            }, client_id)
                
                elif message_type == "clear":
                    if client_id in manager.processors:
                        manager.processors[client_id].clear_sentence()
                        await manager.send_personal_message({
                            "type": "cleared",
                            "message": "Oración limpiada"
                        }, client_id)
                
                elif message_type == "status":
                    if client_id in manager.processors:
                        status = manager.processors[client_id].get_full_status()
                        await manager.send_personal_message({
                            "type": "status",
                            "data": status
                        }, client_id)
                        
            except asyncio.TimeoutError:
                # Enviar ping cada timeout
                success = await manager.send_personal_message({
                    "type": "ping",
                    "message": "keep_alive",
                    "timestamp": asyncio.get_event_loop().time()
                }, client_id)
                
                if not success:
                    break
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Error en WebSocket {client_id}: {e}")
        manager.disconnect(client_id)

# ========== CONFIGURACIÓN DE SHUTDOWN PARA RENDER ==========
def signal_handler(sig, frame):
    logger.info("Cerrando servidor...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    import uvicorn
    
    # Puerto específico para Render
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Iniciando servidor en puerto {port}")
    logger.info("📡 Endpoints disponibles:")
    logger.info("   POST /process - Procesar frame (HTTP)")
    logger.info("   POST /clear - Limpiar oración")
    logger.info("   GET /status/{client_id} - Estado del procesador")
    logger.info("   WS /ws/{client_id} - WebSocket (respaldo)")
    
    # Configuración optimizada para Render
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        # Configuración específica para Render
        timeout_keep_alive=30,     # 30 segundos (menor que límite de Render)
        timeout_graceful_shutdown=30,
        # Configuración WebSocket optimizada
        ws_ping_interval=20,       # Ping cada 20 segundos
        ws_ping_timeout=10,        # Timeout de pong: 10 segundos
        access_log=False,          # Desactivar para mejor performance
        # Configuración adicional para estabilidad
        loop="asyncio",
        http="httptools",
        limit_max_requests=10000,
        backlog=2048
    )