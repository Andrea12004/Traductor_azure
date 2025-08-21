import os
import asyncio
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging

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

# Almacenar conexiones activas
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}
        self.processors: dict = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.processors[client_id] = SignLanguageProcessor()
        logger.info(f"Cliente {client_id} conectado")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.processors:
            del self.processors[client_id]
        logger.info(f"Cliente {client_id} desconectado")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error enviando mensaje a {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Traductor LSC WebSocket API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "connections": len(manager.active_connections)}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Recibir mensaje del cliente
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "frame":
                # Procesar frame
                frame_data = message.get("data")
                if frame_data:
                    try:
                        # Decodificar frame
                        frame_bytes = base64.b64decode(frame_data)
                        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Procesar con tu lógica
                            processor = manager.processors[client_id]
                            result = processor.process_frame(frame)
                            
                            # Enviar resultado
                            await manager.send_personal_message({
                                "type": "prediction",
                                "data": result,
                                "timestamp": message.get("timestamp")
                            }, client_id)
                        
                    except Exception as e:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": f"Error procesando frame: {str(e)}"
                        }, client_id)
            
            elif message_type == "clear":
                # Limpiar oración
                if client_id in manager.processors:
                    manager.processors[client_id].clear_sentence()
                    await manager.send_personal_message({
                        "type": "cleared",
                        "message": "Oración limpiada"
                    }, client_id)
            
            elif message_type == "status":
                # Obtener estado completo
                if client_id in manager.processors:
                    status = manager.processors[client_id].get_full_status()
                    await manager.send_personal_message({
                        "type": "status",
                        "data": status
                    }, client_id)
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Error en WebSocket {client_id}: {e}")
        manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)