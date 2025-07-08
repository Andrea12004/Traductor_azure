from fastapi import FastAPI, UploadFile, File
import shutil
import os
from app.utils.evaluar import evaluar_video  # Tu funci�n de evaluaci�n
from uuid import uuid4

app = FastAPI()

@app.get("/")
def read_root():
    return {"mensaje": "API Traductor LSC funcionando"}

@app.post("/traducir-video")
async def traducir_video(file: UploadFile = File(...)):
    temp_dir = "app/temp"
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"{uuid4().hex}_{file.filename}"
    file_path = os.path.join(temp_dir, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    resultado = evaluar_video(file_path)
    return {"traduccion": resultado}
