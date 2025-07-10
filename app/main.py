from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from keras.models import load_model
from app.utils.constants import MODEL_PATH, WORDS_JSON_PATH, MODEL_FRAMES
from app.utils.helpers import get_word_ids, words_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Cargar el modelo una sola vez
model = load_model(MODEL_PATH)
word_ids = get_word_ids(WORDS_JSON_PATH)

class KeypointsInput(BaseModel):
    keypoints: list

@app.get("/")
def root():
    return {"message": "API de predicción funcionando"}

@app.post("/predecir/")
def predecir_keypoints(data: KeypointsInput):
    try:
        if not data.keypoints:
            raise HTTPException(status_code=400, detail="Lista de keypoints vacía")

        # Asegurarse que la entrada tenga longitud adecuada
        keypoints = data.keypoints
        if len(keypoints) < MODEL_FRAMES:
            # Rellenar con ceros si tiene menos de los frames requeridos
            padding = [ [0]*len(keypoints[0]) ] * (MODEL_FRAMES - len(keypoints))
            keypoints.extend(padding)
        else:
            keypoints = keypoints[:MODEL_FRAMES]

        keypoints_np = np.array([keypoints])
        result = model.predict(keypoints_np)[0]
        max_index = np.argmax(result)
        palabra = words_text.get(word_ids[max_index].split('-')[0])

        return {
            "prediccion": palabra,
            "confianza": float(result[max_index])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
