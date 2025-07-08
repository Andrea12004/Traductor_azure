import os
import cv2

# CONFIGURACI�N GENERAL
MIN_LENGTH_FRAMES = 10  # M�nima cantidad de frames necesarios
MODEL_FRAMES = 40  # Cantidad de frames normalizados para el modelo
LENGTH_KEYPOINTS = 258 # Cantidad de keypoints usados en el modelo

# RUTAS PRINCIPALES
ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")

# RUTAS ESPEC�FICAS
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words.json")

# PAR�METROS PARA VISUALIZACI�N
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)


words_text = {
    "gracias": "GRACIAS",
    "hola":"HOLA",
    "como_estas":"¿COMO ESTAS?",
    "Buenos_dias":"BUENOS DIAS",
    "Buenas_tardes":"BUENAS TARDES",
    "Buenas_noches":"BUENAS NOCHES",
}
