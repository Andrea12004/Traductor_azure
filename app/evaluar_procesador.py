import os
import cv2
import json
import time
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from utils.helpers import *
from utils.constants import (
    MODEL_STATIC_PATH, 
    MODEL_PATH, 
    WORDS_JSON_PATH,
    MODEL_FRAMES_STATIC,
    # otras que necesites
)
# from text_to_speech import text_to_speech  # Lo comentamos para la API

# ---------- CONFIGURACIÓN  ----------
MOVEMENT_THRESHOLD = 0.015  # Umbral para detectar movimiento
STATIC_FRAMES_REQUIRED = 20  # Frames quietos para activar modo estático
PREDICTION_COOLDOWN = 30  # Frames de cooldown entre predicciones
DYNAMIC_MIN_FRAMES = max(5, MIN_LENGTH_FRAMES // 2)  # Mínimo de frames para dinámica

# ---------- FUNCIONES AUXILIARES  ----------
def load_static_labels(json_path):
    """Carga solo las etiquetas de palabra estática del JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("word_static", [])

def interpolate_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    
    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
            interpolated_keypoints.append(interpolated_point.tolist())
    
    return interpolated_keypoints

def normalize_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length < target_length:
        return interpolate_keypoints(keypoints, target_length)
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]
    else:
        return keypoints

def calculate_movement(kp_current, kp_previous):
    """Calcula el movimiento entre dos frames de keypoints"""
    if kp_previous is None:
        return 0.0
    
    try:
        curr = np.array(kp_current)
        prev = np.array(kp_previous)
        movement = np.mean(np.abs(curr - prev))
        return movement
    except:
        return 0.0

def extract_hand_keypoints(kp_frame):
    """Extrae keypoints de mano siguiendo la lógica del código estático"""
    pose_size = 132
    cara_size = 1404
    mano_size = 63
    
    # Slicing para cada mano
    mano_izquierda = kp_frame[pose_size + cara_size : pose_size + cara_size + mano_size]
    mano_derecha = kp_frame[-mano_size:]
    
    # Elegir mano con datos
    if np.max(mano_derecha) > 0:
        return np.array(mano_derecha, dtype="float32")
    elif np.max(mano_izquierda) > 0:
        return np.array(mano_izquierda, dtype="float32")
    else:
        return np.zeros(mano_size, dtype="float32")

# ---------- CLASE PRINCIPAL  ----------
class SignLanguageProcessor:
    def __init__(self):
        # Variables de instancia
        self.sentence = []
        self.kp_sequence = []  # Para acumular keypoints
        self.previous_kp = None
        self.static_counter = 0  # Contador de frames estáticos consecutivos
        self.cooldown_counter = 0
        self.hands_present = False
        self.hands_were_present = False
        
        # Cargar modelos
        print("Cargando modelos...")
        self.word_static = load_static_labels(LETTERS_JSON_PATH)
        self.word_ids = get_word_ids(WORDS_JSON_PATH)
        self.model_static = load_model(MODEL_STATIC_PATH)
        self.model_dynamic = load_model(MODEL_PATH)
        self.holistic_model = Holistic()
        print("Modelos cargados exitosamente")

    def process_frame(self, frame, static_threshold=0.8, dynamic_threshold=0.7):
        """Función principal para procesar cada frame"""
        
        # Tu lógica original de MediaPipe
        results = mediapipe_detection(frame, self.holistic_model)
        self.hands_present = there_hand(results)
        
        # Reducir cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        
        # Variable para retornar resultado
        result = {"message": "Frame procesado", "word": None, "confidence": None, "type": None}
        
        if self.hands_present:
            # ========= HAY MANOS EN CÁMARA  =========
            self.hands_were_present = True
            
            # Extraer keypoints
            kp_frame = extract_keypoints(results)
            self.kp_sequence.append(kp_frame)
            
            # Calcular movimiento para determinar si es estática
            movement = calculate_movement(kp_frame, self.previous_kp)
            self.previous_kp = kp_frame.copy()
            
            # Contar frames estáticos
            if movement < MOVEMENT_THRESHOLD:
                self.static_counter += 1
            else:
                self.static_counter = 0  # Resetear si hay movimiento
            
            # ====== PREDICCIÓN ESTÁTICA ======
            # Si llevamos suficientes frames quietos, predecir estática
            if self.static_counter >= STATIC_FRAMES_REQUIRED and self.cooldown_counter == 0:
                mano = extract_hand_keypoints(kp_frame)
                
                if np.max(mano) > 0:
                    # Normalizar
                    data = mano / np.max(mano)
                    
                    # Predicción estática
                    res = self.model_static.predict(np.expand_dims(data, axis=0), verbose=0)[0]
                    pred_idx = np.argmax(res)
                    confidence = res[pred_idx]
                    
                    print(f"[ESTÁTICA] {self.word_static[pred_idx]} ({confidence*100:.2f}%)")
                    
                    if confidence >= static_threshold:
                        sent = self.word_static[pred_idx]
                        self.sentence.insert(0, sent)
                        # text_to_speech(sent)  # Comentado para API
                        self.cooldown_counter = PREDICTION_COOLDOWN
                        
                        # RETORNAR RESULTADO PARA LA API
                        result = {
                            "word": sent,
                            "confidence": float(confidence),
                            "type": "static", 
                            "message": "Palabra estática detectada"
                        }
                        
                # Resetear contador después de predicción
                self.static_counter = 0
        
        else:
            # ========= NO HAY MANOS EN CÁMARA  =========
            self.static_counter = 0
            self.previous_kp = None
            
            # Si había manos antes y ahora no → EVALUAR DINÁMICA
            if self.hands_were_present and len(self.kp_sequence) >= DYNAMIC_MIN_FRAMES and self.cooldown_counter == 0:
                print(f"[MANOS SALIERON] Evaluando {len(self.kp_sequence)} frames como DINÁMICA")
                
                try:
                    # Normalizar secuencia
                    kp_normalized = normalize_keypoints(self.kp_sequence, int(MODEL_FRAMES))
                    
                    # Predicción dinámica
                    res = self.model_dynamic.predict(np.expand_dims(kp_normalized, axis=0), verbose=0)[0]
                    pred_idx = np.argmax(res)
                    confidence = res[pred_idx]
                    
                    # Obtener nombre de la palabra dinámica
                    if pred_idx < len(self.word_ids):
                        word_id = self.word_ids[pred_idx].split('-')[0]
                        word_name = words_text.get(word_id, word_id) if 'words_text' in globals() else word_id
                    else:
                        word_name = f"Clase_{pred_idx}"
                    
                    print(f"[DINÁMICA] {word_name} ({confidence*100:.2f}%)")
                    
                    if confidence > dynamic_threshold and pred_idx < len(self.word_ids):
                        word_id = self.word_ids[pred_idx].split('-')[0]
                        sent = words_text.get(word_id, word_id)
                        self.sentence.insert(0, sent)
                        # text_to_speech(sent)  # Comentado para API
                        self.cooldown_counter = PREDICTION_COOLDOWN
                        
                        # RETORNAR RESULTADO PARA LA API
                        result = {
                            "word": sent,
                            "confidence": float(confidence),
                            "type": "dynamic",
                            "message": "Palabra dinámica detectada"
                        }
                        
                except Exception as e:
                    print(f"Error en predicción dinámica: {e}")
            
            # Resetear variables para próxima secuencia
            self.hands_were_present = False
            self.kp_sequence = []
        
        return result

    def clear_sentence(self):
        """Limpiar la oración (para cuando React Native lo pida)"""
        self.sentence.clear()
        print("Oración limpiada")

    def get_sentence(self):
        """Obtener la oración actual (si React Native la quiere)"""
        return self.sentence.copy()
    
    def get_full_status(self):
        """Obtener estado completo del procesador"""
        return {
            "sentence": self.sentence.copy(),
            "hands_present": self.hands_present,
            "static_counter": self.static_counter,
            "cooldown_counter": self.cooldown_counter,
            "frames_accumulated": len(self.kp_sequence)
        }


# ---------- FUNCIONES GLOBALES DE COMPATIBILIDAD (OPCIONAL)  ----------
# Estas son para mantener compatibilidad con código que use las funciones globales

# Variables globales para las funciones de compatibilidad
sentence = []
kp_sequence = []  # Para acumular keypoints
previous_kp = None
static_counter = 0  # Contador de frames estáticos consecutivos
cooldown_counter = 0
hands_present = False
hands_were_present = False

# Cargar modelos globales (para compatibilidad)
print("Cargando modelos globales...")
word_static = load_static_labels(LETTERS_JSON_PATH)
word_ids = get_word_ids(WORDS_JSON_PATH)
model_static = load_model(MODEL_STATIC_PATH)
model_dynamic = load_model(MODEL_PATH)
holistic_model = Holistic()
print("Modelos globales cargados exitosamente")

def process_frame_simple(frame, static_threshold=0.8, dynamic_threshold=0.7):
    """Función global para compatibilidad con código existente"""
    global sentence, kp_sequence, previous_kp, static_counter, cooldown_counter
    global hands_present, hands_were_present
    
    # Tu lógica original de MediaPipe
    results = mediapipe_detection(frame, holistic_model)
    hands_present = there_hand(results)
    
    # Reducir cooldown
    if cooldown_counter > 0:
        cooldown_counter -= 1
    
    # Variable para retornar resultado
    result = {"message": "Frame procesado", "word": None, "confidence": None, "type": None}
    
    if hands_present:
        # ========= HAY MANOS EN CÁMARA  =========
        hands_were_present = True
        
        # Extraer keypoints
        kp_frame = extract_keypoints(results)
        kp_sequence.append(kp_frame)
        
        # Calcular movimiento para determinar si es estática
        movement = calculate_movement(kp_frame, previous_kp)
        previous_kp = kp_frame.copy()
        
        # Contar frames estáticos
        if movement < MOVEMENT_THRESHOLD:
            static_counter += 1
        else:
            static_counter = 0  # Resetear si hay movimiento
        
        # ====== PREDICCIÓN ESTÁTICA ======
        # Si llevamos suficientes frames quietos, predecir estática
        if static_counter >= STATIC_FRAMES_REQUIRED and cooldown_counter == 0:
            mano = extract_hand_keypoints(kp_frame)
            
            if np.max(mano) > 0:
                # Normalizar
                data = mano / np.max(mano)
                
                # Predicción estática
                res = model_static.predict(np.expand_dims(data, axis=0), verbose=0)[0]
                pred_idx = np.argmax(res)
                confidence = res[pred_idx]
                
                print(f"[ESTÁTICA] {word_static[pred_idx]} ({confidence*100:.2f}%)")
                
                if confidence >= static_threshold:
                    sent = word_static[pred_idx]
                    sentence.insert(0, sent)
                    # text_to_speech(sent)  # Comentado para API
                    cooldown_counter = PREDICTION_COOLDOWN
                    
                    # RETORNAR RESULTADO PARA LA API
                    result = {
                        "word": sent,
                        "confidence": float(confidence),
                        "type": "static", 
                        "message": "Palabra estática detectada"
                    }
                    
            # Resetear contador después de predicción
            static_counter = 0
    
    else:
        # ========= NO HAY MANOS EN CÁMARA  =========
        static_counter = 0
        previous_kp = None
        
        # Si había manos antes y ahora no → EVALUAR DINÁMICA
        if hands_were_present and len(kp_sequence) >= DYNAMIC_MIN_FRAMES and cooldown_counter == 0:
            print(f"[MANOS SALIERON] Evaluando {len(kp_sequence)} frames como DINÁMICA")
            
            try:
                # Normalizar secuencia
                kp_normalized = normalize_keypoints(kp_sequence, int(MODEL_FRAMES))
                
                # Predicción dinámica
                res = model_dynamic.predict(np.expand_dims(kp_normalized, axis=0), verbose=0)[0]
                pred_idx = np.argmax(res)
                confidence = res[pred_idx]
                
                # Obtener nombre de la palabra dinámica
                if pred_idx < len(word_ids):
                    word_id = word_ids[pred_idx].split('-')[0]
                    word_name = words_text.get(word_id, word_id) if 'words_text' in globals() else word_id
                else:
                    word_name = f"Clase_{pred_idx}"
                
                print(f"[DINÁMICA] {word_name} ({confidence*100:.2f}%)")
                
                if confidence > dynamic_threshold and pred_idx < len(word_ids):
                    word_id = word_ids[pred_idx].split('-')[0]
                    sent = words_text.get(word_id, word_id)
                    sentence.insert(0, sent)
                    # text_to_speech(sent)  # Comentado para API
                    cooldown_counter = PREDICTION_COOLDOWN
                    
                    # RETORNAR RESULTADO PARA LA API
                    result = {
                        "word": sent,
                        "confidence": float(confidence),
                        "type": "dynamic",
                        "message": "Palabra dinámica detectada"
                    }
                    
            except Exception as e:
                print(f"Error en predicción dinámica: {e}")
        
        # Resetear variables para próxima secuencia
        hands_were_present = False
        kp_sequence = []
    
    return result

def clear_sentence():
    """Limpiar la oración (para cuando React Native lo pida)"""
    global sentence
    sentence.clear()
    print("Oración limpiada")

def get_sentence():
    """Obtener la oración actual (si React Native la quiere)"""
    return sentence.copy()