import os
import cv2
import json
import time
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from helpers import *
from constants import *
from text_to_speech import text_to_speech

# ---------- CONFIGURACIÓN ----------
MOVEMENT_THRESHOLD = 0.015  # Umbral para detectar movimiento
STATIC_FRAMES_REQUIRED = 15  # Frames quietos para activar modo estático
PREDICTION_COOLDOWN = 30  # Frames de cooldown entre predicciones
DYNAMIC_MIN_FRAMES = max(5, MIN_LENGTH_FRAMES // 2)  # Mínimo de frames para dinámica

# ---------- FUNCIONES AUXILIARES ----------
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



def extract_hand_keypoints_static(kp_frame):
    mano_size = 63  # 21 puntos * 3 coordenadas

    mano_izquierda = kp_frame[-2*mano_size:-mano_size]
    mano_derecha  = kp_frame[-mano_size:]

    # Mano derecha → devolver tal cual
    if np.sum(mano_derecha) > 0:
        return np.array(mano_derecha, dtype="float32")

    # Mano izquierda → reflejo simple en X
    elif np.sum(mano_izquierda) > 0:
        mano = mano_izquierda.reshape((21, 3))
        mano[:, 0] *= -1  # espejo horizontal
        return mano.flatten().astype("float32")

    # Ninguna mano
    else:
        return np.zeros(mano_size, dtype="float32")


   

def extract_hand_keypoints_dynamic(kp_frame):
    """
    Dinámica: devuelve ambas manos tal cual.
    """
    pose_size=132
    cara_size=1404
    mano_size=63

    mano_izquierda = kp_frame[pose_size + cara_size : pose_size + cara_size + mano_size]
    mano_derecha   = kp_frame[pose_size + cara_size + mano_size : pose_size + cara_size + mano_size*2]

    return np.concatenate([mano_izquierda, mano_derecha])

# ---------- FUNCIÓN PRINCIPAL HÍBRIDA SIMPLIFICADA ----------
def evaluate_hybrid_simple(src=None, static_threshold=0.8, dynamic_threshold=0.7):
    
    # Cargar modelos y datos
    word_static = load_static_labels(LETTERS_JSON_PATH)
    word_ids = get_word_ids(WORDS_JSON_PATH)
    model_static = load_model(MODEL_STATIC_PATH)
    model_dynamic = load_model(MODEL_PATH)
    
    # Variables de estado
    sentence = []
    kp_sequence = []  # Para acumular keypoints
    previous_kp = None
    static_counter = 0  # Contador de frames estáticos consecutivos
    cooldown_counter = 0
    hands_present = False
    hands_were_present = False
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(src or 0)
     
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            results = mediapipe_detection(frame, holistic_model)
            hands_present = there_hand(results)
            
            # Reducir cooldown
            if cooldown_counter > 0:
                cooldown_counter -= 1
            
            if hands_present:
                # ========= HAY MANOS EN CÁMARA =========
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
                if static_counter >= STATIC_FRAMES_REQUIRED and cooldown_counter == 0:
                    mano = extract_hand_keypoints_static(kp_frame)
                    
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
                            text_to_speech(sent)
                            cooldown_counter = PREDICTION_COOLDOWN
                            
                    # Resetear contador después de predicción
                    static_counter = 0
            
            else:
                # ========= NO HAY MANOS EN CÁMARA =========
                static_counter = 0
                previous_kp = None
                
                if hands_were_present and len(kp_sequence) >= DYNAMIC_MIN_FRAMES and cooldown_counter == 0:
                    print(f"[MANOS SALIERON] Evaluando {len(kp_sequence)} frames como DINÁMICA")
                    
                    try:
                        kp_sequence_hands = [extract_hand_keypoints_dynamic(kp) for kp in kp_sequence]
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
                            text_to_speech(sent)
                            cooldown_counter = PREDICTION_COOLDOWN
                            
                    except Exception as e:
                        print(f"Error en predicción dinámica: {e}")
                
                # Resetear variables para próxima secuencia
                hands_were_present = False
                kp_sequence = []
            
            # ============ INTERFAZ DE USUARIO ============
            if not src:
                if hands_present:
                    if static_counter >= STATIC_FRAMES_REQUIRED // 2:
                        bg_color = (100, 255, 100)  # Verde = Estática activa
                        mode_text = "ESTATICA ACTIVA"
                    else:
                        bg_color = (255, 180, 100)  # Naranja = Acumulando
                        mode_text = "ACUMULANDO"
                else:
                    if hands_were_present:
                        bg_color = (255, 100, 100)  # Rojo = Evaluando dinámica
                        mode_text = "EVALUANDO DINAMICA"
                    else:
                        bg_color = (128, 128, 128)  # Gris = Esperando
                        mode_text = "ESPERANDO"
                
                # Fondo
                cv2.rectangle(frame, (0, 0), (640, 90), bg_color, -1)
                
                # Texto de la oración
                sentence_text = ' | '.join(sentence[-4:]) if sentence else "Esperando señas..."
                cv2.putText(frame, sentence_text, (10, 25), FONT, 0.7, (255, 255, 255), 2)
                
                # Estado y modo
                cv2.putText(frame, mode_text, (10, 50), FONT, 0.6, (255, 255, 255), 2)
                
                # Información detallada
                if hands_present:
                    detail = f"Frames: {len(kp_sequence)} | Estaticos: {static_counter}/{STATIC_FRAMES_REQUIRED}"
                else:
                    detail = f"Secuencia anterior: {len(kp_sequence)} frames"
                
                cv2.putText(frame, detail, (10, 70), FONT, 0.4, (200, 200, 200), 1)
                
                # Cooldown
                if cooldown_counter > 0:
                    cv2.putText(frame, f"Cooldown: {cooldown_counter}", (450, 50), FONT, 0.5, (255, 255, 0), 1)
                
                draw_keypoints(frame, results)
                cv2.imshow('Traductor LSC - Ambas Manos', frame)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    sentence.clear()
                    print("Oración limpiada")
        
        video.release()
        cv2.destroyAllWindows()
        return sentence

# ---------- MAIN ----------
if __name__ == "__main__":
    try:
        
        result = evaluate_hybrid_simple()
        
        print("\n=== TRADUCCIÓN FINALIZADA ===")
        if result:
            print("Oración final:", ' | '.join(result))
        else:
            print("No se detectaron señas")
            
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()