# app/evaluar_procesador.py - Versión optimizada para React Native
import os
import cv2
import json
import time
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from tensorflow.keras.models import load_model   # ✅ cambiado a tensorflow.keras
import logging
from utils.helpers import *
from utils.constants import *

logger = logging.getLogger(__name__)

# ========== CONFIGURACIÓN OPTIMIZADA PARA MÓVIL ==========
class MobileConfig:
    MOVEMENT_THRESHOLD = 0.012
    STATIC_FRAMES_REQUIRED = 15
    PREDICTION_COOLDOWN = 20
    DYNAMIC_MIN_FRAMES = max(5, MIN_LENGTH_FRAMES // 3)

    MAX_SEQUENCE_LENGTH = 60
    CLEANUP_THRESHOLD = 100

    SKIP_FRAMES = 2

    STATIC_CONFIDENCE = 0.75
    DYNAMIC_CONFIDENCE = 0.65

# ========== FUNCIONES AUXILIARES OPTIMIZADAS ==========
def load_static_labels(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("word_static", [])
    except Exception as e:
        logger.error(f"Error cargando etiquetas estáticas: {e}")
        return ["A", "B", "C", "D", "E"]

def interpolate_keypoints_fast(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    if current_length < 2:
        return keypoints + [keypoints[-1]] * (target_length - current_length)

    indices = np.linspace(0, current_length - 1, target_length)
    keypoints_array = np.array(keypoints)

    interpolated = []
    for i in indices:
        lower_idx = int(i)
        upper_idx = min(lower_idx + 1, current_length - 1)
        weight = i - lower_idx
        if weight == 0 or lower_idx == upper_idx:
            interpolated.append(keypoints[lower_idx])
        else:
            interpolated_point = ((1 - weight) * keypoints_array[lower_idx] +
                                  weight * keypoints_array[upper_idx]).tolist()
            interpolated.append(interpolated_point)
    return interpolated

def normalize_keypoints_fast(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length < target_length:
        return interpolate_keypoints_fast(keypoints, target_length)
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]
    else:
        return keypoints

def calculate_movement_fast(kp_current, kp_previous):
    if kp_previous is None:
        return 1.0
    try:
        pose_size = 132
        cara_size = 1404
        hand_curr = np.array(kp_current[pose_size + cara_size:])
        hand_prev = np.array(kp_previous[pose_size + cara_size:])
        movement = np.sum(np.abs(hand_curr - hand_prev)) / len(hand_curr)
        return movement
    except Exception as e:
        logger.warning(f"Error calculando movimiento: {e}")
        return 1.0

def extract_hand_keypoints_fast(kp_frame):
    pose_size = 132
    cara_size = 1404
    mano_size = 63
    try:
        mano_derecha = kp_frame[-mano_size:]
        mano_izquierda = kp_frame[pose_size + cara_size: pose_size + cara_size + mano_size]
        if np.sum(mano_derecha) > np.sum(mano_izquierda):
            return np.array(mano_derecha, dtype=np.float32)
        elif np.sum(mano_izquierda) > 0:
            return np.array(mano_izquierda, dtype=np.float32)
        else:
            return np.zeros(mano_size, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Error extrayendo keypoints de mano: {e}")
        return np.zeros(mano_size, dtype=np.float32)

# ========== CLASE PRINCIPAL OPTIMIZADA ==========
class SignLanguageProcessor:
    def __init__(self):
        self.config = MobileConfig()
        self.sentence = []
        self.kp_sequence = []
        self.previous_kp = None
        self.static_counter = 0
        self.cooldown_counter = 0
        self.hands_present = False
        self.hands_were_present = False
        self.frame_skip_counter = 0
        self.prediction_count = 0
        self.last_prediction_time = 0
        self.processing_times = []
        self._load_models()

    def _load_models(self):
        try:
            logger.info("Cargando modelos para procesador móvil...")
            self.word_static = load_static_labels(LETTERS_JSON_PATH)
            self.word_ids = get_word_ids(WORDS_JSON_PATH)
            self.model_static = load_model(MODEL_STATIC_PATH)
            self.model_dynamic = load_model(MODEL_PATH)
            self.holistic_model = Holistic(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            logger.info("Modelos cargados exitosamente para móvil")
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            self.word_static = ["A", "B", "C", "D", "E"]
            self.word_ids = ["hola", "gracias"]
            self.model_static = None
            self.model_dynamic = None
            self.holistic_model = None

    def process_frame(self, frame, static_threshold=None, dynamic_threshold=None):
        start_time = time.time()
        static_threshold = static_threshold or self.config.STATIC_CONFIDENCE
        dynamic_threshold = dynamic_threshold or self.config.DYNAMIC_CONFIDENCE
        self.frame_skip_counter += 1
        if self.frame_skip_counter % (self.config.SKIP_FRAMES + 1) != 0:
            return {"message": "Frame skipped for performance", "skipped": True}
        if self.holistic_model is None:
            return {"error": "Modelos no disponibles", "word": None}

        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        try:
            results = mediapipe_detection(frame, self.holistic_model)
            self.hands_present = there_hand(results)
        except Exception as e:
            logger.error(f"Error en detección MediaPipe: {e}")
            return {"error": f"Error en detección: {str(e)}"}

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        result = {"message": "Frame procesado", "word": None, "confidence": None, "type": None}
        try:
            if self.hands_present:
                result.update(self._process_hands_present(results, frame, static_threshold))
            else:
                result.update(self._process_hands_absent(dynamic_threshold))
        except Exception as e:
            logger.error(f"Error en procesamiento de frame: {e}")
            result["error"] = str(e)

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        result["processing_time"] = processing_time
        self._periodic_cleanup()
        return result

    # 👇 Aquí sigo corrigiendo para cerrar bien diccionarios y try/except
    def _process_hands_present(self, results, frame, static_threshold):
        self.hands_were_present = True
        result = {}
        try:
            kp_frame = extract_keypoints(results)
            if len(self.kp_sequence) >= self.config.MAX_SEQUENCE_LENGTH:
                self.kp_sequence.pop(0)
            self.kp_sequence.append(kp_frame)
            movement = calculate_movement_fast(kp_frame, self.previous_kp)
            self.previous_kp = kp_frame.copy()
            if movement < self.config.MOVEMENT_THRESHOLD:
                self.static_counter += 1
            else:
                self.static_counter = 0
            if (self.static_counter >= self.config.STATIC_FRAMES_REQUIRED and
                self.cooldown_counter == 0 and self.model_static is not None):
                result.update(self._predict_static(kp_frame, static_threshold))
        except Exception as e:
            logger.error(f"Error procesando manos presentes: {e}")
            result["error"] = str(e)
        return result

    def _process_hands_absent(self, dynamic_threshold):
        result = {}
        self.static_counter = 0
        self.previous_kp = None
        if (self.hands_were_present and
            len(self.kp_sequence) >= self.config.DYNAMIC_MIN_FRAMES and
            self.cooldown_counter == 0 and self.model_dynamic is not None):
            result.update(self._predict_dynamic(dynamic_threshold))
        self.hands_were_present = False
        self.kp_sequence = []
        return result

    def _predict_static(self, kp_frame, threshold):
        try:
            mano = extract_hand_keypoints_fast(kp_frame)
            if np.sum(mano) > 0:
                max_val = np.max(mano)
                if max_val > 0:
                    data = mano / max_val
                else:
                    return {"message": "Sin datos válidos de mano"}
                res = self.model_static.predict(np.expand_dims(data, axis=0), verbose=0)[0]
                pred_idx = np.argmax(res)
                confidence = float(res[pred_idx])
                logger.info(f"[ESTÁTICA] {self.word_static[pred_idx]} ({confidence*100:.1f}%)")
                if confidence >= threshold:
                    sent = self.word_static[pred_idx]
                    self.sentence.insert(0, sent)
                    self.cooldown_counter = self.config.PREDICTION_COOLDOWN
                    self.prediction_count += 1
                    self.static_counter = 0
                    return {
                        "word": sent,
                        "confidence": confidence,
                        "type": "static",
                        "message": "Signo estático detectado"
                    }
            self.static_counter = 0
            return {"message": "Evaluando signo estático..."}
        except Exception as e:
            logger.error(f"Error en predicción estática: {e}")
            return {"error": f"Error predicción estática: {str(e)}"}

    def _predict_dynamic(self, threshold):
        try:
            logger.info(f"[DINÁMICO] Evaluando {len(self.kp_sequence)} frames")
            kp_normalized = normalize_keypoints_fast(self.kp_sequence, int(MODEL_FRAMES))
            res = self.model_dynamic.predict(np.expand_dims(kp_normalized, axis=0), verbose=0)[0]
            pred_idx = np.argmax(res)
            confidence = float(res[pred_idx])
            if pred_idx < len(self.word_ids):
                word_id = self.word_ids[pred_idx].split('-')[0]
                word_name = words_text.get(word_id, word_id)
                logger.info(f"[DINÁMICO] {word_name} ({confidence*100:.1f}%)")
                if confidence > threshold:
                    sent = words_text.get(word_id, word_id)
                    self.sentence.insert(0, sent)
                    self.cooldown_counter = self.config.PREDICTION_COOLDOWN
                    self.prediction_count += 1
                    return {
                        "word": sent,
                        "confidence": confidence,
                        "type": "dynamic",
                        "message": "Signo dinámico detectado"
                    }
            return {"message": "Evaluando signo dinámico..."}   # ✅ Cerrado correctamente
        except Exception as e:
            logger.error(f"Error en predicción dinámica: {e}")
            return {"error": f"Error predicción dinámica: {str(e)}"}

    def _periodic_cleanup(self):
        if self.prediction_count >= self.config.CLEANUP_THRESHOLD:
            self.sentence = self.sentence[:10]
            self.prediction_count = 0
