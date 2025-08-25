# app/main.py
import os, cv2, base64, json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model

from utils.helpers import mediapipe_detection, there_hand, extract_keypoints, get_word_ids
from utils.constants import (
    MODEL_FRAMES, MODEL_PATH, MODEL_STATIC_PATH,
    WORDS_JSON_PATH, LETTERS_JSON_PATH
)

# ------- util mano para tu vector 258 -------
def extract_hand_keypoints_from_258(kp_frame):
    pose_size = 33*4
    hand_size = 21*3
    lh = kp_frame[pose_size : pose_size + hand_size]
    rh = kp_frame[pose_size + hand_size : pose_size + 2*hand_size]
    if np.max(rh) > 0: return np.array(rh, dtype="float32")
    if np.max(lh) > 0: return np.array(lh, dtype="float32")
    return np.zeros(hand_size, dtype="float32")

# ------- parámetros de tu lógica -------
MOVEMENT_THRESHOLD = 0.015
STATIC_FRAMES_REQUIRED = 20
PREDICTION_COOLDOWN = 30
DYNAMIC_MIN_FRAMES = max(5, 10 // 2)  # o usa tu MIN_LENGTH_FRAMES si lo importas

def load_static_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Asegúrate que words.json tenga la clave "word_static"
    return data.get("word_static", [])

def normalize_keypoints(seq, target_length):
    cur = len(seq)
    if cur == target_length: return seq
    if cur < target_length:
        idx = np.linspace(0, cur-1, target_length)
        out = []
        for i in idx:
            li, ui = int(np.floor(i)), int(np.ceil(i))
            w = i - li
            if li == ui: out.append(seq[li])
            else: out.append(((1-w)*np.array(seq[li]) + w*np.array(seq[ui])).tolist())
        return out
    step = cur / target_length
    inds = np.arange(0, cur, step).astype(int)[:target_length]
    return [seq[i] for i in inds]

class SessionProcessor:
    def __init__(self):
        self.word_static = load_static_labels(LETTERS_JSON_PATH)
        self.word_ids = get_word_ids(WORDS_JSON_PATH) or []
        self.model_static = load_model(MODEL_STATIC_PATH)
        self.model_dynamic = load_model(MODEL_PATH)

        self.kp_sequence = []
        self.previous_kp = None
        self.static_counter = 0
        self.cooldown_counter = 0
        self.hands_present = False
        self.hands_were_present = False
        self.sentence = []

        self.holistic = Holistic()

    def release(self):
        self.holistic.close()

    def process_frame(self, frame_bgr):
        results = mediapipe_detection(frame_bgr, self.holistic)
        self.hands_present = there_hand(results)

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        events = []

        if self.hands_present:
            self.hands_were_present = True
            kp_frame = extract_keypoints(results)
            self.kp_sequence.append(kp_frame)

            movement = float(np.mean(np.abs(np.array(kp_frame) - np.array(self.previous_kp)))) if self.previous_kp is not None else 0.0
            self.previous_kp = kp_frame.copy()

            self.static_counter = self.static_counter + 1 if movement < MOVEMENT_THRESHOLD else 0

            if self.static_counter >= STATIC_FRAMES_REQUIRED and self.cooldown_counter == 0:
                mano = extract_hand_keypoints_from_258(kp_frame)
                if np.max(mano) > 0:
                    data = mano / np.max(mano)
                    res = self.model_static.predict(np.expand_dims(data, 0), verbose=0)[0]
                    idx = int(np.argmax(res)); conf = float(res[idx])
                    label = self.word_static[idx] if idx < len(self.word_static) else f"IDX_{idx}"
                    events.append({"type":"static_pred", "label":label, "confidence":conf})
                    if conf >= 0.8:
                        self.sentence.insert(0, label)
                        events.append({"type":"accepted", "label":label, "mode":"static"})
                        self.cooldown_counter = PREDICTION_COOLDOWN
                self.static_counter = 0

        else:
            self.static_counter = 0
            self.previous_kp = None
            if self.hands_were_present and len(self.kp_sequence) >= DYNAMIC_MIN_FRAMES and self.cooldown_counter == 0:
                try:
                    kp_norm = normalize_keypoints(self.kp_sequence, int(MODEL_FRAMES))
                    res = self.model_dynamic.predict(np.expand_dims(kp_norm, 0), verbose=0)[0]
                    idx = int(np.argmax(res)); conf = float(res[idx])
                    label = (self.word_ids[idx].split('-')[0] if idx < len(self.word_ids) else f"IDX_{idx}")
                    events.append({"type":"dynamic_pred", "label":label, "confidence":conf})
                    if conf >= 0.7 and idx < len(self.word_ids):
                        self.sentence.insert(0, label)
                        events.append({"type":"accepted", "label":label, "mode":"dynamic"})
                        self.cooldown_counter = PREDICTION_COOLDOWN
                except Exception as e:
                    events.append({"type":"error", "message":str(e)})
            self.hands_were_present = False
            self.kp_sequence = []

        events.append({
            "type":"state",
            "hands_present": bool(self.hands_present),
            "static_counter": int(self.static_counter),
            "cooldown": int(self.cooldown_counter),
            "last_sentence": self.sentence[:4],
        })
        return events

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = SessionProcessor()
    try:
        while True:
            msg = await ws.receive_text()
            m = json.loads(msg)
            if m.get("type") == "frame":
                b = base64.b64decode(m["data"])
                frame = np.frombuffer(b, dtype=np.uint8)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                frame = cv2.resize(frame, (640, 480))
                events = session.process_frame(frame)
                await ws.send_text(json.dumps({"events": events}))
    except WebSocketDisconnect:
        pass
    finally:
        session.release()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=10000)
