import numpy as np
from keras.models import load_model
from app.utils.constants import *
from app.utils.helpers import *

def evaluar_keypoints(keypoints, threshold=0.8):
    model = load_model(MODEL_PATH)
    word_ids = get_word_ids(WORDS_JSON_PATH)
    
    res = model.predict(np.expand_dims(keypoints, axis=0))[0]

    if res[np.argmax(res)] > threshold:
        word_id = word_ids[np.argmax(res)].split('-')[0]
        return words_text.get(word_id)

    return None

