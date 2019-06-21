"""Flask web server serving text_recognizer predictions."""
# From https://github.com/UnitedIncome/serverless-python-requirements
try:
    import unzip_requirements  # pylint: disable=unused-import
except ImportError:
    pass

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.applications.mobilenet import preprocess_input

import numpy as np
import cv2
import pickle
from rdp import rdp

from tensorflow import keras
import sys
sys.modules['keras'] = keras

app = Flask(__name__) 
CORS(app) 


def top_3_predictions(predictions):
    return np.argsort(-predictions, axis=1)[:, :3]

SIZE = 64
BASE_SIZE = 256
N_LABELS = 340

sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)

model = MobileNet(input_shape=(SIZE, SIZE, 1), alpha=1., weights=None, classes=N_LABELS)
model.load_weights('model.h5')

with open ('labels', 'rb') as fp:
    labels = pickle.load(fp)

def draw(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def predict(x):
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        predictions = model.predict(x, verbose=0)
        top_3_preds = top_3_predictions(predictions)
        top_3_labels = [labels[prediction] for prediction in top_3_preds[0]]

    return jsonify(top_3_labels)


@app.route('/')
def index():
    return 'Hello World!'

@app.route('/v1/getLabels', methods=['GET'])
def getLabels():
    return jsonify(labels)

@app.route('/v1/predictRaw', methods=['POST'])
def predictRaw():
    raw_strokes = request.get_json()
    #app.logger.warning(raw_strokes)

    x = np.zeros((1, SIZE, SIZE, 1))
    x[0, :, :, 0] = draw(raw_strokes, size=64, lw=6, time_color=True)
    x = preprocess_input(x).astype(np.float32)

    return predict(x)


@app.route('/v1/predictSimplified', methods=['POST'])
def predictSimplified():

    raw_strokes = request.get_json()
    simplifed_points = [rdp(stroke, epsilon=2.0) for stroke in raw_strokes]
    simplified_strokes = [np.array(stroke).transpose() for stroke in simplifed_points]
    app.logger.warning(simplified_strokes)

    x = np.zeros((1, SIZE, SIZE, 1))
    x[0, :, :, 0] = draw(simplified_strokes, size=64, lw=6, time_color=True)
    x = preprocess_input(x).astype(np.float32)

    return predict(x)

def main():
    app.run(host='0.0.0.0', port=8000, debug=False) 

if __name__ == '__main__':
    main()
