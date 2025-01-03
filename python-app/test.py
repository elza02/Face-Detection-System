
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array
from mtcnn import MTCNN
import cv2
import json
import pandas as pd
import ast
import PIL

def get_mtcnn():
    return MTCNN()
def preprocess_image(img):
    """Preprocess an image and perform face detection using a lazily initialized MTCNN."""
    # np_arr = np.frombuffer(image_bytes, np.uint8)
    # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # if img is None:
    #     return None

    # Initialize MTCNN detector lazily
    detector = MTCNN()  # MTCNN is reinitialized on each executor
    results = detector.detect_faces(img)
    if not results:
        return None

    y, x, width, height = results[0]['box']
    img_t = img[y:y + height, x:x + width]
    img_t = cv2.resize(img_t, (224, 224))
    img_t = img_to_array(img_t)
    img_t = np.expand_dims(img_t, axis=0)
    img_t = preprocess_input(img_t)

    return np.ascontiguousarray(img_t)



def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")

    with open(model_path, 'rb') as f:
        tflite_model = f.read()

    return tflite_model



bc_model = load_tflite_model('/app/models/latent_model.tflite')
def predict_batch(images):
    interpreter = tf.lite.Interpreter(model_content=bc_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    for img in images:
        interpreter.set_tensor(input_details['index'], img)
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_details['index'])[0])
    print('predictions',predictions)
    return predictions

if __name__ == '__main__':
    image_path = "/app/dataset/face.png" 
    img = cv2.imread(image_path)
    img = preprocess_image(img)
    result = predict_batch(img)
    print(result)

