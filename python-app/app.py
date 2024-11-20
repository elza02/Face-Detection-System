import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array
from mtcnn import MTCNN
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, BinaryType, StringType
import cv2
import json

# Initialize Spark session
spark = SparkSession.builder \
    .appName("KafkaSparkIntegration") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.cores", "2") \
    .getOrCreate()

# Path to the TensorFlow Lite model file
tflite_model_path = '/app/models/latent_model.tflite'

# Predefined image hash dictionary for recognition
image_hash = {
    'Ayoub': {
        'hash': np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0,0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1,1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]),
        'fname': 'Ayoub Boulmeghras',
        'Age': '21',
        'Adresse': 'Agadir, Morocoo',
        'Job': 'Student'
    },
    'Akram': {
        'hash': np.array([0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,1,0,1,0,1,1,0,1,0,0]),
        'fname': 'Akram EL MOUDEN',
        'Age': '22',
        'Adresse': 'Tilila, AGADIR',
        'Job': 'Student'
    },
    'Josef': {
        'hash': np.array([1,1,1,1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,1,0,1,0,1,1,1,0,1,1,1,1,0,0,1,1,0]),
        'fname': 'Josef ABOKAYO',
        'Age': '29',
        'Adresse': 'Texas, USA',
        'Job': 'ML Engineer'
    },
    'David': {
        'hash': np.array([1,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,1,1,1,0,0,1,0,0,0,1,1,1,0,0,0,1,0,1,0,0]),
        'fname': 'David CHARELETON',
        'Age': '36',
        'Adresse': 'NYC, USA',
        'Job': 'Teacher'
    }
    
}

# Load and broadcast TensorFlow Lite model
def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")

    with open(model_path, 'rb') as f:
        tflite_model = f.read()

    return tflite_model

bc_model = spark.sparkContext.broadcast(load_tflite_model(tflite_model_path))

# Load and broadcast MTCNN detector
def get_mtcnn():
    return MTCNN()

# bc_mtcnn = spark.sparkContext.broadcast(get_mtcnn())

# Preprocess image for prediction
def preprocess_image(image_bytes):
    """Preprocess an image and perform face detection using a lazily initialized MTCNN."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

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

# Perform batch predictions with TensorFlow Lite
def predict_batch(images):
    interpreter = tf.lite.Interpreter(model_content=bc_model.value)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    for img in images:
        interpreter.set_tensor(input_details['index'], img)
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_details['index'])[0])

    return predictions

# Find nearest neighbor from predefined hashes
def find_nearest_neighbor(prediction):
    min_distance = float('inf')
    nearest_neighbor = None
    for name, data in image_hash.items():
        distance = np.sum(np.abs(data['hash'] - prediction))
        if distance < min_distance:
            min_distance = distance
            nearest_neighbor = data
    return nearest_neighbor, min_distance


# UDF for prediction
def predict_udf(image_bytes):
    try:
        img = preprocess_image(image_bytes)
        if img is None:
            return json.dumps({'status': 'error', 'message': 'No face detected'})

        prediction = predict_batch([img])[0]
        nearest_neighbor, distance = find_nearest_neighbor(prediction)

        if nearest_neighbor and distance < 15:  # Check if distance is less than 15
            return json.dumps({
                'status': 'success',
                'data': {
                    'name': nearest_neighbor.get('fname', 'N/A'),
                    'age': nearest_neighbor.get('Age', 'N/A'),
                    'address': nearest_neighbor.get('Adresse', 'N/A'),
                    'job': nearest_neighbor.get('Job', 'N/A')
                }
            })
        else:  # If no match or distance >= 15
            return json.dumps({
                'status': 'error',
                'message': 'The person does not exist in the dataset or is not a close match'
            })
    except Exception as e:
        return json.dumps({'status': 'error', 'message': str(e)})

predict_udf = udf(predict_udf, StringType())

# Kafka schema
schema = StructType([
    StructField("value", BinaryType(), True)
])

# Read from Kafka
kafka_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "imageTest") \
    .load()

# Apply prediction UDF
predictions_df = kafka_df.withColumn("value", predict_udf(kafka_df.value))

# Write predictions back to Kafka
query = predictions_df \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("topic", "prediction-topic") \
    .option("checkpointLocation", "/tmp/kafka-checkpoint") \
    .start()

spark.streams.awaitAnyTermination()





