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
import pandas as pd
import ast



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
# Load and broadcast TensorFlow Lite model
def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")

    with open(model_path, 'rb') as f:
        tflite_model = f.read()

    return tflite_model



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

# def threshold_predictions(predictions, threshold=0.5):
#     binary_predictions = np.array(predictions >= threshold, dtype=np.float32)
#     return binary_predictions

# Perform batch predictions with TensorFlow Lite

bc_model = load_tflite_model(tflite_model_path)
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


    return predictions

def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    # Note: tf.keras.losses.cosine_similarity returns the negative cosine similarity,
    # so we use the negative sign to get the positive cosine similarity.
    return -tf.keras.losses.cosine_similarity(a, b, axis=-1)

def safe_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        print(f"Skipping malformed embedding: {value}")
        return None  # Or provide a default embedding, e.g., [0.0] * 64

def find_nearest_neighbor(prediction, df_csv):
    model_embeddings = tf.convert_to_tensor(np.array(prediction), dtype=tf.float32)  # Shape: (15, 64)
    model_embeddings = tf.expand_dims(model_embeddings, axis=0)  # Shape: (1, 15, 64)

    similarity = 0.0
    matching_row_id = -1

    for idx, row in df_csv.iterrows():
        row_embeddings = tf.convert_to_tensor(np.array(row['embedding']), dtype=tf.float32)  # Shape: (15, 64)
        row_embeddings = tf.expand_dims(row_embeddings, axis=0)  # Shape: (1, 15, 64)

        s = cosine_similarity(row_embeddings, model_embeddings)

        if s.numpy().item() > similarity:
            similarity = s.numpy().item()
            matching_row_id = idx

    if matching_row_id + 1 < 10:
        imageURL = f"/home/zakaria/Downloads/GTdb_crop2/P{matching_row_id + 1}/s0{matching_row_id + 1}_01.jpg"
    else:
        imageURL = f"/home/zakaria/Downloads/GTdb_crop2/P{matching_row_id + 1}/s{matching_row_id + 1}_01.jpg"

    data = {
        's': similarity,
        'name': df_csv.iloc[matching_row_id]['name'],
        'person_id': matching_row_id + 1,
        'age': df_csv.iloc[matching_row_id]['age'].item(),
        'nationality': df_csv.iloc[matching_row_id]['nationality'],
        'job': df_csv.iloc[matching_row_id]['job'],
        'imageURL': imageURL
    }
    return data, similarity

df_csv = pd.read_csv('/app/embeddings/person_embeddings_combined.csv')
df_csv['embedding'] = df_csv['embedding'].apply(safe_eval)  
# UDF for prediction
def predict_udf(image_bytes):
    try:
        img = preprocess_image(image_bytes)
        if img is None:
            return json.dumps({'status': 'error', 'message': 'No face detected'})

        prediction = predict_batch([img])[0]
        # nearest_neighbor, distance = find_nearest_neighbor(prediction)
        
        
        data, similarity = find_nearest_neighbor(prediction, df_csv)
        # if nearest_neighbor and distance < 15:  # Check if distance is less than 15
        if data and similarity > 0.5:  # Check if distance is less than 15
            return json.dumps({
                'status': 'success',                                                                                            
                'data': {
                    'similarity': data.get('s', 'N/A'),
                    'ID': data.get('person_id', 'N/A'),
                    'name': data.get('name', 'N/A'),
                    'age': data.get('age', 'N/A'),
                    'nationality': data.get('nationality', 'N/A'),
                    'job': data.get('job', 'N/A'),
                    'imageURL' : data.get('imageURL', 'N/A'),
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





