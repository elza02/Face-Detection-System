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
import time
import logging
import faiss
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("KafkaSparkIntegration") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.cores", "2") \
    .config("spark.logConf", "true") \
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


class HaarCascadeClassifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        return cls._instance
def safe_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        print(f"Skipping malformed embedding: {value}")
        return None  # Or provide a default embedding, e.g., [0.0] * 64

df_csv = pd.read_csv('/app/embeddings/person_embeddings_combined.csv')

# Convert the 'embedding' column from strings to NumPy arrays
df_csv['embedding'] = df_csv['embedding'].apply(ast.literal_eval)

# Convert the list of embeddings to a NumPy array
embeddings = np.array(df_csv['embedding'].tolist(), dtype=np.float32)

# Normalize the embeddings (optional but recommended for cosine similarity)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
# df_csv['embedding'] = df_csv['embedding'].apply(safe_eval) 
# Load embeddings from CSV


def preprocess_image(image_bytes):
    start_time = time.time()

    # Decode the image bytes into a NumPy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        logging.error("Failed to decode image")
        return None

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the singleton Haar Cascade classifier
    face_cascade = HaarCascadeClassifier()

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Scale factor for multi-scale detection
        minNeighbors=5,   # Minimum number of neighbors for a region to be considered a face
        minSize=(30, 30)  # Minimum face size
    )

    # If no faces are detected, return None
    if len(faces) == 0:
        logging.warning("No faces detected")
        return None

    # Extract the first detected face
    (x, y, width, height) = faces[0]

    # Crop the face region
    img_t = img[y:y + height, x:x + width]

    # Resize the face region to 224x224 (required for VGG16 preprocessing)
    img_t = cv2.resize(img_t, (224, 224))

    # Convert the image to an array and preprocess it for VGG16
    img_t = img_to_array(img_t)
    img_t = np.expand_dims(img_t, axis=0)
    img_t = preprocess_input(img_t)

    end_time = time.time()
    logging.info(f"preprocess_image took {end_time - start_time:.4f} seconds")

    return np.ascontiguousarray(img_t)

bc_model = load_tflite_model(tflite_model_path)
def predict_batch(images):
    start_time = time.time()
    interpreter = tf.lite.Interpreter(model_content=bc_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    for img in images:
        interpreter.set_tensor(input_details['index'], img)
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_details['index'])[0])

    end_time = time.time()
    logging.info(f"predict_batch {end_time - start_time:.4f} seconds")

    return predictions

# def cosine_similarity(a, b):
#     """Compute cosine similarity between two tensors."""
#     # Note: tf.keras.losses.cosine_similarity returns the negative cosine similarity,
#     # so we use the negative sign to get the positive cosine similarity.
#     return -tf.keras.losses.cosine_similarity(a, b, axis=-1)







# Build FAISS index
dimension = embeddings.shape[1]

# Create a FAISS index for inner product (cosine similarity)
index = faiss.IndexFlatIP(dimension)

# Add the embeddings to the index
index.add(embeddings)

# def find_nearest_neighbor(prediction, index, df_csv):
#     start_time = time.time()
#     prediction = np.array(prediction, dtype=np.float32).reshape(1, -1)
#     distances, indices = index.search(prediction, 1)  # Find the nearest neighbor
#     matching_row_id = indices[0][0]
#     similarity = 1 / (1 + distances[0][0])  # Convert L2 distance to similarity

#     # Retrieve person information
#     data = {
#         's': similarity,
#         'name': df_csv.iloc[matching_row_id]['name'],
#         'person_id': matching_row_id + 1,
#         'age': df_csv.iloc[matching_row_id]['age'].item(),
#         'nationality': df_csv.iloc[matching_row_id]['nationality'],
#         'job': df_csv.iloc[matching_row_id]['job'],
#         'imageURL': f"/path/to/images/P{matching_row_id + 1}/s{matching_row_id + 1}_01.jpg"
#     }

#     end_time = time.time()
#     logging.info(f"find_nearest_neighbor took {end_time - start_time:.4f} seconds")
#     return data, similarity

def find_nearest_neighbor(query_vector, index, df_csv, k=1):
    """
    Find the nearest neighbor using FAISS.

    Args:
        query_vector (np.array): The query vector (model's output).
        index (faiss.Index): The FAISS index.
        df_csv (pd.DataFrame): The CSV data with person information.
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        dict: Information about the nearest neighbor.
    """
    # Normalize the query vector
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Reshape the query vector to match FAISS input format
    query_vector = np.array([query_vector], dtype=np.float32)

    # Perform the search
    distances, indices = index.search(query_vector, k)

    # Retrieve the nearest neighbor's information
    matching_row_id = indices[0][0]
    similarity = float(distances[0][0])  # Convert float32 to native float

    # Get person information from the CSV
    data = {
        's': similarity,
        'name': df_csv.iloc[matching_row_id]['name'],
        'person_id': int(matching_row_id + 1),  # Convert to native int
        'age': int(df_csv.iloc[matching_row_id]['age']),  # Convert to native int
        'nationality': df_csv.iloc[matching_row_id]['nationality'],
        'job': df_csv.iloc[matching_row_id]['job'],
        'imageURL': f"/path/to/images/P{matching_row_id + 1}/s{matching_row_id + 1}_01.jpg"
    }

    return data, similarity 
 
# UDF for prediction
def predict_udf(image_bytes):
    start_time = time.time()
    try:
        # Preprocess the image using Haar Cascade
        img = preprocess_image(image_bytes)
        if img is None:
            return json.dumps({'status': 'error', 'message': 'No face detected'})

        # Perform prediction
        prediction = predict_batch([img])[0]

        # Find the nearest neighbor
        data, similarity = find_nearest_neighbor(prediction, index, df_csv)

        if data and similarity > 0.5:  # Check if similarity is greater than 0.5
            end_time = time.time()
            logging.info(f"predict_udf took {end_time - start_time:.4f} seconds")

            return json.dumps({
                'status': 'success',
                'data': {
                    'similarity': data.get('s', 'N/A'),
                    'ID': data.get('person_id', 'N/A'),
                    'name': data.get('name', 'N/A'),
                    'age': data.get('age', 'N/A'),
                    'nationality': data.get('nationality', 'N/A'),
                    'job': data.get('job', 'N/A'),
                    'imageURL': data.get('imageURL', 'N/A'),
                }
            })
        else:  # If no match or similarity <= 0.5
            end_time = time.time()
            logging.info(f"predict_udf took {end_time - start_time:.4f} seconds")

            return json.dumps({
                'status': 'error',
                'message': 'The person does not exist in the dataset or is not a close match'
            })
    except Exception as e:
        end_time = time.time()
        logging.info(f"predict_udf took {end_time - start_time:.4f} seconds")

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





