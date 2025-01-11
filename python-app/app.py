import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, BinaryType, StringType
import cv2
import json
import pandas as pd
import ast
import faiss

# Initialize Spark session with optimized settings
spark = SparkSession.builder \
    .appName("KafkaSparkIntegration") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.cores", "2") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.default.parallelism", "8") \
    .config("spark.streaming.kafka.consumer.cache.enabled", "true") \
    .config("spark.streaming.kafka.maxRatePerPartition", "100") \
    .config("spark.streaming.backpressure.enabled", "true") \
    .config("spark.streaming.kafka.consumer.poll.ms", "512") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

# Initialize face cascade
face_cascade = None
def get_face_cascade():
    global face_cascade
    if face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
        face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

# Path to the TensorFlow Lite model file
tflite_model_path = '/app/models/latent_model.tflite'

# Load and broadcast TensorFlow Lite model
def load_tflite_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")
    with open(model_path, 'rb') as f:
        tflite_model = f.read()
    return tflite_model

# Preprocess image for prediction
def preprocess_image(image_bytes):
    """Preprocess an image and perform face detection using Haar Cascade."""
    try:
        # Convert bytes to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get face cascade classifier
        cascade = get_face_cascade()
        
        # Detect faces
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        # Get the largest face if multiple faces are detected
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        
        # Resize and preprocess for the model
        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = preprocess_input(face_img)
        
        return np.ascontiguousarray(face_img)
        
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return None

# Load TFLite model content
bc_model = load_tflite_model(tflite_model_path)

def predict_single(image, interpreter, input_details, output_details):
    """Generate embedding for a single image using TFLite model."""
    interpreter.set_tensor(input_details['index'], image)
    interpreter.invoke()
    return interpreter.get_tensor(output_details['index'])[0]

def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    return -tf.keras.losses.cosine_similarity(a, b, axis=-1)

def safe_eval(value):
    """Safely evaluate string representation of embeddings."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        print(f"Skipping malformed embedding: {value}")
        return None  # Or provide a default embedding, e.g., [0.0] * 64

df_csv = pd.read_csv('/app/embeddings/person_embeddings_combined.csv')

# Convert the 'embedding' column from strings to NumPy arrays
df_csv['embedding'] = df_csv['embedding'].apply(safe_eval)
    
# Convert the list of embeddings to a NumPy array
embeddings = np.array(df_csv['embedding'].tolist(), dtype=np.float32)

# Normalize the embeddings (optional but recommended for cosine similarity)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
# Load embeddings from CSV
    
# Build FAISS index
dimension = embeddings.shape[1]

# Create a FAISS index for inner product (cosine similarity)
index = faiss.IndexFlatIP(dimension)

# Add the embeddings to the index
index.add(embeddings)

def find_nearest_neighbor(prediction, index, df_csv, k=1):
    """
    Find the nearest neighbor using FAISS.

    Args:
        prediction (np.array): The query vector (model's output).
        index (faiss.Index): The FAISS index.
        df_csv (pd.DataFrame): The CSV data with person information.
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        dict: Information about the nearest neighbor.
    """
    # Normalize the query vector
    prediction = prediction / np.linalg.norm(prediction)

    # Reshape the query vector to match FAISS input format
    prediction = np.array([prediction], dtype=np.float32)

    # Perform the search
    distances, indices = index.search(prediction, k)

    # Retrieve the nearest neighbor's information
    matching_row_id = indices[0][0]
    similarity = float(distances[0][0])  # Convert float32 to native float

    if matching_row_id + 1 < 10:
        imageURL = f"/s0{matching_row_id + 1}/01.jpg"
    else:
        imageURL = f"/s{matching_row_id + 1}/01.jpg"

    # Get person information from the CSV
    data = {
        's': similarity,
        'name': df_csv.iloc[matching_row_id]['name'],
        'person_id': int(matching_row_id + 1),  # Convert to native int
        'age': int(df_csv.iloc[matching_row_id]['age']),  # Convert to native int
        'nationality': df_csv.iloc[matching_row_id]['nationality'],
        'job': df_csv.iloc[matching_row_id]['job'],
        'imageURL': imageURL
    }

    return data, similarity

df_csv = pd.read_csv('/app/embeddings/person_embeddings_combined.csv')
df_csv['embedding'] = df_csv['embedding'].apply(safe_eval)  

def predict_udf(image_bytes):
    try:
        # Initialize TFLite interpreter for this worker
        interpreter = tf.lite.Interpreter(model_content=bc_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        img = preprocess_image(image_bytes)
        if img is None:
            return json.dumps({'status': 'error', 'message': 'No face detected'})

        prediction = predict_single(img, interpreter, input_details, output_details)
        
        data, similarity = find_nearest_neighbor(prediction, index, df_csv)
        if data and similarity > 0.5: 
            return json.dumps({
                'status': 'success',                                                                                            
                'data': {
                    'ID': data['person_id'],
                    'name': data['name'],
                    'age': data['age'],
                    'nationality': data['nationality'],
                    'job': data['job'],
                    'similarity': f"{similarity:.2%}",
                    'imageURL': data['imageURL']
                }
            })
        else:  
            return json.dumps({
                'status': 'error',
                'message': 'The person does not exist in the dataset or is not a close match'
            })
    except Exception as e:
        return json.dumps({
            'status': 'error',
            'message': str(e)
        })

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