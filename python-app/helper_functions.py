import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array
from mtcnn import MTCNN
import cv2
import pandas as pd


#function to load the model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

#function to preprocess the images
#in my case i didn't use the MTCNN, 
# because my task was to extract the embedding from a prepared dataset
def preprocess_image(image_bytes):
    """Preprocess an image and perform face detection using a lazily initialized MTCNN."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # # Initialize MTCNN detector lazily
    # detector = MTCNN()  # MTCNN is reinitialized on each executor
    # results = detector.detect_faces(img)
    # if not results:
    #     print("No face detected in image.")
    #     return None  # Return None if no face is detected

    # y, x, width, height = results[0]['box']
    # img_t = img[y:y + height, x:x + width]
    
    # if img_t.size == 0:  # Check if img_t is empty
    #     print("Detected face region is empty.")
    #     return None  # Skip this image if the face region is empty

    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return np.ascontiguousarray(img)


#img to extract the embeddings
def predict_batch(images, interpreter):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    for img in images:
        interpreter.set_tensor(input_details['index'], img)
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_details['index'])[0])

    return predictions

#this function assemble the preprocessing and the embedding extraction
def generate_embedding(image_path, model):
    with open(image_path, 'rb') as img_file:
        image_bytes = img_file.read()
        preprocessed_image = preprocess_image(image_bytes)
        if preprocessed_image is None:
            return None
        embeddings = predict_batch([preprocessed_image], model)
        if embeddings is None:
            print('embeddings are none type')
        else:
            return embeddings[0]  # Assuming embeddings are a single array
        
#this one is used to extract the embeddings for each person, 
#then store them in csv file for futur processing
def process_face_dataset(dataset_path, output_csv_path, model):
    """
    Processes a dataset of face images and generates embeddings for each person,
    storing each embedding as a list in the DataFrame.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        output_csv_path (str): Path to save the resulting CSV file.
        model: The preloaded model for generating embeddings.
    """
    # Create a list to store the results
    results = []

    # Iterate over each person's directory (P1, P2, ..., P50)
    for person_dir in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_dir)

        if os.path.isdir(person_path) and person_dir.startswith('P'):
            person_id = person_dir[1:]  # Extract person ID (P1 -> 1)

            # Collect all images for this person
            images = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
            
            # Ensure exactly 15 images are processed
            if len(images) < 15:
                print(f"Skipping {person_id}: not enough images ({len(images)})")
                continue
            if len(images) > 15:
                print(f"Skipping {person_id}: more than 15 images ({len(images)})")
                continue

            # Process the selected 15 images
            for pose_id, image_name in enumerate(images, start=1):  # Enumerate for pose_id
                # Get the full image path
                image_path = os.path.join(person_path, image_name)
                print(f"Processing image: {image_path}")

                # Generate embedding for the image
                embedding = generate_embedding(image_path, model)
                if embedding is not None:
                    # Store the result as a list
                    results.append([person_id, pose_id, embedding])
                else:
                    print(f"Failed to process image: {image_path}")

    # Ensure results are not empty
    if not results:
        print("No valid results to process.")
        return

    # Convert the results to a pandas DataFrame
    columns = ['person_id', 'pose_id', 'embedding']
    df = pd.DataFrame(results, columns=columns)

    # Save the results to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Embeddings saved to {output_csv_path}")

#this func is used to sort the csv file based on the person & pose id
def sort_csv_by_person_and_pose(input_csv_path, output_csv_path):
    """
    Sorts a CSV file by 'person_id' and then by 'pose_id' in ascending order.
    
    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the sorted CSV file.
    """
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(input_csv_path)

    # Convert 'person_id' and 'pose_id' to integers for correct sorting
    df['person_id'] = df['person_id'].astype(int)
    df['pose_id'] = df['pose_id'].astype(int)

    # Sort the DataFrame by 'person_id' and then by 'pose_id'
    df_sorted = df.sort_values(by=['person_id', 'pose_id'], ascending=[True, True])

    # Save the sorted DataFrame to a new CSV file
    df_sorted.to_csv(output_csv_path, index=False)
    print(f"Sorted CSV saved to {output_csv_path}")


# class TransformerEncoder(nn.Module):
#     def __init__(self, input_dim, num_heads, ff_dim, num_layers):
#         super(TransformerEncoder, self).__init__()
        
#         # Transformer Encoder Layers
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim, 
#             nhead=num_heads, 
#             dim_feedforward=ff_dim,
#             activation='relu',
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Linear projection to output a single representational vector
#         self.output_layer = nn.Linear(input_dim, input_dim)

#     def forward(self, x):
#         # Pass through Transformer Encoder
#         encoded_output = self.encoder(x)  # Shape: (batch_size, seq_len, input_dim)
        
#         # Pooling: Take the mean of the sequence output
#         pooled_output = torch.mean(encoded_output, dim=1)  # Shape: (batch_size, input_dim)
        
#         # Project to the representational vector
#         representational_vector = self.output_layer(pooled_output)  # Shape: (batch_size, input_dim)
        
#         return representational_vector

# # Hyperparameters
# input_dim = 64
# num_heads = 4
# ff_dim = 128
# num_layers = 2
# batch_size = 10
# num_epochs = 50
# learning_rate = 0.001

# # Model, optimizer, and loss
# model = TransformerEncoder(input_dim, num_heads, ff_dim, num_layers)
# model.load_state_dict(torch.load('/app/models/Transfomer_weights/transformer_model.pth', map_location=torch.device('cpu')))

def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    return F.cosine_similarity(a, b, dim=-1)