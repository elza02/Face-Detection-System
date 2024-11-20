import tensorflow as tf

def create_and_save_latent_model(original_model_path, modified_model_path, optimized_model_path):
    model = tf.keras.models.load_model(original_model_path)
    # Modify the model to output from the 'latent' layer
    latent_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('latent').output)
    latent_model.compile(optimizer='adam', loss='binary_crossentropy')

    # Save the modified model
    latent_model.save(modified_model_path)

    # Convert the model to TensorFlow Lite format with optimization
    converter = tf.lite.TFLiteConverter.from_keras_model(latent_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the optimized TensorFlow Lite model
    with open(optimized_model_path, 'wb') as f:
        f.write(tflite_model)

# Path to the original model file
original_model_path = '/app/models/final_weights.h5'
# Path to save the modified model
modified_model_path = '/app/models/latent_model.h5'
# Path to save the optimized TensorFlow Lite model
optimized_model_path = '/app/models/latent_model.tflite'

# Create, optimize, and save the models
create_and_save_latent_model(original_model_path, modified_model_path, optimized_model_path)

