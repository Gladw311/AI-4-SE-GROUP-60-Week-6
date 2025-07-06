import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size as per your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Test the model on sample images
def test_model_on_images(image_paths):
    for img_path in image_paths:
        img_array = load_and_preprocess_image(img_path)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        print(f"Image: {img_path}, Predicted class: {predicted_class}")

# Sample image paths for testing
sample_images = [
    "data/test/sample_image1.jpg",
    "data/test/sample_image2.jpg",
    # Add more sample image paths as needed
]

# Run the test
test_model_on_images(sample_images)