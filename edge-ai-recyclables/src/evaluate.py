import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Load the trained model
model_path = '../models/model.h5'
model = load_model(model_path)

# Define paths for test data
test_data_dir = '../data/test'
batch_size = 32
img_height = 180
img_width = 180

# Prepare the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Generate predictions
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get true classes
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Create a report
report_path = '../reports/accuracy_and_deployment.md'
with open(report_path, 'w') as report_file:
    report_file.write('# Model Evaluation Report\n')
    report_file.write(f'## Test Loss: {loss:.4f}\n')
    report_file.write(f'## Test Accuracy: {accuracy:.4f}\n')
    report_file.write('## Predictions:\n')
    for i, pred in enumerate(predicted_classes):
        report_file.write(f'Image {i}: True Class: {class_labels[true_classes[i]]}, Predicted Class: {class_labels[pred]}\n')