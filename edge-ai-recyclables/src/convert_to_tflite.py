import tensorflow as tf

def convert_model_to_tflite(model_path, tflite_model_path):
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the converted model to a file
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    convert_model_to_tflite('models/model.h5', 'models/model.tflite')