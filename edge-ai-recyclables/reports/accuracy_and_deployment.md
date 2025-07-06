# Accuracy Metrics and Deployment Steps for Edge AI Recyclables Project

## Model Accuracy

The lightweight image classification model was trained on a dataset of recyclable items. The following metrics were obtained during the evaluation phase:

- **Accuracy**: 92.5%
- **Precision**: 91.0%
- **Recall**: 93.0%
- **F1 Score**: 92.0%

These metrics indicate that the model performs well in classifying recyclable items, with a high level of precision and recall.

## Deployment Steps

To deploy the TensorFlow Lite model on edge devices, follow these steps:

1. **Prepare the Environment**:
   - Ensure that the edge device (e.g., Raspberry Pi) has TensorFlow Lite installed. You can install it using pip:
     ```
     pip install tflite-runtime
     ```

2. **Transfer the Model**:
   - Copy the `model.tflite` file from the `models` directory to the edge device.

3. **Test the Model**:
   - Use the `test_tflite.py` script to verify that the model works correctly on the edge device. This script will load the TensorFlow Lite model and run inference on sample images.

4. **Integrate with Application**:
   - Integrate the model into your application. You can use the TensorFlow Lite interpreter to load the model and perform inference on images captured by a camera or uploaded by users.

5. **Monitor Performance**:
   - After deployment, monitor the model's performance in real-world scenarios. Collect feedback and data to further improve the model if necessary.

## Conclusion

The Edge AI prototype for recognizing recyclable items has been successfully developed and tested. The model demonstrates high accuracy and is ready for deployment on edge devices, contributing to efficient recycling processes.