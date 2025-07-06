# Edge AI Recyclables Project

This project aims to develop a lightweight image classification model using TensorFlow Lite, specifically designed to recognize recyclable items. The model is trained on a dataset of images and is optimized for deployment on edge devices such as Raspberry Pi.

## Project Structure

- **data/**: Contains the training and testing datasets.
  - **train/**: Directory with training images of recyclable items.
  - **test/**: Directory with test images for model evaluation.
  
- **notebooks/**: Contains Jupyter notebooks for training the model.
  - **training_colab.ipynb**: Notebook for training the model in Google Colab.

- **src/**: Source code for training, evaluating, and converting the model.
  - **train.py**: Script to define and train the image classification model.
  - **evaluate.py**: Script to evaluate the model's performance on the test dataset.
  - **convert_to_tflite.py**: Script to convert the trained model to TensorFlow Lite format.
  - **test_tflite.py**: Script to test the TensorFlow Lite model on sample images.

- **models/**: Contains the trained and converted models.
  - **model.h5**: Saved Keras model after training.
  - **model.tflite**: Converted TensorFlow Lite model for deployment.

- **reports/**: Contains reports on model performance and deployment.
  - **accuracy_and_deployment.md**: Report detailing accuracy metrics and deployment steps.

- **requirements.txt**: Lists the Python dependencies required for the project.

## Setup Instructions

1. Clone the repository to your local machine.
2. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```
3. Prepare your dataset by placing images in the `data/train` and `data/test` directories.

## Usage

- To train the model, run the `training_colab.ipynb` notebook in Google Colab.
- After training, use `convert_to_tflite.py` to convert the model to TensorFlow Lite format.
- Evaluate the model's performance using `evaluate.py`.
- Test the TensorFlow Lite model with `test_tflite.py`.

## Conclusion

This project provides a comprehensive approach to developing an Edge AI prototype for recognizing recyclable items, leveraging TensorFlow Lite for efficient deployment on edge devices.