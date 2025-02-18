import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

app = Flask(__name__, static_folder="../static", template_folder="../templates")

# Load the trained model (relative path based on the folder structure)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "digit_recognition_model.keras")
model = keras.models.load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image'].split(',')[1]  # Remove the base64 header
    img = Image.open(BytesIO(base64.b64decode(image_data)))

    # Preprocess the image
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to MNIST format
    img = np.array(img)

    # Invert and normalize
    img = 255 - img  # Convert black-on-white to white-on-black
    img = img.astype(np.float32) / 255.0  # Normalize

    # Crop the digit to remove extra whitespace
    def crop_image(img_array):
        mask = img_array > 0.1  # Threshold for detecting digit pixels
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img_array  # No digit drawn
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return img_array[y0:y1+1, x0:x1+1]

    cropped_img = crop_image(img)
    img = Image.fromarray(cropped_img * 255).resize((20, 20))
    img = np.array(img) / 255.0

    # Pad to 28x28 like MNIST
    img = np.pad(img, ((4, 4), (4, 4)), mode='constant')
    img = img.reshape(1, 28, 28, 1)

    # Debugging logs
    print(f"Image min/max: {img.min()}, {img.max()}")  # Check if correctly normalized

    # Model prediction
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    print(f"Prediction: {predicted_digit}")

    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)
