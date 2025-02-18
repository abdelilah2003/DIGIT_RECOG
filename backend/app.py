import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('/Applications/XAMPP/xamppfiles/htdocs/python/Digit_Recog/backend/model/digit_recognition_model.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']

    # Decode the base64 image
    image_data = image_data.split(',')[1]
    img = Image.open(BytesIO(base64.b64decode(image_data)))

    # Preprocess the image
    img = img.convert('L')  # Grayscale
    img = img.resize((28, 28))  # Resize
    img = np.array(img)

    # Invert and normalize
    img = 255 - img  # MNIST uses white-on-black, canvas is black-on-white
    img = img.astype(np.float32) / 255.0  # Ensure float32 dtype

    # Center the digit by cropping whitespace
    def crop_image(img_array):
        mask = img_array > 0.1  # Threshold to detect drawn pixels
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img_array  # No digit drawn
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cropped = img_array[y0:y1+1, x0:x1+1]
        return cropped

    # Apply cropping and scaling
    cropped_img = crop_image(img)
    img = Image.fromarray(cropped_img * 255).resize((20, 20))  # Resize to 20x20
    img = np.array(img) / 255.0

    # Pad to 28x28 (like MNIST)
    img = np.pad(img, ((4, 4), (4, 4)), mode='constant')  # Add 4px padding
    img = img.reshape(1, 28, 28, 1)

    # Debugging logs
    print(f"Image min/max: {img.min()}, {img.max()}")  # Should NOT be 0,0

    # Prediction
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    print(f"Prediction: {predicted_digit}")

    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)