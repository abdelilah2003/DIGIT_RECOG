import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Load MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values (0-255 → 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data to add a single color channel
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),  # Reduces overfitting
    keras.layers.Dense(10, activation='softmax')  # 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,   # Rotate images slightly
    zoom_range=0.1,      # Randomly zoom in/out
    width_shift_range=0.1,  # Shift images horizontally
    height_shift_range=0.1  # Shift images vertically
)

# Callbacks
checkpoint = ModelCheckpoint(
    'digit_recognition_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# Train the model
model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=10,  # Increased epochs
    validation_data=(x_test, y_test),
    callbacks=[checkpoint, reduce_lr]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Attempt to save the model
model_save_path = "/Applications/XAMPP/xamppfiles/htdocs/python/Digit_Recog/backend/model/digit_recognition_model2.keras"
try:
    model.save(model_save_path)
    # Check if model file is saved successfully
    if os.path.exists(model_save_path):
        print("Model saved successfully!")
    else:
        print("Model failed to save.")
except Exception as e:
    print(f"Error saving model: {e}")

# --- Testing the Model on some MNIST Test Images ---
for i in range(5):  # Test on 5 sample test images
    img_test = x_test[i].reshape(1, 28, 28, 1)
    prediction = model.predict(img_test)
    predicted_digit = np.argmax(prediction)
    print(f"Predicted digit for test image {i}: {predicted_digit} (True label: {y_test[i]})")
