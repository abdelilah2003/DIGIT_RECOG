# Handwritten Digit Recognition  

This project is a web-based handwritten digit recognition system using a Convolutional Neural Network (CNN). Users can draw a digit (0-9) on a canvas, and the model predicts the digit in real-time with **high accuracy, even for complex and hard-to-read handwritten digits.**  

## ⚠️ Important Warning
Before running the project, **ensure that the file paths in the code match your system's structure**.  
For example, the path to the trained model in `app.py`:
```python
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "digit_recognition_model.keras")
```
If your directory structure differs, adjust the paths accordingly to prevent loading issues.


## Table of Contents  
- [Project Overview](#project-overview)  
- [Folder Structure](#folder-structure)  
- [Technologies Used](#technologies-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Accuracy & Proof](#model-accuracy--proof)  
- [Model Training](#model-training)  
- [Endpoints](#endpoints)  

## Project Overview  
This project uses a trained CNN model on the MNIST dataset to recognize handwritten digits. The frontend provides an interactive canvas for drawing digits, while the backend processes the image and returns a prediction.  

## Folder Structure  
```
DIGIT_RECOG/
│── backend/                   # Backend API (Flask)
│   ├── app.py                 # Main API script
│   ├── model/
│   │   ├── train_model.py          # Model training script
│   │   ├── digit_recognition_model.keras  # Pre-trained model
│   ├── requirements.txt        # Dependencies
│
│── static/                   # Web frontend
│   ├── index.html              # Web interface
│   ├── script.js               # Handles drawing & API calls
│   ├── styles.css              # UI styling
│
│── templates/                 # Flask HTML templates
│   ├── index.html              # Main web interface
│
│── README.md                   # Documentation
```  

## Technologies Used  
- **Frontend:** HTML, CSS, JavaScript (Canvas API for drawing)  
- **Backend:** Flask (Python-based web framework)  
- **Machine Learning:** TensorFlow & Keras (Convolutional Neural Network for digit recognition)  
- **Dataset:** MNIST (Modified National Institute of Standards and Technology database)  

## Installation  
### 1. Clone the repository  
```sh
git clone https://github.com/abdelilah2003/DIGIT_RECOG.git  
cd DIGIT_RECOG  
```  

### 2. Set up a virtual environment  
```sh
python -m venv venv  
source venv/bin/activate  # On Windows use: venv\Scripts\activate  
```  

### 3. Install dependencies  
```sh
pip install -r backend/requirements.txt  
```  

### 4. Run the Flask server  
```sh
python backend/app.py  
```  

### 5. Open the application  
Go to `http://127.0.0.1:5000/` in your browser.  

## Usage  
1. Draw a digit (0-9) in the provided canvas.  
2. Click **Predict** to send the image to the backend.  
3. The backend processes the image and predicts the digit.  
4. The result is displayed on the webpage.  

## Model Accuracy & Proof  
Our CNN model achieves **exceptional accuracy**, even on difficult-to-read handwritten digits. Below are some examples demonstrating the model’s robustness:  

**Prediction on Hard-to-Read Digits:**  

<div>
<img width="215" alt="Image" src="https://github.com/user-attachments/assets/82425a7a-6c9e-46b4-ad05-41a5be33b22e" />

<img width="215" alt="Image" src="https://github.com/user-attachments/assets/d69faad1-3f6a-4507-8074-c570f49dc87b" />

<img width="215" alt="Image" src="https://github.com/user-attachments/assets/21c484b8-4767-443b-bc1e-9c03aad0a47f" />

<img width="215" alt="Image" src="https://github.com/user-attachments/assets/084ff937-980f-4220-a8f2-40a9cd818e70" />
</div>
 

## Model Training  
If you want to retrain the model, navigate to the `backend/model/` directory and run:  
```sh
python train_model.py  
```  
This script:  
- Loads the MNIST dataset  
- Normalizes and augments the data  
- Trains a CNN model  
- Saves the trained model as `digit_recognition_model.keras`

  The model was trained using the Adam optimizer, Sparse Categorical Crossentropy loss function, and Accuracy as the evaluation metric.
Here is the training output for 10 epochs:
<img width="1068" alt="Image" src="https://github.com/user-attachments/assets/c585413f-1e1b-4449-b9c9-245c4f78d26d" />
After 10 epochs, the model achieved an accuracy of 99.55% on the MNIST test dataset, demonstrating its high precision in recognizing handwritten digits.

## Endpoints  
### `GET /`  
Serves the main web interface.  

### `POST /predict`  
- **Request:** JSON with a base64 image  
- **Response:** JSON with the predicted digit  
- **Example Response:**  
```json
{
  "digit": 3
}
```  

---  
Developed by **Abdelilah SAAFLAOUKETE** 🚀  
