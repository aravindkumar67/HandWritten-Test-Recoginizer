import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
model = load_model('Dl_Model1')

label_dict = {}
for char in range(65, 91):  
    label_dict[chr(char)] = char - 65 
for char in range(97, 123):  
    label_dict[chr(char)] = char - 97 + 26
for char in range(48, 58): 
    label_dict[chr(char)] = char - 48 + 52

reverse_dict = {v: k for k, v in label_dict.items()}

# Function to preprocess and predict text from an image
def predict_text(image):
    image = cv2.resize(image, (128, 128))  
    image = image / 255.0  
    image = image.reshape(1, 128, 128, 1)
    
    predictions = model.predict(image)

    predicted_text = model.predict(image)

    return reverse_dict[predicted_text.argmax()]

@app.route('/')
def index():
    return render_template('DL_Proj.html')

@app.route('/extract_text', methods=['POST'])
def extract_text():
    image = request.files['image']
    if image:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        predicted_word = predict_text(image)
        return predicted_word
    else:
        return 'Error: No valid image uploaded.'

if __name__ == '__main__':
    app.run(debug=True)