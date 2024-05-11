from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from keras.models import model_from_json
json_file = open('66k-50epochs.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the pre-trained MobileNetV2 model
loaded_model.load_weights("66k-mobilenetv2-50epochs-76.85.h5")
loaded_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define a list of class labels corresponding to skin diseases
class_labels = ["Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
'Basal cell carcinoma',
'Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)',
'Dermatofibroma',
'Melanoma',
'Melanocytic nevi',
'Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)',
'Vitiligo']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the POST request
        file = request.files['image']

        # Read and preprocess the image
        image = Image.open(file)
        image = image.resize((224, 224))
        image = np.array(image)
        image = image / 1.0
        image = np.expand_dims(image, axis=0)

        # Make a prediction
        predictions = loaded_model.predict(image)
        score = max(predictions[0]*100)
        score = str(score)

        # Get the predicted class label
        predicted_class = class_labels[np.argmax(predictions)]
        
        

        return {'score': score, 'prediction':predicted_class}

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port = 8080)
