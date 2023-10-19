from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow import keras
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the deepfake model
model = keras.models.load_model('deepfake_model.h5')

# Set the image dimensions
image_height = 640
image_width = 480

# Define a function to process image uploads
def process_image(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)  # Save the uploaded file
    img = image.load_img(file_path, target_size=(image_height, image_width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Define the home route
@app.route('/')
def home():
    return render_template('upload.html', prediction=None)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('upload.html', prediction="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('upload.html', prediction="No file selected")

    if file:
        img = process_image(file)
        predictions = model.predict(img)
        class_names = ['fake', 'real']
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]

        return render_template('upload.html', prediction=predicted_class, image_path=f'uploads/{file.filename}')

if __name__ == '__main__':
    # Create the 'uploads' directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
