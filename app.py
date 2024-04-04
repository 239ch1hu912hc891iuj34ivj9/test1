from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load pre-trained machine learning model
model = tf.keras.models.load_model("model.h5")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("file[]")
    results = []
    for file in uploaded_files:
        img = Image.open(file)
        img = img.convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        results.append((file.filename, predictions.tolist()[0]))

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
