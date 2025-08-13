from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import librosa
import tensorflow as tf
import logging
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = r'C:\Users\saive\Desktop\RESPIRE\Respiratory_Sound_Database\dir\static\uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model
model = tf.keras.models.load_model('diagnosis_GRU_CNN_1.keras')
classes = ['COPD', 'URTI', 'Bronchiolitis', 'Pneumonia', 'Healthy']

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the HTML file for the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for upload.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a .wav or .mp3 file.'}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Load and preprocess the audio file
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=52)
        mfcc = np.mean(mfcc.T, axis=0)
        mfcc = mfcc.reshape(1, 1, -1)

        # Predict using the model
        prediction = model.predict(mfcc)
        predicted_class = classes[np.argmax(prediction)]

        return jsonify({'prediction': predicted_class})
    except ValueError as ve:
        logger.error(f"Model input shape mismatch: {str(ve)}")
        return jsonify({'error': 'Model input shape mismatch.', 'details': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'An internal error occurred.', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
