from flask import Flask, request, render_template
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
import os

app = Flask(__name__, template_folder="template")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    """Render the main page with upload form"""
    return render_template('submit_image.html')

def load_model(model_type='cnn'):
    """
    Load the appropriate pre-trained model
    Args:
        model_type: 'cnn' or 'resnet'
    Returns:
        Loaded TensorFlow model
    """
    model_path = f"models/furniture_model_{model_type.upper()}.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)

def prepare_image(img_bytes, target_size=(180, 180)):
    """
    Preprocess image for model prediction
    Args:
        img_bytes: Binary image data
        target_size: Target dimensions (width, height)
    Returns:
        Processed image array
    """
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize(target_size)
    img = img.convert("RGB")
    img = np.array(img) / 255.0  # Normalize to [0,1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

def predict_class(img_array, model_type='cnn'):
    """
    Make prediction using the specified model
    Args:
        img_array: Preprocessed image array
        model_type: 'cnn' or 'resnet'
    Returns:
        Predicted class name
    """
    class_names = ['Chair', 'Sofa', 'TV', 'Table']
    model = load_model(model_type)
    predictions = model.predict(img_array)
    return class_names[np.argmax(predictions[0])]

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'imagefile' not in request.files:
        return "No image file provided", 400

    file = request.files['imagefile']
    if not file or file.filename == '':
        return "Invalid image file", 400

    try:
        # Get image data
        img_bytes = file.read()
        
        # Get model selection (default to CNN)
        model_type = request.form.get('model_type', 'cnn').lower()
        if model_type not in ['cnn', 'resnet']:
            model_type = 'cnn'  # Fallback to CNN if invalid
        
        # Preprocess and predict
        img_array = prepare_image(img_bytes)
        prediction = predict_class(img_array, model_type)
        
        # Convert image for display
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        
        return render_template(
            'submit_image.html',
            img_data=img_str,
            prediction=prediction,
            model_used=model_type.upper()
        )
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        app.logger.error(error_msg)
        return error_msg, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)