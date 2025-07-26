from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#   Load the Keras model and class names
MODEL_PATH = "lfw_cnn_model.keras"
CLASS_NAMES_PATH = "class_names.json"
model = None
class_names = None

def load_model():
    """Load the Keras model"""
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise Exception(f"Failed to load model: {str(e)}")
    return model

def load_class_names():
    """Load class names from JSON file"""
    global class_names
    if class_names is None:
        try:
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)
            print(f"✅ Class names loaded successfully from {CLASS_NAMES_PATH}")
        except Exception as e:
            print(f"❌ Failed to load class names: {e}")
            raise Exception(f"Failed to load class names: {str(e)}")
    return class_names

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, img_size=100):
    """Preprocess image for the face recognition model"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize pixel values (0-255 to 0-1)
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_face(image_path):
    """Predict face recognition using local Keras model"""
    try:
        # Load model and class names
        model = load_model()
        class_names = load_class_names()
        
        # Preprocess image
        processed_img = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(processed_img, verbose=0)
        
        # Get the predicted class index
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get the person name
        person_name = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else f"Unknown_{predicted_class_idx}"
        
        return {
            "prediction": person_name,
            "confidence": confidence
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not allowed. Please upload: ' + ', '.join(ALLOWED_EXTENSIONS)
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_face(filepath)
        
        # Return only prediction result
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_path': MODEL_PATH,
        'class_names_path': CLASS_NAMES_PATH,
        'model_loaded': model is not None,
        'class_names_loaded': class_names is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 