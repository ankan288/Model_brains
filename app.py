from flask import Flask, request, jsonify, render_template_string
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static')

# Available models
MODELS = {
    'cnn_h5': {
        'path': os.path.join(os.path.dirname(__file__), 'simple_cnn_model.h5'),
        'name': 'Simple CNN Model (H5)',
        'description': '2-channel CNN model',
        'classes': {0: 'Healthy/Normal', 1: 'Benign Tumor', 2: 'Malignant Tumor'}
    },
    'keras_model': {
        'path': os.path.join(os.path.dirname(__file__), 'my_model.keras'),
        'name': 'Keras Model',
        'description': 'Modern Keras format model',
        'classes': {0: 'No Tumor', 1: 'Glioma', 2: 'Meningioma', 3: 'Pituitary'}
    },
    'brain_tumor_detector': {
        'path': os.path.join(os.path.dirname(__file__), 'brain_tumor_detector.h5'),
        'name': 'Brain Tumor Detector (H5)',
        'description': 'Specialized brain tumor detection model',
        'classes': {0: 'No Tumor', 1: 'Tumor Detected'}
    }
}

# Current active model
current_model = None
current_model_key = None
input_size = (224, 224)
input_channels = 3

def get_available_models():
    """Get list of available models that exist on disk"""
    available = {}
    for key, info in MODELS.items():
        if os.path.exists(info['path']):
            available[key] = info
    return available

def load_selected_model(model_key):
    global current_model, current_model_key, input_size, input_channels
    
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}")
    
    model_info = MODELS[model_key]
    model_path = model_info['path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model: {model_info['name']}")
    current_model = load_model(model_path)
    current_model_key = model_key
    
    # Determine input shape
    shape = current_model.input_shape
    if isinstance(shape, (list, tuple)) and len(shape) >= 4:
        _, h, w, c = shape[:4]
    elif isinstance(shape, (list, tuple)) and len(shape) == 3:
        h, w, c = input_size[0], input_size[1], input_channels
    else:
        h, w, c = input_size[0], input_size[1], input_channels
    
    try:
        input_size = (int(h), int(w))
        input_channels = int(c)
    except Exception:
        input_size = (224, 224)
        input_channels = 3
        
    print(f"Loaded {model_info['name']}. Input size={input_size}, channels={input_channels}")
    return model_info['name']

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    # convert to needed channels
    if input_channels == 1:
        img = img.convert('L')
    elif input_channels == 2:
        # For 2-channel: convert to grayscale + alpha, or use LA mode
        # If original has alpha, preserve it; otherwise create artificial alpha
        if img.mode in ('RGBA', 'LA'):
            img = img.convert('LA')  # Grayscale + Alpha
        else:
            # Convert to grayscale and add full alpha channel
            gray = img.convert('L')
            img = Image.new('LA', gray.size)
            img.paste(gray, (0, 0))
            # Set alpha to full opacity
            alpha_band = Image.new('L', gray.size, 255)
            img.putalpha(alpha_band)
    else:
        img = img.convert('RGB')
    
    img = img.resize(input_size)
    arr = np.array(img).astype('float32') / 255.0
    
    # Handle different channel dimensions
    if input_channels == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    elif input_channels == 2 and arr.ndim == 2:
        # If we got a single channel but need 2, duplicate it
        arr = np.stack([arr, arr], axis=-1)
    
    # ensure shape (1, H, W, C)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return arr


@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/models')
def get_models():
    """API endpoint to get available models"""
    available = get_available_models()
    return jsonify({
        'models': available,
        'current': current_model_key
    })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """API endpoint to switch between models"""
    data = request.get_json()
    model_key = data.get('model_key')
    
    if not model_key:
        return jsonify({'error': 'model_key required'}), 400
    
    try:
        model_name = load_selected_model(model_key)
        return jsonify({
            'success': True,
            'model': model_name,
            'input_size': input_size,
            'input_channels': input_channels
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if current_model is None:
        return jsonify({'error': 'No model loaded. Please select a model first.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    try:
        img_bytes = f.read()
        x = preprocess_image(img_bytes)
        preds = current_model.predict(x)
        preds = np.asarray(preds).ravel()
        # prepare top results
        top_k = min(10, preds.size)  # Show top 10 instead of 5
        top_idx = preds.argsort()[::-1][:top_k]
        
        # Get class labels for current model
        class_labels = MODELS.get(current_model_key, {}).get('classes', {})
        
        results = []
        for i in top_idx:
            class_name = class_labels.get(int(i), f'Class {int(i)}')
            results.append({
                'class': int(i), 
                'class_name': class_name,
                'probability': float(preds[i])
            })
        
        # Also show some basic stats
        predicted_class_idx = int(np.argmax(preds))
        predicted_class_name = class_labels.get(predicted_class_idx, f'Class {predicted_class_idx}')
        
        stats = {
            'total_classes': int(preds.size),
            'max_probability': float(np.max(preds)),
            'min_probability': float(np.min(preds)),
            'mean_probability': float(np.mean(preds)),
            'predicted_class': predicted_class_idx,
            'predicted_class_name': predicted_class_name
        }
        
        return jsonify({
            'predictions': results, 
            'stats': stats,
            'raw': preds.tolist()[:50]  # Only show first 50 raw values to reduce clutter
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load the first available model by default
    available_models = get_available_models()
    if available_models:
        first_model = list(available_models.keys())[0]
        load_selected_model(first_model)
        print(f"Server starting with default model: {MODELS[first_model]['name']}")
    else:
        print("Warning: No models found! Place models in the directory.")
    
    # debug=True for development only
    app.run(host='0.0.0.0', port=5000, debug=True)
