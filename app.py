# app.py - Flask Backend API
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import os
import sys

# Import model predictor
from diabetes_model import DiabetesPredictor

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Initialize predictor
predictor = DiabetesPredictor()

# Load model saat startup
app._got_first_request
def load_model():
    """Load model saat aplikasi pertama kali dijalankan"""
    try:
        # Coba load model yang sudah ada
        predictor.load_model()
        print("Model berhasil dimuat!")
    except:
        print("Model tidak ditemukan, akan melatih model baru...")
        # Jika model tidak ada, latih model baru
        X, y = predictor.load_and_prepare_data()
        predictor.train_model(X, y)
        predictor.save_model()
        print("Model baru berhasil dilatih dan disimpan!")

@app.route('/')
def home():
    """Halaman utama"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_diabetes():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features
        features = [
            data.get('pregnancies', 0),
            data.get('glucose', 0),
            data.get('blood_pressure', 0),
            data.get('skin_thickness', 0),
            data.get('insulin', 0),
            data.get('bmi', 0),
            data.get('diabetes_pedigree', 0),
            data.get('age', 0)
        ]
        
        # Validate features
        if any(f < 0 for f in features):
            return jsonify({'error': 'All values must be non-negative'}), 400
        
        # Make prediction
        result = predictor.predict_risk(features)
        
        # Add input data to response
        result['input_data'] = {
            'pregnancies': features[0],
            'glucose': features[1],
            'blood_pressure': features[2],
            'skin_thickness': features[3],
            'insulin': features[4],
            'bmi': features[5],
            'diabetes_pedigree': features[6],
            'age': features[7]
        }
        
        return jsonify(result)
    
    except Exception as e:
        print("Error during prediction:", str(e))  # Log the error
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get information about model features"""
    feature_info = {
        'features': [
            {
                'name': 'pregnancies',
                'description': 'Jumlah kehamilan',
                'type': 'integer',
                'min': 0,
                'max': 20
            },
            {
                'name': 'glucose',
                'description': 'Konsentrasi glukosa plasma (mg/dL)',
                'type': 'float',
                'min': 0,
                'max': 200
            },
            {
                'name': 'blood_pressure',
                'description': 'Tekanan darah diastolik (mmHg)',
                'type': 'float',
                'min': 0,
                'max': 150
            },
            {
                'name': 'skin_thickness',
                'description': 'Ketebalan lipatan kulit trisep (mm)',
                'type': 'float',
                'min': 0,
                'max': 100
            },
            {
                'name': 'insulin',
                'description': 'Insulin serum 2 jam (mu U/ml)',
                'type': 'float',
                'min': 0,
                'max': 846
            },
            {
                'name': 'bmi',
                'description': 'Body Mass Index (kg/m²)',
                'type': 'float',
                'min': 0,
                'max': 70
            },
            {
                'name': 'diabetes_pedigree',
                'description': 'Fungsi silsilah diabetes',
                'type': 'float',
                'min': 0,
                'max': 2.5
            },
            {
                'name': 'age',
                'description': 'Usia (tahun)',
                'type': 'integer',
                'min': 0,
                'max': 120
            }
        ]
    }
    return jsonify(feature_info)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Panggil fungsi untuk load/latih model sebelum server dijalankan
    load_model()

    app.run(debug=True, host='0.0.0.0', port=5000)
