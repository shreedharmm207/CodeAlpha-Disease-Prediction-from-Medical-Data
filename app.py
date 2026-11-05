"""
app.py
Production-grade Flask Web Application & REST API
Clinical Decision Support System for Disease Prediction from Medical Data
Benchmarks: SVM, Logistic Regression, Random Forest, XGBoost
Datasets: UCI Heart Disease, Pima Indians Diabetes, Wisconsin Breast Cancer
"""

import os
import io
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS

from src.predictor import (
    CLINICAL_SPECS,
    get_metadata,
    predict_single_patient,
    batch_process_cohort
)
from src.pdf_generator import create_clinical_pdf

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Returns clinical specs, input parameters, ranges, and presets for all 3 diseases."""
    diseases_response = {}
    for key, spec in CLINICAL_SPECS.items():
        diseases_response[key] = {
            'key': key,
            'name': spec['name'],
            'subheading': spec['subheading'],
            'dataset_source': spec['dataset_source'],
            'features_meta': spec['features_meta'],
            'presets': spec['presets']
        }
    return jsonify({
        'status': 'success',
        'diseases': diseases_response,
        'available_models': [
            {'key': 'ensemble', 'name': 'Ensemble Consensus (Soft Voting)', 'badge': 'Recommended Gold Standard'},
            {'key': 'xgboost', 'name': 'XGBoost Classifier', 'badge': 'Extreme Gradient Boosted'},
            {'key': 'random_forest', 'name': 'Random Forest Classifier', 'badge': 'Ensemble Bagging'},
            {'key': 'svm', 'name': 'Support Vector Machine (SVM)', 'badge': 'RBF Kernel with Platt Scaling'},
            {'key': 'logistic_regression', 'name': 'Logistic Regression', 'badge': 'L2 Regularized Baseline'}
        ]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Single patient inference endpoint."""
    try:
        data = request.get_json(force=True)
        disease_key = data.get('disease', 'heart')
        model_key = data.get('model', 'ensemble')
        patient_inputs = data.get('inputs', {})
        
        result = predict_single_patient(disease_key, model_key, patient_inputs)
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch cohort inference from uploaded CSV or sample cohort."""
    try:
        disease_key = request.form.get('disease', 'heart')
        model_key = request.form.get('model', 'ensemble')
        use_sample = request.form.get('use_sample', 'false').lower() == 'true'
        
        if use_sample:
            sample_file_map = {
                'heart': 'sample_heart_batch.csv',
                'diabetes': 'sample_diabetes_batch.csv',
                'breast_cancer': 'sample_breast_cancer_batch.csv'
            }
            file_path = os.path.join(DATA_DIR, sample_file_map.get(disease_key, 'sample_heart_batch.csv'))
            df = pd.read_csv(file_path)
        else:
            if 'file' not in request.files:
                return jsonify({'status': 'error', 'message': 'No CSV file uploaded'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'status': 'error', 'message': 'No selected file'}), 400
            df = pd.read_csv(file)
            
        cohort_summary = batch_process_cohort(disease_key, model_key, df)
        return jsonify({'status': 'success', 'data': cohort_summary})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    """Returns full comparative evaluation metrics across SVM, LogReg, RF, XGBoost."""
    try:
        meta = get_metadata()
        return jsonify({'status': 'success', 'benchmarks': meta})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/dataset-stats/<disease_key>', methods=['GET'])
def get_dataset_stats(disease_key):
    """Returns dataset summary stats, column distributions, and UCI provenance."""
    try:
        meta = get_metadata()
        if disease_key not in meta:
            return jsonify({'status': 'error', 'message': 'Disease not found'}), 404
            
        file_map = {
            'heart': 'heart_clean.csv',
            'diabetes': 'diabetes_clean.csv',
            'breast_cancer': 'breast_cancer_clean.csv'
        }
        df = pd.read_csv(os.path.join(DATA_DIR, file_map[disease_key]))
        
        # Calculate summary statistics for front-end analytics
        summary_stats = df.describe().round(2).to_dict()
        correlation_matrix = df.corr().round(2).to_dict()
        
        return jsonify({
            'status': 'success',
            'metadata': meta[disease_key],
            'summary_stats': summary_stats,
            'correlation_matrix': correlation_matrix,
            'total_records': len(df),
            'features_count': len(meta[disease_key]['features'])
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/export-pdf', methods=['POST'])
def export_pdf():
    """Generates and serves clinical PDF report for the current prediction."""
    try:
        data = request.get_json(force=True)
        pdf_bytes = create_clinical_pdf(data)
        
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={'Content-Disposition': 'attachment;filename=Clinical_Diagnostic_Report.pdf'}
        )
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/download-sample-batch/<disease_key>', methods=['GET'])
def download_sample_batch(disease_key):
    """Allows downloading sample CSV template for cohort batch screening."""
    file_map = {
        'heart': 'sample_heart_batch.csv',
        'diabetes': 'sample_diabetes_batch.csv',
        'breast_cancer': 'sample_breast_cancer_batch.csv'
    }
    filename = file_map.get(disease_key, 'sample_heart_batch.csv')
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=filename, mimetype='text/csv')
    return jsonify({'status': 'error', 'message': 'Sample batch file not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" Disease Prediction System")
    print(f" Models: SVM, Logistic Regression, Random Forest, XGBoost")
    print(f" Server running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
