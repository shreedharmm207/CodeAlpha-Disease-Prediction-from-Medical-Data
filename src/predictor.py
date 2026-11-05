"""
predictor.py
Clinical Diagnostic Inference & Decision Support Engine
Provides calibrated probability scoring, risk tiering, feature contribution analysis,
multi-algorithm consensus, and batch cohort processing.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Cache for loaded joblib models
_MODEL_CACHE = {}
_METADATA_CACHE = None

def get_metadata():
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
        with open(meta_path, 'r') as f:
            _METADATA_CACHE = json.load(f)
    return _METADATA_CACHE

def load_model(disease_key, model_key):
    cache_key = f"{disease_key}_{model_key}"
    if cache_key not in _MODEL_CACHE:
        file_path = os.path.join(MODELS_DIR, f"{cache_key}.joblib")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        _MODEL_CACHE[cache_key] = joblib.load(file_path)
    return _MODEL_CACHE[cache_key]

# Clinical reference definitions and recommendations
CLINICAL_SPECS = {
    'heart': {
        'name': 'Heart Disease',
        'subheading': 'Predict the risk of heart disease based on patient vitals and test results',
        'dataset_source': 'UCI Cleveland Dataset (303 patients)',
        'target_classes': {
            0: {'label': 'No Heart Disease (Normal)', 'tier': 'Low Risk', 'color': 'emerald'},
            1: {'label': 'Heart Disease Detected', 'tier': 'High Risk', 'color': 'rose'}
        },
        'features_meta': [
            {'key': 'age', 'label': 'Age', 'unit': 'years', 'min': 25, 'max': 80, 'step': 1, 'default': 54, 'group': 'Demographics', 'normal': '18-65'},
            {'key': 'sex', 'label': 'Gender', 'unit': '', 'type': 'select', 'options': [{'val': 1, 'label': 'Male'}, {'val': 0, 'label': 'Female'}], 'default': 1, 'group': 'Demographics'},
            {'key': 'cp', 'label': 'Chest Pain Type', 'unit': '', 'type': 'select', 'options': [
                {'val': 1, 'label': 'Typical Angina'},
                {'val': 2, 'label': 'Atypical Angina'},
                {'val': 3, 'label': 'Non-Anginal Pain'},
                {'val': 4, 'label': 'Asymptomatic'}
            ], 'default': 3, 'group': 'Symptoms'},
            {'key': 'trestbps', 'label': 'Resting Blood Pressure', 'unit': 'mm Hg', 'min': 90, 'max': 200, 'step': 1, 'default': 130, 'group': 'Vitals', 'normal': '< 120'},
            {'key': 'chol', 'label': 'Serum Cholesterol', 'unit': 'mg/dL', 'min': 120, 'max': 570, 'step': 1, 'default': 240, 'group': 'Blood Test', 'normal': '< 200'},
            {'key': 'fbs', 'label': 'Fasting Blood Sugar > 120 mg/dL', 'unit': '', 'type': 'select', 'options': [{'val': 0, 'label': 'No (Normal ≤ 120)'}, {'val': 1, 'label': 'Yes (> 120 mg/dL)'}], 'default': 0, 'group': 'Blood Test'},
            {'key': 'restecg', 'label': 'Resting ECG', 'unit': '', 'type': 'select', 'options': [
                {'val': 0, 'label': 'Normal'},
                {'val': 1, 'label': 'ST-T Wave Abnormality'},
                {'val': 2, 'label': 'Left Ventricular Hypertrophy'}
            ], 'default': 0, 'group': 'ECG'},
            {'key': 'thalach', 'label': 'Max Heart Rate Achieved', 'unit': 'bpm', 'min': 70, 'max': 210, 'step': 1, 'default': 150, 'group': 'Stress Test', 'normal': '130-180'},
            {'key': 'exang', 'label': 'Exercise Induced Angina', 'unit': '', 'type': 'select', 'options': [{'val': 0, 'label': 'No'}, {'val': 1, 'label': 'Yes'}], 'default': 0, 'group': 'Stress Test'},
            {'key': 'oldpeak', 'label': 'ST Depression (Exercise vs Rest)', 'unit': 'mm', 'min': 0.0, 'max': 6.2, 'step': 0.1, 'default': 1.0, 'group': 'Stress Test', 'normal': '< 1.0'},
            {'key': 'slope', 'label': 'Slope of Peak Exercise ST', 'unit': '', 'type': 'select', 'options': [
                {'val': 1, 'label': 'Upsloping'},
                {'val': 2, 'label': 'Flat'},
                {'val': 3, 'label': 'Downsloping'}
            ], 'default': 1, 'group': 'Stress Test'},
            {'key': 'ca', 'label': 'Major Vessels Colored (Fluoroscopy)', 'unit': 'vessels', 'min': 0, 'max': 3, 'step': 1, 'default': 0, 'group': 'Vessels', 'normal': '0'},
            {'key': 'thal', 'label': 'Thalassemia / Stress Result', 'unit': '', 'type': 'select', 'options': [
                {'val': 3, 'label': 'Normal'},
                {'val': 6, 'label': 'Fixed Defect'},
                {'val': 7, 'label': 'Reversible Defect'}
            ], 'default': 3, 'group': 'Stress Test'}
        ],
        'presets': [
            {
                'id': 'heart_normal',
                'name': 'Healthy Sample',
                'badge': 'Low Risk',
                'badge_color': 'emerald',
                'data': {'age': 42, 'sex': 1, 'cp': 3, 'trestbps': 118, 'chol': 185, 'fbs': 0, 'restecg': 0, 'thalach': 172, 'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3}
            },
            {
                'id': 'heart_borderline',
                'name': 'Moderate Risk Sample',
                'badge': 'Moderate Risk',
                'badge_color': 'amber',
                'data': {'age': 57, 'sex': 0, 'cp': 2, 'trestbps': 142, 'chol': 255, 'fbs': 0, 'restecg': 1, 'thalach': 145, 'exang': 0, 'oldpeak': 1.2, 'slope': 2, 'ca': 1, 'thal': 3}
            },
            {
                'id': 'heart_high_risk',
                'name': 'High Risk Sample',
                'badge': 'High Risk',
                'badge_color': 'rose',
                'data': {'age': 67, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 286, 'fbs': 1, 'restecg': 2, 'thalach': 108, 'exang': 1, 'oldpeak': 2.6, 'slope': 2, 'ca': 3, 'thal': 7}
            }
        ]
    },
    'diabetes': {
        'name': 'Diabetes',
        'subheading': 'Predict the probability of diabetes using blood sugar, insulin, and BMI',
        'dataset_source': 'UCI Pima Indians Diabetes Dataset (768 patients)',
        'target_classes': {
            0: {'label': 'Non-Diabetic (Normal)', 'tier': 'Low Risk', 'color': 'emerald'},
            1: {'label': 'Diabetes Detected', 'tier': 'High Risk', 'color': 'rose'}
        },
        'features_meta': [
            {'key': 'pregnancies', 'label': 'Pregnancies', 'unit': '', 'min': 0, 'max': 17, 'step': 1, 'default': 1, 'group': 'History', 'normal': '0-3'},
            {'key': 'glucose', 'label': 'Glucose Level', 'unit': 'mg/dL', 'min': 50, 'max': 210, 'step': 1, 'default': 115, 'group': 'Blood Test', 'normal': '70-140'},
            {'key': 'blood_pressure', 'label': 'Blood Pressure', 'unit': 'mm Hg', 'min': 40, 'max': 130, 'step': 1, 'default': 72, 'group': 'Vitals', 'normal': '60-80'},
            {'key': 'skin_thickness', 'label': 'Skin Thickness', 'unit': 'mm', 'min': 8, 'max': 60, 'step': 1, 'default': 25, 'group': 'Body Measurements', 'normal': '15-30'},
            {'key': 'insulin', 'label': 'Insulin Level', 'unit': 'µU/mL', 'min': 15, 'max': 450, 'step': 1, 'default': 90, 'group': 'Blood Test', 'normal': '16-166'},
            {'key': 'bmi', 'label': 'BMI (Body Mass Index)', 'unit': 'kg/m²', 'min': 16.0, 'max': 55.0, 'step': 0.1, 'default': 26.5, 'group': 'Body Measurements', 'normal': '18.5-24.9'},
            {'key': 'pedigree_function', 'label': 'Diabetes Pedigree Function', 'unit': '', 'min': 0.08, 'max': 2.4, 'step': 0.01, 'default': 0.38, 'group': 'Genetics', 'normal': '< 0.5'},
            {'key': 'age', 'label': 'Age', 'unit': 'years', 'min': 21, 'max': 85, 'step': 1, 'default': 35, 'group': 'Demographics', 'normal': '21-65'}
        ],
        'presets': [
            {
                'id': 'diab_normal',
                'name': 'Healthy Sample',
                'badge': 'Normal',
                'badge_color': 'emerald',
                'data': {'pregnancies': 1, 'glucose': 88, 'blood_pressure': 66, 'skin_thickness': 20, 'insulin': 60, 'bmi': 22.4, 'pedigree_function': 0.24, 'age': 28}
            },
            {
                'id': 'diab_borderline',
                'name': 'Moderate Risk Sample',
                'badge': 'Moderate Risk',
                'badge_color': 'amber',
                'data': {'pregnancies': 3, 'glucose': 132, 'blood_pressure': 78, 'skin_thickness': 30, 'insulin': 140, 'bmi': 29.8, 'pedigree_function': 0.52, 'age': 45}
            },
            {
                'id': 'diab_high_risk',
                'name': 'High Risk Sample',
                'badge': 'High Risk',
                'badge_color': 'rose',
                'data': {'pregnancies': 6, 'glucose': 178, 'blood_pressure': 88, 'skin_thickness': 42, 'insulin': 235, 'bmi': 38.2, 'pedigree_function': 0.98, 'age': 54}
            }
        ]
    },
    'breast_cancer': {
        'name': 'Breast Cancer',
        'subheading': 'Predict whether a tumor biopsy is Benign or Malignant based on cell features',
        'dataset_source': 'UCI Wisconsin Diagnostic Breast Cancer (569 cases)',
        'target_classes': {
            0: {'label': 'Benign (Non-Cancerous)', 'tier': 'Low Risk', 'color': 'emerald'},
            1: {'label': 'Malignant (Cancerous)', 'tier': 'High Risk', 'color': 'rose'}
        },
        'features_meta': [
            {'key': 'radius_mean', 'label': 'Mean Nuclear Radius', 'unit': 'µm', 'min': 6.0, 'max': 29.0, 'step': 0.1, 'default': 13.5, 'group': 'Nuclear Dimensions', 'normal': '8-14'},
            {'key': 'texture_mean', 'label': 'Mean Texture (Gray-scale SD)', 'unit': 'score', 'min': 9.0, 'max': 39.0, 'step': 0.1, 'default': 18.0, 'group': 'Chromatin Texture', 'normal': '10-20'},
            {'key': 'perimeter_mean', 'label': 'Mean Nuclear Perimeter', 'unit': 'µm', 'min': 40.0, 'max': 190.0, 'step': 0.5, 'default': 88.0, 'group': 'Nuclear Dimensions', 'normal': '50-95'},
            {'key': 'area_mean', 'label': 'Mean Nuclear Area', 'unit': 'µm²', 'min': 140.0, 'max': 2500.0, 'step': 5.0, 'default': 560.0, 'group': 'Nuclear Dimensions', 'normal': '200-650'},
            {'key': 'smoothness_mean', 'label': 'Mean Smoothness (Radius variation)', 'unit': 'index', 'min': 0.05, 'max': 0.17, 'step': 0.005, 'default': 0.095, 'group': 'Contour Regularity', 'normal': '0.07-0.11'},
            {'key': 'compactness_mean', 'label': 'Mean Compactness (Perim²/Area - 1)', 'unit': 'index', 'min': 0.02, 'max': 0.35, 'step': 0.005, 'default': 0.09, 'group': 'Contour Regularity', 'normal': '0.03-0.12'},
            {'key': 'concavity_mean', 'label': 'Mean Concavity (Severity of indents)', 'unit': 'index', 'min': 0.0, 'max': 0.45, 'step': 0.005, 'default': 0.04, 'group': 'Nuclear Atypia', 'normal': '0.0-0.08'},
            {'key': 'concave_points_mean', 'label': 'Mean Concave Points (# indents)', 'unit': 'index', 'min': 0.0, 'max': 0.20, 'step': 0.002, 'default': 0.03, 'group': 'Nuclear Atypia', 'normal': '0.0-0.05'},
            {'key': 'symmetry_mean', 'label': 'Mean Symmetry', 'unit': 'index', 'min': 0.10, 'max': 0.31, 'step': 0.005, 'default': 0.18, 'group': 'Nuclear Morphology', 'normal': '0.14-0.22'},
            {'key': 'fractal_dimension_mean', 'label': 'Mean Fractal Dimension', 'unit': 'index', 'min': 0.04, 'max': 0.10, 'step': 0.002, 'default': 0.062, 'group': 'Nuclear Morphology', 'normal': '0.05-0.07'}
        ],
        'presets': [
            {
                'id': 'bc_benign',
                'name': 'Benign Sample',
                'badge': 'Low Risk',
                'badge_color': 'emerald',
                'data': {'radius_mean': 11.5, 'texture_mean': 14.8, 'perimeter_mean': 73.2, 'area_mean': 402.0, 'smoothness_mean': 0.086, 'compactness_mean': 0.058, 'concavity_mean': 0.021, 'concave_points_mean': 0.018, 'symmetry_mean': 0.165, 'fractal_dimension_mean': 0.059}
            },
            {
                'id': 'bc_atypical',
                'name': 'Borderline Sample',
                'badge': 'Moderate Risk',
                'badge_color': 'amber',
                'data': {'radius_mean': 14.8, 'texture_mean': 20.2, 'perimeter_mean': 96.5, 'area_mean': 685.0, 'smoothness_mean': 0.103, 'compactness_mean': 0.118, 'concavity_mean': 0.085, 'concave_points_mean': 0.052, 'symmetry_mean': 0.192, 'fractal_dimension_mean': 0.066}
            },
            {
                'id': 'bc_malignant',
                'name': 'Malignant Sample',
                'badge': 'High Risk',
                'badge_color': 'rose',
                'data': {'radius_mean': 19.8, 'texture_mean': 25.4, 'perimeter_mean': 132.0, 'area_mean': 1240.0, 'smoothness_mean': 0.118, 'compactness_mean': 0.215, 'concavity_mean': 0.245, 'concave_points_mean': 0.138, 'symmetry_mean': 0.235, 'fractal_dimension_mean': 0.074}
            }
        ]
    }
}

def generate_clinical_recommendations(disease_key, risk_percent, top_risk_factors):
    """Evidence-based clinical decision support guidance."""
    recs = []
    urgency = "Routine Monitoring"
    
    if disease_key == 'heart':
        if risk_percent >= 65:
            urgency = "Immediate Cardiology Referral Required"
            recs.append("Order urgent 12-lead Electrocardiogram (ECG) and High-Sensitivity Cardiac Troponin I (hs-cTnI).")
            recs.append("Schedule Transthoracic Echocardiogram (TTE) to assess left ventricular ejection fraction and regional wall motion.")
            recs.append("Consider non-invasive Coronary Computed Tomography Angiography (CCTA) or direct invasive angiogram.")
            recs.append("Review guideline-directed medical therapy (high-intensity Statin, ACE-inhibitor/ARB, Aspirin per ACC/AHA guidelines).")
        elif risk_percent >= 30:
            urgency = "Cardiovascular Secondary Prevention Workup"
            recs.append("Conduct Exercise Treadmill Stress Testing (Bruce Protocol) to evaluate ST-segment ischemic threshold.")
            recs.append("Perform 24-hour Ambulatory Blood Pressure Monitoring (ABPM) and comprehensive fasting lipid subfraction panel.")
            recs.append("Implement Mediterranean dietary regimen, sodium restriction (< 2,000 mg/day), and 150 min/wk aerobic conditioning.")
        else:
            urgency = "Standard Preventive Health Maintenance"
            recs.append("Annual blood pressure checkup and baseline fasting lipid profile every 3-5 years.")
            recs.append("Maintain optimal cardiovascular lifestyle: smoke-free status, balanced nutrition, and regular physical activity.")

    elif disease_key == 'diabetes':
        if risk_percent >= 65:
            urgency = "Endocrine Diagnostic Confirmation Needed"
            recs.append("Confirm with laboratory Glycated Hemoglobin (HbA1c) and repeat Fasting Plasma Glucose (FPG).")
            recs.append("Evaluate microvascular complications: Urine Albumin-to-Creatinine Ratio (UACR) and dilated retinal exam.")
            recs.append("Consider initiation of First-Line Metformin pharmacotherapy alongside lifestyle intervention (ADA Standards of Care).")
            recs.append("Referral to Certified Diabetes Care and Education Specialist (CDCES) for medical nutrition therapy.")
        elif risk_percent >= 30:
            urgency = "Pre-Diabetes Targeted Intervention"
            recs.append("Order 75g 2-hour Oral Glucose Tolerance Test (OGTT) and HbA1c screening.")
            recs.append("Enroll patient in structured National Diabetes Prevention Program (DPP) targeting 5-7% weight reduction.")
            recs.append("Re-assess metabolic biomarkers in 6 months; monitor home fasting glucose weekly.")
        else:
            urgency = "Normoglycemic Routine Surveillance"
            recs.append("Periodic blood glucose screening every 3 years per ADA asymptomatic adult guidelines.")
            recs.append("Sustain active lifestyle and low-glycemic Mediterranean or DASH dietary pattern.")

    elif disease_key == 'breast_cancer':
        if risk_percent >= 65:
            urgency = "Urgent Surgical Oncology / Breast Specialist Evaluation"
            recs.append("Perform diagnostic bilateral Digital Breast Tomosynthesis (3D Mammography) with targeted Ultrasound.")
            recs.append("Proceed to Image-Guided Core Needle Biopsy (CNB) for definitive histological grading and ER/PR/HER2 IHC staining.")
            recs.append("Order Breast MRI with contrast for precise extent-of-disease staging.")
            recs.append("Present case at Multidisciplinary Tumor Board (MDTB) consultation.")
        elif risk_percent >= 30:
            urgency = "Diagnostic Breast Imaging Follow-up (BI-RADS 3/4a)"
            recs.append("Obtain high-resolution supplemental Breast Ultrasound of the target quadrant.")
            recs.append("Short-interval (3 to 6 month) follow-up imaging to confirm morphological stability.")
            recs.append("Obtain comprehensive family pedigree to determine eligibility for BRCA1/BRCA2 genetic panel testing.")
        else:
            urgency = "Routine Age-Appropriate Mammographic Screening"
            recs.append("Continue routine annual or biennial screening mammography as indicated by age guidelines.")
            recs.append("Clinical breast examination during annual well-woman gynecologic visit.")

    return {
        'urgency': urgency,
        'recommendations': recs,
        'action_timeframe': 'Within 24-48 Hours' if risk_percent >= 65 else ('Within 2-4 Weeks' if risk_percent >= 30 else 'Routine Annual Checkup')
    }

def analyze_feature_contributions(disease_key, input_dict, metadata):
    """
    Computes individual patient feature impacts relative to healthy baseline population.
    Provides explainable AI insights showing why the risk score was adjusted.
    """
    stats = metadata[disease_key]['stats']
    contributions = []
    
    for key, val in input_dict.items():
        if key not in stats:
            continue
        feat_stat = stats[key]
        mean_h = feat_stat['healthy_mean']
        std_h = feat_stat['std'] if feat_stat['std'] > 0 else 1.0
        
        # Z-score relative to healthy population
        z_score = (val - mean_h) / std_h
        
        # Clinical directionality check
        impact_dir = 'neutral'
        impact_pct = round(abs(z_score) * 12, 1)
        
        if z_score > 0.4:
            impact_dir = 'elevating'
            desc = f"Value ({val}) is above healthy baseline (mean {mean_h:.1f})"
        elif z_score < -0.4:
            impact_dir = 'protective'
            desc = f"Value ({val}) is below healthy average (mean {mean_h:.1f})"
        else:
            impact_dir = 'neutral'
            desc = f"Value ({val}) is within expected baseline (mean {mean_h:.1f})"
            
        contributions.append({
            'feature': key,
            'value': val,
            'baseline_mean': round(mean_h, 2),
            'z_score': round(float(z_score), 2),
            'direction': impact_dir,
            'impact_magnitude': min(100.0, impact_pct),
            'description': desc
        })
        
    # Sort by absolute z-score descending
    contributions.sort(key=lambda x: abs(x['z_score']), reverse=True)
    return contributions

def predict_single_patient(disease_key, model_key, input_data):
    """
    Executes clinical prediction pipeline for a single patient record.
    Returns: probability, risk tier, diagnostic label, consensus breakdown, explainability, recommendations.
    """
    meta = get_metadata()
    if disease_key not in meta:
        raise ValueError(f"Unknown disease: {disease_key}")
        
    features = meta[disease_key]['features']
    
    # Extract feature vector in correct order
    vector = []
    clean_inputs = {}
    for f in features:
        if f not in input_data:
            # Default fallback to median if omitted
            clean_inputs[f] = meta[disease_key]['stats'][f]['median']
        else:
            clean_inputs[f] = float(input_data[f])
        vector.append(clean_inputs[f])
        
    df_vector = pd.DataFrame([vector], columns=features)
    
    # Run selected model
    model = load_model(disease_key, model_key)
    prob_disease = float(model.predict_proba(df_vector)[0][1])
    risk_pct = round(prob_disease * 100, 1)
    
    # Risk Stratification Tier
    if risk_pct >= 65.0:
        tier = 'Elevated Risk'
        color = 'rose'
        diag_code = 1
    elif risk_pct >= 30.0:
        tier = 'Moderate / Borderline'
        color = 'amber'
        diag_code = 1 if risk_pct >= 50.0 else 0
    else:
        tier = 'Low Risk'
        color = 'emerald'
        diag_code = 0
        
    class_info = CLINICAL_SPECS[disease_key]['target_classes'][1 if risk_pct >= 50.0 else 0]
    
    # Compute Multi-Model Consensus (SVM, LogReg, RF, XGBoost)
    all_models = ['logistic_regression', 'svm', 'random_forest', 'xgboost']
    consensus_list = []
    positive_votes = 0
    
    for m_key in all_models:
        m_obj = load_model(disease_key, m_key)
        m_prob = float(m_obj.predict_proba(df_vector)[0][1])
        m_pred = 1 if m_prob >= 0.50 else 0
        positive_votes += m_pred
        consensus_list.append({
            'model': m_key,
            'name': m_key.replace('_', ' ').title(),
            'probability': round(m_prob * 100, 1),
            'prediction': m_pred,
            'label': 'High Risk' if m_prob >= 0.50 else 'Low Risk'
        })
        
    consensus_summary = {
        'total_models': len(all_models),
        'positive_agreements': positive_votes,
        'negative_agreements': len(all_models) - positive_votes,
        'consensus_confidence': round(max(positive_votes, len(all_models) - positive_votes) / len(all_models) * 100, 1),
        'model_breakdowns': consensus_list
    }
    
    # Feature contributions for explanation
    contributions = analyze_feature_contributions(disease_key, clean_inputs, meta)
    
    # Clinical recommendations
    recs = generate_clinical_recommendations(disease_key, risk_pct, contributions[:3])
    
    return {
        'disease_key': disease_key,
        'disease_name': CLINICAL_SPECS[disease_key]['name'],
        'model_used': model_key,
        'risk_percentage': risk_pct,
        'risk_tier': tier,
        'tier_color': color,
        'diagnostic_label': class_info['label'],
        'clinical_urgency': recs['urgency'],
        'action_timeframe': recs['action_timeframe'],
        'recommendations': recs['recommendations'],
        'consensus': consensus_summary,
        'feature_contributions': contributions[:6],
        'patient_features': clean_inputs
    }

def batch_process_cohort(disease_key, model_key, df_records):
    """Processes uploaded CSV cohort with batch inference."""
    meta = get_metadata()
    features = meta[disease_key]['features']
    
    # Check if patient_id present, else generate
    has_pid = 'patient_id' in df_records.columns
    pids = df_records['patient_id'].tolist() if has_pid else [f"PT-{1000+i}" for i in range(len(df_records))]
    
    # Clean features
    X_batch = df_records[[c for c in features if c in df_records.columns]].copy()
    for f in features:
        if f not in X_batch.columns:
            X_batch[f] = meta[disease_key]['stats'][f]['median']
    X_batch = X_batch[features]
    
    model = load_model(disease_key, model_key)
    probabilities = model.predict_proba(X_batch)[:, 1]
    
    results = []
    high_risk_count = 0
    moderate_risk_count = 0
    low_risk_count = 0
    
    for i in range(len(X_batch)):
        prob = float(probabilities[i])
        pct = round(prob * 100, 1)
        if pct >= 65.0:
            tier = 'Elevated Risk'
            high_risk_count += 1
            action = 'Urgent Clinical Review'
        elif pct >= 30.0:
            tier = 'Moderate / Borderline'
            moderate_risk_count += 1
            action = 'Secondary Follow-up'
        else:
            tier = 'Low Risk'
            low_risk_count += 1
            action = 'Routine Surveillance'
            
        row_dict = {
            'patient_id': pids[i],
            'risk_percentage': pct,
            'tier': tier,
            'action': action
        }
        # Include top 3 raw values for summary table
        for col in features[:4]:
            row_dict[col] = df_records[col].iloc[i] if col in df_records.columns else X_batch[col].iloc[i]
        results.append(row_dict)
        
    return {
        'total_patients': len(results),
        'high_risk_count': high_risk_count,
        'moderate_risk_count': moderate_risk_count,
        'low_risk_count': low_risk_count,
        'high_risk_percentage': round(high_risk_count / len(results) * 100, 1) if results else 0,
        'results': results
    }
