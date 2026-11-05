"""
data_loader.py
Medical Data Ingestion, Cleaning, and Standardization Pipeline
Loads authentic UCI benchmarks:
  1. UCI Cleveland Heart Disease
  2. UCI Pima Indians Diabetes
  3. UCI Breast Cancer Wisconsin (Diagnostic)
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def clean_heart_data():
    raw_path = os.path.join(DATA_DIR, 'raw_heart.csv')
    cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv(raw_path, names=cols, na_values='?')
    
    # Impute missing clinical values (ca has 4 missing, thal has 2 missing)
    df['ca'] = df['ca'].fillna(df['ca'].median()).astype(int)
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0]).astype(int)
    
    # Binary classification: 0 = No heart disease, 1-4 = Presence of heart disease
    df['target'] = (df['target'] > 0).astype(int)
    
    # Ensure types
    int_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope', 'ca', 'thal', 'target']
    for c in int_cols:
        df[c] = df[c].astype(int)
    df['oldpeak'] = df['oldpeak'].astype(float)
    
    out_path = os.path.join(DATA_DIR, 'heart_clean.csv')
    df.to_csv(out_path, index=False)
    print(f"[OK] Heart dataset cleaned: {df.shape[0]} records, {df['target'].sum()} positive cases -> {out_path}")
    return df

def clean_diabetes_data():
    raw_path = os.path.join(DATA_DIR, 'raw_diabetes.csv')
    df = pd.read_csv(raw_path)
    
    # Standardize column naming
    rename_map = {
        'Pregnancies': 'pregnancies',
        'Glucose': 'glucose',
        'BloodPressure': 'blood_pressure',
        'SkinThickness': 'skin_thickness',
        'Insulin': 'insulin',
        'BMI': 'bmi',
        'DiabetesPedigreeFunction': 'pedigree_function',
        'Age': 'age',
        'Outcome': 'outcome'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Replace non-physiological zeros with medians
    zero_fields = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
    for col in zero_fields:
        median_val = df.loc[df[col] > 0, col].median()
        df[col] = df[col].replace(0, median_val)
        
    out_path = os.path.join(DATA_DIR, 'diabetes_clean.csv')
    df.to_csv(out_path, index=False)
    print(f"[OK] Diabetes dataset cleaned: {df.shape[0]} records, {df['outcome'].sum()} positive cases -> {out_path}")
    return df

def clean_breast_cancer_data():
    bc = load_breast_cancer(as_frame=True)
    df_raw = bc.frame
    
    # Select 10 core FNA (Fine Needle Aspirate) cell nucleus mean features
    mean_cols = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension'
    ]
    df = df_raw[mean_cols].copy()
    
    # Rename for clean API / DB access
    clean_names = {
        'mean radius': 'radius_mean',
        'mean texture': 'texture_mean',
        'mean perimeter': 'perimeter_mean',
        'mean area': 'area_mean',
        'mean smoothness': 'smoothness_mean',
        'mean compactness': 'compactness_mean',
        'mean concavity': 'concavity_mean',
        'mean concave points': 'concave_points_mean',
        'mean symmetry': 'symmetry_mean',
        'mean fractal dimension': 'fractal_dimension_mean'
    }
    df.rename(columns=clean_names, inplace=True)
    
    # In sklearn breast cancer: 0 is malignant, 1 is benign.
    # Map to diagnosis: 1 = Malignant (High Risk), 0 = Benign (Low Risk)
    df['diagnosis'] = (df_raw['target'] == 0).astype(int)
    
    out_path = os.path.join(DATA_DIR, 'breast_cancer_clean.csv')
    df.to_csv(out_path, index=False)
    print(f"[OK] Breast cancer dataset cleaned: {df.shape[0]} records, {df['diagnosis'].sum()} malignant -> {out_path}")
    return df

def generate_sample_batches(heart_df, diab_df, bc_df):
    """Generates sample batch CSVs for clinical cohort upload testing."""
    # Heart batch: 20 samples with patient IDs
    sample_heart = heart_df.drop(columns=['target']).sample(n=20, random_state=101).copy()
    sample_heart.insert(0, 'patient_id', [f'P-CRD-{1000+i}' for i in range(len(sample_heart))])
    sample_heart.to_csv(os.path.join(DATA_DIR, 'sample_heart_batch.csv'), index=False)
    
    # Diabetes batch: 20 samples
    sample_diab = diab_df.drop(columns=['outcome']).sample(n=20, random_state=102).copy()
    sample_diab.insert(0, 'patient_id', [f'P-DIA-{2000+i}' for i in range(len(sample_diab))])
    sample_diab.to_csv(os.path.join(DATA_DIR, 'sample_diabetes_batch.csv'), index=False)
    
    # Breast cancer batch: 20 samples
    sample_bc = bc_df.drop(columns=['diagnosis']).sample(n=20, random_state=103).copy()
    sample_bc.insert(0, 'patient_id', [f'P-ONC-{3000+i}' for i in range(len(sample_bc))])
    sample_bc.to_csv(os.path.join(DATA_DIR, 'sample_breast_cancer_batch.csv'), index=False)
    print("[OK] Sample batch cohort files created for screening tests.")

if __name__ == '__main__':
    print("Running medical dataset ingestion & cleaning pipeline...")
    h = clean_heart_data()
    d = clean_diabetes_data()
    b = clean_breast_cancer_data()
    generate_sample_batches(h, d, b)
    print("Data ingestion pipeline complete!")
