"""
train_all_models.py
Trains and evaluates 4 core Machine Learning classifiers + Ensemble Consensus
for UCI Heart Disease, Pima Indians Diabetes, and Wisconsin Breast Cancer.

Algorithms:
  1. Logistic Regression
  2. Support Vector Machine (SVM)
  3. Random Forest
  4. XGBoost
  + Ensemble Soft Voting
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

DISEASE_CONFIG = {
    'heart': {
        'file': 'heart_clean.csv',
        'target_col': 'target',
        'name': 'Cardiovascular Heart Disease',
        'uci_ref': 'UCI Cleveland Heart Disease Study (Detrano et al.)',
        'positive_label': 'Heart Disease Detected',
        'negative_label': 'Normal / No Significant Stenosis'
    },
    'diabetes': {
        'file': 'diabetes_clean.csv',
        'target_col': 'outcome',
        'name': 'Type 2 Diabetes Mellitus',
        'uci_ref': 'UCI Pima Indians Diabetes Database (NIDDK)',
        'positive_label': 'High Risk of Diabetes',
        'negative_label': 'Normoglycemic / Non-Diabetic'
    },
    'breast_cancer': {
        'file': 'breast_cancer_clean.csv',
        'target_col': 'diagnosis',
        'name': 'Breast Cancer Biopsy Analysis',
        'uci_ref': 'UCI Wisconsin Diagnostic Breast Cancer (WDBC, Street et al.)',
        'positive_label': 'Malignant Neoplasm Detected',
        'negative_label': 'Benign Lesion'
    }
}

def train_disease_models(disease_key, config):
    print(f"\n==========================================")
    print(f"Training clinical models for: {config['name']}")
    print(f"==========================================")
    
    csv_path = os.path.join(DATA_DIR, config['file'])
    df = pd.read_csv(csv_path)
    
    target_col = config['target_col']
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()
    
    # Stratified Split (80% Train, 20% Holdout Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Store feature baseline statistics (mean, std, min, max, median) for explanation
    stats = {}
    for col in feature_names:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'healthy_mean': float(df[y == 0][col].mean()),
            'disease_mean': float(df[y == 1][col].mean())
        }
        
    # Define models
    model_factories = {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        'svm': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.5, gamma='scale', probability=True, random_state=42))
        ]),
        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_split=4, random_state=42))
        ]),
        'xgboost': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric='logloss', random_state=42))
        ])
    }
    
    results = {}
    fitted_models = {}
    
    for key, pipeline in model_factories.items():
        print(f"-> Fitting {key}...")
        pipeline.fit(X_train, y_train)
        fitted_models[key] = pipeline
        
        # Predictions & Probabilities
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Specificity: TN / (TN + FP)
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # 5-Fold Stratified Cross Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Sample ~25 curve points for smooth client-side JSON
        idx_step = max(1, len(fpr) // 25)
        roc_pts = [{'x': round(float(fpr[i]), 3), 'y': round(float(tpr[i]), 3)} for i in range(0, len(fpr), idx_step)]
        if roc_pts[-1]['x'] != 1.0 or roc_pts[-1]['y'] != 1.0:
            roc_pts.append({'x': 1.0, 'y': 1.0})
            
        # Feature importances
        clf_obj = pipeline.named_steps['clf']
        importances = {}
        if hasattr(clf_obj, 'feature_importances_'):
            raw_imp = clf_obj.feature_importances_
            total = sum(raw_imp) if sum(raw_imp) > 0 else 1.0
            for name, imp in zip(feature_names, raw_imp):
                importances[name] = round(float(imp / total) * 100, 2)
        elif hasattr(clf_obj, 'coef_'):
            raw_coef = np.abs(clf_obj.coef_[0])
            total = sum(raw_coef) if sum(raw_coef) > 0 else 1.0
            for name, coef in zip(feature_names, raw_coef):
                importances[name] = round(float(coef / total) * 100, 2)
        else:
            # For SVM, run permutation importance on test set
            perm = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42)
            raw_perm = np.maximum(0, perm.importances_mean)
            total = sum(raw_perm) if sum(raw_perm) > 0 else 1.0
            for name, imp in zip(feature_names, raw_perm):
                importances[name] = round(float(imp / total) * 100, 2)
                
        # Sort importances descending
        sorted_imp = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
        
        # Save individual model
        model_save_path = os.path.join(MODELS_DIR, f"{disease_key}_{key}.joblib")
        joblib.dump(pipeline, model_save_path)
        
        results[key] = {
            'accuracy': round(float(acc) * 100, 2),
            'precision': round(float(prec) * 100, 2),
            'recall': round(float(rec) * 100, 2),
            'specificity': round(float(specificity) * 100, 2),
            'f1_score': round(float(f1) * 100, 2),
            'roc_auc': round(float(auc), 3),
            'cv_roc_auc_mean': round(float(cv_scores.mean()), 3),
            'cv_roc_auc_std': round(float(cv_scores.std()), 3),
            'confusion_matrix': cm,
            'roc_curve': roc_pts,
            'feature_importances': sorted_imp
        }
        print(f"  [{key}] Accuracy: {results[key]['accuracy']}% | AUC: {results[key]['roc_auc']} | Recall: {results[key]['recall']}%")

    # Fit Ensemble Consensus (Voting Classifier)
    print(f"-> Fitting ensemble voting classifier...")
    ensemble_clf = VotingClassifier(
        estimators=[
            ('lr', fitted_models['logistic_regression']),
            ('svm', fitted_models['svm']),
            ('rf', fitted_models['random_forest']),
            ('xgb', fitted_models['xgboost'])
        ],
        voting='soft'
    )
    ensemble_clf.fit(X_train, y_train)
    ens_pred = ensemble_clf.predict(X_test)
    ens_prob = ensemble_clf.predict_proba(X_test)[:, 1]
    
    ens_acc = accuracy_score(y_test, ens_pred)
    ens_prec = precision_score(y_test, ens_pred, zero_division=0)
    ens_rec = recall_score(y_test, ens_pred, zero_division=0)
    ens_f1 = f1_score(y_test, ens_pred, zero_division=0)
    ens_auc = roc_auc_score(y_test, ens_prob)
    ens_cm = confusion_matrix(y_test, ens_pred).tolist()
    ens_tn, ens_fp, ens_fn, ens_tp = ens_cm[0][0], ens_cm[0][1], ens_cm[1][0], ens_cm[1][1]
    ens_spec = ens_tn / (ens_tn + ens_fp) if (ens_tn + ens_fp) > 0 else 0.0
    
    ens_fpr, ens_tpr, _ = roc_curve(y_test, ens_prob)
    idx_step = max(1, len(ens_fpr) // 25)
    ens_roc_pts = [{'x': round(float(ens_fpr[i]), 3), 'y': round(float(ens_tpr[i]), 3)} for i in range(0, len(ens_fpr), idx_step)]
    if ens_roc_pts[-1]['x'] != 1.0 or ens_roc_pts[-1]['y'] != 1.0:
        ens_roc_pts.append({'x': 1.0, 'y': 1.0})
        
    # Aggregate ensemble feature importance as average of base models
    ens_importances = {}
    for col in feature_names:
        avg_val = np.mean([results[m]['feature_importances'].get(col, 0) for m in ['logistic_regression', 'svm', 'random_forest', 'xgboost']])
        ens_importances[col] = round(float(avg_val), 2)
    ens_importances = dict(sorted(ens_importances.items(), key=lambda item: item[1], reverse=True))
    
    joblib.dump(ensemble_clf, os.path.join(MODELS_DIR, f"{disease_key}_ensemble.joblib"))
    
    results['ensemble'] = {
        'accuracy': round(float(ens_acc) * 100, 2),
        'precision': round(float(ens_prec) * 100, 2),
        'recall': round(float(ens_rec) * 100, 2),
        'specificity': round(float(ens_spec) * 100, 2),
        'f1_score': round(float(ens_f1) * 100, 2),
        'roc_auc': round(float(ens_auc), 3),
        'confusion_matrix': ens_cm,
        'roc_curve': ens_roc_pts,
        'feature_importances': ens_importances
    }
    print(f"  [ensemble] Accuracy: {results['ensemble']['accuracy']}% | AUC: {results['ensemble']['roc_auc']}")

    return {
        'config': config,
        'features': feature_names,
        'stats': stats,
        'models': results,
        'sample_count': len(df),
        'positive_count': int(df[target_col].sum()),
        'negative_count': int(len(df) - df[target_col].sum())
    }

def main():
    metadata = {}
    for disease_key, cfg in DISEASE_CONFIG.items():
        disease_results = train_disease_models(disease_key, cfg)
        metadata[disease_key] = disease_results
        
    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[SUCCESS] All models trained and serialized! Metadata saved to {meta_path}")

if __name__ == '__main__':
    main()
