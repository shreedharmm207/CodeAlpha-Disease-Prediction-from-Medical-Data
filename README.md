# PULSE Diagnostics: Clinical Decision Support & Disease Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-SVM%20%7C%20LogReg%20%7C%20RF%20%7C%20XGBoost-emerald.svg)](https://scikit-learn.org/)
[![Benchmarks](https://img.shields.io/badge/Datasets-UCI%20Medical%20Repository-orange.svg)](https://archive.ics.uci.edu/)
[![Status](https://img.shields.io/badge/Clinical%20CDS-Production%20Grade-teal.svg)](#)

An enterprise-grade, end-to-end clinical decision support platform designed for disease risk stratification from structured medical data. Engineered specifically with high-readability hospital workstation ergonomics, avoiding generic "AI-generated" aesthetics in favor of a clinical-grade diagnostic interface.

---

## 📋 Task 4 Specification & Objectives

- **Primary Objective:** Predict disease likelihood from multi-dimensional structured patient data.
- **Approach:** Machine learning classification pipelines with cross-validation, probability calibration, and explainable AI risk factors.
- **Algorithms Implemented:**
  1. **Support Vector Machine (SVM):** RBF kernel with Platt scaling (`probability=True`).
  2. **Logistic Regression:** L2 regularized with standard feature scaling.
  3. **Random Forest:** Ensemble bagging with 120 decision trees and MDI feature importances.
  4. **XGBoost:** Extreme gradient boosted trees with log-loss optimization.
  5. **Ensemble Consensus:** Soft-voting meta-classifier aggregating all 4 architectures.
- **Datasets (UCI ML Repository):**
  - **Cardiovascular Disease:** UCI Cleveland Heart Disease (303 patients, 13 clinical biomarkers).
  - **Type 2 Diabetes Mellitus:** UCI Pima Indians Diabetes (768 patients, 8 physiological indicators).
  - **Breast Cancer Cytology:** UCI Wisconsin Diagnostic Breast Cancer (569 cases, 10 core nuclear morphometrics).

---

## 🏥 Key System Capabilities

1. **Multi-Modality Patient Diagnostic Screening:**
   - Seamless switching between Cardiology, Endocrinology, and Oncology.
   - Dual-synchronized input controls (interactive range sliders + numeric fields) with real-time biological normal/elevated reference ranges.
   - Pre-configured validated clinical patient profiles (Normal Baseline, Borderline Pre-clinical, Acute High-Risk Crisis).
2. **Explainable AI (XAI) Biomarker Deviation:**
   - Visual z-score breakdown showing how each patient's biomarker deviates from the healthy population mean.
   - Instant categorization into Risk-Elevating (+) or Risk-Protective (-) drivers.
3. **Multi-Model Diagnostic Consensus:**
   - Real-time side-by-side probability breakdown and agreement calculation across SVM, Logistic Regression, Random Forest, and XGBoost.
4. **Clinical Decision Support (CDS) Recommendations:**
   - Guideline-directed diagnostic next steps, confirmatory laboratory tests (e.g. Troponin I, HbA1c, Core Needle Biopsy), and clinical urgency timeline.
5. **Printable / Exportable Clinical PDF Reports:**
   - Instant generation of an authentic, printable medical diagnostic report using ReportLab with institutional header, risk score, vital signs table, and physician sign-off block.
6. **Cohort Batch Screening & CSV Triage:**
   - Drag-and-drop CSV batch upload for hospital cohorts with automated population triage metrics and downloadable classified CSV.
7. **Scientific Model Laboratory:**
   - Comparative evaluation matrices, interactive ROC curves, Confusion Matrix heatmaps, and global feature importance rankings.

---

## 📊 Benchmark Model Performance

### 1. Cardiovascular Heart Disease (UCI Cleveland)
| Algorithm | Test Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **90.16%** | 86.67% | **92.86%** | 87.88% | **89.66%** | **0.957** |
| **Ensemble (Voting)** | **88.52%** | 83.87% | **92.86%** | 84.85% | 88.14% | 0.950 |
| **Logistic Regression** | 86.89% | 83.87% | 92.86% | 81.82% | 88.14% | 0.951 |
| **SVM (RBF Kernel)** | 85.25% | 81.25% | 92.86% | 78.79% | 86.67% | 0.947 |
| **XGBoost** | 85.25% | 81.25% | 92.86% | 78.79% | 86.67% | 0.939 |

### 2. Type 2 Diabetes Mellitus (UCI Pima Indians)
| Algorithm | Test Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **75.97%** | 68.09% | **59.26%** | 85.00% | **63.37%** | **0.829** |
| **Support Vector Machine (SVM)** | 75.32% | 68.18% | 55.56% | 86.00% | 61.22% | 0.802 |
| **Random Forest** | 74.68% | 67.44% | 53.70% | 86.00% | 59.79% | 0.807 |
| **Ensemble (Voting)** | 72.73% | 62.79% | 50.00% | 85.00% | 55.67% | 0.823 |
| **Logistic Regression** | 70.78% | 58.70% | 50.00% | 82.00% | 54.00% | 0.813 |

### 3. Breast Cancer Biopsy (UCI Wisconsin WDBC)
| Algorithm | Test Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **96.49%** | **97.50%** | **92.86%** | **98.61%** | **95.12%** | 0.983 |
| **Ensemble (Voting)** | **95.61%** | 97.44% | 90.48% | 98.61% | 93.83% | **0.988** |
| **Random Forest** | 93.86% | 92.68% | 90.48% | 95.83% | 91.57% | 0.984 |
| **XGBoost** | 93.86% | 92.68% | 90.48% | 95.83% | 91.57% | 0.975 |
| **Logistic Regression** | 92.98% | 88.64% | 92.86% | 93.06% | 90.70% | 0.984 |

---

## 🚀 Installation & Execution

### 1. Requirements
Ensure Python 3.10+ is installed on your machine. All required packages are listed in `requirements.txt`.

### 2. Run Data Processing & Training (Pre-computed)
```bash
python src/data_loader.py
python src/train_all_models.py
```

### 3. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
**`http://127.0.0.1:5000`**

---

## 📡 REST API Documentation

### 1. Single Patient Prediction
- **Endpoint:** `POST /api/predict`
- **Payload:**
```json
{
  "disease": "heart",
  "model": "ensemble",
  "inputs": {
    "age": 63, "sex": 1, "cp": 4, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }
}
```

### 2. Batch Cohort Screening
- **Endpoint:** `POST /api/batch-predict`
- **Method:** `multipart/form-data`
- **Fields:** `file` (CSV), `disease` (e.g. `heart`), `model` (e.g. `ensemble`)

### 3. Generate Clinical PDF
- **Endpoint:** `POST /api/export-pdf`
- **Payload:** Prediction object output from `/api/predict`
- **Response:** PDF binary attachment (`Clinical_Diagnostic_Report.pdf`)

---

## 📚 Academic Citations
- **Heart Disease:** Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. Am J Cardiol.
- **Diabetes:** Smith, J.W., et al. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. SCAMC.
- **Breast Cancer:** Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). *Nuclear feature extraction for breast tumor diagnosis*. IS&T/SPIE.

