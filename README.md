# Comprehensive Machine Learning Pipeline on Heart Disease UCI Dataset

## 📌 Project Overview
This project builds a **full end-to-end ML pipeline** to predict heart disease using the **UCI Heart Disease Dataset**.  
It includes **data preprocessing, PCA, feature selection, supervised and unsupervised models, hyperparameter tuning, and deployment**.  
A bonus **Streamlit UI** is included for real-time predictions, with deployment via **Ngrok**.

---

## 🎯 Objectives
- Data cleaning, encoding, and scaling.
- Dimensionality reduction using PCA.
- Feature selection via RFE, Chi-Square, and Random Forest importance.
- Supervised classification with Logistic Regression, Decision Tree, Random Forest, SVM.
- Unsupervised learning with K-Means and Hierarchical Clustering.
- Model optimization using GridSearchCV / RandomizedSearchCV.
- Save the final pipeline (`final_model.pkl`) for reproducibility.
- [Bonus] Deploy Streamlit UI with Ngrok for live access.

---

## 🛠️ Tools & Libraries
- **Python**
- **Pandas, NumPy, Scikit-learn**
- **Matplotlib, Seaborn**
- **Streamlit** [Bonus]
- **Ngrok** [Bonus]
- **joblib / pickle** for saving models

---

# ❤️ Heart Disease Prediction App

A machine learning web application that predicts the likelihood of heart disease based on patient health metrics.

## 🔗 Live Demo

**🌐 Public URL:** https://mariko-uncalcareous-phototelegraphically.ngrok-free.dev

**📱 Access from anywhere:**
- Open the link in any browser
- Enter patient data  
- Get instant heart disease risk prediction
- See interactive risk gauge

⚠️ **Note:** This is a demo link that may expire after some time.

---

## 📂 File Structure
Heart_Disease_Project/
│── data/

│ ├── heart_disease.csv

│── notebooks/

│ ├── 01_data_preprocessing.ipynb

│ ├── 02_pca_analysis.ipynb

│ ├── 03_feature_selection.ipynb

│ ├── 04_supervised_learning.ipynb

│ ├── 05_unsupervised_learning.ipynb

│ ├── 06_hyperparameter_tuning.ipynb

│── models/

│ ├── final_model.pkl

│── ui/

│ ├── app.py # Streamlit UI

│── deployment/

│ ├── ngrok_setup.txt

│── results/

│ ├── evaluation_metrics.txt

│── requirements.txt

│── README.md

│── .gitignore

---

## ⚙️ Setup Instructions

### 2️⃣ Create a Virtual Environment
It’s recommended to use a virtual environment to keep dependencies isolated.

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```
**On Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```
### 3️⃣ Install Dependencies
Install all required libraries from requirements.txt:

```bash
pip install -r requirements.txt
```
### 4️⃣ Verify Installation
Run Python and check that key packages are available:

```bash
python
>>> import pandas, sklearn, streamlit, joblib
```
