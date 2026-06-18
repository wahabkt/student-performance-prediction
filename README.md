# 🎓 Student Performance Prediction System
**HITEC University Taxila — AI Lab Project**

A desktop ML application that predicts whether a student will **Pass or Fail**
based on study habits, attendance, and lifestyle factors.

---

## ⚙️ Setup (Do This Once)

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python generate_dataset.py
```

### 3. Train the Models
```bash
python train_model.py
```
This trains 3 models, picks the best, saves it, and generates all evaluation charts.

### 4. Launch the Application
```bash
python app.py
```

---

## 📁 Project Structure
```
student_performance/
├── app.py                  ← Main desktop application
├── train_model.py          ← ML training pipeline
├── generate_dataset.py     ← Dataset generator
├── requirements.txt        ← Python dependencies
├── data/
│   └── student_data.csv    ← Generated after step 2
├── model/
│   ├── best_model.pkl      ← Saved best model
│   ├── scaler.pkl          ← StandardScaler
│   ├── feature_names.pkl   ← Feature list
│   └── best_model_name.pkl ← Name of best model
└── assets/
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.png
    └── class_distribution.png
```

---

## 🤖 Models
| Model | Typical Accuracy |
|---|---|
| Logistic Regression | ~87% |
| SVM | ~85% |
| Random Forest | ~84% |

---

## 📊 Features Used
| Feature | Description |
|---|---|
| study_hours | Daily study hours (0–12) |
| attendance_% | Class attendance percentage |
| previous_grade | Previous exam grade (0–100) |
| sleep_hours | Daily sleep (3–10 hrs) |
| extra_activities | Participates in activities (0/1) |
| internet_access | Has internet at home (0/1) |
| parental_education | 0=None, 1=School, 2=Higher |
| absences | Number of absences |
| health_status | Self-rated health (1–5) |
| motivation_level | Self-rated motivation (1–5) |
