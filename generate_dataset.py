"""
Generate a realistic Student Performance dataset for training.
Run this once to create student_data.csv
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

study_hours     = np.random.normal(5, 2, n).clip(0, 12)
attendance      = np.random.normal(75, 15, n).clip(30, 100)
prev_grade      = np.random.normal(65, 15, n).clip(20, 100)
sleep_hours     = np.random.normal(7, 1.5, n).clip(3, 10)
extra_activities= np.random.randint(0, 2, n)           # 0 or 1
internet_access = np.random.randint(0, 2, n)
parental_edu    = np.random.randint(0, 3, n)            # 0=none,1=school,2=higher
absences        = np.random.randint(0, 20, n)
health          = np.random.randint(1, 6, n)            # 1-5 scale
motivation      = np.random.randint(1, 6, n)

# Composite score to derive final grade
score = (
    study_hours * 3.5
    + attendance * 0.25
    + prev_grade * 0.35
    + sleep_hours * 1.2
    + extra_activities * 4
    + internet_access * 3
    + parental_edu * 2.5
    - absences * 1.5
    + health * 1.8
    + motivation * 2.2
    + np.random.normal(0, 8, n)
)

# Map score to pass/fail (0 = Fail, 1 = Pass)
threshold = np.percentile(score, 35)
final_result = (score >= threshold).astype(int)

df = pd.DataFrame({
    "study_hours":      np.round(study_hours, 1),
    "attendance_%":     np.round(attendance, 1),
    "previous_grade":   np.round(prev_grade, 1),
    "sleep_hours":      np.round(sleep_hours, 1),
    "extra_activities": extra_activities,
    "internet_access":  internet_access,
    "parental_education": parental_edu,
    "absences":         absences,
    "health_status":    health,
    "motivation_level": motivation,
    "result":           final_result          # 0=Fail, 1=Pass
})

df.to_csv("data/student_data.csv", index=False)
print(f"Dataset saved: {len(df)} rows, {df['result'].mean()*100:.1f}% pass rate")
