"""
train_model.py
Trains Logistic Regression, Random Forest, and SVM on student performance data.
Saves the best model and all evaluation plots.
Run: python train_model.py
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

warnings.filterwarnings("ignore")
os.makedirs("assets", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/student_data.csv")
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Pass rate: {df['result'].mean()*100:.1f}%\n")

X = df.drop("result", axis=1)
y = df["result"]
FEATURE_NAMES = X.columns.tolist()

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 3. Define models ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred  = model.predict(X_test_s)
    acc     = accuracy_score(y_test, y_pred)
    cv      = cross_val_score(model, X_train_s, y_train, cv=5).mean()
    results[name] = {
        "model": model, "y_pred": y_pred,
        "accuracy": acc, "cv_accuracy": cv,
        "report": classification_report(y_test, y_pred,
                                        target_names=["Fail","Pass"],
                                        output_dict=True)
    }
    print(f"{name:<22} Test Acc: {acc*100:.2f}%  CV Acc: {cv*100:.2f}%")

# ── 4. Pick best model ────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best      = results[best_name]
print(f"\n✔ Best model: {best_name} ({best['accuracy']*100:.2f}%)")

joblib.dump(best["model"], "model/best_model.pkl")
joblib.dump(scaler,        "model/scaler.pkl")
joblib.dump(FEATURE_NAMES, "model/feature_names.pkl")
joblib.dump(best_name,     "model/best_model_name.pkl")
print("Model saved to model/")

# ── 5. Plot: Model Comparison Bar Chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
names  = list(results.keys())
accs   = [results[n]["accuracy"]*100 for n in names]
colors = ["#4F8EF7" if n != best_name else "#22C55E" for n in names]
bars   = ax.bar(names, accs, color=colors, width=0.5, zorder=3)
ax.set_ylim(60, 100)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Model Comparison – Test Accuracy", fontsize=14, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
ax.set_axisbelow(True)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig("assets/model_comparison.png", dpi=150)
plt.close()

# ── 6. Plot: Confusion Matrix ─────────────────────────────────────────────────
cm   = confusion_matrix(y_test, best["y_pred"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail","Pass"], yticklabels=["Fail","Pass"],
            linewidths=1, linecolor="white", ax=ax, annot_kws={"size":16})
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Confusion Matrix – {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=150)
plt.close()

# ── 7. Plot: ROC Curve ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, res in results.items():
    proba    = res["model"].predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc  = auc(fpr, tpr)
    lw       = 2.5 if name == best_name else 1.5
    ax.plot(fpr, tpr, lw=lw, label=f"{name} (AUC={roc_auc:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves – All Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("assets/roc_curve.png", dpi=150)
plt.close()

# ── 8. Plot: Feature Importance (Random Forest) ───────────────────────────────
rf   = results["Random Forest"]["model"]
imp  = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
colors_f = ["#4F8EF7"] * len(imp)
colors_f[-1] = "#22C55E"
ax.barh(imp.index, imp.values, color=colors_f)
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("Feature Importance – Random Forest", fontsize=13, fontweight="bold")
ax.xaxis.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("assets/feature_importance.png", dpi=150)
plt.close()

# ── 9. Plot: Class Distribution ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
counts  = df["result"].value_counts()
ax.pie(counts, labels=["Pass","Fail"], autopct="%1.1f%%",
       colors=["#22C55E","#EF4444"], startangle=90,
       wedgeprops={"edgecolor":"white","linewidth":2})
ax.set_title("Dataset Class Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/class_distribution.png", dpi=150)
plt.close()

print("\nAll plots saved to assets/")
print("Training complete ✔")
