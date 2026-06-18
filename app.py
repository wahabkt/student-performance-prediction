"""
app.py  –  Student Performance Prediction System
Desktop application built with CustomTkinter.
Run: python app.py
"""

import os, sys
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import joblib
import numpy as np

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
NAME_PATH   = os.path.join(BASE_DIR, "model", "best_model_name.pkl")
ASSETS_DIR  = os.path.join(BASE_DIR, "assets")


def load_model():
    model       = joblib.load(MODEL_PATH)
    scaler      = joblib.load(SCALER_PATH)
    model_name  = joblib.load(NAME_PATH)
    return model, scaler, model_name


# ── Helper: image panel ───────────────────────────────────────────────────────
def make_image_label(parent, filename, width=None, height=None):
    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(path):
        return ctk.CTkLabel(parent, text=f"[{filename} not found]")
    img = Image.open(path)
    if width and height:
        img = img.resize((width, height), Image.LANCZOS)
    elif width:
        ratio = width / img.width
        img   = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    lbl   = tk.Label(parent, image=photo, bg="#1a1a2e", bd=0)
    lbl.image = photo          # keep reference
    return lbl


# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Student Performance Prediction System")
        self.geometry("1100x720")
        self.minsize(900, 620)
        self.configure(fg_color="#0f0f23")

        try:
            self.model, self.scaler, self.model_name = load_model()
        except FileNotFoundError:
            messagebox.showerror("Error",
                "Model files not found.\nRun train_model.py first.")
            sys.exit(1)

        self._build_sidebar()
        self._build_main()
        self.show_predict()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=210, corner_radius=0,
                                    fg_color="#13132b")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo / title
        ctk.CTkLabel(self.sidebar, text="🎓",
                     font=ctk.CTkFont(size=40)).pack(pady=(30, 4))
        ctk.CTkLabel(self.sidebar, text="Student\nPerformance\nPredictor",
                     font=ctk.CTkFont(size=15, weight="bold"),
                     justify="center").pack(pady=(0, 30))

        self.nav_buttons = {}
        nav_items = [
            ("🔮  Predict",    self.show_predict),
            ("📊  Analytics",  self.show_analytics),
            ("ℹ️  About",      self.show_about),
        ]
        for label, cmd in nav_items:
            btn = ctk.CTkButton(
                self.sidebar, text=label, anchor="w",
                fg_color="transparent", hover_color="#2a2a4a",
                font=ctk.CTkFont(size=14), height=44,
                command=cmd
            )
            btn.pack(fill="x", padx=12, pady=4)
            self.nav_buttons[label] = btn

        # Model badge at bottom
        ctk.CTkLabel(self.sidebar, text="Active Model",
                     font=ctk.CTkFont(size=10),
                     text_color="#888").place(relx=0.5, rely=0.88, anchor="center")
        ctk.CTkLabel(self.sidebar, text=self.model_name,
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#4F8EF7",
                     wraplength=180, justify="center").place(
                         relx=0.5, rely=0.93, anchor="center")

    # ── Main content area ─────────────────────────────────────────────────────
    def _build_main(self):
        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color="#0f0f23")
        self.main.pack(side="left", fill="both", expand=True)

    def _clear_main(self):
        for w in self.main.winfo_children():
            w.destroy()

    # ── Page: Predict ─────────────────────────────────────────────────────────
    def show_predict(self):
        self._clear_main()

        scroll = ctk.CTkScrollableFrame(self.main, fg_color="#0f0f23")
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(scroll, text="Student Performance Prediction",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(scroll,
                     text="Fill in the student details below and click Predict.",
                     font=ctk.CTkFont(size=13), text_color="#aaa").pack(
                         anchor="w", pady=(2, 18))

        # ── Input grid ────────────────────────────────────────────────────────
        grid = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=12)
        grid.pack(fill="x", pady=(0, 16))
        grid.columnconfigure((0, 1), weight=1)

        self.inputs = {}

        fields = [
            # (label, key, type, default, extra)
            ("Study Hours / Day",     "study_hours",       "slider", 5,   (0, 12)),
            ("Attendance (%)",        "attendance_%",      "slider", 75,  (30, 100)),
            ("Previous Grade",        "previous_grade",    "slider", 65,  (0, 100)),
            ("Sleep Hours / Day",     "sleep_hours",       "slider", 7,   (3, 10)),
            ("Extra Activities",      "extra_activities",  "option", 1,   ["Yes (1)", "No (0)"]),
            ("Internet Access",       "internet_access",   "option", 1,   ["Yes (1)", "No (0)"]),
            ("Parental Education",    "parental_education","option", 1,   ["None (0)","School (1)","Higher (2)"]),
            ("Absences",              "absences",          "slider", 3,   (0, 20)),
            ("Health Status (1–5)",   "health_status",     "slider", 3,   (1, 5)),
            ("Motivation Level (1–5)","motivation_level",  "slider", 3,   (1, 5)),
        ]

        for i, (label, key, ftype, default, extra) in enumerate(fields):
            row, col = divmod(i, 2)
            cell = ctk.CTkFrame(grid, fg_color="transparent")
            cell.grid(row=row, column=col, padx=20, pady=12, sticky="ew")

            ctk.CTkLabel(cell, text=label,
                         font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w")

            if ftype == "slider":
                lo, hi = extra
                var = ctk.DoubleVar(value=default)
                val_lbl = ctk.CTkLabel(cell, textvariable=var,
                                       font=ctk.CTkFont(size=12),
                                       text_color="#4F8EF7")
                val_lbl.pack(anchor="e")

                def make_cmd(v=var, l=val_lbl, is_int=(key in
                        ["absences","health_status","motivation_level",
                         "extra_activities","internet_access","parental_education"])):
                    def cmd(val):
                        v.set(round(float(val), 1) if not is_int else int(float(val)))
                    return cmd

                sl = ctk.CTkSlider(cell, from_=lo, to=hi, variable=var,
                                   command=make_cmd(),
                                   button_color="#4F8EF7",
                                   progress_color="#4F8EF7")
                sl.pack(fill="x")
                self.inputs[key] = var

            elif ftype == "option":
                var = ctk.StringVar(value=extra[default])
                opt = ctk.CTkOptionMenu(cell, variable=var, values=extra,
                                        fg_color="#2a2a4a",
                                        button_color="#4F8EF7")
                opt.pack(fill="x", pady=(4, 0))
                self.inputs[key] = var

        # ── Result area ───────────────────────────────────────────────────────
        self.result_frame = ctk.CTkFrame(scroll, fg_color="#1a1a2e",
                                         corner_radius=12)
        self.result_frame.pack(fill="x", pady=12)

        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Result will appear here after prediction.",
            font=ctk.CTkFont(size=15), text_color="#888")
        self.result_label.pack(pady=24)

        # ── Predict button ────────────────────────────────────────────────────
        ctk.CTkButton(
            scroll, text="🔮  Predict Performance",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=52, corner_radius=10,
            fg_color="#4F8EF7", hover_color="#3b7de8",
            command=self._predict
        ).pack(fill="x", pady=(8, 0))

    def _predict(self):
        try:
            raw = {}
            for key, var in self.inputs.items():
                val = var.get()
                if isinstance(val, str):
                    # extract integer from option strings like "Yes (1)"
                    raw[key] = int(val.split("(")[1].rstrip(")"))
                else:
                    raw[key] = float(val)

            feature_order = [
                "study_hours", "attendance_%", "previous_grade",
                "sleep_hours", "extra_activities", "internet_access",
                "parental_education", "absences", "health_status",
                "motivation_level"
            ]
            X = np.array([[raw[f] for f in feature_order]])
            X_s = self.scaler.transform(X)
            pred = self.model.predict(X_s)[0]
            prob = self.model.predict_proba(X_s)[0]
            conf = prob[pred] * 100

            # Clear old result
            for w in self.result_frame.winfo_children():
                w.destroy()

            if pred == 1:
                emoji, label, color = "✅", "PASS", "#22C55E"
                tip = "Great profile! Keep up the consistent study habits."
            else:
                emoji, label, color = "❌", "FAIL", "#EF4444"
                tip = "Consider increasing study hours and reducing absences."

            ctk.CTkLabel(self.result_frame,
                         text=f"{emoji}  Predicted Result: {label}",
                         font=ctk.CTkFont(size=20, weight="bold"),
                         text_color=color).pack(pady=(20, 6))
            ctk.CTkLabel(self.result_frame,
                         text=f"Confidence: {conf:.1f}%",
                         font=ctk.CTkFont(size=14),
                         text_color="#ccc").pack()

            # Confidence bar
            bar_frame = ctk.CTkFrame(self.result_frame, fg_color="#0f0f23",
                                     corner_radius=8, height=14)
            bar_frame.pack(fill="x", padx=30, pady=10)
            bar_frame.pack_propagate(False)
            fill = ctk.CTkFrame(bar_frame, fg_color=color,
                                corner_radius=8, height=14)
            fill.place(relwidth=conf / 100, relheight=1)

            ctk.CTkLabel(self.result_frame, text=tip,
                         font=ctk.CTkFont(size=12), text_color="#aaa",
                         wraplength=500).pack(pady=(4, 20))

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    # ── Page: Analytics ───────────────────────────────────────────────────────
    def show_analytics(self):
        self._clear_main()

        scroll = ctk.CTkScrollableFrame(self.main, fg_color="#0f0f23")
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(scroll, text="Model Analytics & Evaluation",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(scroll, text="Visual evaluation of all trained models.",
                     font=ctk.CTkFont(size=13), text_color="#aaa").pack(
                         anchor="w", pady=(2, 18))

        charts = [
            ("model_comparison.png",   "Accuracy Comparison – All Models"),
            ("confusion_matrix.png",   f"Confusion Matrix – {self.model_name}"),
            ("roc_curve.png",          "ROC Curves – All Models"),
            ("feature_importance.png", "Feature Importance – Random Forest"),
            ("class_distribution.png", "Dataset Class Distribution"),
        ]
        for filename, title in charts:
            card = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=12)
            card.pack(fill="x", pady=10)
            ctk.CTkLabel(card, text=title,
                         font=ctk.CTkFont(size=14, weight="bold")).pack(
                             anchor="w", padx=18, pady=(14, 6))
            lbl = make_image_label(card, filename, width=700)
            lbl.pack(padx=18, pady=(0, 14))

    # ── Page: About ───────────────────────────────────────────────────────────
    def show_about(self):
        self._clear_main()

        scroll = ctk.CTkScrollableFrame(self.main, fg_color="#0f0f23")
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(scroll, text="About This Project",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(scroll, text="HITEC University Taxila — AI Lab Project",
                     font=ctk.CTkFont(size=13), text_color="#aaa").pack(
                         anchor="w", pady=(2, 20))

        sections = [
            ("🎯 Objective",
             "Predict whether a student will pass or fail based on study habits, "
             "attendance, previous grades, and lifestyle factors using supervised "
             "Machine Learning algorithms."),
            ("📁 Dataset",
             "A synthetic dataset of 1,000 student records was generated with 10 "
             "features including study hours, attendance %, previous grade, sleep "
             "hours, extra activities, internet access, parental education level, "
             "absences, health status, and motivation level."),
            ("🤖 Models Trained",
             "Three classification models were trained and compared:\n"
             "  • Logistic Regression\n"
             "  • Random Forest Classifier (150 estimators)\n"
             "  • Support Vector Machine (SVM)\n\n"
             f"Best performing model: {self.model_name}"),
            ("📊 Evaluation Metrics",
             "Models were evaluated using:\n"
             "  • Test Set Accuracy\n"
             "  • 5-Fold Cross-Validation Accuracy\n"
             "  • Confusion Matrix\n"
             "  • ROC-AUC Curve\n"
             "  • Classification Report (Precision, Recall, F1)"),
            ("🔧 Tech Stack",
             "Python 3.x | scikit-learn | pandas | NumPy\n"
             "matplotlib | seaborn | CustomTkinter | Pillow | joblib"),
            ("🏫 Project Details",
             "Course: Artificial Intelligence Lab\n"
             "Department: Computer Science\n"
             "University: HITEC University Taxila\n"
             "Instructor: Ms. Urwa Farooq"),
        ]

        for title, body in sections:
            card = ctk.CTkFrame(scroll, fg_color="#1a1a2e", corner_radius=12)
            card.pack(fill="x", pady=8)
            ctk.CTkLabel(card, text=title,
                         font=ctk.CTkFont(size=15, weight="bold")).pack(
                             anchor="w", padx=18, pady=(14, 4))
            ctk.CTkLabel(card, text=body,
                         font=ctk.CTkFont(size=13), text_color="#ccc",
                         justify="left", wraplength=780).pack(
                             anchor="w", padx=18, pady=(0, 14))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
