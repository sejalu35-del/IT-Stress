import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import joblib

# =========================
# Load model + encoders + features
# =========================
try:
    bundle = joblib.load("models/random_forest_bundle.pkl")
    model = bundle["model"]
    encoders = bundle["encoders"]
    feature_names = bundle["features"]
    print("✅ Model, encoders & features loaded successfully!")
except:
    messagebox.showerror("Error", "❌ Could not load model bundle!")
    exit()

# Detect categorical features
categorical_cols = list(encoders.keys())
numeric_cols = [c for c in feature_names if c not in categorical_cols]

# =========================
# Build GUI
# =========================
root = tk.Tk()
root.title("Stress Prediction - Random Forest Model")
root.geometry("550x700")

tk.Label(root, text="Enter Feature Values", font=("Arial", 11, "bold")).pack(pady=10)

form_frame = tk.Frame(root)
form_frame.pack(pady=10)

entries = {}

# Numeric fields
for feature in numeric_cols:
    row = tk.Frame(form_frame)
    label = tk.Label(row, text=f"{feature}:", width=20, anchor="w")
    entry = tk.Entry(row, width=20)
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    label.pack(side=tk.LEFT)
    entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    entries[feature] = entry

# Dropdowns for categorical fields
for feature in categorical_cols:
    row = tk.Frame(form_frame)
    label = tk.Label(row, text=f"{feature}:", width=20, anchor="w")
    combo = ttk.Combobox(row, values=list(encoders[feature].classes_), state="readonly", width=18)
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    label.pack(side=tk.LEFT)
    combo.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    entries[feature] = combo

# =========================
# Prediction function
# =========================
def predict():
    try:
        values = []
        for feature in feature_names:
            val = entries[feature].get()
            if feature in categorical_cols:
                val = encoders[feature].transform([val])[0]
            else:
                val = float(val)
            values.append(val)

        input_data = np.array(values).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        # Map prediction to human-readable text
        if prediction == 1:
            result_text = "Stressed 😟"
        else:
            result_text = "Not Stressed 🙂"

        messagebox.showinfo("Prediction", f"Predicted Stress Condition: {result_text}")

    except Exception as e:
        messagebox.showerror("Error", f"❌ {e}")

# =========================
# Button
# =========================
btn = tk.Button(root, text="Predict Stress Level", command=predict,
                bg="Green", fg="Black", font=("Arial", 12, "bold"))
btn.pack(pady=20)

root.mainloop()
