import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

# =========================================
# Load Data
# =========================================
data = pd.read_csv("stress.csv")

print("\n🔎  First rows of data:")
print(data.head())

print("\n🔎 Missing values check:")
print(data.isnull().sum())

# Ensure target column exists
if "label" not in data.columns:
    raise ValueError("❌ Dataset must have a 'label' column!")

# Drop rows with missing labels
data = data.dropna(subset=["label"])

# =========================================
# Prepare Features (only original 12)
# =========================================
drop_cols = [c for c in ["label", "person"] if c in data.columns]
X = data.drop(columns=drop_cols, errors="ignore")
y = data["label"]

print(f"\n✅ Features being used: {list(X.columns)}")
print(f"✅ Total features: {X.shape[1]}")

# Encode categorical features
categorical_cols = X.select_dtypes(include=["object"]).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le  # Save encoder for GUI use later

# =========================================
# Train-test split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# Train Random Forest
# =========================================
rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# =========================================
# Results
# =========================================
acc = accuracy_score(y_test, y_pred) * 100
print(f"\n🌲 Random Forest Accuracy: {acc:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(rf_classifier, X_test, y_test, cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

# =========================================
# Feature Importances
# =========================================
importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", figsize=(8, 6), title="Feature Importances")
plt.show()

# =========================================
# Save Model + Encoders + Feature Names
# =========================================
os.makedirs("models", exist_ok=True)

bundle = {
    "model": rf_classifier,
    "encoders": encoders,
    "features": list(X.columns)  # keep exact order
}
joblib.dump(bundle, "models/random_forest_bundle.pkl")

print("\n✅ Model, encoders, and features saved in 'models/random_forest_bundle.pkl'!")
