
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib

# =========================================
# Load the data
# =========================================
data = pd.read_csv('data.csv')

# Age classification (career stage)
def AgeClassify(Age):
    if 18 <= Age < 40:
        return 1  # Young professional
    elif 40 <= Age < 60:
        return 2  # Mid-career
    else:
        return 3  # Senior professional

data['AgeClass'] = data['Age'].apply(AgeClassify)

# =========================================
# NEW FEATURE ENGINEERING
# =========================================
data['stress_level'] = (data['Heartrate'] * 0.6) + (data['Skinconductivity'] * 0.4)
data['burnout_risk'] = (data['Hoursworked'] * 0.7) + (data['meetingsattended'] * 0.3)
data['work_life_balance'] = data['Hoursworked'] / (data['meetingsattended'] + 1)
data['productivity_score'] = (1 / (1 + data['stress_level'])) * 100
data['engagement_score'] = data['meetingsattended'] / data['meetingsattended'].max()

# =========================================
# Prepare data
# =========================================
X = data.drop(['label', 'person'], axis=1)  # Drop ID + target
Y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=13)

# =========================================
# Train Models
# =========================================
# Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%')

# XGBoost
xgb_classifier = xgb.XGBClassifier(
    learning_rate=0.3,
    max_depth=7,
    n_estimators=200,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
print(f'XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb) * 100:.2f}%')

# KNN
kn_classifier = KNeighborsClassifier()
kn_classifier.fit(X_train, y_train)
y_pred_kn = kn_classifier.predict(X_test)
print(f'KNN Accuracy: {accuracy_score(y_test, y_pred_kn) * 100:.2f}%')

# Decision Tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
print(f'Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt) * 100:.2f}%')

# =========================================
# Confusion Matrices
# =========================================
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

ConfusionMatrixDisplay.from_estimator(rf_classifier, X_test, y_test, ax=axs[0, 0], cmap='Blues')
axs[0, 0].set_title('Random Forest')

ConfusionMatrixDisplay.from_estimator(xgb_classifier, X_test, y_test, ax=axs[0, 1], cmap='Blues')
axs[0, 1].set_title('XGBoost')

ConfusionMatrixDisplay.from_estimator(kn_classifier, X_test, y_test, ax=axs[1, 0], cmap='Blues')
axs[1, 0].set_title('KNN')

ConfusionMatrixDisplay.from_estimator(dt_classifier, X_test, y_test, ax=axs[1, 1], cmap='Blues')
axs[1, 1].set_title('Decision Tree')

plt.tight_layout()
plt.show()

# =========================================
# Feature Importances
# =========================================
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

xgb.plot_importance(xgb_classifier, ax=axs[0])
axs[0].set_title('XGBoost Feature Importances')

importances_rf = pd.Series(rf_classifier.feature_importances_, index=X.columns)
importances_rf.sort_values().plot(kind='barh', ax=axs[1])
axs[1].set_title('Random Forest Feature Importances')

plt.show()

# Print top features
print("\n===== Feature Importance (Random Forest) =====")
rf_top10 = importances_rf.sort_values(ascending=False).head(10)
print(rf_top10)

print("\n===== Feature Importance (XGBoost) =====")
xgb_importances = pd.Series(xgb_classifier.feature_importances_, index=X.columns)
xgb_top10 = xgb_importances.sort_values(ascending=False).head(10)
print(xgb_top10)

new_features = ['stress_level', 'burnout_risk', 'work_life_balance', 'productivity_score', 'engagement_score']
print("\n📌 Note: Mental health & productivity features influence check:")
for f in new_features:
    if f in rf_top10.index or f in xgb_top10.index:
        print(f"   → {f} is among the top predictors ✅")
    else:
        print(f"   → {f} has lower impact (not in top 10)")

# =========================================
# Save Models
# =========================================
joblib.dump(rf_classifier, "random_forest_model.pkl")
joblib.dump(dt_classifier, "decision_tree_model.pkl")
joblib.dump(kn_classifier, "knn_model.pkl")
xgb_classifier.save_model("xgboost_model.json")

print("\n✅ All trained models saved successfully!")
