import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data.csv')
print(data.columns)

# Define the AgeClassify function
def AgeClassify(Age):
    if 18 <= Age < 40:
        return 1
    elif 40 <= Age < 60:
        return 2
    else:
        return 3

# Apply Age classification
data['Age'] = data['Age'].apply(AgeClassify)

# Separate features and target
X = data.drop(['label'], axis=1)
Y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=13)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
print(f'Random Forest Accuracy: {accuracy_rf}')

# Train XGBoost classifier
xgb_classifier = xgb.XGBClassifier(learning_rate=0.3, max_depth=7, n_estimators=200)
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb) * 100
print(f'XGBoost Accuracy: {accuracy_xgb}')

# Save the trained XGBoost model
model_filename = "xgboost_model.pkl"
xgb_classifier.save_model(model_filename)
print(f"XGBoost model saved to {model_filename}")

# Train K-Nearest Neighbors classifier
kn_classifier = KNeighborsClassifier()
kn_classifier.fit(X_train, y_train)
y_pred_kn = kn_classifier.predict(X_test)
accuracy_kn = accuracy_score(y_test, y_pred_kn) * 100
#print(f'K-Nearest Neighbors Accuracy: {accuracy_kn}')

# Train Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt) * 100
#print(f'Decision Tree Accuracy: {accuracy_dt}')



fig, axs = plt.subplots(2, 2, figsize=(5, 4))

# Plot confusion matrix for Random Forest
plot_confusion_matrix(rf_classifier, X_test, y_test, ax=axs[0, 0], cmap='Blues')
axs[0, 0].title.set_text('Random Forest Confusion Matrix')

# Plot confusion matrix for XGBoost
plot_confusion_matrix(xgb_classifier, X_test, y_test, ax=axs[0, 1], cmap='Blues')
axs[0, 1].title.set_text('XGBoost Confusion Matrix')

# Plot confusion matrix for K-Nearest Neighbors
plot_confusion_matrix(kn_classifier, X_test, y_test, ax=axs[1, 0], cmap='Blues')
axs[1, 0].title.set_text('K-Nearest Neighbors Confusion Matrix')

# Plot confusion matrix for Decision Tree
plot_confusion_matrix(dt_classifier, X_test, y_test, ax=axs[1, 1], cmap='Blues')
axs[1, 1].title.set_text('Decision Tree Confusion Matrix')

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
# Plot feature importances for XGBoost and Random Forest
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
xgb.plot_importance(xgb_classifier, ax=axs[0])
axs[0].title.set_text('XGBoost Feature Importances')
importances_rf = pd.Series(rf_classifier.feature_importances_, index=X.columns)
importances_rf.sort_values().plot(kind='barh', ax=axs[1])
axs[1].title.set_text('Random Forest Feature Importances')
plt.show()
