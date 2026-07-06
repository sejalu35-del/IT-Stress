import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv('data.csv')
print(data.columns)

# Drop unnecessary columns
X = data.drop(['label'], axis=1)
Y = data['label']

# Function to classify customer age
def AgeClassify(Age):
    if 18 <= Age < 40:
        return 1
    elif 40 <= Age < 60:
        return 2
    else:
        return 3

data['Age'] = data['Age'].apply(AgeClassify)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=13)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
print(f'Random Forest Accuracy: {accuracy_rf:.2f}%')

# Train XGBoost classifier
xgb_classifier = xgb.XGBClassifier(learning_rate=0.3, max_depth=7, n_estimators=200)
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb) * 100
print(f'XGBoost Accuracy: {accuracy_xgb:.2f}%')

# Save the trained XGBoost model
model_filename = "xgboost_model.pkl"
xgb_classifier.save_model(model_filename)
print(f"XGBoost model saved to {model_filename}")

# Train K-Nearest Neighbors classifier
kn_classifier = KNeighborsClassifier()
kn_classifier.fit(X_train, y_train)
y_pred_kn = kn_classifier.predict(X_test)
accuracy_kn = accuracy_score(y_test, y_pred_kn) * 100
print(f'K-Nearest Neighbors Accuracy: {accuracy_kn:.2f}%')

# Train Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt) * 100
print(f'Decision Tree Accuracy: {accuracy_dt:.2f}%')

def plot_bar_graph(stress_value):
    categories = ['Low', 'Moderate', 'High']
    values = [0, 0, 0]
    values[stress_value - 1] = 1  
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(categories, values, color=['green', 'yellow', 'red'])
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, 
                 f'{bar.get_height() * 100:.0f}%', 
                 ha='center', fontsize=12, color='black')
    
    plt.ylabel('Stress Level')
    plt.title('Stress Level Bar Graph')
    plt.ylim(0, 1.1)
    plt.show()

def predict_stress():
    # Get values from the text boxes
    person = int(input("Enter person: "))
    Heartrate = int(input("Enter heart rate: "))  
    skinconductivity = int(input("Enter skin conductivity: "))
    Hourate = float(input("Enter hr rate: "))
    Age = int(input("Enter Age: "))
    meetingsattended = int(input("Meetings attended: "))
    
    # Preprocess the input data
    input_data = pd.DataFrame({
        'person': [person],
        'Heartrate': [Heartrate],  
        'Skinconductivity': [skinconductivity], 
        'Hoursworked': [Hourate],   
        'Age': [Age],
        'meetingsattended': [meetingsattended]
    })
    
    # Make prediction
    prediction = xgb_classifier.predict(input_data)[0]
    
    # Display prediction and graph
    if prediction == 1:
        stress_level = 'High'
        stress_value = 3
    else:
        if Heartrate > 90 or skinconductivity > 90:
            stress_level = 'Moderate'
            stress_value = 2
        else:
            stress_level = 'Low'
            stress_value = 1

    print(f"Stress Level: {stress_level}")
    
   
    plot_bar_graph(stress_value)

predict_stress()