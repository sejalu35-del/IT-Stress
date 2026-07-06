import joblib
import numpy as np
import sqlite3
import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, flash
import os

app = Flask(__name__)
app.secret_key = 'xbsklaoe'

# ==========================
# Load trained model bundle
# ==========================
try:
    bundle = joblib.load("models/random_forest_bundle.pkl")  # updated path
    model = bundle["model"]
    encoders = bundle["encoders"]
    feature_names = bundle["features"]
    print("✅ Model, encoders & features loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Could not load model bundle: {e}")

# Detect categorical & numeric features
categorical_cols = list(encoders.keys())
numeric_cols = [c for c in feature_names if c not in categorical_cols]

# ==========================
# Database (SQLite)
# ==========================
DB_NAME = "stress_db.sqlite3"

def create_users_table():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

create_users_table()

# ==========================
# Routes
# ==========================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']

            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                        (name, email, password))
            conn.commit()
            conn.close()
            flash("Record Added Successfully", "success")
        except Exception as e:
            flash(f"Error Insert Operation: {str(e)}", "danger")
        return redirect(url_for("login"))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash("Username and password are required", "danger")
            return redirect(url_for('login'))

        try:
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE name=? AND password=?", (username, password))
            data = cur.fetchone()
            conn.close()

            if data:
                session["username"] = data[1]
                flash("Login Successful", "success")
                return redirect(url_for('checking'))
            else:
                flash("Incorrect username or password", "danger")
                return redirect(url_for('login'))
        except Exception as e:
            flash(f"Database Error: {str(e)}", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for('login'))

@app.route('/training')
def training():
    if 'username' not in session:
        flash("Please login to access this page", "info")
        return redirect(url_for('login'))
    return render_template('training.html')

# ==========================
# Plotting
# ==========================
def plot_bar_graph(prediction):
    categories = ['Not Stressed', 'Stressed']
    values = [0, 0]
    values[prediction] = 1

    plt.figure(figsize=(6, 4))
    bars = plt.bar(categories, values, color=['green', 'red'])

    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{bar.get_height() * 100:.0f}%',
                 ha='center', fontsize=12, color='black')

    plt.ylabel('Prediction')
    plt.title('Stress Prediction Result')
    plt.ylim(0, 1.1)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# ==========================
# Stress Prediction
# ==========================
@app.route('/checking', methods=['GET', 'POST'])
def checking():
    if 'username' not in session:
        flash("Please login to access this page", "info")
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            values = []
            for feature in feature_names:
                val = request.form.get(feature)

                if feature in categorical_cols:  # categorical
                    val = encoders[feature].transform([val])[0]
                else:  # numeric
                    val = float(val)

                values.append(val)

            input_data = np.array(values).reshape(1, -1)
            prediction = model.predict(input_data)[0]

            result_text = "Stressed 😟" if prediction == 1 else "Not Stressed 🙂"
            plot_url = plot_bar_graph(prediction)

            return redirect(url_for('result', result=result_text, plot_url=plot_url))
        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")

    return render_template('checking.html', feature_names=feature_names,
                           categorical_cols=categorical_cols,
                           numeric_cols=numeric_cols,
                           encoders=encoders)

@app.route('/result')
def result():
    result = request.args.get('result')
    plot_url = request.args.get('plot_url')
    return render_template('result.html', result=result, plot_url=plot_url)

# ==========================
# Run App
# ==========================
if __name__ == '__main__':
    app.run(debug=True)
