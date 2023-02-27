from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'pregnancies': [int(request.form['pregnancies'])],
        'glucose': [int(request.form['glucose'])],
        'blood_pressure': [int(request.form['blood_pressure'])],
        'skin_thickness': [int(request.form['skin_thickness'])],
        'insulin': [int(request.form['insulin'])],
        'bmi': [float(request.form['bmi'])],
        'dpf': [float(request.form['dpf'])],
        'age': [int(request.form['age'])]
    }

    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(df, diabetes.target, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return render_template('index.html', prediction=predictions[0])

if __name__ == '_main_':
    app.run(debug=True)
