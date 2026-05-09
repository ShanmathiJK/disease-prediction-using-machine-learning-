from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your models
diabetes_model = pickle.load(open('models/diabetes_model.pkl', 'rb'))
heart_model = pickle.load(open('models/heart_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form/diabetes')
def diabetes_form():
    return render_template('diabetes.html')

@app.route('/form/heart')
def heart_form():
    return render_template('heart.html')

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        features = [float(request.form[key]) for key in request.form]
        prediction = diabetes_model.predict([features])[0]
        probability = diabetes_model.predict_proba([features])[0][1] * 100
        result = f"Diabetes: {'Positive' if prediction else 'Negative'} (Risk: {probability:.2f}%)"
        return render_template('result.html', result=result)
    except Exception as e:
        return str(e), 500

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        features = [float(request.form[key]) for key in request.form]
        prediction = heart_model.predict([features])[0]
        probability = heart_model.predict_proba([features])[0][1] * 100
        result = f"Heart Disease: {'Positive' if prediction else 'Negative'} (Risk: {probability:.2f}%)"
        return render_template('result.html', result=result)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
