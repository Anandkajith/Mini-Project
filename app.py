from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
# Load the trained model
# with open('model\model1.pkl', 'rb') as file:
#     model = pickle.load(file)

model=joblib.load("model1.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        SPX = float(request.form['SPX'])
        USO = float(request.form['USO'])
        SLV = float(request.form['SLV'])
        EUR_USD = float(request.form['EUR/USD'])

        user_data = {
            'SPX': [SPX],
            'USO': [USO],
            'SLV': [SLV],
            'EUR/USD': [EUR_USD]
        }
        user_df = pd.DataFrame(user_data)

        predicted_labels = model.predict(user_df)

        return render_template('result.html', prediction=predicted_labels[0])

if __name__ == '__main__':
    app.run(debug=True)
