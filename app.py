from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)


import os

base_dir = os.path.dirname(__file__)

with open(os.path.join(base_dir, "models/preprocessor.pkl"), "rb") as f:
    preprocessor = pickle.load(f)

with open(os.path.join(base_dir, "models/classifier.pkl"), "rb") as f:
    classifier = pickle.load(f)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    row = pd.DataFrame([{
        "Gender":           data["gender"],
        "Seniorcitizen":    int(data["seniorCitizen"]),
        "Partner":          data["partner"],
        "Dependents":       data["dependents"],
        "Tenure":           float(data["tenure"]),
        "Phoneservice":     data["phoneService"],
        "Multiplelines":    data["multipleLines"],
        "Internetservice":  data["internetService"],
        "Onlinesecurity":   data["onlineSecurity"],
        "Onlinebackup":     data["onlineBackup"],
        "Deviceprotection": data["deviceProtection"],
        "Techsupport":      data["techSupport"],
        "Streamingtv":      data["streamingTV"],
        "Streamingmovies":  data["streamingMovies"],
        "Contract":         data["contract"],
        "Paperlessbilling": data["paperlessBilling"],
        "Paymentmethod":    data["paymentMethod"],
        "Monthlycharges":   float(data["monthlyCharges"]),
        "Totalcharges":     float(data["tenure"]) * float(data["monthlyCharges"]),
    }])

    X = preprocessor.transform(row)
    prediction  = classifier.predict(X)[0]
    probability = classifier.predict_proba(X)[0]

    return jsonify({
        "churn":       bool(prediction),
        "probability": round(float(probability[1]) * 100, 1),
        "accuracy":    82.0
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)