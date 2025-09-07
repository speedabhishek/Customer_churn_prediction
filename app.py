from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model (dictionary wrapper)
with open("churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

# Extract scikit-learn model
if isinstance(model_data, dict):
    # Common keys: 'model', 'classifier', 'clf'
    if "model" in model_data:
        model = model_data["model"]
    elif "classifier" in model_data:
        model = model_data["classifier"]
    else:
        # take the first estimator object inside the dict
        model = next((v for v in model_data.values() if hasattr(v, "predict")), None)
else:
    model = model_data

if model is None:
    raise ValueError("❌ Could not find a valid model inside the pickle file.")

# Load encoders
with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        senior_citizen = int(request.form["SeniorCitizen"])
        tenure = float(request.form["tenure"])
        monthly_charges = float(request.form["MonthlyCharges"])
        total_charges = float(request.form["TotalCharges"])
        contract = request.form["Contract"]

        input_data = {
            "gender": "Male",
            "SeniorCitizen": senior_citizen,
            "Partner": "No",
            "Dependents": "No",
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        df = pd.DataFrame([input_data])

        # Apply encoders
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # ✅ Now this will work because `model` is an estimator
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] * 100

        result = f"Prediction: {'Churn' if prediction == 1 else 'Not Churn'} (Churn Probability: {probability:.2f}%)"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
