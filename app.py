import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the trained Decision Tree model
MODEL_PATH = "Decision_Tree_model.pkl"
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form input
        age = float(request.form["Age"])
        gender = float(request.form["Gender"])
        region = float(request.form["Region"])
        occupation = float(request.form["Occupation"])
        income = float(request.form["Income"])

        # Arrange features into numpy array matching model training order
        features = np.array([[age, gender, region, occupation, income]])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Class Output: {prediction}",
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.get_json(force=True)
        # Expected keys: Age, Gender, Region, Occupation, Income
        features = np.array(
            [
                [
                    float(data["Age"]),
                    float(data["Gender"]),
                    float(data["Region"]),
                    float(data["Occupation"]),
                    float(data["Income"]),
                ]
            ]
        )

        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
