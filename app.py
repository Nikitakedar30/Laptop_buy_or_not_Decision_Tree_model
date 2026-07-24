import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# Load the trained Decision Tree model
MODEL_PATH = "Decision_Tree_model.pkl"
model = joblib.load(MODEL_PATH)

# HTML template with embedded CSS styling
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Decision Tree Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: #f4f6f9;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .card {
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            width: 100%;
            max-width: 480px;
            padding: 35px 30px;
        }

        h2 {
            color: #2c3e50;
            margin-bottom: 8px;
            text-align: center;
            font-size: 24px;
        }

        p.subtitle {
            color: #7f8c8d;
            text-align: center;
            font-size: 14px;
            margin-bottom: 25px;
        }

        .form-group {
            margin-bottom: 18px;
        }

        label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            color: #34495e;
            margin-bottom: 6px;
        }

        input[type="number"] {
            width: 100%;
            padding: 12px 14px;
            border: 1.5px solid #dcdfe6;
            border-radius: 6px;
            font-size: 15px;
            color: #2c3e50;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            outline: none;
        }

        input[type="number"]:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.15);
        }

        button {
            width: 100%;
            padding: 14px;
            background-color: #3498db;
            border: none;
            border-radius: 6px;
            color: #ffffff;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
            transition: background-color 0.2s ease;
        }

        button:hover {
            background-color: #2980b9;
        }

        .result-box {
            margin-top: 25px;
            padding: 16px;
            background-color: #e8f4fc;
            border-left: 4px solid #3498db;
            border-radius: 4px;
            text-align: center;
            color: #1a5276;
            font-size: 16px;
            font-weight: 600;
        }

        .error-box {
            background-color: #fceae8;
            border-left-color: #e74c3c;
            color: #922b21;
        }
    </style>
</head>
<body>

<div class="card">
    <h2>Model Predictor</h2>
    <p class="subtitle">Enter feature details to run the Decision Tree model</p>

    <form action="/predict" method="post">
        <div class="form-group">
            <label for="Age">Age</label>
            <input type="number" step="any" id="Age" name="Age" placeholder="e.g. 35" required>
        </div>

        <div class="form-group">
            <label for="Gender">Gender (Encoded)</label>
            <input type="number" step="any" id="Gender" name="Gender" placeholder="e.g. 0 or 1" required>
        </div>

        <div class="form-group">
            <label for="Region">Region (Encoded)</label>
            <input type="number" step="any" id="Region" name="Region" placeholder="e.g. 1" required>
        </div>

        <div class="form-group">
            <label for="Occupation">Occupation (Encoded)</label>
            <input type="number" step="any" id="Occupation" name="Occupation" placeholder="e.g. 2" required>
        </div>

        <div class="form-group">
            <label for="Income">Income</label>
            <input type="number" step="any" id="Income" name="Income" placeholder="e.g. 50000" required>
        </div>

        <button type="submit">Predict</button>
    </form>

    {% if prediction_text %}
        <div class="result-box {% if 'Error' in prediction_text %}error-box{% endif %}">
            {{ prediction_text }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_LAYOUT)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from incoming form request
        age = float(request.form["Age"])
        gender = float(request.form["Gender"])
        region = float(request.form["Region"])
        occupation = float(request.form["Occupation"])
        income = float(request.form["Income"])

        # Create input array matching feature input order: Age, Gender, Region, Occupation, Income
        features = np.array([[age, gender, region, occupation, income]])

        # Execute prediction
        prediction = model.predict(features)[0]

        return render_template_string(
            HTML_LAYOUT,
            prediction_text=f"Predicted Class Output: {prediction}",
        )
    except Exception as e:
        return render_template_string(
            HTML_LAYOUT, prediction_text=f"Error: {str(e)}"
        )


@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.get_json(force=True)
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
