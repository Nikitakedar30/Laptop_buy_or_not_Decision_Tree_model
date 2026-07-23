import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained DecisionTree model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Decision_Tree_model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception:
    model = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Decision Tree Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: var(--text-main);
            padding: 20px;
            overflow-x: hidden;
        }

        .container {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 480px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: slideUp 0.8s ease-out forwards;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        h2 {
            font-size: 1.75rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            text-align: center;
            color: var(--text-sub);
            font-size: 0.9rem;
            margin-bottom: 30px;
        }

        .input-group {
            margin-bottom: 20px;
            animation: fadeIn 0.5s ease-out forwards;
            opacity: 0;
        }

        .input-group:nth-child(1) { animation-delay: 0.2s; }
        .input-group:nth-child(2) { animation-delay: 0.3s; }
        .input-group:nth-child(3) { animation-delay: 0.4s; }
        .input-group:nth-child(4) { animation-delay: 0.5s; }
        .input-group:nth-child(5) { animation-delay: 0.6s; }

        @keyframes fadeIn {
            to { opacity: 1; }
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input {
            width: 100%;
            padding: 12px 16px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 1rem;
            outline: none;
            transition: all 0.3s ease;
        }

        input:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
        }

        button {
            width: 100%;
            padding: 14px;
            margin-top: 10px;
            background: var(--accent-color);
            border: none;
            border-radius: 12px;
            color: #fff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        button:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        }

        button:active {
            transform: translateY(0);
        }

        #result {
            margin-top: 25px;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            display: none;
            animation: pulse 0.5s ease-in-out;
        }

        .result-success {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid #10b981;
            color: #34d399;
        }

        .result-error {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #ef4444;
            color: #f87171;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>ML Predictor</h2>
        <p class="subtitle">Enter feature values to generate prediction</p>
        
        <form id="predictForm">
            <div class="input-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="Age" placeholder="e.g., 30" required>
            </div>
            <div class="input-group">
                <label for="gender">Gender (Encoded)</label>
                <input type="number" id="gender" name="Gender" placeholder="e.g., 0 or 1" required>
            </div>
            <div class="input-group">
                <label for="region">Region (Encoded)</label>
                <input type="number" id="region" name="Region" placeholder="e.g., 1" required>
            </div>
            <div class="input-group">
                <label for="occupation">Occupation (Encoded)</label>
                <input type="number" id="occupation" name="Occupation" placeholder="e.g., 2" required>
            </div>
            <div class="input-group">
                <label for="income">Income</label>
                <input type="number" id="income" name="Income" placeholder="e.g., 50000" step="any" required>
            </div>
            <button type="submit">Predict Class</button>
        </form>

        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'none';

            const payload = {
                Age: parseFloat(document.getElementById('age').value),
                Gender: parseFloat(document.getElementById('gender').value),
                Region: parseFloat(document.getElementById('region').value),
                Occupation: parseFloat(document.getElementById('occupation').value),
                Income: parseFloat(document.getElementById('income').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    resultDiv.className = 'result-success';
                    resultDiv.innerText = `Prediction Class: ${data.prediction}`;
                } else {
                    resultDiv.className = 'result-error';
                    resultDiv.innerText = `Error: ${data.error}`;
                }
            } catch (err) {
                resultDiv.className = 'result-error';
                resultDiv.innerText = 'Server request failed.';
            }
            resultDiv.style.display = 'block';
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model pickle file not loaded'}), 500
    
    try:
        data = request.get_json(force=True)
        # Features ordered: ['Age', 'Gender', 'Region', 'Occupation', 'Income']
        features = [
            float(data['Age']),
            float(data['Gender']),
            float(data['Region']),
            float(data['Occupation']),
            float(data['Income'])
        ]
        
        prediction = model.predict([features])[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Expose app instance for Vercel WSGI environment
app = app
