from flask import Flask, request, render_template_string
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model at startup
with open('impact_on_student.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML template embedded directly into app.py for single-file deployment
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Prediction Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <style>
        body { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; }
    </style>
</head>
<body class="py-10 px-4 font-sans text-gray-800">
    <div class="max-w-4xl mx-auto bg-white rounded-2xl shadow-xl overflow-hidden">
        
        <div class="bg-gradient-to-r header-gradient bg-indigo-600 px-8 py-6 text-white text-center">
            <h1 class="text-3xl font-bold tracking-tight">Predictive Analytics Dashboard</h1>
            <p class="text-indigo-100 mt-2">Enter the metrics below to generate a real-time prediction</p>
        </div>

        <div class="p-8">
            {% if prediction_text %}
            <div class="mb-8 p-6 rounded-xl border {% if '1' in prediction_text or 'Churn' in prediction_text or 'High' in prediction_text %} bg-red-50 border-red-200 text-red-900 {% else %} bg-green-50 border-green-200 text-green-900 {% endif %} text-center">
                <h3 class="text-lg font-semibold uppercase tracking-wider mb-1">Prediction Result</h3>
                <p class="text-2xl font-bold">{{ prediction_text }}</p>
            </div>
            {% endif %}

            <form action="/predict" method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Credit Score</label>
                    <input type="number" name="credit_score" value="600" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Country</label>
                    <select name="country" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                        <option value="0">France (0)</option>
                        <option value="1">Germany (1)</option>
                        <option value="2">Spain (2)</option>
                    </select>
                </div>

                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Gender</label>
                    <select name="gender" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                        <option value="1">Male (1)</option>
                        <option value="0">Female (0)</option>
                    </select>
                </div>

                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Age</label>
                    <input type="number" name="age" value="35" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Tenure (Years)</label>
                    <input type="number" name="tenure" value="5" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Balance ($)</label>
                    <input type="number" step="any" name="balance" value="0.0" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Number of Products</label>
                    <input type="number" name="products_number" value="1" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Has Credit Card?</label>
                    <select name="credit_card" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                        <option value="1">Yes (1)</option>
                        <option value="0">No (0)</option>
                    </select>
                </div>

                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Is Active Member?</label>
                    <select name="active_member" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                        <option value="1">Yes (1)</option>
                        <option value="0">No (0)</option>
                    </select>
                </div>

                <div>
                    <label class="block text-sm font-semibold text-gray-700 mb-1">Estimated Salary ($)</label>
                    <input type="number" step="any" name="estimated_salary" value="50000.0" required class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                </div>

                <div class="md:col-span-2 text-center mt-4">
                    <button type="submit" class="w-full md:w-auto px-8 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-lg shadow-lg transition-colors duration-200 cursor-pointer">
                        Run Prediction
                    </button>
                </div>
            </form>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features in the correct order mandated by the model
        feature_ordered_keys = [
            'credit_score', 'country', 'gender', 'age', 'tenure',
            'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary'
        ]
        
        # Build dictionary from input data
        input_data = {}
        for key in feature_ordered_keys:
            val = request.form.get(key)
            input_data[key] = [float(val)]
            
        # Convert into a DataFrame to ensure feature names match perfectly
        df_input = pd.DataFrame(input_data)
        
        # Execute prediction
        prediction = model.predict(df_input)[0]
        
        # Construct human-readable output
        output_text = f"Class {prediction}"
        
        return render_template_string(HTML_TEMPLATE, prediction_text=output_text)
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error processing input: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
