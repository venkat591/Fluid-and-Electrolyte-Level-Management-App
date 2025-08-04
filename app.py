from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained models and encoders
models_and_encoders = joblib.load(r'D:\Desktop\fluid\model\models_and_encoders.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions."""
    try:
        # Extract form inputs
        sodium = float(request.form['sodium'])
        potassium = float(request.form['potassium'])
        calcium = float(request.form['calcium'])
        magnesium = float(request.form['magnesium'])
        chloride = float(request.form['chloride'])

        # Prepare a dictionary of input data
        input_features = {
            'Sodium (Na+) (mEq/L)': sodium,
            'Potassium (K+) (mEq/L)': potassium,
            'Calcium (Ca2+) (mg/dL)': calcium,
            'Magnesium (Mg2+) (mg/dL)': magnesium,
            'Chloride (Cl-) (mEq/L)': chloride,
        }

        # Perform predictions for each target
        predictions = {}
        for target, feature in feature_target_map.items():
            model = models_and_encoders[f"{target}_model"]
            feature_value = np.array([[input_features[feature[0]]]])  # Match single-feature input
            pred = model.predict(feature_value)
            pred_label = models_and_encoders["label_encoders"][target].inverse_transform(pred)[0]
            predictions[target] = pred_label

        # Pass predictions back to the template
        return render_template('index.html', predictions=predictions)

    except Exception as e:
        return f"An error occurred: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True)
