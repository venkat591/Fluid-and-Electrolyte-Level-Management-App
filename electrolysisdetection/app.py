from flask import Flask, request, render_template
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import matplotlib
matplotlib.use('Agg')  

app = Flask(__name__)

# Load the model, scaler, and label encoders using joblib
MODEL_PATH = 'multi_target_rf_model.joblib'
SCALER_PATH = 'scaler.joblib'
ENCODERS_PATH = 'label_encoders.joblib'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
except Exception as e:
    print(f"Error loading model components: {e}")
    model = None

# Create folder for graphs if not exists
os.makedirs("static/graphs", exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model is not loaded. Please check the server logs.", 500

    try:
        # Get input values from form
        name = request.form['name']
        age = float(request.form['age'])
        gender = request.form['gender'].strip().lower()  # Ensure case-insensitivity
        if gender not in ['male', 'female']:
            return f"Invalid gender value: {gender}. Please select 'male' or 'female'.", 400

        gender_encoded = 1 if gender == 'male' else 0  # Map to numerical values
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        sodium = float(request.form['sodium'])
        potassium = float(request.form['potassium'])
        calcium = float(request.form['calcium'])
        magnesium = float(request.form['magnesium'])
        chloride = float(request.form['chloride'])

        # Prepare input for model
        input_features = np.array([[age, gender_encoded, weight, height, sodium, potassium, calcium, magnesium, chloride]])
        input_scaled = scaler.transform(input_features)

        # Predict
        predictions = model.predict(input_scaled)[0]
        status_labels = {}
        for i, key in enumerate(['Sodium Status', 'Potassium Status', 'Calcium Status', 'Magnesium Status', 'Chloride Status']):
            status_labels[key] = label_encoders[key].inverse_transform([int(predictions[i])])[0]

        # Generate graph
        # plt.figure(figsize=(10, 6))
        # electrolytes = list(status_labels.keys())
        # statuses = [status_labels[key] for key in electrolytes]
        # status_numeric = [label_encoders[key].transform([status_labels[key]])[0] for key in electrolytes]

        # plt.bar(electrolytes, status_numeric, color='skyblue')
        # plt.title(f'Predicted Electrolyte Statuses for {name}')
        # plt.xlabel('Electrolytes')
        # plt.ylabel('Status (Encoded)')
        # plt.tight_layout()

        # Generate radar chart for comparison
        required_levels = {
            'Sodium Status': 'Normal',  # Example target values
            'Potassium Status': 'Normal',
            'Calcium Status': 'Normal',
            'Magnesium Status': 'Normal',
            'Chloride Status': 'Normal'
        }

        # Map of actual levels based on predictions
        actual_levels = {
            'Sodium Status': status_labels['Sodium Status'],
            'Potassium Status': status_labels['Potassium Status'],
            'Calcium Status': status_labels['Calcium Status'],
            'Magnesium Status': status_labels['Magnesium Status'],
            'Chloride Status': status_labels['Chloride Status']
        }

        # Numeric values for plotting (e.g., encode statuses as numeric values for comparison)
        numeric_required = [label_encoders[key].transform([required_levels[key]])[0] for key in required_levels]
        numeric_actual = [label_encoders[key].transform([actual_levels[key]])[0] for key in actual_levels]

        # Radar chart for comparison
        labels = list(required_levels.keys())  # Electrolytes
        num_vars = len(labels)

        # Values for radar chart
        required_values = np.array(numeric_required)
        actual_values = np.array(numeric_actual)

        # Angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Make the plot circular
        required_values = np.concatenate((required_values, [required_values[0]]))
        actual_values = np.concatenate((actual_values, [actual_values[0]]))
        angles += angles[:1]

        # Create radar plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Plot the required and actual levels
        ax.plot(angles, required_values, label='Required', color='blue', linewidth=3)
        ax.fill(angles, required_values, color='blue', alpha=0.2)

        ax.plot(angles, actual_values, label='Actual', color='red', linewidth=3)
        ax.fill(angles, actual_values, color='red', alpha=0.2)

        # Add labels and title
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_title(f'Comparison of Required vs Actual Electrolyte Levels for {name}', size=15)

        # Display the legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))


        # Save graph
        # graph_filename = f"{name.replace(' ', '_')}_graph.png"
        # graph_path = os.path.join('static/graphs', graph_filename)
        # plt.savefig(graph_path)
        # plt.close()

        # Save graph
        graph_filename = f"{name.replace(' ', '_')}_radar_comparison_graph.png"
        graph_path = os.path.join('static/graphs', graph_filename)
        plt.tight_layout()
        plt.savefig(graph_path)
        plt.close()

        return render_template(
            'result.html',
            name=name,
            predictions=status_labels,
            graph_filename=graph_filename  # Pass the graph filename to the template
        )
    except KeyError as ke:
        return f"Missing form field: {str(ke)}", 400
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
