from flask import Flask, render_template, request, send_file, flash, redirect
import pandas as pd
import pickle
import os
from werkzeug.utils import secure_filename
from flask import Flask, send_from_directory
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'UPLOAD_FOLDER'
app.secret_key = "supersecretkey"

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set of required columns for the model excluding 'SuspiciousFlag'
REQUIRED_INPUT_COLUMNS = set(model.feature_names_in_) - {'SuspiciousFlag'}
@app.route('/help', methods=['GET', 'POST'])
def help():
    return render_template('help.html')
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Make a prediction on the uploaded file
            try:
                df = pd.read_csv(filepath)
                # Check if all required columns are present
                if not REQUIRED_INPUT_COLUMNS.issubset(set(df.columns)):
                    missing_columns = REQUIRED_INPUT_COLUMNS - set(df.columns)
                    flash(f"Missing required columns: {', '.join(missing_columns)}")
                    return redirect(request.url)

                # Validate data and make predictions
                df['Prediction'], df['Error'] = zip(*df.apply(lambda row: predict_fraud(row), axis=1))
                output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + filename)
                df.to_csv(output_filepath, index=False)
                return send_file(output_filepath, as_attachment=True)
            except Exception as e:
                flash(str(e))
                return redirect(request.url)

    return render_template('index.html')

def predict_fraud(row):
    try:
        # Ensure 'SuspiciousFlag' is in the row and set its value to 0
        row['SuspiciousFlag'] = 0
        
        # Convert the row to a DataFrame
        input_data = pd.DataFrame([row])
        
        # If 'Category' is in input_data, encode it
        if 'Category' in input_data.columns:
            # Load the label encoder
            with open('label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
                
            # Check if the category is in the trained classes
            trained_categories = set(label_encoder.classes_)
            input_data['Category'] = input_data['Category'].apply(lambda x: x if x in trained_categories else 'Other')
            
            # Transform the 'Category' column
            input_data['Category'] = label_encoder.transform(input_data['Category'])
        
        # Ensure the order of columns matches the order during the model training
        input_data = input_data[model.feature_names_in_]
        
        # Make a prediction
        prediction = model.predict(input_data)
        
        # Convert prediction to "fraudulent" or "non-fraudulent"
        prediction_label = "fraudulent" if prediction[0] == 1 else "non-fraudulent"
        return prediction_label, "Success"
    except ValueError as e:
        return None, f"Error in row: {row.name}. Details: {str(e)}"


@app.route('/download_sample')
def download_sample():
    current_dir = os.getcwd()
    return send_from_directory(directory=current_dir, path='sample.csv', as_attachment=True)
@app.route('/download', methods=['GET'])
def download_file():
    # Your file download logic here
    flash('File has been downloaded')
    return redirect('/')
if __name__ == "__main__":
    app.run(debug=True)
