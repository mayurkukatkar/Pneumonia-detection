import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from utils.model import PneumoniaModel
from utils.preprocessing import preprocess_image
from utils.gradcam import generate_gradcam
from utils.report_generator import generate_report

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pneumonia-detection-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize the model
model = PneumoniaModel()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        preprocessed_img = preprocess_image(file_path)
        
        # Get predictions
        class_prediction, class_probabilities, severity_prediction = model.predict(preprocessed_img)
        
        # Generate Grad-CAM visualization
        gradcam_path = generate_gradcam(model, preprocessed_img, os.path.join(app.config['UPLOAD_FOLDER'], f'gradcam_{filename}'))
        
        # Generate medical report
        report = generate_report(class_prediction, severity_prediction, class_probabilities)
        
        # Prepare results for display
        results = {
            'original_image': 'uploads/' + filename,
            'gradcam_image': 'uploads/gradcam_' + filename,
            'class_prediction': class_prediction,
            'class_probabilities': class_probabilities,
            'severity': severity_prediction,
            'report': report
        }
        
        return render_template('results.html', results=results)
    
    flash('File type not allowed')
    return redirect(request.url)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)