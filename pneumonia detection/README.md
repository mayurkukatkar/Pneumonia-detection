# Pneumonia Detection System

A deep learning-based system that detects pneumonia from chest X-ray images using a pre-trained ResNet50 model. The system classifies images into Normal, Bacterial Pneumonia, or Viral Pneumonia, predicts severity levels (Mild, Moderate, Severe), and generates Grad-CAM heatmaps to highlight affected lung regions. Additionally, it provides an AI-powered medical report summarizing the diagnosis.

## Key Features

- **Multi-Class Classification**: Distinguishes between Normal, Bacterial Pneumonia, and Viral Pneumonia
- **Severity Detection**: Classifies severity as Mild, Moderate, or Severe
- **Grad-CAM Heatmap**: Highlights infection areas in the lung
- **AI-Powered Report Generation**: Summarizes findings in a medical report format

## Technologies Used

- TensorFlow and Keras for deep learning
- Pre-trained ResNet50 model with fine-tuning
- OpenCV for image processing
- Grad-CAM for visualization
- Flask for web application deployment

## Project Structure

```
pneumonia-detection/
├── app.py                  # Flask application
├── requirements.txt        # Project dependencies
├── static/                 # Static files for web app
│   ├── css/                # CSS styles
│   ├── js/                 # JavaScript files
│   └── uploads/            # Temporary storage for uploaded images
├── templates/              # HTML templates
├── models/                 # Saved model files
├── utils/                  # Utility functions
│   ├── gradcam.py          # Grad-CAM implementation
│   ├── model.py            # Model definition and training
│   ├── preprocessing.py    # Image preprocessing functions
│   └── report_generator.py # AI report generation
└── data/                   # Data directory (for development)
    ├── train/              # Training data
    ├── val/                # Validation data
    └── test/               # Test data
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`

## Usage

1. Access the web interface at `http://localhost:5000`
2. Upload a chest X-ray image
3. View the classification results, severity assessment, Grad-CAM visualization, and AI-generated medical report

