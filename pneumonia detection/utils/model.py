import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np

class PneumoniaModel:
    def __init__(self, model_path=None):
        """Initialize the pneumonia detection model.
        
        Args:
            model_path: Path to saved model weights. If None, uses a pre-trained model.
        """
        self.img_size = (224, 224)  # ResNet50 default input size
        self.class_names = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
        self.severity_levels = ['Mild', 'Moderate', 'Severe']
        
        # Create the base model from pre-trained ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add classification head for pneumonia detection
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Classification output (3 classes: Normal, Bacterial Pneumonia, Viral Pneumonia)
        class_output = Dense(len(self.class_names), activation='softmax', name='class_output')(x)
        
        # Severity output (3 levels: Mild, Moderate, Severe)
        severity_output = Dense(len(self.severity_levels), activation='softmax', name='severity_output')(x)
        
        # Create the final model with two outputs
        self.model = Model(inputs=base_model.input, outputs=[class_output, severity_output])
        
        # If a model path is provided, load the weights
        if model_path:
            self.model.load_weights(model_path)
        else:
            # For demonstration purposes, we'll use the pre-trained model
            # In a real scenario, you would fine-tune this model on pneumonia data
            print("Using pre-trained model. For production, fine-tune on pneumonia dataset.")
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss={
                'class_output': 'categorical_crossentropy',
                'severity_output': 'categorical_crossentropy'
            },
            metrics={
                'class_output': 'accuracy',
                'severity_output': 'accuracy'
            }
        )
    
    def predict(self, preprocessed_img):
        """Make predictions on a preprocessed image.
        
        Args:
            preprocessed_img: A preprocessed image ready for model input
            
        Returns:
            class_prediction: Predicted class (Normal, Bacterial, Viral)
            class_probabilities: Probability for each class
            severity_prediction: Predicted severity level (Mild, Moderate, Severe)
        """
        # Ensure image has batch dimension
        if len(preprocessed_img.shape) == 3:
            preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        
        # Get model predictions
        class_preds, severity_preds = self.model.predict(preprocessed_img)
        
        # Get the predicted class and severity
        class_idx = np.argmax(class_preds[0])
        severity_idx = np.argmax(severity_preds[0])
        
        class_prediction = self.class_names[class_idx]
        severity_prediction = self.severity_levels[severity_idx]
        
        # Format probabilities as percentages
        class_probabilities = {}
        for i, class_name in enumerate(self.class_names):
            class_probabilities[class_name] = float(class_preds[0][i] * 100)
        
        return class_prediction, class_probabilities, severity_prediction
    
    def get_model(self):
        """Return the Keras model for external use (e.g., Grad-CAM)."""
        return self.model