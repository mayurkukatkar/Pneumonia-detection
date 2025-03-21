import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess an image for the pneumonia detection model.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image (default: 224x224 for ResNet50)
        
    Returns:
        preprocessed_img: A preprocessed image ready for model input
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert from BGR to RGB (OpenCV loads images in BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the target size
    img = cv2.resize(img, target_size)
    
    # Convert to float32 and normalize
    img = img.astype(np.float32)
    
    # Apply ResNet50 preprocessing (mean subtraction, etc.)
    preprocessed_img = preprocess_input(img)
    
    return preprocessed_img

def normalize_image_for_display(img):
    """Normalize an image for display purposes.
    
    Args:
        img: Input image
        
    Returns:
        normalized_img: Normalized image suitable for display
    """
    # Ensure the image is in float format
    img = img.astype(np.float32)
    
    # Normalize to [0, 1] range
    img_min = np.min(img)
    img_max = np.max(img)
    
    if img_max > img_min:
        normalized_img = (img - img_min) / (img_max - img_min)
    else:
        normalized_img = img
    
    # Convert to uint8 for display
    normalized_img = (normalized_img * 255).astype(np.uint8)
    
    return normalized_img