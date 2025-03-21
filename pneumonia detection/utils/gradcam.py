import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model

def generate_gradcam(model_instance, preprocessed_img, output_path, layer_name='conv5_block3_out'):
    """Generate Grad-CAM visualization for the given image.
    
    Args:
        model_instance: Instance of PneumoniaModel class
        preprocessed_img: Preprocessed image (should be same as used for prediction)
        output_path: Path to save the Grad-CAM visualization
        layer_name: Name of the layer to use for Grad-CAM (default: last conv layer of ResNet50)
        
    Returns:
        output_path: Path to the saved Grad-CAM visualization
    """
    # Get the model
    model = model_instance.get_model()
    
    # Ensure image has batch dimension
    if len(preprocessed_img.shape) == 3:
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(layer_name).output, 
            model.output[0]  # We use the class output for Grad-CAM
        ]
    )
    
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Cast inputs to float32
        inputs = tf.cast(preprocessed_img, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        
        # Get the predicted class index
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Gradient of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by corresponding gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # ReLU
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to match the original image size
    original_img = cv2.imread(output_path.replace('gradcam_', ''))
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * 0.4 + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Save the Grad-CAM visualization
    cv2.imwrite(output_path, superimposed_img)
    
    return output_path