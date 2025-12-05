# model_utils.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image
import cv2

# Custom loss functions (from your notebooks)
def dice_loss(y_true, y_pred):
    """Dice loss for segmentation"""
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    """Combined BCE and Dice loss"""
    return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def load_model(model_path: str = "models/current_model.keras"):
    """Load U-Net model with custom objects"""
    custom_objects = {
        'dice_loss': dice_loss,
        'bce_dice_loss': bce_dice_loss
    }
    
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"✅ Model loaded from {model_path}")
    return model

def preprocess_image(image: Image.Image, size: int = 512) -> np.ndarray:
    """Preprocess image for model input"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize
    image = image.resize((size, size))
    
    # Normalize to [0, 1]
    img_array = np.array(image) / 255.0
    
    # Add batch and channel dimensions
    img_array = np.expand_dims(img_array, axis=0)  # Batch
    img_array = np.expand_dims(img_array, axis=-1)  # Channel
    
    return img_array

def predict_mask(model, image: Image.Image) -> tuple:
    """Generate prediction and calculate loss"""
    # Preprocess
    img_array = preprocess_image(image)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0]
    
    # Calculate pseudo-loss (uncertainty score)
    # High entropy = high uncertainty = hard case
    epsilon = 1e-7
    pred_clipped = np.clip(prediction, epsilon, 1 - epsilon)
    entropy = -pred_clipped * np.log(pred_clipped) - (1 - pred_clipped) * np.log(1 - pred_clipped)
    avg_entropy = np.mean(entropy)
    
    # Convert to PIL Image
    mask_vis = (prediction.squeeze() * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_vis)
    
    return mask_pil, float(avg_entropy)

def postprocess_mask(mask_dict, threshold: float = 0.5, min_area: int = 500) -> Image.Image:
    """Post-process mask from ImageEditor"""
    # Extract composite image from ImageEditor dict
    if isinstance(mask_dict, dict):
        if 'composite' in mask_dict and mask_dict['composite'] is not None:
            mask = mask_dict['composite']
        elif 'background' in mask_dict and mask_dict['background'] is not None:
            mask = mask_dict['background']
        else:
            # If layers exist, composite them
            if 'layers' in mask_dict and mask_dict['layers']:
                mask = mask_dict['layers'][0]
            else:
                return None
    else:
        mask = mask_dict
    
    # Convert to numpy
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    # Binarize
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    _, binary_mask = cv2.threshold(mask, int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255
    
    return Image.fromarray(filtered_mask)
