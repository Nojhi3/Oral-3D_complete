# app.py - WITH YOLO + CONFIDENCE + BOUNDING BOXES
import gradio as gr
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import database as db
import model_utils
import numpy as np
import cv2

# Initialize
db.init_database()
os.makedirs("data/corrections", exist_ok=True)
os.makedirs("data/original_images", exist_ok=True)

# Load UNet model
MODEL = model_utils.load_model("models/current_model.keras")

# Load YOLO model (if available)
try:
    from ultralytics import YOLO
    YOLO_MODEL = YOLO("models/yolo_dental.pt")  # Your trained YOLO model
    YOLO_ENABLED = True
except:
    YOLO_ENABLED = False
    print("⚠️ YOLO not available - install ultralytics or add model")

CURRENT_IMAGE_PATH = None
CURRENT_PRED_PATH = None
CURRENT_LOSS = None

def get_tooth_bounding_boxes(mask_binary, min_area=100):
    """Extract bounding boxes from segmentation mask"""
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_binary.astype(np.uint8), connectivity=8
    )
    
    boxes = []
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
            
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        boxes.append({
            'x': x, 'y': y, 'width': w, 'height': h,
            'area': stats[i, cv2.CC_STAT_AREA]
        })
    
    return boxes

def run_yolo_detection(image, threshold=0.5):
    """Run YOLO detection and return results"""
    if not YOLO_ENABLED:
        return []
    
    results = YOLO_MODEL(image, conf=threshold, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'class': result.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            })
    
    return detections

def draw_boxes_and_predictions(image, seg_boxes, yolo_detections):
    """Draw segmentation boxes and YOLO predictions with smaller black text"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load a smaller font
    try:
        # Smaller font size: 12 instead of 16
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            # Fallback for Windows
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
    
    # Draw segmentation boxes (green outline, black text)
    for idx, box in enumerate(seg_boxes):
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        
        # Green box
        draw.rectangle([x, y, x+w, y+h], outline="lime", width=2)
        
        # Smaller black text with white background for readability
        label = f"Tooth #{idx+1}"
        
        # Get text size for background
        bbox = draw.textbbox((x, y-18), label, font=font)
        
        # Draw white background rectangle for text
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill="white")
        
        # Draw black text on white background
        draw.text((x, y-18), label, fill="black", font=font)
    
    # Draw YOLO detections (red outline, black text with confidence)
    for det in yolo_detections:
        x1, y1, x2, y2 = det['bbox']
        
        # Red box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Smaller label with confidence
        label = f"{det['class']}: {det['confidence']:.2f}"
        
        # Get text size for background
        bbox = draw.textbbox((x1, y1-18), label, font=font)
        
        # Draw white background
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill="white")
        
        # Draw black text
        draw.text((x1, y1-18), label, fill="black", font=font)
    
    return img_draw


def predict_segmentation(xray_image):
    """Generate initial segmentation prediction with boxes"""
    global CURRENT_IMAGE_PATH, CURRENT_PRED_PATH, CURRENT_LOSS
    
    if xray_image is None:
        return None, None, None, "❌ Please upload an X-ray image first"
    
    # Generate UNet prediction
    pred_mask, loss_score = model_utils.predict_mask(MODEL, xray_image)
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CURRENT_IMAGE_PATH = f"data/original_images/img_{timestamp}.png"
    CURRENT_PRED_PATH = f"data/corrections/pred_{timestamp}.png"
    
    xray_image.save(CURRENT_IMAGE_PATH)
    pred_mask.save(CURRENT_PRED_PATH)
    CURRENT_LOSS = loss_score
    
    # Extract bounding boxes from segmentation
    mask_array = np.array(pred_mask)
    seg_boxes = get_tooth_bounding_boxes(mask_array > 127, min_area=100)
    
    # Run YOLO detection (if available)
    yolo_detections = run_yolo_detection(xray_image, threshold=0.5)
    
    # Create visualization with boxes
    vis_image = draw_boxes_and_predictions(xray_image, seg_boxes, yolo_detections)
    
    # Create status message
    status = f"""✅ Prediction generated successfully!

📊 **Segmentation Results:**
   • Uncertainty Score: {loss_score:.4f}
   • Teeth Detected (UNet): {len(seg_boxes)}
   {"⚠️ HIGH UNCERTAINTY - Priority case!" if loss_score > 0.3 else "✓ Low uncertainty"}

"""
    
    if YOLO_ENABLED:
        status += f"""🎯 **YOLO Detection Results:**
   • Total Detections: {len(yolo_detections)}
"""
        for det in yolo_detections:
            status += f"   • {det['class']}: {det['confidence']:.2%}\n"
    else:
        status += "⚠️ YOLO detection not available\n"
    
    status += """
📝 **Next Step:** Edit the mask in ImageEditor below
• Use white brush for teeth regions
• Use black brush for background
• Zoom with mouse wheel for precision
"""
    
    return pred_mask, vis_image, loss_score, status

def save_correction(editor_dict):
    """Save corrected mask to database"""
    global CURRENT_IMAGE_PATH, CURRENT_PRED_PATH, CURRENT_LOSS
    
    if CURRENT_IMAGE_PATH is None:
        return "❌ No prediction to save. Generate a prediction first."
    
    if editor_dict is None:
        return "❌ No edits made. Please edit the mask before saving."
    
    # Extract composite image from ImageEditor
    if isinstance(editor_dict, dict):
        mask_image = editor_dict.get("composite") or editor_dict.get("background")
    else:
        mask_image = editor_dict
    
    if mask_image is None:
        return "❌ Could not extract mask from editor."
    
    # Convert to PIL
    if isinstance(mask_image, np.ndarray):
        if len(mask_image.shape) == 3:
            mask_image = Image.fromarray(mask_image).convert('L')
        else:
            mask_image = Image.fromarray(mask_image)
    
    if mask_image.mode != 'L':
        mask_image = mask_image.convert('L')
    
    # Save corrected mask
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    corrected_path = f"data/corrections/corrected_{timestamp}.png"
    mask_image.save(corrected_path)
    
    # Save to database
    correction_id = db.save_correction(
        original_img_path=CURRENT_IMAGE_PATH,
        predicted_mask_path=CURRENT_PRED_PATH,
        corrected_mask_path=corrected_path,
        loss_score=CURRENT_LOSS,
        model_version="v1"
    )
    
    stats = db.get_stats()
    progress_bar = "█" * min(stats['pending'], 20) + "░" * max(0, 20 - stats['pending'])
    
    result = f"""✅ Correction saved! (ID: {correction_id})

📊 Progress: [{progress_bar}] {stats['pending']}/20
Total: {stats['total']} | Used: {stats['used']} | Pending: {stats['pending']}

{"🎯 READY! Run: python retraining.py" if stats['pending'] >= 20 else f"⚠️ Need {20-stats['pending']} more"}"""
    
    return result

def refresh_stats():
    stats = db.get_stats()
    progress_bar = "█" * min(stats['pending'], 20) + "░" * max(0, 20 - stats['pending'])
    percentage = (stats['pending'] / 20) * 100 if stats['pending'] <= 20 else 100
    
    return f"""📊 Active Learning Progress
[{progress_bar}] {stats['pending']}/20 ({percentage:.0f}%)

Total: {stats['total']} | Used: {stats['used']} | Avg Loss: {stats['avg_loss']:.4f}

{'🎯 Ready to retrain!' if stats['pending'] >= 20 else f'⚠️ Need {20-stats["pending"]} more'}"""

# Build Gradio Interface
with gr.Blocks(
    title="Tooth Segmentation Active Learning + YOLO",
    theme=gr.themes.Soft(),
    fill_height=True
) as demo:
    
    gr.Markdown("""
    # 🦷 Dental X-ray Segmentation with Active Learning + YOLO
    **AI-powered tooth detection with bounding boxes and confidence scores**
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload X-ray")
            input_xray = gr.Image(
                type="pil",
                label="Dental X-ray",
                image_mode="L",
                height=400
            )
            predict_btn = gr.Button("🔮 Generate Prediction + Boxes", variant="primary", size="lg")
            loss_display = gr.Number(label="🎯 Uncertainty Score", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### 🎯 Detection Results")
            visualization = gr.Image(
                label="Segmentation + YOLO Detection",
                height=500
            )
    
    status_output = gr.Textbox(
        label="📋 Detection Status",
        lines=12,
        interactive=False,
        show_copy_button=True
    )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### ✏️ Correct Segmentation Mask")
            
            mask_editor = gr.ImageEditor(
                type="pil",
                image_mode="L",
                crop_size="1:1",
                brush=gr.Brush(
                    default_size=20,
                    colors=["#FFFFFF", "#000000"],
                    default_color="#FFFFFF"
                ),
                eraser=gr.Eraser(default_size=25),
                label="ImageEditor (Zoom with mouse wheel)",
                height=600
            )
            
            save_btn = gr.Button("💾 Save Correction", variant="primary", size="lg")
            correction_result = gr.Textbox(label="📊 Save Result", lines=6, interactive=False)
            
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 Refresh Stats", size="lg")
            stats_display = gr.Textbox(label="📈 Statistics", lines=8, interactive=False)
    
    # Event handlers
    predict_btn.click(
        predict_segmentation,
        inputs=input_xray,
        outputs=[mask_editor, visualization, loss_display, status_output]
    )
    
    save_btn.click(
        save_correction,
        inputs=mask_editor,
        outputs=correction_result
    )
    
    refresh_btn.click(
        refresh_stats,
        outputs=stats_display
    )
    
    demo.load(refresh_stats, outputs=stats_display)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🦷 TOOTH SEGMENTATION + YOLO DETECTION")
    print("="*80)
    print(f"\n📦 Configuration:")
    print(f"   • UNet Model: Loaded")
    print(f"   • YOLO Detection: {'Enabled ✓' if YOLO_ENABLED else 'Disabled ✗'}")
    print("\n🚀 Starting server...")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )



