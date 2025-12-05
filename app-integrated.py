# app.py - DUAL CORRECTION SYSTEM WITH INTEGRATED ANNOTATOR
import gradio as gr
import os
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import database as db
import model_utils
import numpy as np
import cv2
from gradio_image_annotation import image_annotator

# Initialize directories
db.init_database()
os.makedirs("data/corrections", exist_ok=True)
os.makedirs("data/original_images", exist_ok=True)
os.makedirs("data/yolo_annotations", exist_ok=True)

# Load UNet model
print("🔄 Loading UNet segmentation model...")
MODEL = model_utils.load_model("models/current_model.keras")
print("✅ UNet model loaded successfully!")

# Load YOLO model (if available)
try:
    from ultralytics import YOLO
    print("🔄 Loading YOLO detection model...")
    YOLO_MODEL = YOLO("models/yolo_dental.pt")
    YOLO_ENABLED = True
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    YOLO_ENABLED = False
    print(f"⚠️ YOLO not available: {e}")

# Global state
CURRENT_IMAGE_PATH = None
CURRENT_PRED_MASK_PATH = None
CURRENT_LOSS = None
CURRENT_YOLO_BOXES = []
TOOTH_CLASSES = ["Cavity", "Damaged", "Implant", "Caries"]

# Colors for each class (RGB format for gradio_image_annotation)
CLASS_COLORS = [
    (255, 0, 0),      # Red - Cavity
    (0, 0, 255),      # Blue - Damaged
    (255, 255, 0),    # Yellow - Implant
    (255, 0, 255),    # Magenta - Caries
]

def get_tooth_bounding_boxes(mask_binary, min_area=100):
    """Extract bounding boxes from segmentation mask"""
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
    global CURRENT_YOLO_BOXES
    
    if not YOLO_ENABLED:
        CURRENT_YOLO_BOXES = []
        return []
    
    results = YOLO_MODEL(image, conf=threshold, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                'class': result.names[int(box.cls)],
                'confidence': float(box.conf),
                'x1': float(x1), 'y1': float(y1),
                'x2': float(x2), 'y2': float(y2)
            })
    
    CURRENT_YOLO_BOXES = detections
    return detections

def convert_yolo_to_annotator_format(image, yolo_boxes):
    """Convert YOLO detections to gradio_image_annotation format"""
    if image is None:
        return None
    
    img_array = np.array(image)
    
    # Build annotation data structure
    annotation_data = {
        "image": img_array,
        "boxes": []
    }
    
    for det in yolo_boxes:
        # Convert from x1,y1,x2,y2 to xmin,ymin,xmax,ymax format
        box = {
            "xmin": int(det['x1']),
            "ymin": int(det['y1']),
            "xmax": int(det['x2']),
            "ymax": int(det['y2']),
            "label": det['class'],
            "color": CLASS_COLORS[TOOTH_CLASSES.index(det['class'])] if det['class'] in TOOTH_CLASSES else CLASS_COLORS[0]
        }
        annotation_data["boxes"].append(box)
    
    return annotation_data

def draw_boxes_on_image(image, seg_boxes, yolo_boxes):
    """Draw segmentation and YOLO boxes on image"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
    
    # Draw segmentation boxes (green)
    for idx, box in enumerate(seg_boxes):
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        draw.rectangle([x, y, x+w, y+h], outline="lime", width=2)
        label = f"Seg #{idx+1}"
        bbox = draw.textbbox((x, y-18), label, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill="white")
        draw.text((x, y-18), label, fill="black", font=font)
    
    # Draw YOLO boxes (red)
    for idx, det in enumerate(yolo_boxes):
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = f"{det['class']}: {det['confidence']:.2f}"
        bbox = draw.textbbox((x1, y1-18), label, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill="white")
        draw.text((x1, y1-18), label, fill="black", font=font)
    
    return img_draw

def predict_and_generate(xray_image):
    """Generate predictions from both models"""
    global CURRENT_IMAGE_PATH, CURRENT_PRED_MASK_PATH, CURRENT_LOSS, CURRENT_YOLO_BOXES
    
    if xray_image is None:
        return None, None, None, None, "❌ Please upload an X-ray image first"
    
    # Generate UNet segmentation prediction
    pred_mask, loss_score = model_utils.predict_mask(MODEL, xray_image)
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CURRENT_IMAGE_PATH = f"data/original_images/img_{timestamp}.png"
    CURRENT_PRED_MASK_PATH = f"data/corrections/pred_{timestamp}.png"
    
    xray_image.save(CURRENT_IMAGE_PATH)
    pred_mask.save(CURRENT_PRED_MASK_PATH)
    CURRENT_LOSS = loss_score
    
    # Extract segmentation boxes
    mask_array = np.array(pred_mask)
    seg_boxes = get_tooth_bounding_boxes(mask_array > 127, min_area=100)
    
    # Run YOLO detection
    yolo_detections = run_yolo_detection(xray_image, threshold=0.5)
    
    # Create visualization
    vis_image = draw_boxes_on_image(xray_image, seg_boxes, yolo_detections)
    
    # Convert YOLO predictions to annotator format
    annotator_data = convert_yolo_to_annotator_format(xray_image, yolo_detections)
    
    # Status message
    status = f"""✅ Predictions generated successfully!

📊 **Segmentation Results (UNet):**
   • Uncertainty Score: {loss_score:.4f}
   • Teeth Detected: {len(seg_boxes)}
   {"⚠️ HIGH UNCERTAINTY - Review needed!" if loss_score > 0.3 else "✓ Low uncertainty"}

"""
    
    if YOLO_ENABLED:
        status += f"""🎯 **YOLO Detection Results:**
   • Total Detections: {len(yolo_detections)}
"""
        for det in yolo_detections:
            status += f"   • {det['class']}: {det['confidence']:.1%}\n"
    else:
        status += "⚠️ YOLO detection not available\n"
    
    status += """
📝 **Next Steps:**
1. Review segmentation mask (left) - edit with white/black brush if needed
2. Review YOLO boxes (right) - YOLO detections are pre-loaded! Edit/add/remove boxes as needed
3. Save both corrections separately
"""
    
    return pred_mask, vis_image, annotator_data, loss_score, status

def save_segmentation_correction(editor_dict):
    """Save corrected segmentation mask"""
    global CURRENT_IMAGE_PATH, CURRENT_PRED_MASK_PATH, CURRENT_LOSS
    
    if CURRENT_IMAGE_PATH is None:
        return "❌ No prediction to save. Generate a prediction first."
    
    if editor_dict is None:
        return "❌ No edits made. Please edit the mask before saving."
    
    # Extract mask from ImageEditor
    if isinstance(editor_dict, dict):
        mask_image = editor_dict.get("composite") or editor_dict.get("background")
    else:
        mask_image = editor_dict
    
    if mask_image is None:
        return "❌ Could not extract mask from editor."
    
    # Convert to PIL grayscale
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
        predicted_mask_path=CURRENT_PRED_MASK_PATH,
        corrected_mask_path=corrected_path,
        loss_score=CURRENT_LOSS,
        model_version="v1"
    )
    
    stats = db.get_stats()
    progress_bar = "█" * min(stats['pending'], 20) + "░" * max(0, 20 - stats['pending'])
    
    return f"""✅ Segmentation correction saved! (ID: {correction_id})

📊 Progress: [{progress_bar}] {stats['pending']}/20
Total: {stats['total']} | Used: {stats['used']} | Pending: {stats['pending']}

{"🎯 READY! Run: python retraining.py" if stats['pending'] >= 20 else f"⚠️ Need {20-stats['pending']} more corrections"}"""

def save_yolo_correction(annotation_data):
    """Save YOLO annotation corrections from gradio_image_annotation"""
    global CURRENT_IMAGE_PATH
    
    if annotation_data is None:
        return "❌ No annotations to save. Generate predictions first."
    
    if CURRENT_IMAGE_PATH is None:
        return "❌ No image context. Generate predictions first."
    
    try:
        boxes = annotation_data.get('boxes', [])
        
        if len(boxes) == 0:
            return "⚠️ No boxes found in annotations. Add at least one box."
        
        # Get image dimensions
        image = Image.open(CURRENT_IMAGE_PATH)
        img_width, img_height = image.size
        
        # Save JSON format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"data/yolo_annotations/anno_{timestamp}.json"
        
        json_data = {
            'image_path': CURRENT_IMAGE_PATH,
            'image_width': img_width,
            'image_height': img_height,
            'boxes': []
        }
        
        # Convert to YOLO format and save
        txt_path = f"data/yolo_annotations/anno_{timestamp}.txt"
        class_map = {cls: idx for idx, cls in enumerate(TOOTH_CLASSES)}
        
        with open(txt_path, 'w') as f:
            for box in boxes:
                # Extract box coordinates
                xmin = box['xmin']
                ymin = box['ymin']
                xmax = box['xmax']
                ymax = box['ymax']
                label = box['label']
                
                # Convert to YOLO format (normalized center_x, center_y, width, height)
                center_x = ((xmin + xmax) / 2) / img_width
                center_y = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                class_id = class_map.get(label, 0)
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                # Add to JSON
                json_data['boxes'].append({
                    'class': label,
                    'class_id': class_id,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height
                })
        
        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Count total annotations
        total_annotations = len([f for f in os.listdir("data/yolo_annotations") if f.endswith('.txt')])
        
        return f"""✅ YOLO annotations saved!
📄 Files: {json_path}
📦 Total boxes: {len(boxes)}
📊 Total annotated images: {total_annotations}

Classes: {', '.join(set(b['label'] for b in boxes))}
"""
    
    except Exception as e:
        return f"❌ Error saving annotations: {str(e)}"

def refresh_stats():
    """Get current training progress stats"""
    stats = db.get_stats()
    progress_bar = "█" * min(stats['pending'], 20) + "░" * max(0, 20 - stats['pending'])
    percentage = (stats['pending'] / 20) * 100 if stats['pending'] <= 20 else 100
    
    # Count YOLO annotations
    yolo_count = len([f for f in os.listdir("data/yolo_annotations") if f.endswith('.txt')])
    
    return f"""📊 Active Learning Progress

**Segmentation (UNet):**
[{progress_bar}] {stats['pending']}/20 ({percentage:.0f}%)
Total: {stats['total']} | Used: {stats['used']} | Avg Loss: {stats['avg_loss']:.4f}

**Detection (YOLO):**
Annotated Images: {yolo_count}

{'🎯 Ready to retrain models!' if stats['pending'] >= 20 else f'⚠️ Need {20-stats["pending"]} more segmentation corrections'}
"""

# Build Gradio Interface
with gr.Blocks(title="Dental X-ray Dual Correction System", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🦷 Dental X-ray AI with Dual Correction System
    **AI-powered tooth segmentation + detection with active learning**
    
    Upload → AI Predicts → Review & Correct → Save → Retrain Models
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload X-ray Image")
            input_xray = gr.Image(
                type="pil",
                label="Dental X-ray",
                height=400
            )
            predict_btn = gr.Button(
                "🔮 Generate AI Predictions",
                variant="primary",
                size="lg"
            )
            loss_display = gr.Number(
                label="🎯 Uncertainty Score",
                interactive=False
            )
            
        with gr.Column(scale=2):
            gr.Markdown("### 🎯 AI Predictions Visualization")
            visualization = gr.Image(
                label="Segmentation (Green) + YOLO (Red)",
                height=500
            )
    
    status_output = gr.Textbox(
        label="📋 Prediction Status",
        lines=15,
        interactive=False,
        show_copy_button=True
    )
    
    gr.Markdown("---")
    gr.Markdown("## ✏️ Correction Tools")
    
    with gr.Row():
        # Segmentation correction
        with gr.Column(scale=1):
            gr.Markdown("### 🖌️ Correct Segmentation Mask")
            gr.Markdown("""
            **Instructions:**
            - White brush = teeth regions
            - Black brush = background
            - Right-click to erase
            """)
            
            mask_editor = gr.ImageEditor(
                type="pil",
                image_mode="L",
                brush=gr.Brush(
                    default_size=20,
                    colors=["#FFFFFF", "#000000"],
                    default_color="#FFFFFF"
                ),
                eraser=gr.Eraser(default_size=25),
                label="Segmentation Mask Editor",
                height=600
            )
            
            save_seg_btn = gr.Button(
                "💾 Save Segmentation Correction",
                variant="primary",
                size="lg"
            )
            seg_result = gr.Textbox(
                label="Save Result",
                lines=6,
                interactive=False
            )
        
        # YOLO correction with new annotator
        with gr.Column(scale=1):
            gr.Markdown("### 📦 Correct YOLO Boxes")
            gr.Markdown("""
            **Instructions:**
            - **C** = Create boxes (click & drag)
            - **D** = Drag/move boxes
            - **Delete** = Remove selected box
            - **Drag corners** = Resize box
            - YOLO predictions are pre-loaded as editable boxes!
            """)
            
            yolo_annotator = image_annotator(
                label="YOLO Box Annotator (YOLO detections pre-loaded)",
                label_list=TOOTH_CLASSES,
                label_colors=CLASS_COLORS,
                height=600
            )
            
            save_yolo_btn = gr.Button(
                "💾 Save YOLO Correction",
                variant="primary",
                size="lg"
            )
            yolo_result = gr.Textbox(
                label="Save Result",
                lines=6,
                interactive=False
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh Statistics", size="lg")
        stats_display = gr.Textbox(
            label="📈 Training Progress",
            lines=12,
            interactive=False
        )
    
    # Event handlers
    predict_btn.click(
        predict_and_generate,
        inputs=input_xray,
        outputs=[mask_editor, visualization, yolo_annotator, loss_display, status_output]
    )
    
    save_seg_btn.click(
        save_segmentation_correction,
        inputs=mask_editor,
        outputs=seg_result
    )
    
    save_yolo_btn.click(
        save_yolo_correction,
        inputs=yolo_annotator,
        outputs=yolo_result
    )
    
    refresh_btn.click(
        refresh_stats,
        outputs=stats_display
    )
    
    demo.load(refresh_stats, outputs=stats_display)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🦷 DENTAL X-RAY DUAL CORRECTION SYSTEM")
    print("="*80)
    print(f"\n📦 Models Status:")
    print(f"   • UNet Segmentation: ✓ Loaded")
    print(f"   • YOLO Detection: {'✓ Loaded' if YOLO_ENABLED else '✗ Not Available'}")
    print(f"\n📂 Output Directories:")
    print(f"   • Original Images: data/original_images/")
    print(f"   • Segmentation Masks: data/corrections/")
    print(f"   • YOLO Annotations: data/yolo_annotations/")
    print(f"\n🦷 Tooth Classes: {', '.join(TOOTH_CLASSES)}")
    print("\n🚀 Starting Gradio server...")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )