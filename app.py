# app.py - GRADIO 5.24 OPTIMIZED WITH PROFESSIONAL ImageEditor
import gradio as gr
import os
from datetime import datetime
from PIL import Image
import database as db
import model_utils
import numpy as np

# Initialize
db.init_database()
os.makedirs("data/corrections", exist_ok=True)
os.makedirs("data/original_images", exist_ok=True)

# Load model
MODEL = model_utils.load_model("models/current_model.keras")
CURRENT_IMAGE_PATH = None
CURRENT_PRED_PATH = None
CURRENT_LOSS = None

def predict_segmentation(xray_image):
    """Generate initial segmentation prediction"""
    global CURRENT_IMAGE_PATH, CURRENT_PRED_PATH, CURRENT_LOSS
    
    if xray_image is None:
        return None, None, "❌ Please upload an X-ray image first"
    
    # Generate prediction
    pred_mask, loss_score = model_utils.predict_mask(MODEL, xray_image)
    
    # Save files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CURRENT_IMAGE_PATH = f"data/original_images/img_{timestamp}.png"
    CURRENT_PRED_PATH = f"data/corrections/pred_{timestamp}.png"
    
    xray_image.save(CURRENT_IMAGE_PATH)
    pred_mask.save(CURRENT_PRED_PATH)
    CURRENT_LOSS = loss_score
    
    status = f"""✅ Prediction generated successfully!

📊 Uncertainty Score: {loss_score:.4f}
{"⚠️ HIGH UNCERTAINTY - Priority for correction!" if loss_score > 0.3 else "✓ Low uncertainty - still useful"}

🎨 Edit the mask using the ImageEditor below:
• Use mouse wheel to ZOOM in/out
• Click and DRAG to pan when zoomed
• White brush = teeth regions
• Black brush = background
• Eraser tool available in toolbar
• Undo/Redo built-in!
"""
    
    return pred_mask, loss_score, status

def save_correction(editor_dict):
    """Save corrected mask from ImageEditor"""
    global CURRENT_IMAGE_PATH, CURRENT_PRED_PATH, CURRENT_LOSS
    
    if CURRENT_IMAGE_PATH is None:
        return "❌ No prediction to save. Generate a prediction first."
    
    if editor_dict is None:
        return "❌ No edits made. Please edit the mask before saving."
    
    # Extract composite image from ImageEditor
    if isinstance(editor_dict, dict):
        # Gradio 5.24 returns {"background": img, "layers": [...], "composite": img}
        mask_image = editor_dict.get("composite") or editor_dict.get("background")
    else:
        mask_image = editor_dict
    
    if mask_image is None:
        return "❌ Could not extract mask from editor. Please try again."
    
    # Convert to PIL Image
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
    
    # Get stats
    stats = db.get_stats()
    progress_bar = "█" * min(stats['pending'], 20) + "░" * max(0, 20 - stats['pending'])
    percentage = (stats['pending'] / 20) * 100 if stats['pending'] <= 20 else 100
    
    result = f"""✅ Correction saved successfully! (ID: {correction_id})

📊 Training Progress:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[{progress_bar}] {stats['pending']}/20 ({percentage:.0f}%)

• Total corrections: {stats['total']}
• Used for training: {stats['used']}
• Pending retraining: {stats['pending']}
• Average loss: {stats['avg_loss']:.4f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
    
    if stats['pending'] >= 20:
        result += """🎯 READY FOR RETRAINING!

Run this command in your terminal:
    python retraining.py

This will fine-tune the model using your corrections.
The model will learn from its mistakes and improve!
"""
    else:
        needed = 20 - stats['pending']
        result += f"""⚠️ Need {needed} more correction{"s" if needed > 1 else ""} for retraining

Keep going! Focus on high uncertainty cases (score > 0.3) for maximum impact.
Each correction helps the model learn better tooth segmentation.
"""
    
    return result

def refresh_stats():
    """Get current statistics with visual progress"""
    stats = db.get_stats()
    progress_bar = "█" * min(stats['pending'], 20) + "░" * max(0, 20 - stats['pending'])
    percentage = (stats['pending'] / 20) * 100 if stats['pending'] <= 20 else 100
    
    return f"""📊 Active Learning Progress
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Collection Status:**
[{progress_bar}] {stats['pending']}/20 ({percentage:.0f}%)

**Statistics:**
• Total corrections collected: {stats['total']}
• Used for training: {stats['used']}
• Pending retraining: {stats['pending']}
• Average uncertainty: {stats['avg_loss']:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{'🎯 READY TO RETRAIN! Run: python retraining.py' if stats['pending'] >= 20 else f'⚠️ Need {20 - stats["pending"]} more corrections to reach training threshold'}

**Active Learning Benefits:**
• Model learns from its mistakes
• Focuses on hard cases (high uncertainty)
• Improves iteratively with minimal data
• Human expertise guides the learning process
"""

# Build Gradio 5.24 Interface
with gr.Blocks(
    title="Tooth Segmentation Active Learning System",
    theme=gr.themes.Soft(),
    fill_height=True
) as demo:
    
    gr.Markdown("""
    # 🦷 Dental X-ray Segmentation with Active Learning
    
    **Professional-grade AI-assisted tooth segmentation with human-in-the-loop learning**
    
    Upload dental X-rays → AI predicts → You correct → Model learns and improves
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Step 1: Upload Dental X-ray")
            input_xray = gr.Image(
                type="pil",
                label="Upload Dental X-ray Image",
                image_mode="L",
                height=450,
                sources=["upload", "clipboard"]  # Gradio 5 feature
            )
            
            predict_btn = gr.Button(
                "🔮 Generate AI Prediction", 
                variant="primary", 
                size="lg",
                scale=1
            )
            
            loss_display = gr.Number(
                label="🎯 Prediction Uncertainty Score",
                info="Higher = More uncertain = Priority for correction",
                interactive=False
            )
            
        with gr.Column(scale=2):
            gr.Markdown("### ✏️ Step 2: Edit Segmentation with Professional Tools")
            
            
            # GRADIO 5.24 ImageEditor - Professional Edition
            mask_editor = gr.ImageEditor(
                type="pil",
                image_mode="L",
                crop_size="1:1",  # Square aspect ratio
                brush=gr.Brush(
                    default_size=20,
                    colors=["#FFFFFF", "#000000", "#808080"],
                    default_color="#FFFFFF"
                ),
                eraser=gr.Eraser(default_size=25),
                label="🎨 Professional ImageEditor (Zoom with mouse wheel, Pan with drag)",
                height=700,
                sources=["upload"]  # Only allow manual loading
            )
    
    status_output = gr.Textbox(
        label="📋 Status & Instructions",
        lines=10,
        interactive=False,
        show_copy_button=True  # Gradio 5 feature
    )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            save_btn = gr.Button(
                "💾 Save Correction for Training",
                variant="primary",
                size="lg",
                icon="💾"  # Gradio 5 feature
            )
            
            correction_result = gr.Textbox(
                label="📊 Save Result & Training Progress",
                lines=16,
                interactive=False,
                show_copy_button=True
            )
            
        with gr.Column(scale=1):
            refresh_btn = gr.Button(
                "🔄 Refresh Statistics",
                size="lg",
                icon="🔄"
            )
            
            stats_display = gr.Textbox(
                label="📈 Active Learning Statistics",
                lines=16,
                interactive=False
            )
    
    gr.Markdown("---")
    
    with gr.Accordion("📚 Active Learning Guide & Best Practices", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 🎓 How Active Learning Works
                
                **The Workflow:**
                1. **AI Prediction**: Model generates initial segmentation
                2. **Uncertainty Scoring**: System calculates confidence level
                3. **Human Correction**: You fix errors using professional tools
                4. **Data Collection**: Corrections stored in SQLite database
                5. **Batch Training**: After 20 corrections, retrain model
                6. **Model Improvement**: Updated model performs better
                7. **Continuous Iteration**: Process repeats indefinitely
                
                **Why It Works:**
                - Model learns from its own mistakes
                - Focuses on hard cases (high uncertainty)
                - Requires minimal labeled data
                - Improves iteratively with each correction
                - Human expertise guides the learning
                
                **Scientific Basis:**
                - Query-by-uncertainty sampling strategy
                - Fine-tuning with hard negative examples
                - Reduces labeling effort by 50-70%
                - Converges faster than random sampling
                """)
                
            with gr.Column():
                gr.Markdown("""
                ### 🎨 Professional Editing Best Practices
                
                **Prioritization Strategy:**
                - ⭐ **High uncertainty cases first** (score > 0.3)
                - Focus on obvious segmentation errors
                - Challenging overlapping teeth regions
                - Partially visible or ambiguous teeth
                - Metal fillings or dental work areas
                
                **Editing Techniques:**
                - **Zoom in** (mouse wheel) for precision at boundaries
                - Use **larger brush** for filling main tooth regions
                - Use **smaller brush** for detailed edges
                - **Eraser** for quick corrections
                - **Black brush** for background cleanup
                - **Undo button** (bottom toolbar) for mistakes
                
                **Quality Guidelines:**
                - Be consistent with tooth boundaries
                - White (255) = Tooth regions only
                - Black (0) = Background and soft tissue
                - Include tooth roots if clearly visible
                - Exclude bone structures unless teeth
                - Mark metal fillings as part of tooth
                
                **Zoom & Navigation:**
                - Scroll wheel: Zoom in/out (up to 300%)
                - Click + drag: Pan around canvas when zoomed
                - Reset zoom: Double-click canvas
                - Layer panel: Bottom of editor for complex edits
                """)
    
    # Event Handlers - Clean and Simple
    predict_btn.click(
        fn=predict_segmentation,
        inputs=input_xray,
        outputs=[mask_editor, loss_display, status_output],
        api_name="predict"  # Gradio 5 feature - API endpoint name
    )
    
    save_btn.click(
        fn=save_correction,
        inputs=mask_editor,
        outputs=correction_result,
        api_name="save"
    )
    
    refresh_btn.click(
        fn=refresh_stats,
        outputs=stats_display,
        api_name="stats"
    )
    
    # Load stats on app startup
    demo.load(
        fn=refresh_stats,
        outputs=stats_display
    )

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🦷 TOOTH SEGMENTATION ACTIVE LEARNING SYSTEM")
    print("="*80)
    print("\n📦 Configuration:")
    print("   • Python 3.10.11")
    print("   • TensorFlow 2.16.1 + Keras 3.0")
    print("   • Gradio 5.24.0 (Professional ImageEditor with zoom/pan)")
    print("   • SQLite database for correction tracking")
    print("\n🎨 Features:")
    print("   • Mouse wheel zoom (up to 3x magnification)")
    print("   • Click & drag panning")
    print("   • Professional brush and eraser tools")
    print("   • Built-in undo/redo")
    print("   • Layer support for complex edits")
    print("\n🚀 Starting server...")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=True  # Gradio 5 feature - shows API docs
    )
