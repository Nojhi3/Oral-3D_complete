"""
Simple YOLO-style Annotator
Just upload image and annotate - that's it!
"""

import gradio as gr
from gradio_image_annotation import image_annotator

# Tooth classes
TOOTH_CLASSES = ["Incisor", "Canine", "Premolar", "Molar", "Wisdom", "Damaged"]

# Colors for each class
CLASS_COLORS = [
    (255, 0, 0),      # Red - Incisor
    (0, 0, 255),      # Blue - Canine  
    (255, 255, 0),    # Yellow - Premolar
    (255, 0, 255),    # Magenta - Molar
    (0, 255, 255),    # Cyan - Wisdom
    (255, 128, 0)     # Orange - Damaged
]

# Build simple interface
with gr.Blocks(title="Simple Annotator", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🦷 Simple Dental X-ray Annotator
    Upload image → Press C to create boxes → Press D to drag boxes → Annotate!
    """)
    
    # Just the annotator
    annotator = image_annotator(
        label="Upload image and start annotating (Press C to create, D to drag)",
        label_list=TOOTH_CLASSES,
        label_colors=CLASS_COLORS,
        height=700
    )
    
    gr.Markdown("""
    ### 🎮 Controls:
    - **C** = Create boxes (click & drag)
    - **D** = Drag/move boxes
    - **Delete** = Remove selected box
    - **Drag corners** = Resize box
    """)

if __name__ == "__main__":
    print("🚀 Starting Simple Annotator...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )