# yolo_tester.py - Standalone YOLO Dental Detection Tester
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import glob
from pathlib import Path

# Configuration
YOLO_MODEL_PATH = "models/yolo_dental.pt"  # Update path if different
TEST_IMAGES_DIR = "test_images/"  # Put test X-rays here
OUTPUT_DIR = "yolo_test_results/"
CONF_THRESHOLDS = [0.1, 0.25, 0.4, 0.5]  # Test different thresholds
IMG_SIZE = 640  # YOLO input size

# Dental classes (match your app)
TOOTH_CLASSES = ["Cavity", "Damaged", "Implant", "Caries"]
CLASS_COLORS = {
    'Cavity': (255, 0, 0),      # Red
    'Damaged': (0, 0, 255),     # Blue
    'Implant': (255, 255, 0),   # Yellow
    'Caries': (255, 0, 255)     # Magenta
}

def load_yolo_model():
    """Load YOLO model with error handling"""
    try:
        print(f"🔄 Loading YOLO model from: {YOLO_MODEL_PATH}")
        model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        print("💡 Check: model file exists? ultralytics installed?")
        return None

def test_single_image(model, image_path, conf_threshold=0.25):
    """Test YOLO on single image with detailed output"""
    print(f"\n🦷 Testing: {image_path}")
    
    # Load and preprocess
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load image: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Run detection
    results = model(img_rgb, conf=conf_threshold, verbose=False, imgsz=IMG_SIZE)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                label = result.names[cls_id]
                
                detections.append({
                    'label': label,
                    'conf': conf,
                    'box': [int(x1), int(y1), int(x2), int(y2)]
                })
                print(f"   🎯 {label}: {conf:.3f} at ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    
    print(f"   📊 Total detections: {len(detections)}")
    return detections, img_rgb

def visualize_results(img_rgb, detections, output_path):
    """Draw detections on image and save"""
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['conf']
        
        # Box color by class
        color = CLASS_COLORS.get(label, (255, 255, 255))
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Label background
        label_text = f"{label} {conf:.2f}"
        bbox = draw.textbbox((x1, y1-25), label_text, font=font)
        draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], 
                      fill=color, outline=color)
        draw.text((x1, y1-25), label_text, fill="black", font=font)
    
    # Save result
    img_pil.save(output_path)
    print(f"   💾 Saved visualization: {output_path}")

def batch_test(model, test_dir):
    """Test on all images in directory"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_dir, ext)))
        test_images.extend(glob.glob(os.path.join(test_dir, ext.upper())))
    
    if not test_images:
        print(f"❌ No test images found in {test_dir}")
        print("💡 Create 'test_images/' folder and add X-ray images")
        return
    
    print(f"\n🔬 Batch testing {len(test_images)} images...")
    
    stats = {cls: 0 for cls in TOOTH_CLASSES}
    stats['total'] = 0
    stats['empty'] = 0
    
    for img_path in test_images:
        for conf_thresh in CONF_THRESHOLDS:
            detections, img_rgb = test_single_image(model, img_path, conf_thresh)
            
            if detections:
                stats['total'] += 1
                for det in detections:
                    stats[det['label']] = stats.get(det['label'], 0) + 1
                
                # Save visualization for first threshold only
                if conf_thresh == CONF_THRESHOLDS[0]:
                    base_name = Path(img_path).stem
                    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_conf{conf_thresh}.png")
                    visualize_results(img_rgb, detections, output_path)
            else:
                stats['empty'] += 1
    
    print("\n📈 BATCH TEST SUMMARY:")
    print(f"Total detections: {stats['total']}")
    print(f"Images with no detections: {stats['empty']}")
    print("\nPer-class detections:")
    for cls in TOOTH_CLASSES:
        print(f"  {cls}: {stats.get(cls, 0)}")

def interactive_test(model):
    """Interactive testing mode"""
    print("\n🎮 INTERACTIVE MODE (Ctrl+C to exit)")
    print("Enter image path or 'batch' for batch testing:")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit' or user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'batch':
                batch_test(model, TEST_IMAGES_DIR)
                break
            elif user_input:
                conf = input("Confidence threshold (default 0.25): ").strip()
                conf = float(conf) if conf else 0.25
                
                detections, img_rgb = test_single_image(model, user_input, conf)
                if detections and img_rgb is not None:
                    base_name = Path(user_input).stem
                    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_interactive.png")
                    visualize_results(img_rgb, detections, output_path)
                    print(f"✅ Check result: {output_path}")
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main testing function"""
    print("="*60)
    print("🦷 YOLO DENTAL MODEL TESTER")
    print("="*60)
    
    # Load model
    model = load_yolo_model()
    if model is None:
        return
    
    # Check for test images
    if os.path.exists(TEST_IMAGES_DIR):
        print(f"\n📁 Found {TEST_IMAGES_DIR} - Ready for batch testing")
        response = input("Run batch test now? (y/n): ").lower()
        if response in ['y', 'yes']:
            batch_test(model, TEST_IMAGES_DIR)
            return
    
    # Interactive mode
    interactive_test(model)
    
    print("\n✅ Testing complete!")
    print(f"📂 Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
