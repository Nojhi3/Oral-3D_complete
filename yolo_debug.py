# yolo_deep_debug.py - Deep YOLO Diagnostics
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch

YOLO_MODEL_PATH = "models/yolo_dental.pt"
TEST_IMAGE = r"data\original_images\img_20251007_112744.png"  # Your failing image

print("🔍 DEEP YOLO DIAGNOSTICS")
print("="*50)

# 1. MODEL INFO
model = YOLO(YOLO_MODEL_PATH)
print(f"\n1️⃣ MODEL INFO:")
print(f"   Classes: {model.names}")
print(f"   NC (num classes): {model.model.nc}")
print(f"   Model type: {type(model.model)}")

# 2. IMAGE LOADING & PREPROCESS
img_bgr = cv2.imread(TEST_IMAGE)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f"\n2️⃣ IMAGE INFO:")
print(f"   Shape: {img_rgb.shape}")
print(f"   Min/Max pixel: {img_rgb.min()}/{img_rgb.max()}")

# 3. RAW PREDICTIONS (NO THRESHOLD)
print(f"\n3️⃣ RAW PREDICTIONS (conf=0.001):")
results = model(img_rgb, conf=0.001, verbose=True, save=False)

print("\n   ALL BOX DATA:")
all_boxes = []
for r in results:
    if r.boxes is not None:
        print(f"   Found {len(r.boxes)} boxes")
        for i, box in enumerate(r.boxes):
            cls = int(box.cls)
            conf = float(box.conf)
            print(f"     Box {i}: class={cls} ({r.names[cls]}) conf={conf:.4f}")
            all_boxes.append((r.names[cls], conf))
    else:
        print("   ❌ NO BOXES DETECTED")

# 4. TEST DIFFERENT INPUT FORMATS
print(f"\n4️⃣ FORMAT TESTS:")
tests = [
    ("Original RGB", img_rgb),
    ("Grayscale", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)[..., None]),
    ("Resized 640x640", cv2.resize(img_rgb, (640, 640))),
    ("Padded 640", model(img_rgb, imgsz=640)[0].orig_img)  # Letterboxed
]

for name, test_img in tests:
    print(f"\n   {name}:")
    r = model(test_img, conf=0.001, verbose=False)
    if r[0].boxes is not None:
        print(f"     ✓ {len(r[0].boxes)} detections")
    else:
        print("     ❌ No detections")

# 5. MODEL SUMMARY (first 3 layers)
print(f"\n5️⃣ MODEL ARCHITECTURE (first layers):")
try:
    print(model.model.model[0])  # First layer info
except:
    print("   Cannot inspect model layers")

print("\n🎯 SUMMARY:")
if all_boxes:
    print("✅ Model capable of predictions - filtering issue")
else:
    print("❌ MODEL BROKEN - no predictions at conf=0.001")
