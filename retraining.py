# retraining.py - DUAL MODEL RETRAINING (UNet + YOLO)
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import database as db
import model_utils


# Configuration
UNET_CONFIG = {
    'batch_size': 4,
    'epochs': 5,
    'learning_rate': 0.0001,
    'min_samples': 20
}

YOLO_CONFIG = {
    'epochs': 50,
    'batch_size': 16,
    'img_size': 640,
    'min_samples': 10,
    'project': 'runs/train',
    'name': 'dental_yolo'
}

TOOTH_CLASSES = ["Incisor", "Canine", "Premolar", "Molar", "Wisdom", "Damaged"]


def load_training_data():
    """Load original training data + corrections"""
    # TODO: Load your original 418 training images
    # For now, just load corrections
    print("📦 Loading UNet training data...")
    
    hardest_cases = db.get_hardest_cases(n=20, unused_only=True)
    
    if len(hardest_cases) < UNET_CONFIG['min_samples']:
        print(f"⚠️ Only {len(hardest_cases)} corrections available. Need {UNET_CONFIG['min_samples']} for retraining.")
        return None, None, []
    
    images = []
    masks = []
    correction_ids = []
    
    for case in hardest_cases:
        # Load image
        img = Image.open(case['image_path']).convert('L')
        img = img.resize((512, 512))
        img_array = np.array(img) / 255.0
        
        # Load mask
        mask = Image.open(case['mask_path']).convert('L')
        mask = mask.resize((512, 512))
        mask_array = np.array(mask) / 255.0
        
        images.append(img_array)
        masks.append(mask_array)
        correction_ids.append(case['id'])
    
    X = np.array(images)[..., np.newaxis]  # Add channel dim
    y = np.array(masks)[..., np.newaxis]
    
    print(f"✅ Loaded {len(images)} hard cases for UNet retraining")
    return X, y, correction_ids


def retrain_unet_model():
    """Retrain UNet model with active learning"""
    print("\n" + "="*80)
    print("🚀 RETRAINING UNET SEGMENTATION MODEL")
    print("="*80)
    
    # Load data
    X_new, y_new, correction_ids = load_training_data()
    
    if X_new is None:
        print("⚠️ Skipping UNet retraining - insufficient data")
        return False
    
    # Load current model
    print("🔄 Loading current model...")
    model = model_utils.load_model("models/current_model.keras")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=UNET_CONFIG['learning_rate']),
        loss=model_utils.bce_dice_loss,
        metrics=['accuracy']
    )
    
    # Fine-tune for 5 epochs (as per your third notebook)
    print(f"\n🔧 Fine-tuning on {len(X_new)} hard cases for {UNET_CONFIG['epochs']} epochs...")
    history = model.fit(
        X_new, y_new,
        batch_size=UNET_CONFIG['batch_size'],
        epochs=UNET_CONFIG['epochs'],
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1)
        ],
        verbose=1
    )
    
    # Save new model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_path = f"models/training_history/unet_{timestamp}.keras"
    os.makedirs("models/training_history", exist_ok=True)
    
    model.save(new_model_path)
    model.save("models/current_model.keras")  # Update current model
    
    # Mark corrections as used
    db.mark_as_used(correction_ids)
    
    print(f"\n✅ UNet retraining complete!")
    print(f"  • Model saved: {new_model_path}")
    print(f"  • Final loss: {history.history['loss'][-1]:.4f}")
    print(f"  • Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    return True


def prepare_yolo_dataset():
    """Prepare YOLO training data from annotations"""
    print("\n" + "="*80)
    print("📦 PREPARING YOLO DATASET")
    print("="*80)
    
    # Check annotation files
    anno_dir = Path("data/yolo_annotations")
    
    if not anno_dir.exists():
        print("⚠️ No YOLO annotations directory found")
        return None
    
    txt_files = list(anno_dir.glob("*.txt"))
    json_files = list(anno_dir.glob("*.json"))
    
    if len(txt_files) < YOLO_CONFIG['min_samples']:
        print(f"⚠️ Not enough annotations! Need {YOLO_CONFIG['min_samples']}, have {len(txt_files)}")
        return None
    
    print(f"✅ Found {len(txt_files)} annotation files")
    
    # Create YOLO dataset structure
    dataset_root = Path("datasets/dental_yolo")
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    # Create train/val split
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    
    for split_dir in [train_dir, val_dir]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # Split data (80/20)
    num_train = int(len(json_files) * 0.8)
    train_files = json_files[:num_train]
    val_files = json_files[num_train:]
    
    print(f"  • Train samples: {len(train_files)}")
    print(f"  • Val samples: {len(val_files)}")
    
    # Copy files to dataset structure
    def copy_to_split(json_files, split_dir):
        copied = 0
        for json_file in json_files:
            try:
                # Load JSON to get image path
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                image_path = Path(data['image_path'])
                txt_file = json_file.with_suffix('.txt')
                
                if not image_path.exists() or not txt_file.exists():
                    print(f"⚠️ Skipping {json_file.stem} - missing files")
                    continue
                
                # Copy image
                dest_image = split_dir / "images" / image_path.name
                shutil.copy(image_path, dest_image)
                
                # Copy annotation
                dest_label = split_dir / "labels" / txt_file.name
                shutil.copy(txt_file, dest_label)
                
                copied += 1
            except Exception as e:
                print(f"⚠️ Error copying {json_file.name}: {e}")
                continue
        
        return copied
    
    train_copied = copy_to_split(train_files, train_dir)
    val_copied = copy_to_split(val_files, val_dir)
    
    print(f"✅ Copied {train_copied} train + {val_copied} val samples")
    
    # Create data.yaml
    yaml_content = f"""# Dental X-ray Detection Dataset
path: {dataset_root.absolute()}
train: train/images
val: val/images

# Classes
nc: {len(TOOTH_CLASSES)}
names: {TOOTH_CLASSES}
"""
    
    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Dataset prepared at: {dataset_root}")
    print(f"✅ Config saved: {yaml_path}")
    
    return yaml_path


def retrain_yolo_model(data_yaml):
    """Retrain YOLO model with new annotations"""
    print("\n" + "="*80)
    print("🚀 RETRAINING YOLO DETECTION MODEL")
    print("="*80)
    
    try:
        from ultralytics import YOLO
        
        # Load existing model or create new one
        model_path = "models/yolo_dental.pt"
        if os.path.exists(model_path):
            print(f"✅ Loading existing model: {model_path}")
            model = YOLO(model_path)
        else:
            print("⚠️ No existing model, using YOLOv8n base")
            model = YOLO('yolov8n.pt')
        
        # Train model
        print(f"\n🔧 Training for {YOLO_CONFIG['epochs']} epochs...")
        results = model.train(
            data=str(data_yaml),
            epochs=YOLO_CONFIG['epochs'],
            imgsz=YOLO_CONFIG['img_size'],
            batch=YOLO_CONFIG['batch_size'],
            project=YOLO_CONFIG['project'],
            name=YOLO_CONFIG['name'],
            patience=10,
            save=True,
            plots=True,
            verbose=True
        )
        
        # Find best model
        best_model_path = Path(YOLO_CONFIG['project']) / YOLO_CONFIG['name'] / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            # Backup old model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.exists(model_path):
                backup_path = f"models/training_history/yolo_backup_{timestamp}.pt"
                os.makedirs("models/training_history", exist_ok=True)
                shutil.copy(model_path, backup_path)
                print(f"✅ Backed up old model to: {backup_path}")
            
            # Copy new model
            shutil.copy(best_model_path, model_path)
            print(f"\n✅ YOLO retraining complete!")
            print(f"  • New model: {model_path}")
            print(f"  • Training results: {best_model_path.parent.parent}")
            
            return True
        else:
            print("⚠️ Training completed but best model not found")
            return False
        
    except ImportError:
        print("❌ YOLO (ultralytics) not installed!")
        print("   Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Error during YOLO training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main retraining pipeline for both models"""
    print("\n" + "="*80)
    print("🦷 DUAL MODEL RETRAINING SYSTEM")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check UNet readiness
    hardest_cases = db.get_hardest_cases(n=20, unused_only=True)
    unet_ready = len(hardest_cases) >= UNET_CONFIG['min_samples']
    
    # Check YOLO readiness
    anno_dir = Path("data/yolo_annotations")
    yolo_samples = len(list(anno_dir.glob("*.txt"))) if anno_dir.exists() else 0
    yolo_ready = yolo_samples >= YOLO_CONFIG['min_samples']
    
    print(f"\n🎯 Training Readiness:")
    print(f"  • UNet: {'✅ Ready' if unet_ready else '❌ Need ' + str(UNET_CONFIG['min_samples'] - len(hardest_cases)) + ' more samples'}")
    print(f"  • YOLO: {'✅ Ready' if yolo_ready else '❌ Need ' + str(YOLO_CONFIG['min_samples'] - yolo_samples) + ' more samples'}")
    
    if not unet_ready and not yolo_ready:
        print("\n❌ Not enough data for either model. Exiting.")
        print("\n💡 Next steps:")
        if not unet_ready:
            needed = UNET_CONFIG['min_samples'] - len(hardest_cases)
            print(f"   • Annotate {needed} more segmentation masks")
        if not yolo_ready:
            needed = YOLO_CONFIG['min_samples'] - yolo_samples
            print(f"   • Annotate {needed} more YOLO boxes")
        return
    
    # Track success
    unet_success = False
    yolo_success = False
    
    # Retrain UNet
    if unet_ready:
        unet_success = retrain_unet_model()
    else:
        needed = UNET_CONFIG['min_samples'] - len(hardest_cases)
        print(f"\n⚠️ Skipping UNet retraining - need {needed} more samples")
    
    # Retrain YOLO
    if yolo_ready:
        data_yaml = prepare_yolo_dataset()
        if data_yaml:
            yolo_success = retrain_yolo_model(data_yaml)
        else:
            print("⚠️ Skipping YOLO retraining due to data preparation error")
    else:
        needed = YOLO_CONFIG['min_samples'] - yolo_samples
        print(f"\n⚠️ Skipping YOLO retraining - need {needed} more samples")
    
    # Final summary
    print("\n" + "="*80)
    print("📊 RETRAINING SUMMARY")
    print("="*80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ Successfully retrained:")
    if unet_success:
        print("  • UNet Segmentation Model → models/current_model.keras")
    if yolo_success:
        print("  • YOLO Detection Model → models/yolo_dental.pt")
    
    if not unet_success and not yolo_success:
        print("  • No models were retrained")
    
    print("\n📁 Backups saved in: models/training_history/")
    
    if unet_success or yolo_success:
        print("\n🚀 Restart the app to use the new models:")
        print("   python app.py")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()