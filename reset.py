# reset_project.py
import os
import shutil
import sqlite3

# Paths (adjust if different)
DATA_DIR = "data"
CORR_DIR = os.path.join(DATA_DIR, "corrections")
IMG_DIR  = os.path.join(DATA_DIR, "original_images")
DB_PATH  = os.path.join(DATA_DIR, "corrections.db")

MODELS_DIR = "models"
CURRENT_MODEL = os.path.join(MODELS_DIR, "current_model.keras")
HISTORY_DIR = os.path.join(MODELS_DIR, "training_history")

def rm_dir_contents(path):
    if os.path.isdir(path):
        for name in os.listdir(path):
            p = os.path.join(path, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

def ensure_dirs():
    os.makedirs(CORR_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

def reset_database():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    os.makedirs(DATA_DIR, exist_ok=True)
    # Recreate empty schema
    import database as db  # your project's module
    db.init_database()

def reset_models(delete_active=False):
    # Clear model history
    rm_dir_contents(HISTORY_DIR)
    # Optionally remove current_model.keras
    if delete_active and os.path.exists(CURRENT_MODEL):
        os.remove(CURRENT_MODEL)

def reset_data():
    rm_dir_contents(CORR_DIR)
    rm_dir_contents(IMG_DIR)

if __name__ == "__main__":
    print("⚠️ This will ERASE corrections, images, DB, and model history.")
    confirm = input("Type 'RESET' to proceed: ").strip()
    if confirm != "RESET":
        print("Aborted.")
        raise SystemExit(0)

    # 1) Clear data folders
    print("🧹 Clearing data folders...")
    reset_data()
    ensure_dirs()

    # 2) Reset database
    print("🗄️ Resetting database...")
    reset_database()

    # 3) Reset models (delete_active=False keeps your base model)
    print("🧠 Resetting models (history only)...")
    reset_models(delete_active=False)

    print("✅ Project reset complete.")
    print("Folders are empty, DB is clean, model history cleared.")
