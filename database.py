# database.py
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os

DB_PATH = "data/corrections.db"

def init_database():
    """Initialize SQLite database with corrections table"""
    os.makedirs("data", exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            original_image_path TEXT NOT NULL,
            predicted_mask_path TEXT NOT NULL,
            corrected_mask_path TEXT NOT NULL,
            loss_score REAL NOT NULL,
            used_for_training INTEGER DEFAULT 0,
            model_version TEXT,
            user_notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

def save_correction(original_img_path: str, predicted_mask_path: str, 
                   corrected_mask_path: str, loss_score: float, 
                   model_version: str = "v1") -> int:
    """Save a correction to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO corrections 
        (timestamp, original_image_path, predicted_mask_path, corrected_mask_path, 
         loss_score, model_version)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, original_img_path, predicted_mask_path, corrected_mask_path, 
          loss_score, model_version))
    
    correction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return correction_id

def get_hardest_cases(n: int = 20, unused_only: bool = True) -> List[Dict]:
    """Get hardest cases (highest loss) for retraining"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = '''
        SELECT id, original_image_path, corrected_mask_path, loss_score 
        FROM corrections 
    '''
    
    if unused_only:
        query += " WHERE used_for_training = 0"
    
    query += " ORDER BY loss_score DESC LIMIT ?"
    
    cursor.execute(query, (n,))
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "image_path": row[1],
            "mask_path": row[2],
            "loss_score": row[3]
        }
        for row in rows
    ]

def mark_as_used(correction_ids: List[int]):
    """Mark corrections as used for training"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    placeholders = ','.join('?' * len(correction_ids))
    cursor.execute(f'''
        UPDATE corrections 
        SET used_for_training = 1 
        WHERE id IN ({placeholders})
    ''', correction_ids)
    
    conn.commit()
    conn.close()

def get_stats() -> Dict:
    """Get correction statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM corrections")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM corrections WHERE used_for_training = 1")
    used = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(loss_score) FROM corrections")
    avg_loss = cursor.fetchone()[0] or 0.0
    
    conn.close()
    
    return {
        "total": total,
        "used": used,
        "pending": total - used,
        "avg_loss": avg_loss
    }
