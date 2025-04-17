import cv2
import numpy as np
# Removed: from ultralytics import YOLO
# Removed: import tensorflow as tf
import sqlite3
from scipy.spatial.distance import cosine
import time
import json
import streamlit as st
import os
from pathlib import Path
import av # Needed for VideoFrame
import queue # For communication
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis

BASE_DIR = Path(os.path.dirname(__file__)).parent # Should resolve to D:\flask

def find_matching_face(embedding, threshold=0.6):
    """Compares an embedding against the database and returns user info if matched."""
    conn = None
    best_match_info = None
    max_similarity = -1.0  # Khởi tạo nhỏ nhất vì similarity nằm trong [-1, 1]

    try:
        db_path = os.path.join(BASE_DIR, 'Database.db')
        if not os.path.exists(db_path):
            print(f"Error: Database not found at {db_path}")
            return None

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT embedding, name, id FROM customers WHERE embedding IS NOT NULL")
        results = cursor.fetchall()

        if embedding is None or embedding.ndim != 1 or embedding.size == 0:
            return None

        for db_embedding_str, name, id in results:
            try:
                if not db_embedding_str:
                    continue
                db_embedding = np.array(json.loads(db_embedding_str), dtype=np.float32)

                if db_embedding.ndim != 1 or db_embedding.size == 0:
                    continue

                # Tính cosine similarity
                similarity = np.dot(embedding, db_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_info = {'name': name, 'id': id, 'similarity': similarity}

            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue

        if best_match_info and max_similarity >= threshold:
            print(f"Match found: {best_match_info['name']} (ID: {best_match_info['id']}), Similarity: {max_similarity:.4f}")
            return {'name': best_match_info['name'], 'id': best_match_info['id']}

        return None

    except sqlite3.Error as e:
        print(f"Database error in find_matching_face: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in find_matching_face: {e}")
        return None
    finally:
        if conn:
            conn.close()



class FaceAuthTransformer:
    """Handles face detection and recognition without Streamlit dependencies."""
    def __init__(self, model_name="det_10g.onnx"):
        """Initializes face detection and recognition models."""
        print("🟢 Initializing FaceAuthTransformer (Flask version)...")

        try:
            model_dir = os.path.join(BASE_DIR, 'models')
            det_model_path = os.path.join(model_dir, model_name)
            if not os.path.exists(det_model_path):
                 raise FileNotFoundError(f"Detection model not found at expected location: {det_model_path}")

            print(f"📦 Loading RetinaFace from {det_model_path}")
            self.det_model = get_model(det_model_path)

            self.det_model.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5)
            print("✅ RetinaFace loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading RetinaFace: {e}")
            raise

        try:
            print("📦 Loading ArcFace (FaceAnalysis)...")
            # Specify CPUExecutionProvider explicitly if not using GPU
            # ArcFace models are typically handled internally by insightface, no explicit path needed usually
            self.arcface_app = FaceAnalysis(name="buffalo_l", root=model_dir, providers=["CPUExecutionProvider"]) # Point root to model dir
            # Use CPU context (ctx_id=-1) or 0 for GPU
            self.arcface_app.prepare(ctx_id=-1)
            print("✅ ArcFace loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading ArcFace: {e}")
            raise

        print("✅ FaceAuthTransformer initialized.")

    def recognize_face(self, frame):
        """ 
        Processes a single frame to detect and recognize a face.

        Args:
            frame: A numpy array representing the image frame (BGR format).

        Returns:
            dict: Containing keys 'match' (user_info dict, False, or None), 
                'bbox' (list [x1, y1, x2, y2] or None), and 
                'confidence' (float or None).
        """
        if frame is None:
            return {'match': None, 'bbox': None, 'confidence': None}

        result = {'match': None, 'bbox': None, 'confidence': None}

        try:
            bboxes, landmarks = self.det_model.detect(frame, max_num=0, metric='default')

            if bboxes is None or len(bboxes) == 0:
                return result

            # Chọn bbox có độ tự tin cao nhất
            best_bbox = None
            best_confidence = -1

            for bbox in bboxes:
                conf = float(bbox[4])
                if conf > best_confidence:
                    best_confidence = conf
                    best_bbox = bbox

            if best_bbox is None or best_confidence < self.det_model.det_thresh:
                result['bbox'] = best_bbox[:4].astype(int).tolist() if best_bbox is not None else None
                result['confidence'] = best_confidence
                result['match'] = None
                return result

            x1, y1, x2, y2 = best_bbox[:4].astype(int)
            result['bbox'] = [x1, y1, x2, y2]
            result['confidence'] = best_confidence

            # Cắt ảnh khuôn mặt theo bbox
            # face_crop = frame[y1:y2, x1:x2]
            face_crop = frame
            if face_crop.size == 0:
                print("❌ Empty crop region.")
                result['match'] = False
                return result

            # Resize về 112x112 để phù hợp với ArcFace
            face_crop_resized = cv2.resize(face_crop, (112, 112))

            # Lấy embedding từ ArcFace
            arcface_faces = self.arcface_app.get(face_crop_resized)

            if not arcface_faces:
                result['match'] = False
                return result

            target_face = arcface_faces[0]
            embedding = target_face.embedding

            if embedding is None or np.linalg.norm(embedding) == 0:
                print("❌ ArcFace embedding extraction failed or embedding is zero.")
                result['match'] = False
                return result

            # So khớp với cơ sở dữ liệu
            match_info = find_matching_face(embedding)

            result['match'] = match_info if match_info else False
            return result

        except Exception as e:
            print(f"❌ Error during face recognition processing: {e}")
            return {'match': None, 'bbox': None, 'confidence': None}
