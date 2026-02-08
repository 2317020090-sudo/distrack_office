import cv2
import numpy as np
import mysql.connector
import winsound
import time
import math
import hashlib
import json
import pwinput  # Tetap butuh ini untuk konfirmasi password
from insightface.app import FaceAnalysis

# ================= KONFIGURASI DATABASE MYSQL =================
DB_CONFIG = {
    "database": "face_dbs",
    "user": "root",
    "password": "",
    "host": "localhost",
    "port": 3306
}

# ================= KONFIGURASI TANTANGAN =================
STAGES = ["DEPAN", "KANAN", "KIRI", "ATAS", "BAWAH"]
STAGE_DURATION = 15 

# ================= INISIALISASI AI =================
print("[INIT] Memuat Model AI (InsightFace)...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ================= HELPER FUNCTIONS =================
def get_db_conn():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR KONEKSI MYSQL] {e}")
        return None

def hash_password(password):
    """Mengubah password menjadi kode acak (SHA-256)"""
    return hashlib.sha256(password.encode()).hexdigest()

def calculate_pose(landmarks):
    left_eye = landmarks[0]; right_eye = landmarks[1]; nose = landmarks[2]
    left_mouth = landmarks[3]; right_mouth = landmarks[4]

    dist_nose_left_eye = math.hypot(nose[0] - left_eye[0], nose[1] - left_eye[1])
    dist_nose_right_eye = math.hypot(nose[0] - right_eye[0], nose[1] - right_eye[1])
    
    if dist_nose_right_eye == 0: ratio_yaw = 0
    else: ratio_yaw = dist_nose_left_eye / dist_nose_right_eye

    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    dist_nose_eyes = nose[1] - eye_center_y
    dist_nose_mouth = (left_mouth[1] + right_mouth[1]) / 2 - nose[1]

    if dist_nose_mouth == 0: ratio_pitch = 0
    else: ratio_pitch = dist_nose_eyes / dist_nose_mouth

    direction = "DEPAN"
    if ratio_yaw < 0.6: direction = "KANAN"
    elif ratio_yaw > 1.6: direction = "KIRI"
    elif ratio_pitch < 0.6: direction = "ATAS" 
    elif ratio_pitch > 1.3: direction = "BAWAH"

    return direction, (ratio_yaw, ratio_pitch)

def draw_ui(frame, current_stage, progress, instruction_text, status_color):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.putText(frame, instruction_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    bar_width = int((progress / STAGE_DURATION) * (w - 100))
    cv2.rectangle(frame, (50, 80), (50 + bar_width, 90), status_color, -1)
    cv2.rectangle(frame, (50, 80), (w - 50, 90), (255, 255, 255), 2)
    return frame

# ================= MAIN FUNCTION =================
def register_user_process():
    print("\n" + "="*40)
    print("   PENDAFTARAN USER BARU (MYSQL) ")
    print("="*40)
    
    # --- 1. INPUT DATA USER ---
    user_name = input("1. Masukkan Nama Lengkap : ").strip()
    user_email = input("2. Masukkan Gmail        : ").strip()
    
    while True:
        # --- PERUBAHAN DI SINI ---
        # Password UTAMA: Pakai input() biasa (Teks Terlihat)
        user_pass = input("3. Buat Password         : ")
        
        # Password KONFIRMASI: Pakai pwinput (Teks Bintang ******)
        user_pass_conf = pwinput.pwinput(prompt="4. Ulangi Password       : ", mask="*")
        
        if user_pass == user_pass_conf and user_pass != "":
            print("✅ Password valid!")
            break
        else:
            print("❌ Password tidak sama atau kosong. Ulangi.\n")

    if not user_name or not user_email: 
        print("Nama dan Email wajib diisi.")
        return

    # --- 2. MULAI KAMERA ---
    cap = cv2.VideoCapture(0)
    stage_idx = 0
    progress_counter = 0
    best_frontal_embedding = None 
    
    print("\n[INFO] Ikuti instruksi gerakan kepala...")
    time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) 
        faces = face_app.get(frame)
        current_instruction = f"MOHON HADAP: {STAGES[stage_idx]}"
        status_color = (0, 255, 255)

        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
            face = faces[0]
            box = face.bbox.astype(int)
            landmarks = face.kps 
            
            pose_direction, ratios = calculate_pose(landmarks)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
            
            target_pose = STAGES[stage_idx]
            
            if pose_direction == target_pose:
                status_color = (0, 255, 0)
                progress_counter += 1
                if target_pose == "DEPAN" and progress_counter > 5:
                    best_frontal_embedding = face.embedding
            else:
                progress_counter = max(0, progress_counter - 1)
                status_color = (0, 0, 255)
                cv2.putText(frame, f"Terdeteksi: {pose_direction}", (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if progress_counter >= STAGE_DURATION:
                stage_idx += 1
                progress_counter = 0
                try: winsound.Beep(1000, 200) 
                except: pass
                
                if stage_idx >= len(STAGES):
                    save_success = finish_registration_mysql(user_name, user_email, user_pass, best_frontal_embedding)
                    
                    if save_success:
                        cv2.putText(frame, "REGISTRASI SUKSES!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                        cv2.imshow("Registration", frame)
                        cv2.waitKey(3000)
                        break
                    else:
                        print("[ERROR] Gagal menyimpan.")
                        break
        else:
            cv2.putText(frame, "Wajah tidak ditemukan", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame = draw_ui(frame, STAGES[stage_idx], progress_counter, current_instruction, status_color)
        cv2.imshow("Registration", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

def finish_registration_mysql(name, email, password, embedding):
    if embedding is None:
        print("[ERROR] Embedding DEPAN tidak terekam.")
        return False
        
    conn = get_db_conn()
    if not conn: return False

    try:
        cur = conn.cursor()
        
        # 1. Hashing Password
        pass_hash = hash_password(password)
        
        # 2. Konversi Array ke JSON String
        emb_json = json.dumps(embedding.tolist())
        
        # 3. Query MySQL
        query = """
            INSERT INTO users (name, email, password_hash, embedding) 
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
            email = VALUES(email),
            password_hash = VALUES(password_hash),
            embedding = VALUES(embedding);
        """
        
        cur.execute(query, (name, email, pass_hash, emb_json))
        conn.commit()
        
        print("\n" + "="*50)
        print(f"✅ PENDAFTARAN USER BERHASIL (MYSQL)!")
        print(f"   Nama  : {name}")
        print(f"   Email : {email}")
        print("="*50)
        return True
    except Exception as e:
        print(f"[DB ERROR] {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    register_user_process()