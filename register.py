import cv2
import numpy as np
import psycopg2
import winsound  # <--- INI YANG KURANG TADI
import time
import math
from insightface.app import FaceAnalysis

# ================= KONFIGURASI DATABASE =================
DB_CONFIG = {
    "dbname": "face_db",
    "user": "postgres",
    "password": "020206",  # <--- JANGAN LUPA GANTI PASSWORDNYA
    "host": "localhost",
    "port": "5432"
}

# ================= KONFIGURASI TANTANGAN =================
# Urutan tantangan gerakan kepala
STAGES = ["DEPAN", "KANAN", "KIRI", "ATAS", "BAWAH"]
STAGE_DURATION = 15  # Butuh berapa frame menahan posisi agar lolos (15 frame ~= 0.5 detik)

# ================= INISIALISASI AI =================
print("[INIT] Memuat Model AI (InsightFace)...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ================= HELPER FUNCTIONS =================
def get_db_conn():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR DB] {e}")
        return None

def calculate_pose(landmarks):
    """
    Menentukan arah wajah berdasarkan 5 titik landmark utama (Mata, Hidung, Mulut)
    Return: "DEPAN", "KANAN", "KIRI", "ATAS", "BAWAH"
    """
    # Landmark indices (InsightFace 5 points):
    # 0: Mata Kiri, 1: Mata Kanan, 2: Hidung, 3: Mulut Kiri, 4: Mulut Kanan
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    # --- 1. HITUNG YAW (KIRI/KANAN) ---
    # Bandingkan jarak hidung ke mata kiri vs mata kanan
    dist_nose_left_eye = math.hypot(nose[0] - left_eye[0], nose[1] - left_eye[1])
    dist_nose_right_eye = math.hypot(nose[0] - right_eye[0], nose[1] - right_eye[1])
    
    # Rasio horizontal
    if dist_nose_right_eye == 0: ratio_yaw = 0
    else: ratio_yaw = dist_nose_left_eye / dist_nose_right_eye

    # --- 2. HITUNG PITCH (ATAS/BAWAH) ---
    # Titik tengah mata
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    # Jarak vertikal hidung ke mata vs hidung ke mulut
    dist_nose_eyes = nose[1] - eye_center_y
    dist_nose_mouth = (left_mouth[1] + right_mouth[1]) / 2 - nose[1]

    if dist_nose_mouth == 0: ratio_pitch = 0
    else: ratio_pitch = dist_nose_eyes / dist_nose_mouth

    # --- 3. TENTUKAN ARAH (THRESHOLD KALIBRASI) ---
    direction = "DEPAN"
    
    # Ambang batas (Sensitivity)
    if ratio_yaw < 0.6: direction = "KANAN"  # Hidung lebih dekat ke mata kanan
    elif ratio_yaw > 1.6: direction = "KIRI"   # Hidung lebih dekat ke mata kiri
    elif ratio_pitch < 0.6: direction = "ATAS" 
    elif ratio_pitch > 1.3: direction = "BAWAH"

    return direction, (ratio_yaw, ratio_pitch)

def draw_ui(frame, current_stage, progress, instruction_text, status_color):
    h, w = frame.shape[:2]
    
    # 1. Overlay Hitam Semi-Transparan di atas
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
    
    # 2. Teks Instruksi Besar
    cv2.putText(frame, instruction_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # 3. Progress Bar (Garis Hijau di bawah teks)
    bar_width = int((progress / STAGE_DURATION) * (w - 100))
    cv2.rectangle(frame, (50, 80), (50 + bar_width, 90), status_color, -1)
    cv2.rectangle(frame, (50, 80), (w - 50, 90), (255, 255, 255), 2) # Border

    return frame

# ================= MAIN FUNCTION =================
def bank_registration():
    print("\n" + "="*40)
    print("   PENDAFTARAN WAJAH ")
    print("="*40)
    
    user_name = input("Masukkan Nama Anda: ").strip()
    if not user_name: return

    cap = cv2.VideoCapture(0)
    
    stage_idx = 0
    progress_counter = 0
    best_frontal_embedding = None 
    
    print("[INFO] Silakan ikuti instruksi di layar...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Flip frame agar seperti cermin
        frame = cv2.flip(frame, 1) 
        
        faces = face_app.get(frame)
        current_instruction = f"MOHON HADAP: {STAGES[stage_idx]}"
        status_color = (0, 255, 255) # Kuning (Waiting)

        if len(faces) > 0:
            # Ambil wajah terbesar
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
            face = faces[0]
            box = face.bbox.astype(int)
            landmarks = face.kps 
            
            # Hitung Pose
            pose_direction, ratios = calculate_pose(landmarks)
            
            # Visualisasi Wajah
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
            
            # --- LOGIKA TANTANGAN ---
            target_pose = STAGES[stage_idx]
            
            if pose_direction == target_pose:
                status_color = (0, 255, 0) # Hijau
                progress_counter += 1
                
                # Simpan embedding saat pose DEPAN
                if target_pose == "DEPAN" and progress_counter > 5:
                    best_frontal_embedding = face.embedding
            else:
                progress_counter = max(0, progress_counter - 1)
                status_color = (0, 0, 255) # Merah
                
                cv2.putText(frame, f"Terdeteksi: {pose_direction}", (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- CEK APAKAH TANTANGAN SELESAI ---
            if progress_counter >= STAGE_DURATION:
                stage_idx += 1
                progress_counter = 0
                
                # BEEP BERHASIL
                try:
                    winsound.Beep(1000, 200) 
                except: pass
                
                if stage_idx >= len(STAGES):
                    # SEMUA TANTANGAN SELESAI
                    save_success = finish_registration(user_name, best_frontal_embedding)
                    if save_success:
                        cv2.putText(frame, "REGISTRASI SUKSES!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                        cv2.imshow("Bank Registration", frame)
                        cv2.waitKey(2000)
                        break
                    else:
                        print("[ERROR] Gagal menyimpan ke DB.")
                        break
        else:
            cv2.putText(frame, "Wajah tidak ditemukan", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame = draw_ui(frame, STAGES[stage_idx], progress_counter, current_instruction, status_color)
        
        cv2.imshow("Bank Registration", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            print("[INFO] Dibatalkan oleh user.")
            break

    cap.release()
    cv2.destroyAllWindows()

def finish_registration(name, embedding):
    """Simpan data final ke PostgreSQL"""
    if embedding is None:
        print("[ERROR] Embedding DEPAN tidak terekam dengan baik.")
        return False
        
    conn = get_db_conn()
    if not conn: return False

    try:
        cur = conn.cursor()
        emb_list = embedding.tolist()
        
        # Query Upsert
        query = """
            INSERT INTO users (name, embedding) 
            VALUES (%s, %s)
            ON CONFLICT (name) 
            DO UPDATE SET embedding = EXCLUDED.embedding;
        """
        cur.execute(query, (name, emb_list))
        conn.commit()
        
        print("\n" + "="*50)
        print(f"✅ PENDAFTARAN BERHASIL!")
        print(f"   Nama: {name}")
        print(f"   Status: Terverifikasi (Liveness Check Passed)")
        print("="*50)
        return True
    except Exception as e:
        print(f"[DB ERROR] {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    bank_registration()