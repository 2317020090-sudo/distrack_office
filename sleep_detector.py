import cv2
import time
import os
import threading
import winsound
import numpy as np
import warnings
import mysql.connector 
import json             
import datetime          
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import math

# ================= INTEGRASI ANTI-SPOOFING =================
import sys
sys.path.append('./src') 
try:
    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    USE_DEEP_ANTISPOOF = True
    print("[INFO] Modul Silent-Face-Anti-Spoofing ditemukan.")
except ImportError:
    USE_DEEP_ANTISPOOF = False
    print("[WARNING] Modul 'src' tidak ditemukan. Mode Anti-Spoofing terbatas.")

warnings.filterwarnings('ignore')

# ================= KONFIGURASI DATABASE MYSQL =================
DB_CONFIG = {
    "database": "face_dbs",   
    "user": "root",          
    "password": "",          
    "host": "localhost",
    "port": 3306             
}

# ================= KONFIGURASI WAKTU & ATURAN =================
WORK_START_HOUR = 7            
WORK_END_HOUR = 17             
WORK_END_MINUTE = 30           

MAX_AWAY_LIMIT_SECONDS = 3600  # Batas keluar 1 jam

# [TAHAP 2] Batas waktu Validasi setelah terdeteksi tidur (10 Detik)
VIOLATION_CONFIRM_SECONDS = 10 

# ================= KONFIGURASI SISTEM =================
YOLO_CONFIDENCE = 0.50
MIN_BOX_AREA = 8000  
FREQUENCY_THRESHOLD = 50.0 
EAR_THRESHOLD = 0.22         # Sensitivitas mata

# [TAHAP 1] Ambang Batas DETEKSI AWAL (Sesuai permintaan Anda)
# Mata harus tertutup 15 detik dulu baru dibilang "TIDUR"
EYE_CLOSED_THRESHOLD = 15.0   
# Wajah harus hilang 30 detik dulu baru dibilang "KEMUNGKINAN TIDUR"
IDLE_THRESHOLD = 5.0         
NO_FACE_THRESHOLD = 5.0       

INSTANT_MOVE_LIMIT = 100
FACE_SIM_THRESHOLD = 0.50
LIVENESS_THRESHOLD = 0.85 
AWAY_TOLERANCE = 5.0        
ALARM_SOUND = "sound/warning.wav"
SKIP_FRAMES_FACE = 3
SKIP_FRAMES_LIVENESS = 60 

# ================= FOLDER =================
os.makedirs("captures/tidur", exist_ok=True)
os.makedirs("captures/kemungkinan_tidur", exist_ok=True)
os.makedirs("captures/active_after_sleep", exist_ok=True) 
os.makedirs("captures/keluar_batas", exist_ok=True)

# ================= MEMORI GLOBAL =================
active_violations = {} 
away_violation_tracker = {} 

# Mapping Hari (Inggris -> Indo)
DAYS_MAP = {
    'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
    'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
}

# ================= FUNGSI DATABASE =================
def get_db_conn():
    try: 
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e: 
        print(f"[KONEKSI ERROR] {e}")
        return None

def load_faces_from_db():
    print("[DB] Memuat dataset wajah dari MySQL...")
    conn = get_db_conn()
    if not conn: return [], [], []
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, embedding FROM users") 
        rows = cur.fetchall()
        if not rows: return [], [], []
        ids, names, embeds = [], [], []
        for row in rows:
            ids.append(row[0])
            names.append(row[1])
            json_str = row[2]
            if isinstance(json_str, str): emb_list = json.loads(json_str)
            else: emb_list = json_str
            embeds.append(np.array(emb_list, dtype=np.float32))
        print(f"[DB] {len(names)} wajah berhasil dimuat.")
        return ids, names, embeds
    except Exception as e: 
        print(f"[LOAD ERROR] {e}")
        return [], [], []
    finally: 
        if conn: conn.close()

# --- FUNGSI 1: MULAI SESI TIDUR ---
def start_violation_session(name, vio_type, frame):
    conn = get_db_conn()
    if not conn: return None
    
    folder = "captures/tidur" if vio_type == "TIDUR" else "captures/kemungkinan_tidur"
    filename = f"START_{name}_{int(time.time())}.jpg"
    full_path = f"{folder}/{filename}" 
    cv2.imwrite(full_path, frame)
    
    db_id = None
    try:
        cur = conn.cursor()
        query = """
            INSERT INTO violation_sessions 
            (name, violation_type, start_image_path, status_sesi, admin_status) 
            VALUES (%s, %s, %s, 'ONGOING', 'PENDING')
        """
        cur.execute(query, (name, vio_type, full_path))
        conn.commit()
        db_id = cur.lastrowid 
        print(f"🔴 [START] {name} mulai {vio_type}. DB ID: {db_id}")
    except Exception as e: 
        print(f"[DB START ERROR] {e}")
    finally: 
        conn.close()
    return db_id

# --- FUNGSI 2: AKHIRI SESI TIDUR ---
def end_violation_session(name, frame):
    if name not in active_violations: return
    
    session_data = active_violations[name]
    db_id = session_data["db_id"]
    # Kita pakai start_ts dari session_data agar durasi akurat sejak awal terdeteksi
    start_ts = session_data["start_ts"]
    
    folder = "captures/active_after_sleep"
    filename = f"END_{name}_{int(time.time())}.jpg"
    full_path = f"{folder}/{filename}"
    cv2.imwrite(full_path, frame)
    
    end_ts = time.time()
    duration_seconds = int(end_ts - start_ts)
    duration_str = str(datetime.timedelta(seconds=duration_seconds)) 
    
    conn = get_db_conn()
    if conn:
        try:
            cur = conn.cursor()
            query = """
                UPDATE violation_sessions 
                SET end_time = NOW(), 
                    end_image_path = %s, 
                    duration_str = %s, 
                    status_sesi = 'FINISHED'
                WHERE id = %s
            """
            cur.execute(query, (full_path, duration_str, db_id))
            conn.commit()
            print(f"🟢 [END] {name} bangun. Durasi: {duration_str}")
        except Exception as e: 
            print(f"[DB END ERROR] {e}")
        finally: 
            conn.close()

# --- FUNGSI 3: SIMPAN PELANGGARAN KELUAR ---
def save_away_violation(name, exit_time_ts, return_time_ts, frame):
    conn = get_db_conn()
    if not conn: return
    
    folder = "captures/keluar_batas"
    filename = f"AWAY_{name}_{int(time.time())}.jpg"
    full_path = f"{folder}/{filename}"
    evidence_frame = frame.copy()
    cv2.putText(evidence_frame, f"BUKTI: {name} TIDAK ADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(full_path, evidence_frame)

    duration_sec = int(return_time_ts - exit_time_ts)
    duration_str = str(datetime.timedelta(seconds=duration_sec))
    total_minutes = duration_sec // 60
    
    exit_dt_full = datetime.datetime.fromtimestamp(exit_time_ts).strftime('%Y-%m-%d %H:%M:%S')
    return_dt_full = datetime.datetime.fromtimestamp(return_time_ts).strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        cur = conn.cursor()
        query = """
            INSERT INTO away_logs 
            (name, exit_time, return_time, duration_str, total_minutes, evidence_path, status_validasi) 
            VALUES (%s, %s, %s, %s, %s, %s, 'PENDING')
        """
        cur.execute(query, (name, exit_dt_full, return_dt_full, duration_str, total_minutes, full_path))
        conn.commit()
        print(f"⚠️ [PELANGGARAN KELUAR] {name} tercatat keluar {duration_str}.")
    except Exception as e: 
        print(f"[DB AWAY ERROR] {e}")
    finally: 
        conn.close()

# --- FUNGSI 4: LOAD & SAVE BREAK LOGS ---
def get_current_indo_day():
    eng_day = datetime.datetime.now().strftime('%A')
    return DAYS_MAP.get(eng_day, eng_day)

def load_break_logs():
    conn = get_db_conn()
    data_sec = {}
    data_count = {} 
    today = get_current_indo_day()
    if not conn: return data_sec, data_count
    try:
        cur = conn.cursor()
        query = "SELECT name, total_seconds, exit_count FROM break_logs WHERE day_name = %s"
        cur.execute(query, (today,))
        rows = cur.fetchall()
        for row in rows: 
            data_sec[row[0]] = float(row[1])
            data_count[row[0]] = int(row[2]) if row[2] else 0
        return data_sec, data_count
    except: 
        return {}, {}
    finally: 
        conn.close()

def save_break_log_bulk(cache_sec, cache_count):
    conn = get_db_conn()
    if not conn: return
    today = get_current_indo_day()
    try:
        cur = conn.cursor()
        for name, seconds in cache_sec.items():
            count = cache_count.get(name, 0)
            query = """
                INSERT INTO break_logs (name, day_name, total_seconds, exit_count, last_updated)
                VALUES (%s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE 
                total_seconds = VALUES(total_seconds), 
                exit_count = VALUES(exit_count),
                last_updated = NOW()
            """
            cur.execute(query, (name, today, seconds, count))
        conn.commit()
    except Exception as e: 
        print(f"[DB UPDATE ERROR] {e}")
    finally: 
        conn.close()

def play_alarm():
    def _run():
        if os.path.exists(ALARM_SOUND):
            winsound.PlaySound(ALARM_SOUND, winsound.SND_FILENAME | winsound.SND_ASYNC)
    threading.Thread(target=_run).start()

# ================= UTILITY WAJAH =================
def cosine_sim(a, b): return np.dot(a, b) / (norm(a) * norm(b))

def recognize_face(embedding):
    if len(known_embeddings) == 0: return "Unknown", None
    best_score, best_name, best_id = 0, "Unknown", None
    for idx, (emb, name) in enumerate(zip(known_embeddings, known_names)):
        sim = cosine_sim(embedding, emb)
        if sim > best_score:
            best_score, best_name = sim, name
            best_id = known_ids[idx]
    if best_score >= FACE_SIM_THRESHOLD: return best_name, best_id
    return "Unknown", None

def calc_ear(lm, idx):
    p1 = lm[idx[0]]; p2 = lm[idx[1]]; p3 = lm[idx[2]] 
    p4 = lm[idx[3]]; p5 = lm[idx[4]]; p6 = lm[idx[5]] 
    v1 = np.linalg.norm(p2 - p6); v2 = np.linalg.norm(p3 - p5)
    h  = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

def point_inside(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def calculate_image_quality(img_face):
    try:
        gray = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except: return 0.0

def check_liveness_ai(frame, face_box):
    if not USE_DEEP_ANTISPOOF: return True, 1.0
    x1, y1, x2, y2 = face_box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: return False, 0.0
    prediction = model_test.predict(frame, [x1, y1, w, h])
    score = prediction[0][np.argmax(prediction)]
    return (np.argmax(prediction) == 1), score

def try_recover_identity(cx, cy, current_time):
    best_name, best_id, min_dist = "Unknown", None, 9999
    for name, data in last_known_positions.items():
        if current_time - data["time"] < 2.0:
            dist = math.hypot(cx - data["pos"][0], cy - data["pos"][1])
            if dist < 150 and dist < min_dist:
                min_dist, best_name, best_id = dist, name, data["db_id"]
    return best_name, best_id

# ================= INIT MODELS =================
print("[INFO] Memuat Model YOLOv8 & InsightFace (Buffalo_L)...")
model = YOLO("yolov8n.pt") 
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(480, 480))

if USE_DEEP_ANTISPOOF:
    model_test = AntiSpoofPredict(device_id=0) 
    image_cropper = CropImage()

# ================= MAIN INIT =================
known_ids, known_names, known_embeddings = load_faces_from_db()
break_database_cache, break_count_cache = load_break_logs()
employee_tracker = {}

for name in known_names:
    if name not in break_database_cache: break_database_cache[name] = 0.0
    if name not in break_count_cache: break_count_cache[name] = 0
    
    employee_tracker[name] = {
        "last_seen": 0, "status": "BELUM HADIR", "is_active_today": False, "exit_start_time": None
    }
    away_violation_tracker[name] = {"is_saved": False}

LEFT_IDX = [35, 36, 33, 39, 40, 41] 
RIGHT_IDX = [89, 90, 87, 93, 94, 95]
last_known_positions = {} 

# ================= MAIN LOOP =================
cap = cv2.VideoCapture(0)
cap.set(3, 640); cap.set(4, 480)

persons = {} 
frame_count = 0
last_time_frame = time.time()
last_db_sync = time.time()

print("[INFO] Sistem Deteksi & Tracking Durasi Berjalan...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 1. CEK HARI & JAM
        now = datetime.datetime.now()
        current_day_str = get_current_indo_day()
        current_hour = now.hour
        current_minute = now.minute
        
        is_working_hours = False
        if (current_hour > WORK_START_HOUR) or (current_hour == WORK_START_HOUR and current_minute >= 0):
            if (current_hour < WORK_END_HOUR) or (current_hour == WORK_END_HOUR and current_minute <= WORK_END_MINUTE):
                is_working_hours = True

        if not is_working_hours:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "SISTEM ISTIRAHAT", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Monitor Absensi Cerdas", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue 

        # === JAM KERJA AKTIF ===
        curr_time = time.time()
        time_delta = curr_time - last_time_frame
        last_time_frame = curr_time
        frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        if curr_time - last_db_sync > 10:
            threading.Thread(target=save_break_log_bulk, args=(break_database_cache.copy(), break_count_cache.copy())).start()
            last_db_sync = curr_time

        results = model.track(frame, persist=True, verbose=False, classes=[0, 67], 
                              conf=YOLO_CONFIDENCE, iou=0.5, tracker="botsort.yaml") 
        
        current_person_ids = []
        device_boxes = []
        person_boxes_map = {}

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, trk_id, cls in zip(boxes, ids, clss):
                if model.names[cls] == "person":
                    w_box = box[2] - box[0]; h_box = box[3] - box[1]
                    if (w_box * h_box) < MIN_BOX_AREA: continue 
                    cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
                    current_person_ids.append(trk_id)
                    person_boxes_map[trk_id] = (box[0], box[1], box[2], box[3], cx, cy)
                elif model.names[cls] == "cell phone": 
                    device_boxes.append(box)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)

        faces = []
        if frame_count % SKIP_FRAMES_FACE == 0:
            faces = face_app.get(frame)

        real_person_count = 0

        for pid in current_person_ids:
            x1, y1, x2, y2, cx, cy = person_boxes_map[pid]
            
            if pid not in persons:
                rec_name, rec_id = try_recover_identity(cx, cy, curr_time)
                persons[pid] = {
                    "last_position": (cx, cy), "last_move_time": time.time(),
                    "eye_closed_start": None, "no_face_start": None,
                    "photo_taken": False, "alarm_played": False,
                    "name": rec_name, "db_id": rec_id,
                    "face_locked": (rec_name != "Unknown"),
                    "is_photo_spatial": False, "is_photo_ai": False,
                    "liveness_score": 0.0, "last_ear": 0.3,
                    "freq_score": 0.0, "is_blur_spoof": False,
                    "violation_start_timer": None 
                }
            p = persons[pid]

            instant_move = abs(cx - p["last_position"][0]) + abs(cy - p["last_position"][1])
            p["last_position"] = (cx, cy)
            
            if instant_move > INSTANT_MOVE_LIMIT:
                p["last_move_time"] = time.time()
                p["eye_closed_start"] = None; p["no_face_start"] = None; p["photo_taken"] = False
                p["violation_start_timer"] = None # Reset validasi jika gerak banyak
            
            idle_time = time.time() - p["last_move_time"]

            if p["name"] != "Unknown":
                last_known_positions[p["name"]] = {"pos": (cx, cy), "time": curr_time, "db_id": p["db_id"]}

            if not p["face_locked"] or p["name"] == "Unknown":
                if len(faces) > 0:
                    for face in faces:
                        fb = face.bbox.astype(int)
                        fcx, fcy = (fb[0]+fb[2])//2, (fb[1]+fb[3])//2
                        if x1 < fcx < x2 and y1 < fcy < y2:
                            res_name, res_id = recognize_face(face.embedding)
                            if res_name != "Unknown":
                                p["name"] = res_name; p["db_id"] = res_id; p["face_locked"] = True 
                            
                            crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                            if crop.size > 0:
                                p["freq_score"] = calculate_image_quality(crop)
                                p["is_blur_spoof"] = p["freq_score"] < FREQUENCY_THRESHOLD

                            if not p["is_blur_spoof"]:
                                lm = np.array(face.landmark_2d_106).astype(int)
                                p["last_ear"] = (calc_ear(lm, LEFT_IDX) + calc_ear(lm, RIGHT_IDX)) / 2
                                if p["last_ear"] < EAR_THRESHOLD:
                                    if p["eye_closed_start"] is None: p["eye_closed_start"] = time.time()
                                else: p["eye_closed_start"] = None
                            else: p["eye_closed_start"] = None

                            if frame_count % SKIP_FRAMES_LIVENESS == 0 and USE_DEEP_ANTISPOOF:
                                is_real, score = check_liveness_ai(frame, fb)
                                p["liveness_score"] = score
                                p["is_photo_ai"] = (not is_real or score < LIVENESS_THRESHOLD)
                            p["no_face_start"] = None
                            break
                    else:
                        if p["no_face_start"] is None: p["no_face_start"] = time.time()
                else:
                    if p["no_face_start"] is None: p["no_face_start"] = time.time()
            else:
                if len(faces) > 0:
                    for face in faces:
                          fb = face.bbox.astype(int)
                          fcx, fcy = (fb[0]+fb[2])//2, (fb[1]+fb[3])//2
                          if x1 < fcx < x2 and y1 < fcy < y2:
                             crop = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                             if crop.size > 0: score = calculate_image_quality(crop) < FREQUENCY_THRESHOLD
                             if not score:
                                 lm = np.array(face.landmark_2d_106).astype(int)
                                 p["last_ear"] = (calc_ear(lm, LEFT_IDX) + calc_ear(lm, RIGHT_IDX)) / 2
                                 if p["last_ear"] < EAR_THRESHOLD:
                                    if p["eye_closed_start"] is None: p["eye_closed_start"] = time.time()
                                 else: p["eye_closed_start"] = None
                             else: p["eye_closed_start"] = None
                             p["no_face_start"] = None
                             break
                    else:
                         if p["no_face_start"] is None: p["no_face_start"] = time.time()

            # --- STATUS PENENTUAN (TAHAP 1: DETEKSI AWAL) ---
            status = "AKTIF"
            p["is_photo_spatial"] = any(point_inside(cx, cy, dbox) for dbox in device_boxes)

            if instant_move > INSTANT_MOVE_LIMIT: status = "AKTIF (GERAK)"
            elif p["is_photo_spatial"]: status = "SPOOF (HP)"
            elif p["is_photo_ai"]: status = f"SPOOF (AI)"
            elif p["eye_closed_start"] and (time.time() - p["eye_closed_start"] > EYE_CLOSED_THRESHOLD):
                # TAHAP 1: Mata tertutup > 15 detik? -> Status TIDUR (Tapi belum direkam)
                if idle_time > 2.0: status = "TIDUR"
                else: status = "AKTIF (MATA)"
            elif p["no_face_start"] and (time.time() - p["no_face_start"] > NO_FACE_THRESHOLD):
                 # TAHAP 1: Wajah hilang > 5 detik & diam lama -> Status KEMUNGKINAN TIDUR
                 if idle_time > IDLE_THRESHOLD: status = "KEMUNGKINAN TIDUR"
                 else: status = "AKTIF (MENUNDUK)"

            if "SPOOF" not in status:
                real_person_count += 1
                if p["name"] != "Unknown" and p["name"] in employee_tracker:
                    employee_tracker[p["name"]]["last_seen"] = curr_time
                    employee_tracker[p["name"]]["status"] = "HADIR"
                    employee_tracker[p["name"]]["is_active_today"] = True

            # ========================================================
            # LOGIKA INTI (TAHAP 2 & 3: VALIDASI & SIMPAN)
            # ========================================================
            
            # Jika Status sudah TIDUR / KEMUNGKINAN TIDUR (dari Tahap 1)
            if status in ["TIDUR", "KEMUNGKINAN TIDUR"] and p["name"] != "Unknown":
                
                # 1. Mulai Timer Validasi 10 Detik
                if p["violation_start_timer"] is None:
                    p["violation_start_timer"] = time.time()
                
                # Hitung waktu validasi
                validation_time = time.time() - p["violation_start_timer"]
                
                # 2. Cek apakah sudah TEREKAM sebelumnya?
                if p["name"] in active_violations:
                    # SUDAH TEREKAM (Tahap 3 Selesai, Lanjut Monitoring)
                    color = (0, 0, 255) # Merah
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Hitung Total Durasi (Sejak validasi dimulai)
                    total_duration = int(validation_time)
                    duration_str = str(datetime.timedelta(seconds=total_duration))
                    
                    label_txt = f"{status} ({duration_str})"
                    cv2.putText(frame, label_txt, (x1, y1-15), 0, 0.6, color, 2)

                else:
                    # BELUM TEREKAM (Masih di Tahap 2: Validasi)
                    if validation_time >= VIOLATION_CONFIRM_SECONDS:
                        # TAHAP 3: Waktu Validasi Habis -> JEPRET!
                        new_id = start_violation_session(p["name"], status, frame)
                        if new_id:
                            active_violations[p["name"]] = {
                                "db_id": new_id,
                                "start_ts": p["violation_start_timer"], 
                                "type": status
                            }
                            if not p["alarm_played"]: play_alarm(); p["alarm_played"] = True
                    else:
                        # Masih Menghitung Mundur (Verifikasi)
                        color = (0, 165, 255) # Orange
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        countdown = VIOLATION_CONFIRM_SECONDS - int(validation_time)
                        cv2.putText(frame, f"Verifikasi... {countdown}s", (x1, y1-35), 0, 0.6, color, 2)
                        cv2.putText(frame, status, (x1, y1-15), 0, 0.6, color, 2)
            
            else:
                # Jika status kembali normal (Bangun)
                p["violation_start_timer"] = None
                p["alarm_played"] = False 
                
                # Jika sebelumnya sedang tidur, tutup sesi database
                if p["name"] in active_violations:
                    end_violation_session(p["name"], frame)
                    del active_violations[p["name"]]
                
                color = (0, 255, 0)
                if "SPOOF" in status: color = (100, 100, 100)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                id_display = f"[{p['db_id']}]" if p['db_id'] is not None else ""
                cv2.putText(frame, f"{id_display} {p['name']} | {status}", (x1, y1-5), 0, 0.5, color, 1)

        cv2.putText(frame, f"Orang: {real_person_count}", (10, 30), 0, 0.5, (0, 255, 0), 1)

        # DASHBOARD KECIL
        cv2.rectangle(frame, (0, 40), (280, 40 + len(known_names)*15 + 20), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS ({current_day_str.upper()}):", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        y = 70
        for idx, name in enumerate(known_names):
            uid = known_ids[idx]
            tracker = employee_tracker[name]
            
            if tracker["is_active_today"]:
                is_away = (curr_time - tracker["last_seen"]) > AWAY_TOLERANCE
                if is_away:
                    if tracker["status"] == "HADIR":
                         tracker["status"] = "KELUAR"
                         tracker["exit_start_time"] = tracker["last_seen"] 
                         away_violation_tracker[name]["is_saved"] = False
                    
                    if tracker["exit_start_time"]: current_duration = curr_time - tracker["exit_start_time"]
                    else: current_duration = 0 
                    
                    break_database_cache[name] += time_delta
                    mins, secs = divmod(int(current_duration), 60)
                    hrs, mins = divmod(mins, 60)
                    leave_time_str = time.strftime('%H:%M', time.localtime(tracker["exit_start_time"]))
                    
                    txt = f"[{uid}] {name}: KELUAR (@{leave_time_str} | {hrs}j {mins}m {secs}s)"
                    clr = (0, 0, 255)
                    
                    if current_duration > MAX_AWAY_LIMIT_SECONDS:
                         if not away_violation_tracker[name]["is_saved"]:
                             save_away_violation(name, tracker["exit_start_time"], curr_time, frame)
                             away_violation_tracker[name]["is_saved"] = True
                             play_alarm()
                else:
                    if tracker["status"] == "KELUAR":
                        break_count_cache[name] += 1
                        tracker["status"] = "HADIR"
                        tracker["exit_start_time"] = None
                        away_violation_tracker[name]["is_saved"] = False
                    txt = f"[{uid}] {name}: HADIR ({break_count_cache[name]}x)"
                    clr = (0, 255, 0)
            else:
                txt = f"[{uid}] {name}: BELUM ABSEN"
                clr = (150, 150, 150)
            cv2.putText(frame, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, clr, 1)
            y += 15

        cv2.imshow("Monitor Absensi Cerdas", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

except Exception as e: 
    print(f"Error Utama: {e}")
finally:
    print("Menutup sesi yang masih aktif...")
    for name in list(active_violations.keys()): 
        end_violation_session(name, frame)
    save_break_log_bulk(break_database_cache, break_count_cache)
    cap.release()
    cv2.destroyAllWindows()
