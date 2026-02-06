import cv2
import time
import os
import threading
import winsound
import numpy as np
import warnings
import psycopg2 
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

# ================= KONFIGURASI DATABASE =================
DB_CONFIG = {
    "dbname": "face_db", 
    "user": "postgres",
    "password": "020206",      # Password sesuai request
    "host": "localhost",
    "port": "5432"
}

# ================= KONFIGURASI SISTEM =================
YOLO_CONFIDENCE = 0.40 
MIN_BOX_AREA = 8000  

FREQUENCY_THRESHOLD = 50.0 
EAR_THRESHOLD = 0.25 
MOVE_THRESHOLD = 3      
INSTANT_MOVE_LIMIT = 5  

IDLE_THRESHOLD = 120
EYE_CLOSED_THRESHOLD = 60 
NO_FACE_THRESHOLD = 5
FACE_SIM_THRESHOLD = 0.50
LIVENESS_THRESHOLD = 0.85 
AWAY_TOLERANCE = 5.0        

ALARM_SOUND = "sound/warning.wav"
SKIP_FRAMES_FACE = 5      
SKIP_FRAMES_LIVENESS = 60 

# ================= FOLDER =================
os.makedirs("captures/tidur", exist_ok=True)
os.makedirs("captures/kemungkinan_tidur", exist_ok=True)
os.makedirs("captures/spoofing", exist_ok=True)

# ================= FUNGSI DATABASE =================
def get_db_conn():
    try: return psycopg2.connect(**DB_CONFIG)
    except: return None

def load_faces_from_db():
    print("[DB] Memuat dataset wajah dari PostgreSQL...")
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
            embeds.append(np.array(row[2], dtype=np.float32))
        print(f"[DB] {len(names)} wajah berhasil dimuat.")
        return ids, names, embeds
    except: return [], [], []
    finally:
        if conn: conn.close()

def load_break_logs():
    conn = get_db_conn()
    data = {}
    if not conn: return data
    try:
        cur = conn.cursor()
        cur.execute("SELECT name, total_seconds FROM break_logs")
        rows = cur.fetchall()
        for row in rows:
            data[row[0]] = float(row[1])
        return data
    except: return {}
    finally: conn.close()

def save_break_log_bulk(cache_data):
    conn = get_db_conn()
    if not conn: return
    try:
        cur = conn.cursor()
        for name, seconds in cache_data.items():
            query = """
                INSERT INTO break_logs (name, total_seconds, last_updated)
                VALUES (%s, %s, NOW())
                ON CONFLICT (name) DO UPDATE 
                SET total_seconds = EXCLUDED.total_seconds, last_updated = NOW();
            """
            cur.execute(query, (name, seconds))
        conn.commit()
    except: pass
    finally: conn.close()

# ================= MODELS =================
print("[INFO] Memuat Model YOLOv8 & InsightFace (Buffalo_L)...")
model = YOLO("yolov8n.pt") 

# [PENTING] RAM SAFE MODE
# Menggunakan buffalo_l tapi dengan ukuran deteksi 320x320 agar tidak error "Bad Allocation"
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(480, 480))

if USE_DEEP_ANTISPOOF:
    model_test = AntiSpoofPredict(device_id=0) 
    image_cropper = CropImage()

# ================= INIT DATA =================
known_ids, known_names, known_embeddings = load_faces_from_db()
break_database_cache = load_break_logs()
employee_tracker = {}
for name in known_names:
    if name not in break_database_cache: break_database_cache[name] = 0.0
    employee_tracker[name] = {"last_seen": 0, "status": "BELUM HADIR", "is_active_today": False}

# ================= UTILITY & RECOVERY =================
LEFT_IDX = [35, 36, 33, 39, 40, 41] 
RIGHT_IDX = [89, 90, 87, 93, 94, 95]
last_known_positions = {} 

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

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

def play_alarm():
    def _run():
        if os.path.exists(ALARM_SOUND):
            winsound.PlaySound(ALARM_SOUND, winsound.SND_FILENAME | winsound.SND_ASYNC)
    threading.Thread(target=_run).start()

# ================= MAIN LOOP =================
cap = cv2.VideoCapture(0)
cap.set(3, 640); cap.set(4, 480)

persons = {} 
frame_count = 0
last_time_frame = time.time()
last_db_sync = time.time()

print("[INFO] Sistem Berjalan...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        curr_time = time.time()
        time_delta = curr_time - last_time_frame
        last_time_frame = curr_time
        frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        if curr_time - last_db_sync > 10:
            threading.Thread(target=save_break_log_bulk, args=(break_database_cache.copy(),)).start()
            last_db_sync = curr_time

        # 2. YOLO TRACKING
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
                    "freq_score": 0.0, "is_blur_spoof": False
                }
            p = persons[pid]

            instant_move = abs(cx - p["last_position"][0]) + abs(cy - p["last_position"][1])
            p["last_position"] = (cx, cy)
            
            if instant_move > INSTANT_MOVE_LIMIT:
                p["last_move_time"] = time.time()
                p["eye_closed_start"] = None; p["no_face_start"] = None; p["photo_taken"] = False
            
            idle_time = time.time() - p["last_move_time"]

            if p["name"] != "Unknown":
                last_known_positions[p["name"]] = {"pos": (cx, cy), "time": curr_time, "db_id": p["db_id"]}

            # SCAN WAJAH
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

            # --- STATUS ---
            status = "AKTIF"
            p["is_photo_spatial"] = any(point_inside(cx, cy, dbox) for dbox in device_boxes)

            if instant_move > INSTANT_MOVE_LIMIT: status = "AKTIF (GERAK)"
            elif p["is_photo_spatial"]: status = "SPOOF (HP)"
            elif p["is_photo_ai"]: status = f"SPOOF (AI)"
            elif p["eye_closed_start"] and (time.time() - p["eye_closed_start"] > EYE_CLOSED_THRESHOLD):
                if idle_time > 2.0: status = "TIDUR"
                else: status = "AKTIF (MATA)"
            elif p["no_face_start"] and (time.time() - p["no_face_start"] > NO_FACE_THRESHOLD):
                 if idle_time > IDLE_THRESHOLD: status = "KEMUNGKINAN TIDUR"
                 else: status = "AKTIF (MENUNDUK)"

            if "SPOOF" not in status:
                real_person_count += 1
                if p["name"] != "Unknown" and p["name"] in employee_tracker:
                    employee_tracker[p["name"]]["last_seen"] = curr_time
                    employee_tracker[p["name"]]["status"] = "HADIR"
                    employee_tracker[p["name"]]["is_active_today"] = True

            # VISUALISASI KHUSUS
            if status == "TIDUR":
                color = (0, 0, 255) # Merah
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                
                header_y1 = max(0, y1 - 50)
                cv2.rectangle(frame, (x1, header_y1), (x2, y1), color, -1)
                
                label = ">>> TIDUR <<<"
                (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                text_x = x1 + (x2 - x1 - w_txt) // 2
                cv2.putText(frame, label, (text_x, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if not p["alarm_played"]: play_alarm(); p["alarm_played"] = True
                if not p["photo_taken"]:
                    cv2.imwrite(f"captures/tidur/{p['name']}_{int(time.time())}.jpg", frame)
                    p["photo_taken"] = True

            elif status == "KEMUNGKINAN TIDUR":
                color = (0, 255, 255) # Kuning
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                
                header_y1 = max(0, y1 - 50)
                cv2.rectangle(frame, (x1, header_y1), (x2, y1), color, -1)
                
                label = ">>> KEMUNGKINAN TIDUR <<<"
                font_scale = 0.6 if (x2-x1) < 200 else 0.7
                (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                text_x = x1 + (x2 - x1 - w_txt) // 2
                cv2.putText(frame, label, (text_x, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
                
                if not p["alarm_played"]: play_alarm(); p["alarm_played"] = True
                if not p["photo_taken"]:
                    cv2.imwrite(f"captures/kemungkinan_tidur/{p['name']}_{int(time.time())}.jpg", frame)
                    p["photo_taken"] = True

            else:
                color = (0, 255, 0) # Hijau
                if "SPOOF" in status: color = (100, 100, 100) # Abu-abu
                p["alarm_played"] = False 
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                id_text = f"ID:{p['db_id']}" if p['db_id'] is not None else "ID:?"
                label = f"{id_text} | {p['name']} | {status}"
                (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_txt, y1), color, -1)
                text_col = (0,0,0) if "AKTIF" in status else (255,255,255)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_col, 1)

        cv2.putText(frame, f"Orang: {real_person_count}", (10, 30), 0, 0.5, (0, 255, 0), 1)

        # ========================================================
        # [DASHBOARD] JAM KELUAR + DURASI DARI 0
        # ========================================================
        cv2.rectangle(frame, (0, 40), (230, 40 + len(known_names)*15 + 20), (0, 0, 0), -1)
        cv2.putText(frame, "STATUS KARYAWAN:", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        y = 70
        for idx, name in enumerate(known_names):
            uid = known_ids[idx]
            tracker = employee_tracker[name]
            
            if tracker["is_active_today"]:
                # Logic Status Keluar
                is_away = (curr_time - tracker["last_seen"]) > AWAY_TOLERANCE
                if is_away:
                    tracker["status"] = "KELUAR"
                    break_database_cache[name] += time_delta
                
                if tracker["status"] == "HADIR":
                    txt, clr = f"[{uid}] {name}: HADIR", (0, 255, 0)
                else:
                    # [PERBAIKAN] Hitung durasi berdasarkan sesi ini saja
                    current_duration = curr_time - tracker["last_seen"]
                    mins, secs = divmod(int(current_duration), 60)
                    
                    leave_time = time.strftime('%H:%M', time.localtime(tracker["last_seen"]))
                    txt = f"[{uid}] {name}: KELUAR (@{leave_time} | {mins}m {secs}s)"
                    clr = (0, 0, 255)
            else:
                txt, clr = f"[{uid}] {name}: -", (150, 150, 150)
            
            cv2.putText(frame, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, clr, 1)
            y += 15

        cv2.imshow("Monitor Absensi", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

except Exception as e:
    print(f"Error Utama: {e}")
finally:
    save_break_log_bulk(break_database_cache)
    cap.release()
    cv2.destroyAllWindows()