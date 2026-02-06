import cv2
import numpy as np
import psycopg2
import winsound
import time
import math
import hashlib
import threading
import random
import asyncio
from insightface.app import FaceAnalysis

# Library Password Bintang
try:
    import pwinput
except ImportError:
    print("⚠️  Library 'pwinput' belum diinstall. Jalankan: pip install pwinput")
    import getpass as pwinput # Fallback ke mode hidden jika belum install

# Library Telegram
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ================= KONFIGURASI =================
DB_CONFIG = {
    "dbname": "face_db",
    "user": "postgres",
    "password": "020206",   # <--- PASSWORD DB
    "host": "localhost",
    "port": "5432"
}

# MASUKKAN TOKEN BOT
TOKEN = "8476797101:AAGUhSDaVHdSFdaUdGHq8z-lKwsITmL0DCw"

# Setting Wajah
STAGES = ["DEPAN", "KANAN", "KIRI", "ATAS", "BAWAH"]
STAGE_DURATION = 5  

# Global Memory
pending_registrations = {} 
bot_ready_event = threading.Event()

# ================= BAGIAN 1: LOGIKA BOT (SHARE CONTACT) =================
def normalize_phone(phone):
    phone = str(phone).strip().replace("-", "").replace(" ", "")
    if phone.startswith("0"): return "62" + phone[1:]
    if phone.startswith("+"): return phone[1:]
    return phone

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    btn = KeyboardButton("📱 KIRIM KONTAK SAYA (KLIK INI)", request_contact=True)
    markup = ReplyKeyboardMarkup([[btn]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        "👋 Halo Admin!\nKlik tombol di bawah untuk verifikasi nomor HP.",
        reply_markup=markup
    )

async def contact_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    contact = update.message.contact
    user_chat_id = update.message.chat.id
    telegram_phone = normalize_phone(contact.phone_number)
    
    if telegram_phone in pending_registrations:
        otp_code = pending_registrations[telegram_phone]['otp']
        pending_registrations[telegram_phone]['chat_id'] = user_chat_id
        pending_registrations[telegram_phone]['verified'] = True 
        
        await update.message.reply_text(f"✅ TERVERIFIKASI!\n🔐 OTP: `{otp_code}`", parse_mode="Markdown")
        print(f"\n[BOT] Kontak diterima: {telegram_phone}")
    else:
        await update.message.reply_text("⚠️ Nomor ini tidak sedang mendaftar di komputer.")

def run_telegram_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.CONTACT, contact_handler))
    bot_ready_event.set()
    print("[SYSTEM] Bot Telegram Siap...")
    app.run_polling(stop_signals=None)

# ================= BAGIAN 2: LOGIKA AI & DB =================
print("[INIT] Memuat Model AI (Buffalo_L)...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

def get_db_conn():
    try: return psycopg2.connect(**DB_CONFIG)
    except: return None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def capture_face_process():
    cap = cv2.VideoCapture(0)
    stage_idx = 0; progress = 0; best_emb = None 
    print("[KAMERA] Ikuti instruksi gerakan kepala...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        faces = face_app.get(frame)
        target = STAGES[stage_idx] if stage_idx < len(STAGES) else "SELESAI"
        
        cv2.rectangle(frame, (0,0), (640, 60), (0,0,0), -1)
        cv2.putText(frame, f"DAFTAR: {target}", (20,40), 0, 0.8, (0, 255, 255), 2)

        if len(faces) > 0:
            face = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]
            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
            progress += 1
            if target == "DEPAN": best_emb = face.embedding
            
            cv2.rectangle(frame, (20, 70), (20+progress*4, 75), (0,255,0), -1)

            if progress >= 20: 
                winsound.Beep(1200, 100); stage_idx += 1; progress = 0
                if stage_idx >= len(STAGES):
                    cv2.putText(frame, "SUKSES!", (200, 240), 0, 2, (0,255,0), 3)
                    cv2.imshow("Register", frame); cv2.waitKey(1000)
                    cap.release(); cv2.destroyAllWindows()
                    return best_emb
        
        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release(); cv2.destroyAllWindows()
    return None

def save_admin_final(name, email, phone, chat_id, password, embedding):
    conn = get_db_conn()
    if not conn: return
    try:
        cur = conn.cursor()
        hashed_pw = hash_password(password)
        emb_list = embedding.tolist()
        
        query = """
            INSERT INTO admins (name, email, telegram_phone, telegram_chat_id, password_hash, embedding) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (name, email, phone, chat_id, hashed_pw, emb_list))
        conn.commit()
        print("\n✅ DATA TERSIMPAN KE TABEL ADMINS!")
    except Exception as e:
        print(f"[ERROR DB] {e}")
    finally: conn.close()

# ================= ALUR UTAMA =================
def main():
    t = threading.Thread(target=run_telegram_bot, daemon=True)
    t.start()
    bot_ready_event.wait()

    print("\n=== PENDAFTARAN ADMIN ===")
    name = input("1. Nama  : ").strip()
    email = input("2. Email : ").strip()
    phone = input("3. No HP : ").strip()
    
    clean_phone = normalize_phone(phone)
    otp = str(random.randint(111111, 999999))
    pending_registrations[clean_phone] = {'otp': otp, 'chat_id': None, 'verified': False}

    print(f"\n[INFO] Menunggu verifikasi nomor: {clean_phone}")
    print("⏳ BUKA TELEGRAM -> KLIK TOMBOL 'KIRIM KONTAK SAYA'...")
    
    final_chat_id = None
    while True:
        if pending_registrations[clean_phone]['verified']:
            final_chat_id = pending_registrations[clean_phone]['chat_id']
            user_otp = input("\n>> Masukkan OTP dari Bot: ")
            if user_otp == otp:
                print("✅ OTP BENAR!"); break
            print("❌ Salah.")
        time.sleep(1)

    print("\n")
    
    # === MODIFIKASI PASSWORD DISINI ===
    while True:
        # Input biasa (Terlihat Jelas)
        p1 = input("4. Buat Password (Terlihat): ")
        
        # Pwinput (Bintang-Bintang ******)
        # Jika belum install pwinput, dia akan otomatis pakai getpass (hidden total)
        if hasattr(pwinput, 'pwinput'):
            p2 = pwinput.pwinput("   Ulangi Password (******): ")
        else:
            p2 = pwinput.getpass("   Ulangi Password (Hidden): ")

        if p1 == p2 and p1: break
        print("❌ Password tidak cocok. Ulangi.")
    # ==================================

    print("\n[INFO] Scan Wajah...")
    time.sleep(2)
    emb = capture_face_process()
    
    if emb is not None:
        save_admin_final(name, email, clean_phone, final_chat_id, p1, emb)
    else:
        print("❌ Gagal.")

if __name__ == "__main__":
    main()