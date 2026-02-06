import psycopg2
import numpy as np

# KONFIGURASI (Sesuaikan Password kamu)
DB_CONFIG = {
    "dbname": "face_db",  # Cek apakah nama db kamu 'face_db' atau 'surveillance_db'?
    "user": "postgres",
    "password": "020206",       # <--- GANTI PASSWORD SESUAI PGADMIN
    "host": "localhost",
    "port": "5432"
}

def cek_koneksi():
    print("--- MULAI DIAGNOSA ---")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print("[OK] Koneksi ke Database Berhasil!")

        # 1. Cek Tabel
        cur.execute("SELECT to_regclass('public.users');")
        if cur.fetchone()[0] is None:
            print("[ERROR] Tabel 'users' TIDAK DITEMUKAN!")
            return
        else:
            print("[OK] Tabel 'users' ditemukan.")

        # 2. Cek Isi Data
        cur.execute("SELECT id, name, embedding FROM users")
        rows = cur.fetchall()
        
        jumlah_data = len(rows)
        print(f"[INFO] Jumlah Wajah Terdaftar: {jumlah_data}")
        
        if jumlah_data == 0:
            print("[PERINGATAN] Database KOSONG. Harap lakukan Pendaftaran Wajah dulu!")
        else:
            print("[OK] Contoh Data Pertama:")
            print(f" - Nama: {rows[0][1]}")
            # Cek apakah embedding valid
            emb = rows[0][2]
            if emb is None or len(emb) == 0:
                print("[ERROR] Data Embedding (Wajah) RUSAK/KOSONG!")
            else:
                print(f" - Embedding Size: {len(emb)} (Data Valid)")

        conn.close()

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        print("Saran: Cek password, nama database, atau apakah PostgreSQL sudah jalan.")

if __name__ == "__main__":
    cek_koneksi()