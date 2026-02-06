import psycopg2
import os
import shutil  # Library untuk menghapus folder beserta isinya

# ================= KONFIGURASI =================
DB_CONFIG = {
    "dbname": "face_db",
    "user": "postgres",
    "password": "020206",  # <--- GANTI PASSWORD
    "host": "localhost",
    "port": "5432"
}

DATASET_PATH = "dataset_faces"  # Nama folder tempat simpan foto

def hapus_total_user():
    print("\n" + "="*50)
    print("   HAPUS USER TOTAL (DATABASE + FOTO)")
    print("="*50)
    
    target_name = input("Masukkan Nama User yang akan dihapus: ").strip()
    
    if not target_name:
        print("[ERROR] Nama tidak boleh kosong.")
        return

    # Konfirmasi Bahaya
    print(f"\n[PERINGATAN] Tindakan ini akan menghapus:")
    print(f"1. Data '{target_name}' dari Database PostgreSQL.")
    print(f"2. Folder foto '{target_name}' dari komputer.")
    confirm = input(f"Apakah Anda YAKIN? Ketik 'HAPUS' untuk konfirmasi: ")
    
    if confirm != 'HAPUS':
        print("[INFO] Dibatalkan.")
        return

    # --- LANGKAH 1: HAPUS DARI DATABASE ---
    print("\n[1/2] Menghapus dari Database...")
    db_success = False
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Hapus Wajah
        cur.execute("DELETE FROM users WHERE name = %s", (target_name,))
        rows_users = cur.rowcount
        
        # Hapus Log Istirahat
        cur.execute("DELETE FROM break_logs WHERE name = %s", (target_name,))
        rows_logs = cur.rowcount

        conn.commit()
        cur.close()
        conn.close()
        
        if rows_users > 0 or rows_logs > 0:
            print(f"   [OK] Data Database terhapus (Users: {rows_users}, Logs: {rows_logs})")
            db_success = True
        else:
            print("   [INFO] User tidak ditemukan di Database (Mungkin sudah dihapus).")
            db_success = True # Tetap lanjut ke hapus folder

    except Exception as e:
        print(f"   [ERROR] Gagal menghapus database: {e}")

    # --- LANGKAH 2: HAPUS FOLDER FOTO ---
    print("[2/2] Menghapus File Foto...")
    folder_path = os.path.join(DATASET_PATH, target_name)
    
    if os.path.exists(folder_path):
        try:
            # shutil.rmtree menghapus folder beserta seluruh isinya
            shutil.rmtree(folder_path)
            print(f"   [OK] Folder '{folder_path}' berhasil dihapus permanen.")
        except Exception as e:
            print(f"   [ERROR] Gagal menghapus folder: {e}")
    else:
        print(f"   [INFO] Folder '{folder_path}' tidak ditemukan di komputer.")

    print("\n" + "="*50)
    print("PROSES SELESAI.")

if __name__ == "__main__":
    hapus_total_user()