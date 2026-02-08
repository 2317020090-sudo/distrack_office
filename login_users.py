import mysql.connector
import hashlib
import time
import pwinput  # <--- Library agar password jadi bintang
import sys

# ================= KONFIGURASI DATABASE MYSQL =================
DB_CONFIG = {
    "database": "face_dbs",
    "user": "root",
    "password": "",
    "host": "localhost",
    "port": 3306
}

# ================= HELPER FUNCTIONS =================
def get_db_conn():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[ERROR KONEKSI MYSQL] {e}")
        return None

def hash_password(password):
    """
    WAJIB SAMA PERSIS dengan fungsi di register_user.py
    Mengubah password jadi kode SHA-256.
    """
    return hashlib.sha256(password.encode()).hexdigest()

def login_process():
    print("\n" + "="*40)
    print("      LOGIN USER (SISTEM ABSENSI)      ")
    print("="*40)

    # 1. Input Email (Teks Biasa)
    email_input = input("📧 Masukkan Email    : ").strip()
    
    # 2. Input Password (LANGSUNG BINTANG ******)
    # Di login, kita tidak kasih lihat teks asli demi keamanan.
    pass_input = pwinput.pwinput(prompt="🔑 Masukkan Password : ", mask="*")

    # 3. Hash Password Inputan User
    pass_hash_input = hash_password(pass_input)

    # 4. Cek ke Database MySQL
    conn = get_db_conn()
    if not conn:
        return

    try:
        cur = conn.cursor(dictionary=True) # Agar hasil query bentuknya Dictionary
        
        # Query: Cari user yang Email DAN Password Hash-nya cocok
        query = "SELECT id, name, email, created_at FROM users WHERE email = %s AND password_hash = %s"
        cur.execute(query, (email_input, pass_hash_input))
        
        user = cur.fetchone() # Ambil satu data

        print("\n⏳ Memeriksa kredensial...")
        time.sleep(1) # Efek loading

        if user:
            # === LOGIN SUKSES ===
            show_dashboard(user)
        else:
            # === LOGIN GAGAL ===
            print("\n❌ LOGIN GAGAL!")
            print("   Email atau Password salah.")
            print("   (Pastikan Anda sudah Register data di MySQL)")

    except Exception as e:
        print(f"[ERROR QUERY] {e}")
    finally:
        if conn.is_connected():
            conn.close()

def show_dashboard(user):
    """
    Tampilan menu setelah berhasil login
    """
    print("\n" + "="*50)
    print(f"✅ SELAMAT DATANG, {user['name'].upper()}!")
    print("="*50)
    
    print("📄 DATA PROFIL ANDA:")
    print(f"   • ID User      : {user['id']}")
    print(f"   • Nama Lengkap : {user['name']}")
    print(f"   • Email        : {user['email']}")
    print(f"   • Terdaftar    : {user['created_at']}")
    
    print("\n[MENU USER]")
    print("1. Mulai Absensi (Deteksi Wajah)")
    print("2. Lihat Laporan Kehadiran")
    print("3. Ubah Password")
    print("4. Keluar")
    print("="*50)
    
    # Simulasi menu sederhana
    while True:
        pilih = input("\nPilih menu (1-4): ")
        if pilih == "4":
            print("Logout berhasil. Sampai jumpa!")
            break
        else:
            print("Fitur ini akan segera hadir!")

if __name__ == "__main__":
    while True:
        login_process()
        
        # Tanya user mau ulang atau keluar
        pilihan = input("\nCoba login user lain? (y/n): ").lower()
        if pilihan != 'y':
            print("Sistem berhenti.")
            break