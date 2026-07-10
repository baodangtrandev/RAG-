import lancedb
import os

dir_old = "/network-volume/RAG-/TRAG-v2/data/lancedb"
dir_new = "/network-volume/RAG-/data/lancedb"

def check_db(db_path):
    if not os.path.exists(db_path):
        print(f"Thư mục không tồn tại: {db_path}")
        return
    db = lancedb.connect(db_path)
    tables = db.table_names()
    print(f"\n📁 Đang kiểm tra DB tại: {db_path}")
    print(f"Tổng số bảng: {len(tables)}")
    for t in tables:
        try:
            tbl = db.open_table(t)
            print(f" - Bảng '{t}': {len(tbl)} chunks (vectors)")
        except Exception as e:
            print(f" - Bảng '{t}': LỖI KHÔNG THỂ ĐỌC ({e})")

if __name__ == "__main__":
    check_db(dir_old)
    check_db(dir_new)
