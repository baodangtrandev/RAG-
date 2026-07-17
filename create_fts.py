import lancedb
import os

db = lancedb.connect(os.environ.get("RAG_DB_URI", "data/lancedb"))
for table_name in db.table_names():
    table = db.open_table(table_name)
    print(f"Tạo FTS index cho bảng: {table_name}")
    try:
        table.create_fts_index("content", replace=True)
        print(f"  -> Thành công: {table_name}")
    except Exception as e:
        print(f"  -> Lỗi: {e}")
