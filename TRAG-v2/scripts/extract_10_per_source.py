import pandas as pd

# Đường dẫn (chạy script từ thư mục gốc TRAG-v2)
input_path = 'data/EnterpriseRAG-Bench/data/documents/test.parquet'
output_path = 'data/EnterpriseRAG-Bench/data/documents/test_10_per_source.parquet'

print(f"Đang đọc dữ liệu từ: {input_path}...")
df = pd.read_parquet(input_path)

print(f"Các nguồn (source_type) có trong dữ liệu: {df['source_type'].unique().tolist()}")

# Gom nhóm theo 'source_type' và lấy 10 document đầu tiên của mỗi nhóm
sampled_df = df.groupby('source_type').head(10).reset_index(drop=True)

# Lưu ra file parquet
sampled_df.to_parquet(output_path)
print(f"\nĐã xuất thành công {len(sampled_df)} documents ra file: {output_path}")

print("\nChi tiết số lượng mẫu lấy được từ mỗi nguồn:")
print(sampled_df['source_type'].value_counts())
