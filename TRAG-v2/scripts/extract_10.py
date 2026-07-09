import pandas as pd

input_path = 'data/EnterpriseRAG-Bench/data/documents/test.parquet'
output_path = 'data/EnterpriseRAG-Bench/data/documents/test_10.parquet'

print(f"Reading first 10 rows from {input_path}...")
df = pd.read_parquet(input_path).head(10)
df.to_parquet(output_path)
print(f"Saved 10 rows to {output_path}")
