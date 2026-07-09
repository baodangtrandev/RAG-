import pandas as pd

try:
    df = pd.read_parquet('data/EnterpriseRAG-Bench/data/documents/test_10_per_source.parquet')
    sources = df['source_type'].unique()
    
    with open('data_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("# Data Analysis Report\n\n")
        for source in sources:
            f.write(f"## Source: {source}\n")
            doc = df[df['source_type'] == source].iloc[0]
            content = str(doc['content'])
            f.write(f"**Title**: {doc['title']}\n")
            f.write(f"**Content length**: {len(content)} characters\n\n")
            f.write("**Preview (first 1000 chars)**:\n")
            f.write("```text\n")
            f.write(content[:1000])
            f.write("\n```\n\n")
            f.write("---\n\n")
    print("Report generated successfully.")
except Exception as e:
    print(f"Error: {e}")
