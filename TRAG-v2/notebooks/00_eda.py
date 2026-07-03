# %% [markdown]
# # Phân tích EDA cho EnterpriseRAG-Bench (test.parquet)
# Notebook này thực hiện Exploratory Data Analysis (EDA) để hiểu cấu trúc và đặc điểm của tập dữ liệu documents.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập hiển thị cho matplotlib/seaborn
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Đường dẫn tới file parquet
parquet_path = "../data/EnterpriseRAG-Bench/data/documents/test.parquet"

# %% [markdown]
# ## 1. Đọc dữ liệu Parquet
# File Parquet là định dạng lưu trữ dạng cột, không cần giải nén mà có thể đọc trực tiếp bằng pandas (yêu cầu cài đặt `pyarrow` hoặc `fastparquet`).

# %%
try:
    df = pd.read_parquet(parquet_path)
    print(f"✅ Đã tải thành công: {len(df):,} dòng (documents)")
except Exception as e:
    print(f"❌ Lỗi khi đọc file: {e}")
    print("Vui lòng cài đặt thư viện: pip install pandas pyarrow matplotlib seaborn")

# %% [markdown]
# ## 2. Khám phá tổng quan (Basic Overview)

# %%
# Xem 5 dòng đầu tiên
print(df.head())

# %%
# Thông tin về các cột và kiểu dữ liệu
df.info()

# %%
# Kiểm tra giá trị bị thiếu (Missing values)
missing_data = df.isnull().sum()
print("Số lượng missing values mỗi cột:")
print(missing_data[missing_data > 0])

# %% [markdown]
# ## 3. Phân bố nguồn tài liệu (Source Type Distribution)
# T-RAG tập trung giải quyết độ nhiễu từ nhiều nguồn khác nhau. Hãy xem tỷ lệ các nguồn tài liệu.

# %%
if 'source' in df.columns:  # Hoặc 'source_type' tùy vào cấu trúc dataset
    source_col = 'source'
elif 'source_type' in df.columns:
    source_col = 'source_type'
else:
    source_col = None

if source_col:
    source_counts = df[source_col].value_counts()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=source_counts.values, y=source_counts.index, palette="viridis")
    plt.title("Phân bố Số lượng Tài liệu theo Nguồn (Source Type)")
    plt.xlabel("Số lượng Documents")
    plt.ylabel("Nguồn (Source)")
    plt.tight_layout()
    plt.show()
else:
    print("Không tìm thấy cột 'source' hoặc 'source_type'")

# %% [markdown]
# ## 4. Phân tích độ dài văn bản (Document Length)
# Phân tích độ dài giúp quyết định chiến lược Chunking cho T-RAG.

# %%
if 'text' in df.columns or 'content' in df.columns:
    text_col = 'text' if 'text' in df.columns else 'content'
    
    # Tính số lượng từ (words) ước tính
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=50, kde=True, color='blue')
    plt.title("Phân bố Độ dài Tài liệu (Số từ)")
    plt.xlabel("Số từ (Word Count)")
    plt.ylabel("Tần suất")
    plt.xlim(0, df['word_count'].quantile(0.95)) # Cắt bỏ 5% outliers dài nhất để dễ nhìn
    plt.tight_layout()
    plt.show()
    
    print("Thống kê mô tả độ dài tài liệu:")
    print(df['word_count'].describe())

# %% [markdown]
# ## 5. Phân tích Yếu tố Thời gian (Temporal Analysis)
# Đây là yếu tố cốt lõi của T-RAG (Time Decay). Chúng ta cần xem định dạng timestamp.

# %%
# Tìm cột thời gian (thường là 'timestamp', 'date', 'created_at', 'updated_at')
time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]

if time_cols:
    time_col = time_cols[0]
    print(f"Sử dụng cột thời gian: {time_col}")
    
    # Chuyển đổi sang datetime nếu chưa phải
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df['parsed_date'] = pd.to_datetime(df[time_col], errors='coerce')
        except:
            df['parsed_date'] = pd.NaT
    else:
        df['parsed_date'] = df[time_col]
        
    # Xóa các dòng NaT để vẽ biểu đồ
    valid_dates = df['parsed_date'].dropna()
    
    if not valid_dates.empty:
        # Group theo năm-tháng
        monthly_counts = valid_dates.dt.to_period('M').value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        monthly_counts.plot(kind='bar', color='coral')
        plt.title("Số lượng tài liệu theo Thời gian (Tháng)")
        plt.xlabel("Tháng")
        plt.ylabel("Số lượng")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
else:
    print("Không tìm thấy cột chứa thông tin thời gian.")

# %% [markdown]
# ## 6. Kết luận ban đầu
# - Cấu trúc dataset: [Sẽ có sau khi chạy]
# - Đặc điểm thời gian: [Sẽ có sau khi chạy]
# - Đặc điểm phân bố: [Sẽ có sau khi chạy]
