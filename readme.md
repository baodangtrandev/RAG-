# PDF to Markdown Converter

## 1. Tạo `venv` trong repo

```powershell
cd D:\Projects\RAG-
python -m venv .venv
```

## 2. Kích hoạt `venv`

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

CMD:

```cmd
.\.venv\Scripts\activate.bat
```

## 3. Cài dependency

```powershell
pip install pypdf
```

## 4. Chạy script convert

```powershell
python .\convert_pdfs_to_md.py
```

Mặc định script sẽ:
- Đọc PDF trong `documents`
- Tạo thư mục `paper-md` (nếu chưa có)
- Xuất file `.md` vào `paper-md`

Tùy chọn:

```powershell
python .\convert_pdfs_to_md.py -i .\documents -o .\paper-md --overwrite
python .\convert_pdfs_to_md.py -i .\documents -r
```
