# PDF to Markdown Converter

## 1. Tạo `venv` trong repo

```powershell
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
python .\documents\convert_pdfs_to_md.py
```

Mặc định script sẽ:
- Quét toàn bộ PDF trong các thư mục con của `documents`
- Tạo thư mục `documents\paper-md` (nếu chưa có)
- Xuất file `.md` vào `documents\paper-md` và giữ nguyên cấu trúc thư mục con

Tùy chọn:

```powershell
python .\documents\convert_pdfs_to_md.py --overwrite
python .\documents\convert_pdfs_to_md.py -i .\documents -o .\documents\paper-md --overwrite
```
