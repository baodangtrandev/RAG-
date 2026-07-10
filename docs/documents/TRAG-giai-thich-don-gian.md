# TRAG — Giải thích dễ hiểu cho người mới

> Không cần biết AI hay kỹ thuật. Chỉ cần đọc như đọc truyện.

---

## Bức tranh tổng thể: Hệ thống tìm kiếm thông minh

Hãy tưởng tượng công ty bạn có **511,962 tờ giấy tài liệu** nằm trong một căn phòng rất lớn. Mỗi tờ là một tài liệu: email, cuộc họp, ticket công việc, tài liệu kỹ thuật...

Có người hỏi: *"Deadline của dự án Cost Optimization Q2 là bao giờ?"*

Bạn phải tìm câu trả lời trong 511,962 tờ đó. Làm thế nào?

**Cách 1 — Người thường:** Đọc từng tờ từ đầu → mất cả đời.

**Cách 2 — RAG thông thường (cách phổ biến hiện nay):** Dùng máy tính tìm kiếm theo nghĩa gần đúng. Nhưng vấn đề là: có đến 50 tờ về "Cost Optimization Q2" nằm chồng chất lên nhau — email, slack, jira ticket, tài liệu họp — máy không biết tờ nào chứa đúng deadline.

**Cách 3 — TRAG (cái chúng ta xây dựng):** Trước khi tìm kiếm, thu hẹp phòng từ 511,962 tờ xuống còn ~500 tờ liên quan nhất. Sau đó mới tìm trong 500 tờ đó. Và cuối cùng, chọn tờ **mới nhất** trong số đó vì deadline có thể đã bị thay đổi.

---

## Ba vấn đề thực sự cần giải quyết

### Vấn đề 1: Căn phòng quá chật chội

Khi công ty có 5,000 tài liệu, máy tính tìm đúng 90% thời gian.
Khi công ty có 511,000 tài liệu, máy tìm đúng chỉ 46% thời gian.

**Tại sao?** Vì máy tính tìm tài liệu bằng cách biến mỗi tờ giấy thành một **điểm trên bản đồ**. Tài liệu về cùng chủ đề nằm gần nhau trên bản đồ đó. Khi có 511K tài liệu, 50 tài liệu về "Cost Optimization" chen chúc quá gần nhau → máy không còn phân biệt được tờ nào chứa thông tin đúng nữa.

**Giống như:** Thư viện có 1,000 cuốn sách — dễ tìm. Thư viện có 500,000 cuốn, tất cả sách về cùng chủ đề xếp cạnh nhau — tìm đúng cuốn rất khó.

### Vấn đề 2: Thông tin bị cũ theo thời gian

Tài liệu doanh nghiệp liên tục được cập nhật:

```
Ngày 10/3: Jira ticket SUP-312868
            Trạng thái: "Đang xử lý"
            Người phụ trách: Miguel Torres

Ngày 14/3: Jira ticket SUP-312868 (cập nhật)
            Trạng thái: "Đã xong" ← thông tin MỚI
            Giải pháp: đã deploy bản vá
```

Nếu hỏi "ticket này đang ở trạng thái gì?" mà máy trả về tờ ngày 10/3 → sai!

**Giống như:** Bạn hỏi "giá xăng hiện tại là bao nhiêu?" mà người ta trả lời bằng tờ báo tháng trước.

### Vấn đề 3: Câu hỏi dùng từ khác với tài liệu

Câu hỏi: *"Khi nào có thể đặt GPU thế hệ mới nhất cho cluster ở châu Âu?"*

Tài liệu gốc viết: *"Reservations for NVIDIA H200 80GB open in eu-central-1 starting Q2 2025"*

Không có từ nào giống nhau! "GPU thế hệ mới nhất" ≠ "H200 80GB". Máy tìm từ khóa sẽ thất bại.

---

## Ba bước của TRAG giải quyết từng vấn đề

```
Câu hỏi của người dùng
        ↓
┌─────────────────────────────────┐
│  BƯỚC 1: Hiểu câu hỏi sâu hơn  │
│  (Structured Query Parser)      │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  BƯỚC 2: Thu hẹp phòng tìm     │
│  kiếm trước, rồi mới tìm       │
│  (Metadata-Filtered Retrieval)  │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  BƯỚC 3: Sắp xếp lại kết quả  │
│  theo độ tươi mới và đa dạng   │
│  (Temporal Recency Reranker)    │
└─────────────────────────────────┘
        ↓
   Câu trả lời chính xác
```

---

## Bước 1 — Hiểu câu hỏi sâu hơn

### Bước 1 làm gì?

Thay vì đưa câu hỏi thô vào máy tìm kiếm, TRAG **phân tích** câu hỏi trước để trích xuất thông tin hữu ích.

Giống như: Thay vì nói với nhân viên thư viện "tìm sách về cost optimization", bạn nói rõ hơn: "tìm sách về cost optimization, xuất bản năm 2026, trong khu vực kỹ thuật phần mềm."

### Ví dụ thực tế

**Câu hỏi đầu vào:**
*"Trong buổi họp kiểm tra bảo mật về backup on-prem với khách hàng y tế vào tháng 2/2025, ai là người tổ chức phía Redwood?"*

**Sau khi phân tích:**
```
Đối tượng liên quan: ["khách hàng y tế", "backup on-prem"]
Thời gian:           Tháng 2, năm 2025
Loại tài liệu:       "buổi họp" → ghi chú cuộc họp (Fireflies)
Thông tin cần tìm:   tên người
Phiên bản cần:       tại thời điểm đó (không phải mới nhất)
```

**Câu hỏi khác:**
*"Trạng thái HIỆN TẠI của vụ webhook bị lỗi là gì?"*

**Sau khi phân tích:**
```
Đối tượng: ["webhook", "lỗi"]
Thời gian: HIỆN TẠI → cần tài liệu mới nhất
Loại tài liệu: "vụ lỗi" → ticket (Jira/Linear)
Thông tin cần tìm: trạng thái
```

**Câu hỏi khác nữa:**
*"Sứ mệnh của công ty Redwood Inference là gì?"*

**Sau khi phân tích:**
```
Đối tượng: ["Redwood Inference"]
Thời gian: không có → tìm bất kỳ đâu
Loại tài liệu: không xác định → tìm toàn bộ
Thông tin cần tìm: sứ mệnh công ty
```

### Ai làm bước này?

Một mô hình AI nhỏ (Llama-3-8B, chạy trên máy của bạn, không cần internet) đọc câu hỏi và điền vào bảng thông tin trên. Chỉ mất khoảng 200ms.

---

## Bước 2 — Thu hẹp phòng tìm kiếm, rồi mới tìm

### Ý tưởng chính

Đây là bước **quan trọng nhất** và **sáng tạo nhất** của TRAG.

**Cách thông thường:**

```
511,962 tài liệu
       ↓ (tìm kiếm trong toàn bộ)
Top 50 tài liệu gần nhất
```

**Cách của TRAG:**

```
511,962 tài liệu
       ↓ (lọc theo thông tin từ Bước 1)
~500 tài liệu khả nghi (đã loại 99.9% không liên quan)
       ↓ (tìm kiếm trong nhóm nhỏ này)
Top 50 tài liệu phù hợp nhất
```

Khi chỉ còn 500 tài liệu, máy tính dễ phân biệt tài liệu đúng hơn rất nhiều. Vấn đề "căn phòng quá chật" biến mất!

### Cách lọc thông minh: Lọc theo thứ tự

Không phải cứ lọc là tốt — phải lọc **theo thứ tự từ hẹp nhất đến rộng nhất**.

**Ví dụ:**
```
Câu hỏi về "Streamly AI, hồ chứa dp-132-usw"

Tùy chọn lọc:
  - Theo tên khách hàng "Streamly AI": còn 340 tài liệu (trong 511K)
  - Theo loại tài liệu "jira + google_drive": còn 31,128 tài liệu

→ Lọc theo tên khách hàng TRƯỚC (hẹp hơn, hiệu quả hơn)
  511K → 340 (lọc tên) → 89 (lọc loại tài liệu)

Nếu lọc ngược lại:
  511K → 31,128 (lọc loại) → 89 (lọc tên)
  → Số tài liệu phải xử lý nhiều hơn nhiều!
```

Nguyên tắc này gọi là **"lọc thứ hẹp nhất trước"** — lấy từ lý thuyết tối ưu hóa cơ sở dữ liệu.

### Hai cách tìm kiếm kết hợp

Sau khi thu hẹp xuống ~500 tài liệu, TRAG dùng **hai cách tìm kiếm cùng lúc** rồi ghép kết quả:

**Cách 1 — Tìm theo nghĩa (Dense):**
Giống như hỏi một người thông minh: "tìm cho tôi những tài liệu nói về GPU thế hệ mới ở châu Âu" — họ hiểu ý bạn dù bạn không dùng đúng từ kỹ thuật.

Máy biến câu hỏi và tài liệu thành các điểm trên bản đồ. Tài liệu nào gần câu hỏi nhất (về nghĩa) → được chọn.

**Cách 2 — Tìm theo từ khóa (Sparse/BM25):**
Giống như máy tìm kiếm Google cũ: đếm xem câu hỏi có bao nhiêu từ trùng với tài liệu.

Ví dụ: "H200 80GB eu-central-1" → tìm tài liệu có chứa những từ đó.

**Tại sao cần cả hai?**

| Loại câu hỏi | Cách 1 (nghĩa) | Cách 2 (từ khóa) |
|-------------|----------------|-----------------|
| "GPU thế hệ mới ở châu Âu" | ✅ Tốt (hiểu ý) | ❌ Thất bại (không khớp từ) |
| "H200 80GB eu-central-1 reservation" | ❌ Có thể nhầm | ✅ Tốt (khớp chính xác) |
| "Deadline dự án Alpha" | ✅ Tốt | ✅ Tốt |

→ Kết hợp cả hai → bao phủ mọi loại câu hỏi.

**Kết quả Bước 2:** 50 tài liệu ứng viên tốt nhất.

---

## Bước 3 — Sắp xếp lại kết quả theo điểm số tổng hợp

### Vấn đề với 50 ứng viên

Sau Bước 2, ta có 50 tài liệu. Nhưng:
- Có thể có 2 phiên bản của cùng một tài liệu (cũ và mới)
- Có thể 10 tài liệu đều nói về cùng một thứ (lãng phí)
- Cần chọn 10 tài liệu tốt nhất để đưa vào câu trả lời

### Công thức điểm số

Mỗi tài liệu được tính **một điểm tổng hợp** từ 3 thành phần:

```
Điểm = (α × Độ phù hợp nội dung)
      + (β × Độ tươi mới)
      + (γ × Độ đa dạng)
```

**Thành phần 1 — Độ phù hợp nội dung (α):**
Một mô hình AI đọc cả câu hỏi lẫn nội dung tài liệu rồi chấm điểm từ 0-1: "tài liệu này trả lời được câu hỏi hay không?"

Đây là cách chính xác nhất nhưng chậm nhất → chỉ chạy trên 50 ứng viên (không phải 511K).

**Thành phần 2 — Độ tươi mới (β):**
Tài liệu mới hơn → điểm cao hơn. Tài liệu cũ hơn → điểm thấp dần.

```
Ví dụ:
  Tài liệu A (tháng 11/2025): điểm tươi mới = 0.30
  Tài liệu B (tháng 2/2026):  điểm tươi mới = 1.00  ← mới hơn → điểm cao hơn

→ Tự động chọn phiên bản mới nhất mà không cần routing!
```

Công thức: điểm tươi mới giảm dần theo thời gian, giống như nhiệt độ cà phê nguội dần.

**Thành phần 3 — Độ đa dạng (γ):**
Tránh chọn 10 tài liệu nói về cùng một thứ. Nếu đã chọn tài liệu A về "oauth bug", tài liệu B cũng về "oauth bug" sẽ bị trừ điểm.

```
Ví dụ:
  Đã chọn: doc về "webhook bug tháng 3"
  Xét tiếp: doc về "webhook bug tháng 3" (rất giống)
  → Điểm đa dạng thấp → không chọn

  Xét tiếp: doc về "giải pháp webhook" (khác góc nhìn)
  → Điểm đa dạng cao → chọn
```

### Tại sao 3 thành phần này giải quyết hết vấn đề?

| Vấn đề | Giải quyết bởi |
|--------|---------------|
| Tài liệu cũ sai thông tin | β (độ tươi mới) → tự động ưu tiên bản mới |
| 2 bản mâu thuẫn về cùng sự kiện | β → chọn bản mới hơn |
| 10 tài liệu giống hệt nhau | γ (đa dạng) → loại bỏ trùng lặp |
| Tài liệu không liên quan | α (nội dung) → điểm thấp → bị loại |

**Không cần routing! Ba con số α, β, γ được máy học tự động từ dữ liệu.**

---

## Toàn bộ luồng xử lý — Ví dụ hoàn chỉnh

### Câu hỏi: *"Tài liệu Confluence nào trong không gian bảo mật, được xuất bản bởi Elena Kim, mô tả về audit ledger phân tầng và bảo toàn bằng chứng cho model rollback?"*

**→ Bước 1: Phân tích câu hỏi**
```
Đối tượng: ["Elena Kim"]
Loại tài liệu: confluence (nói rõ)
Không gian: "bảo mật" → space = security-and-compliance
Thông tin cần tìm: tên tài liệu
Thời gian: không có → bất kỳ phiên bản nào
```

**→ Bước 2: Thu hẹp và tìm kiếm**
```
Lọc theo thứ tự hẹp nhất trước:
  - Tác giả = "Elena Kim":          511K → 1,240 tài liệu (tất cả của Elena Kim)
  - Loại = "confluence":            1,240 → 89 tài liệu
  - Không gian = "security-compliance": 89 → 12 tài liệu

Tìm kiếm trong 12 tài liệu:
  Cách nghĩa: "audit ledger, rollback evidence" → tìm nội dung liên quan
  Cách từ khóa: "audit ledger", "model rollback" → tìm từ trùng

Kết quả top 5:
  - "Sensitivity-tiered Audit Ledger and Rollback Evidence Playbook" — sim=0.96
  - "Security Evidence Collection Guide" — sim=0.71
  - "Model Governance Policy" — sim=0.65
  ...
```

**→ Bước 3: Tính điểm và chọn**
```
Tài liệu "Sensitivity-tiered Audit Ledger...":
  Nội dung phù hợp (α): 0.96 (rất cao)
  Độ tươi mới (β):      0.85 (tài liệu còn hiệu lực)
  Đa dạng (γ):          1.00 (đầu tiên được chọn)
  → ĐIỂM TỔNG: 0.94 ← được chọn làm câu trả lời

Tài liệu "Security Evidence Collection Guide":
  Nội dung phù hợp (α): 0.71
  → ĐIỂM TỔNG: 0.72 ← thấp hơn, chỉ dùng nếu cần thêm context
```

**→ Câu trả lời:**
```
"Playbook: 'Sensitivity-tiered Audit Ledger and Rollback Evidence Playbook'
 — xuất bản bởi Elena Kim trong không gian security-and-compliance."
```

✅ Đúng với câu trả lời mẫu trong dataset!

---

## Hệ thống lưu trữ tài liệu (Ingestion)

Trước khi tìm kiếm, cần **chuẩn bị kho lưu trữ**. Đây là công việc làm một lần khi đưa tài liệu vào hệ thống.

### Mỗi tài liệu được lưu ở 3 nơi đồng thời

**Nơi 1 — Kho từ khóa (như mục lục sách):**
```
Lưu: tác giả, loại tài liệu, ngày tạo, ngày cập nhật, không gian...
Dùng để: lọc nhanh trước khi tìm kiếm
Ví dụ: "tất cả tài liệu của Elena Kim trong space security" → tra mục lục → có ngay
```

**Nơi 2 — Kho bản đồ ý nghĩa (dày đặc nhất):**
```
Mỗi tài liệu được chuyển thành một điểm trên bản đồ 3072 chiều
Dùng để: tìm tài liệu có ý nghĩa gần nhất với câu hỏi
Tốn bộ nhớ nhất nhưng tìm theo nghĩa tốt nhất
```

**Nơi 3 — Kho từ khóa (thưa):**
```
Đếm tần suất từ xuất hiện trong mỗi tài liệu
Dùng để: tìm tài liệu có từ khớp chính xác với câu hỏi
Nhanh, nhẹ, tốt cho câu hỏi kỹ thuật cụ thể
```

### Thêm thông tin thời gian

Khi lưu tài liệu, hệ thống tự động gắn nhãn:
```
Tài liệu Jira SUP-312868 (tạo 10/3):
  → Trạng thái: CŨ (vì có phiên bản mới hơn từ 14/3)

Tài liệu Jira SUP-312868 (cập nhật 14/3):
  → Trạng thái: MỚI NHẤT

→ Bước 3 sẽ ưu tiên tài liệu "MỚI NHẤT" hơn
```

---

## Tài nguyên cần thiết (H100 40GB)

| Công việc | Mô hình AI dùng | Bộ nhớ GPU |
|-----------|----------------|-----------|
| Bước 1: Phân tích câu hỏi | Llama-3-8B (nhỏ, nhanh) | 8 GB |
| Bước 3: Chấm điểm nội dung | BGE-Reranker-v2 (chuyên chấm điểm) | 4 GB |
| Trả lời: Tổng hợp câu trả lời | Qwen2.5-7B (mô hình ngôn ngữ) | 8 GB |
| **Tổng cộng** | | **~20 GB** (còn dư 20 GB) |

Tốc độ xử lý mỗi câu hỏi: khoảng **800 mili-giây** (dưới 1 giây).

---

## Tóm tắt ngắn gọn nhất

```
📄 511,962 tài liệu trong kho

❓ Câu hỏi đến

🔍 Bước 1: "Câu hỏi này hỏi về ai, về thời gian nào, loại tài liệu nào?"
   → Trích xuất thông tin phụ

📁 Bước 2: "Lọc kho còn ~500 tài liệu liên quan, rồi tìm 50 tài liệu tốt nhất"
   → Thu hẹp trước, tìm sau

⭐ Bước 3: "Chấm điểm 50 tài liệu theo: nội dung + độ mới + đa dạng"
   → Chọn 10 tài liệu cuối cùng

💬 Trả lời: "Tổng hợp câu trả lời từ 10 tài liệu đó"
   → Câu trả lời chính xác
```

**Điểm mấu chốt:** Thu hẹp phòng trước khi tìm giải quyết vấn đề chật chội. Ưu tiên tài liệu mới giải quyết vấn đề thông tin cũ. Không cần biết câu hỏi thuộc loại nào — pipeline xử lý tất cả theo cùng một cách.
