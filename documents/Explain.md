### Batch 1: Sự Ảo Tưởng Của RAG Truyền Thống & Cú Tát Từ Bài Paper

#### 1. RAG thực chất là cái quái gì? (Giải thích chống ngáo)

Mày cứ tưởng tượng thế này: Mày đang ngồi rã con phím cơ ra để mod lại. Kiến thức có sẵn trong đầu mày về switch, keycap, stab chính là con LLM. Nhưng tự nhiên mày bốc trúng một con switch lạ hoắc hoặc cần nhớ thông số lực nhấn chính xác của nó. Thay vì ngồi đoán mò (thuật ngữ AI gọi là *Hallucinate*), mày mở datasheet ra đọc (bước **Retrieve** - Truy xuất), sau đó mày mới chốt là nên lube nó bằng loại mỡ nào (bước **Generate** - Sinh text).

RAG (Retrieval-Augmented Generation) chính là cái hành động "lật datasheet" đó. Nó biến một con AI chém gió thành một con AI nói có sách, mách có chứng.

#### 2. Vấn đề hiện tại (The Pain-point)

Cái giới làm AI đang bị ảo tưởng sức mạnh. Mấy cái benchmark (bài test chuẩn) hiện nay toàn lấy dữ liệu sạch sẽ, chuẩn chỉ từ các nguồn công khai như Wikipedia hay bài báo khoa học ra để test.  

Nhưng thực tế thì sao? Dữ liệu nội bộ của công ty là một bãi rác khổng lồ. Nó lộn xộn, ồn ào và chứa những loại tài liệu đéo bao giờ xuất hiện trên public benchmark như ticket hỗ trợ, luồng email, chitchat chửi thề trên Slack.  

#### 3. Bài Paper (EnterpriseRAG-Bench) ra đời để làm gì?

Tụi Onyx làm cái paper này để vả vào mặt thực tế đó. Tụi nó tạo ra cái dataset **EnterpriseRAG-Bench** chứa khoảng 500,000 tài liệu rải đều trên 9 loại nguồn dữ liệu doanh nghiệp (Slack, Gmail, Jira, Confluence, v.v.).  

Đỉnh cao của sự khốn nạn trong cái benchmark này là tụi nó cố tình nhét "nhiễu" (noise) y như đời thật: tài liệu quăng sai thư mục, copy trùng lặp mâu thuẫn nhau, và thông tin cũ chưa được update.  

#### 4. Kết quả phũ phàng (Lý do phải có T-RAG)

Khi tụi nó mang hệ thống RAG xài **Vector Search** (tìm kiếm theo ngữ nghĩa - cái đang hot và xịn nhất hiện nay) vào chạy test trên đống data này, kết quả cực kỳ thảm hại. Nó bị thằng **BM25** (thuật toán tìm kiếm chính xác theo từ khóa cổ điển) đấm vỡ mặt về độ chính xác và khả năng gom đủ tài liệu.  

Nguyên nhân? Vector Search được train bằng data public nên nó đéo hiểu được từ lóng nội bộ, codename dự án, hay mấy định dạng rập khuôn của ticket. Mày hỏi câu nào chứa từ khóa kỹ thuật cụ thể kiểu "Lỗi mã SUP-312868" là nó ngáo mẹ luôn. Còn nữa, vì data quá lớn (500k file), Vector Search bị "chật chội", tìm cái gì cũng thấy na ná nhau nên bốc toàn rác nhét vào mồm AI.  



---



### Batch 2: Giải phẫu T-RAG (Giai đoạn 1 - Lọc nới lỏng & Tìm kiếm lai)

#### 1. Bước 1: Query Parser & Sự ngây thơ của "Hard-Filtering"

Khi người dùng hỏi: *"Trạng thái cái ticket lỗi webhook SUP-312868 của ông Tuấn là gì?"*

- **RAG truyền thống (Ngu ngốc):** Vác nguyên câu đó đi quét Vector Search. Kết quả là nó sẽ trả về hàng đống tài liệu nói về "webhook" nhưng đéo phải của Tuấn, hoặc đéo phải mã SUP-312868.
- **T-RAG (Thông minh):** Nó dùng một con LLM nhỏ (như Llama-3-8B hoặc Qwen) đứng gác cổng. Con này đọc câu hỏi và "bóc tách" (Parse) ra thành các biến số cụ thể:
  - `Topic`: Lỗi webhook
  - `ID`: SUP-312868
  - `Person`: Tuấn
  - `Type`: Ticket

**Nhưng đây mới là cái ăn tiền để mày lấy điểm với thầy:** Trong bản nháp ý tưởng ban đầu, người ta hay dùng **Hard-filtering** (Lọc cứng). Tức là bốc nguyên chữ "Tuấn" đi query SQL `WHERE author = 'Tuấn'`.

Thực tế đẫm máu là gì? Trong database, hệ thống nhân sự lưu tên ổng là "Tuan Nguyen" hoặc "T. Nguyen". Thế là lệnh SQL trả về 0 kết quả. Toang luôn từ vòng gửi xe!

**Cách giải quyết của bản kiến trúc tao chốt cho mày:** Dùng **Soft-Filtering (Lọc nới lỏng)** kết hợp với cơ chế **Fallback**.

Mày dùng SQLite với FTS5 (Full-Text Search) để tìm kiếm mờ (Fuzzy match). Nếu tìm với điều kiện Đầy Đủ mà số lượng tài liệu trả về ít quá (nhỏ hơn 50 cái), hệ thống sẽ tự động rớt cấp: Bỏ điều kiện tên tác giả đi, chỉ tìm theo mã Ticket và Topic. Tự động nới lỏng cho đến khi gom đủ một cái rổ chứa khoảng 500 tài liệu khả nghi (Candidate Documents).

#### 2. Bước 2: Thu hẹp căn phòng (Metadata-Filtered Retrieval)

Cái rổ 500 tài liệu khả nghi ở Bước 1 giải quyết được cái bách nhục lớn nhất của Vector Search: **Sự chật chội**.

Thay vì mò kim đáy biển trong 511,962 tài liệu, giờ đây mày ép bọn Vector DB và thuật toán tìm kiếm chỉ được phép hoạt động trong không gian của 500 tài liệu đã được khoanh vùng.

Lúc này, các điểm dữ liệu trên bản đồ Vector không còn chen chúc nhau nữa. Khoảng cách giữa các tài liệu rác và tài liệu thật sự giãn ra rất rộng, giúp AI dễ dàng bốc trúng đích. Giống như mày đang debug một cục code backend Node.js, thay vì search `console.log` trên toàn bộ source code của project, mày chỉ search trong thư mục `controllers/` thôi.

#### 3. Bước 3: Cú đấm thép Hybrid Search (Tìm kiếm Lai)

Chỉ còn 500 tài liệu, nhưng tìm bằng cách nào? Bài báo EnterpriseRAG-Bench đã chỉ ra sự thảm hại của Vector Search khi độ chính xác chỉ lẹt đẹt ở mức 51.4%, thua cả thuật toán cổ điển BM25 (68.8%).  

Mày không chọn 1 trong 2, mày lấy cả 2:

- **Mũi khoan 1 - Sparse Search (BM25 qua SQLite FTS5):** Băm nhỏ văn bản ra và đếm từ khóa. Cực kỳ bá đạo khi câu hỏi có chứa các mã số chính xác (như `SUP-312868` hay IP `192.168.1.1`).
- **Mũi khoan 2 - Dense Search (Vector qua LanceDB):** Đọc hiểu ý nghĩa. Cực kỳ bá đạo khi người dùng hỏi lòng vòng kiểu "Cái ông làm bảo mật hôm nọ nói gì về mã hóa?".

**Thuật toán dung hợp (RRF - Reciprocal Rank Fusion):**

Chạy cả 2 mũi khoan cùng lúc. Thằng BM25 trả về top 50, thằng Vector trả về top 50. Hệ thống sẽ dùng thuật toán RRF để cộng điểm xếp hạng lại, xáo trộn và chắt lọc ra Top 50 tài liệu tinh túy nhất, không trượt phát nào.

Kết thúc Batch 2. Mày chỉ cần nhớ: **Đừng bao giờ search mù. Phải bóc tách câu hỏi -> Lọc SQL lấy 500 ID -> Quăng ID sang cho Hybrid Search càn quét lấy 50 ID tốt nhất.**



---



### Batch 3: Khâu Chấm Điểm Tuyệt Đối (Reranker) & Hàm Phân Rã Thời Gian

Sau khi giã xong cái Hybrid Search ở Batch 2, mày đang cầm trong tay Top 50 tài liệu ngon nhất. Nhưng đem 50 cái này tọng vào họng con LLM ở cuối pipeline thì nó nghẹn và bắt đầu nói sảng ngay.

Tại sao? Vì trong 50 cái đó có thể chứa 5 version của cùng 1 cái Jira ticket (từ lúc Open đến lúc Closed), hoặc 10 cái tin nhắn Slack nội dung y hệt nhau. Nhiệm vụ của Batch 3 là vắt kiệt 50 thằng ứng viên này xuống còn 10 thằng tinh hoa nhất: **Đúng chủ đề nhất, Mới nhất, và Đa dạng nhất.**

Vũ khí bí mật của mày nằm ở cái công thức Toán học này:

$$\text{Final\_Score} = \text{Relevance} \times e^{-\lambda \Delta t} \times \text{Diversity\_Penalty}$$

Nhớ kỹ, nó là **phép nhân**, đéo phải phép cộng. Nếu dùng phép cộng như thiết kế gốc cùi bắp, một tài liệu đéo liên quan gì (Relevance = 0) nhưng vừa mới tạo cách đây 1 phút (Độ mới max điểm) vẫn có thể leo lên top. Với phép nhân, một nhân tố bằng 0 là cả cụm cút về 0.

Đây là cách từng biến số trong công thức này vận hành:

#### 1. $\text{Relevance}$ (Độ Phù Hợp Cốt Lõi)

Mày lấy 50 tài liệu chạy qua một mô hình **Cross-Encoder** (ví dụ: `BGE-Reranker-v2`).

Khác với Vector Search (Bi-Encoder) băm tài liệu ra thành số rồi tính khoảng cách một cách độc lập và ngu ngơ, Cross-Encoder nó nhét thẳng cả "Câu hỏi" lẫn "Tài liệu" vào chung một lượt để AI đọc hiểu sự liên kết chéo. Quá trình này chậm hơn nhiều, nhưng độ chính xác thì tuyệt đối.

Điểm $\text{Relevance}$ trả về nằm trong khoảng 0 đến 1. Nếu điểm này lẹt đẹt dưới 0.3, thẳng tay loại luôn khỏi bộ nhớ để tiết kiệm tài nguyên tính toán cho 2 biến số tiếp theo.

#### 2. $e^{-\lambda \Delta t}$ (Hàm Suy Giảm Thời Gian - Temporal Decay)

Đây là cú vả cực mạnh vào những hệ thống RAG không biết xử lý dữ liệu lỗi thời.

Giả sử mày có 2 tài liệu giống hệt nhau về nội dung (Relevance bằng nhau). Một cái là ticket tháng trước ghi "Đang xử lý", một cái vừa update sáng nay ghi "Đã xong".

- $\Delta t$: Khoảng lùi thời gian (ví dụ: số ngày tính từ hiện tại về lúc tài liệu được cập nhật cuối cùng). Tài liệu càng cũ thì $\Delta t$ càng lớn.
- $\lambda$: Hệ số kiểm soát tốc độ lỗi thời (Decay rate). Nếu data công ty thay đổi cực lẹ, set $\lambda$ cao. Nếu data thuộc dạng tài liệu chuẩn ít khi đổi, set $\lambda$ thấp.

Khi tài liệu càng cũ, lũy thừa âm làm cho nguyên cái cụm $e^{-\lambda \Delta t}$ tiệm cận về 0. Nghĩa là nội dung có khớp đến mấy mà out-date thì điểm tổng cũng nát. T-RAG tự động chọn ra phiên bản MỚI NHẤT mà đéo cần phải thiết kế cái luồng routing phức tạp hay liên tục ghi đè vào database.

#### 3. $\text{Diversity\_Penalty}$ (Phạt Trùng Lặp bằng MMR)

Giả sử mày lấy được top 3 tài liệu điểm cực cao, nhưng nội dung 3 cái đó chỉ là 3 cái comment "Đã nhận" trên cùng một luồng email. Nếu nhét cả 3 vào câu trả lời, mày đang phí phạm bộ nhớ Context Window của con LLM.

Thuật toán MMR (Maximal Marginal Relevance) sẽ can thiệp. Nếu hệ thống đã bốc một file nói về "Lỗi Webhook" đưa vào danh sách chọn, thì khi xét đến file thứ 2 có ý nghĩa quá giống file 1, nó sẽ dập điểm $\text{Diversity\_Penalty}$ xuống thấp. Bắt buộc hệ thống phải nhường chỗ cho một tài liệu mang góc nhìn khác (ví dụ: "Tài liệu hướng dẫn fix lỗi Webhook").

**Kết quả:** Top 10 tài liệu đi ra từ cái màng lọc này là những thứ hoàn hảo nhất. Lúc này, con LLM cuối cùng (như Qwen hay Llama) chỉ việc đọc 10 tài liệu tinh khiết này và nhả ra câu trả lời cuối cùng cho user.



---



### Batch 4: Vũ Khí Cày Benchmark (Database Local & vLLM Offline Batching)

Mày làm hệ thống để chạy **EnterpriseRAG-Bench** viết báo khoa học, mục tiêu tối thượng không phải là chịu tải hàng triệu user cùng lúc, mà là: **Dễ tái lập (Reproducibility)** và **Tốc độ nuốt batch (Throughput)**.

Đây là lý do tao băm nát cái kiến trúc Microservices rườm rà ban đầu và thay bằng combo "mì ăn liền" nhưng sát thủ này:

#### 1. SQLite + FTS5 (Đánh sập khái niệm Database Server)

Tại sao đéo xài Postgres hay Elasticsearch? Vì hội đồng chấm bài hay những thằng tải code của mày về trên GitHub đéo rảnh để cài Docker hay setup server DB.

- **Cách làm:** Mày nhét toàn bộ 500,000 tài liệu kia vào một file `.sqlite` duy nhất.
- **Độ bá đạo:** SQLite có sẵn cái module **FTS5** (Full-Text Search). Nó biến file SQLite cùi bắp thành một cỗ máy tìm kiếm theo thuật toán BM25. Thế là mày vừa dùng lệnh SQL bình thường để lọc Soft-Filtering (tên tác giả, ngày tháng, loại file), vừa dùng luôn nó để làm cái "mũi khoan Sparse Search" tìm từ khóa chính xác. Gọn gàng, code chạy cái vù đéo cần chờ rớt mạng.

#### 2. LanceDB hoặc Qdrant Local (Mũi Khoan Vector In-memory)

Thay vì dựng Qdrant Server đứng chình ình một góc, mày nhúng thẳng database vào code Python. Dữ liệu vector được load thẳng vào RAM hoặc đọc cực nhanh từ ổ cứng SSD cục bộ. Lúc cái rổ 500 Candidate IDs được bắn sang từ SQLite, thằng LanceDB/Qdrant sẽ lấy danh sách ID đó để filter và chạy Cosine Similarity quét ngữ nghĩa (Dense Search) tắp lự. Không suy hao tốc độ qua cổng HTTP hay gRPC.

#### 3. Vắt kiệt con H100 bằng vLLM (Offline Batching)

Trường mày cấp cho con H100 (40GB - 80GB VRAM) mà mày gọi API từng câu hỏi một thì phí phạm của giời. Mày phải chơi chiến thuật nhồi nhét.

- **Co-hosting:** Nhét cùng lúc con Llama-3-8B (làm Query Parser ở Bước 1), con BGE-Reranker (chấm điểm ở Bước 3) và con Qwen (sinh text cuối) vào chung bộ nhớ VRAM.
- **Offline Batching:** Đừng thiết kế luồng chạy tuần tự 1 câu hỏi từ đầu đến cuối. Mày lấy cái list 500 câu của EnterpriseRAG-Bench, tọng một phát vào họng thằng vLLM. Engine của nó sẽ tự gom batch dưới tầng GPU (PagedAttention) để tính toán song song ma trận. Tốc độ sẽ xé gió, chạy xong cả cái benchmark có khi chưa kịp uống hết ly cà phê phin.

Xong! Toàn bộ bức tranh từ lúc vạch trần sự ngu ngục của RAG truyền thống, chiến thuật chia để trị bằng Soft-Filtering, mài hai mũi khoan Hybrid Search, nén điểm bằng hàm phân rã thời gian, cho đến cách chọn DB chạy local ăn gian tốc độ... tao đã bơm hết vào đầu mày rồi. Hiểu sâu sát từ gốc rễ thế này thì ông thầy nào vặn cũng đéo sợ cứng họng.