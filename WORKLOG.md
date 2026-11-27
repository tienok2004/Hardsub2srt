# Nhật ký phát triển Hardsub2srt

Tệp này giúp phiên làm việc trên máy khác tiếp tục từ kết quả đã kiểm chứng. Đọc
`AGENTS.md` và `README.md` trước khi sửa. Luôn đối chiếu ghi chép với `git log`,
`git status` và mã nguồn hiện tại. Video mẫu là dữ liệu cục bộ ngoài repository;
SRT thử nằm trong `output/` cục bộ và không có trên Git.

## 27-11-2026 — Sửa cảnh báo chuỗi thoát trong `cli.py`

- Docstring của `_clean_path` chứa `C:\Videos\a.mp4`; đổi sang chuỗi thô `r"""` để
  Python 3.12+ không báo `SyntaxWarning: invalid escape sequence`.
- Đã chạy `python -m unittest discover -s tests` trong `.venv` trên Windows: 81
  kiểm thử qua, 5 kiểm thử ROI/OCR bị bỏ qua vì máy này chưa có FFmpeg cục bộ
  (`.tools/`). Cần cài FFmpeg để chạy đủ 86 kiểm thử.

## 26-11-2026 — OCR chữ trong cảnh và tách ngôn ngữ (`a288e5d`)

- `core/captions.py` và `core/pipeline.py`: quét tin nhắn trên điện thoại ở vùng
  giữa khung hình; lọc URL, logo góc khung và chữ lặp. Chữ ngoài lời thoại vào SRT
  trong ngoặc tròn. Chữ Hán dọc giới thiệu nhân vật được ghép theo cột, khử các
  biến thể OCR trùng vị trí.
- `core/language_split.py`: tách từng dòng OCR theo chữ viết trước khi ghép thời
  gian. Tên đầu ra là `<tên phim>.<mã ngôn ngữ>.srt`, ví dụ `1.zh.srt` và
  `1.en.srt`; Web UI cho tải nhiều tệp.
- Đã chạy `python -m unittest discover -s tests` trong `.venv` trên Windows:
  **86 kiểm thử qua**. Chưa kiểm thử các thay đổi này trên Linux/NVIDIA.
- Đã OCR toàn bộ video mẫu `1.mp4` (5 phút 17 giây, 1080×1920, lấy mẫu 3 FPS)
  trên CPU trong khoảng 8 phút 18 giây. SRT cuối có 108 mục tiếng Trung và 89
  mục tiếng Anh. Đã đối chiếu hình: `(赵丽颖：金鸡奖影后)` tại 00:01,667–00:03,667;
  `(陈默：新人导演)` tại 02:22,333–02:24,333; tin nhắn
  `(恭喜，终于等到这一天了。)` tại 00:15,667–00:18,000.
- Giới hạn đã thấy trên video mẫu: còn vài chữ logo `红豆短剧` lọt vào SRT tiếng
  Trung; tin nhắn điện thoại gần 00:59 thiếu đoạn cuối; tiếng Anh đôi khi thiếu
  khoảng trắng hoặc sai chữ. Đây là quan sát trên một video, không phải thước đo
  độ chính xác chung. Tiếng Việt không dấu và câu Nhật chỉ có Hán tự có thể bị
  phân loại nhầm vì OCR hiện không trả nhãn ngôn ngữ từng dòng.
- Hiệu năng chưa được tối ưu thêm theo quyết định của người dùng; trước khi đổi
  kiến trúc cần đo thời gian giải mã, OCR và chất lượng trên video đại diện.

## Các mốc trước đó

- `ca230f9`: thêm `run-web.bat` để mở Web UI trên Windows.
- `7691ee6`: đọc chú thích/biên tập ngoài vùng hội thoại cùng phụ đề.
- `5d695c2`: Web UI, chỉnh ROI trên ảnh và chuẩn bị CUDA cho Vast.ai. Theo lần
  thử trước, Cloudflare Tunnel của phiên Vast.ai không kết nối được; dùng SSH
  port forwarding khi cần. Không đưa địa chỉ VPS hoặc token vào Git.

## Cách ghi mục mới

Thêm mục mới ở đầu tệp với ngày và commit liên quan. Ghi hành vi thực sự đã chạy,
lệnh hoặc môi trường kiểm thử, kết quả, giới hạn chưa xử lý. Không ghi video, SRT
đầu ra, mật khẩu, API key, tunnel token hay khóa SSH vào repository.
