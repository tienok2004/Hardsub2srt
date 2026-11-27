# Hardsub2srt

Ứng dụng trích phụ đề cứng từ video thành SRT bằng OCR. Hỗ trợ Web UI, CLI, xử lý CPU hoặc NVIDIA GPU, tự tìm vùng phụ đề (ROI) và tách tệp theo ngôn ngữ.

Mã nguồn: [github.com/tienok2004/Hardsub2srt](https://github.com/tienok2004/Hardsub2srt).

## Hướng dẫn cài đặt

### Windows

Cần Python 3.11 trở lên và FFmpeg (`ffmpeg`, `ffprobe`) trong `PATH` hoặc trong `.tools/ffmpeg/`. Mở PowerShell tại thư mục dự án:

```powershell
py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

Trên máy NVIDIA GPU có CUDA phù hợp, dùng `requirements-gpu.txt` thay cho `requirements-cpu.txt` khi tạo môi trường mới. Kiểm tra FFmpeg bằng `ffmpeg -version` và `ffprobe -version` nếu cài qua `PATH`.

### Linux/VPS có NVIDIA GPU

Cần driver NVIDIA hoạt động (`nvidia-smi`) và Python 3.11 trở lên. Trên Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg libgl1 libgomp1 zlib1g
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` ghim ONNX Runtime GPU 1.26.0 và cài thư viện CUDA 12/cuDNN 9 qua pip. Xác minh GPU với mô hình OCR thật:

```bash
model_path="$(.venv/bin/python -c 'from pathlib import Path; import rapidocr; print(Path(rapidocr.__file__).parent / "models" / "PP-OCRv6_det_small.onnx")')"
.venv/bin/python -m core.gpu --model "$model_path" --require-cuda
```

Kết quả cần có `Phiên CUDA: đã xác minh` và `Provider được chọn: CUDAExecutionProvider`. Chỉ thấy `CUDA provider: có` chưa đủ để xác nhận GPU đang xử lý OCR. Máy Linux không có GPU có thể cài `requirements-cpu.txt`.

### Docker Compose trên PC/VPS có NVIDIA GPU

Cần Docker Engine, Docker Compose và NVIDIA Container Toolkit. Từ thư mục dự án:

```bash
mkdir -p input output
docker compose up --build -d
docker compose logs -f
```

Ảnh Docker dùng CUDA 12.8/cuDNN 9 và chuẩn bị mô hình OCR lúc build. Thư mục `input/` và `output/` của máy chủ được gắn vào `/input` và `/output` trong container; `config.yaml` được gắn chỉ đọc.

### Vast.ai

Trên Windows, chạy `python package_vast.py` rồi chuyển `dist/hardsub2srt-vast.tar.gz` lên instance bằng SCP. Trong container Ubuntu/Debian có driver NVIDIA:

```bash
tar -xzf ~/hardsub2srt-vast.tar.gz -C ~
cd ~/hardsub2srt
bash setup-vast.sh
```

Script cài gói hệ thống, môi trường GPU và `cloudflared`, sau đó kiểm tra CUDA bằng mô hình OCR thật. Cần Python 3.11+; có thể chạy lại script nếu cài đặt bị ngắt.

## Hướng dẫn chạy app, cấu hình khi chạy

### Web UI

Trên Windows, nhấp đúp `run-web.bat`, hoặc chạy:

```powershell
& .\.venv\Scripts\python.exe .\app.py
```

Trên Linux, chạy `.venv/bin/python app.py`. Mở `http://localhost:7860` trên máy chạy app. Windows mặc định nghe tại `127.0.0.1`; Linux mặc định nghe tại `0.0.0.0`. Có thể đặt `HARDSUB_WEB_HOST` và `HARDSUB_WEB_PORT` để đổi địa chỉ/cổng. Đặt đồng thời `HARDSUB_WEB_USER` và `HARDSUB_WEB_PASSWORD` để bật xác thực.

Web UI trên Windows cục bộ cho chọn tệp/thư mục bằng hộp thoại, không tải video lên. Khi truy cập máy chủ từ xa, nhập đường dẫn **trên máy chủ** hoặc tải lên một video. Nút **Xem trước ROI** cho kéo và chỉnh khung trực tiếp; bốn tọa độ cập nhật cùng khung. **Bắt đầu xử lý** hiển thị tiến độ và nhật ký. **Dừng và lưu** ghi phụ đề đã quét thành `.partial.srt`; đóng tab khi đang chạy cũng dừng và lưu. Với một video, giao diện cho tải các tệp SRT; bản phục vụ tải xuống được sao chép vào `web_cache/`.

Trong Docker, video ở `/input`, phụ đề ở `/output`. Nếu chạy trên VPS, có thể truy cập riêng bằng SSH port forwarding.

### CLI

Trên Windows, nhấp đúp `run-cli.bat` hoặc chạy trong PowerShell:

```powershell
& .\.venv\Scripts\python.exe .\cli.py
& .\.venv\Scripts\python.exe .\cli.py --input 'D:\video\sample.mp4' --output .\output --lang auto --fps 3
```

Không truyền `--input` thì CLI hỏi đường dẫn video hoặc thư mục; có thể dán đường dẫn từ Explorer kể cả dấu ngoặc kép. Trên Linux, thay lệnh Python bằng `.venv/bin/python`. Để xử lý cả thư mục con:

```powershell
& .\.venv\Scripts\python.exe .\cli.py --input 'D:\video' --output .\output --recursive --preserve-subfolders
```

CLI mặc định bỏ qua video đã có SRT; `--overwrite` buộc xử lý lại. Một video lỗi không dừng cả lô. Nhấn `Ctrl+C` một lần để lưu phụ đề đã quét thành `<tên video>.<mã ngôn ngữ>.partial.srt`; nhấn lần nữa để thoát ngay. Lần chạy sau xử lý lại video từ đầu.

### Cấu hình xử lý

Giá trị mặc định nằm trong `config.yaml`; CLI nhận `--config` để chọn tệp khác. Các tùy chọn chính:

| Tùy chọn | Tác dụng |
| --- | --- |
| `--lang auto\|ch\|vi\|ja\|en` | Chế độ OCR; `auto` có thể đọc nhiều ngôn ngữ trong cùng video. |
| `--fps 3` | Số khung hình lấy mẫu mỗi giây video. |
| `--roi X_MIN Y_MIN X_MAX Y_MAX` | ROI thủ công, tọa độ tỷ lệ từ 0 đến 1. |
| `--no-auto-roi` | Quét toàn khung hình. |
| `--captions merged\|separate\|off` | Ghi chú thích/chữ trong cảnh chung SRT, tệp riêng hoặc tắt. |
| `--recursive --preserve-subfolders` | Quét thư mục con và giữ cấu trúc đầu ra. |
| `--overwrite` | Xử lý lại video đã có SRT. |
| `--watermark-term 'tên logo'` | Thêm chữ cần lọc; có thể lặp tùy chọn. |
| `--ffmpeg ... --ffprobe ...` | Đường dẫn FFmpeg khi không dùng `PATH` hoặc `.tools/ffmpeg/`. |

ROI tự động có thể chọn toàn khung hình nếu phụ đề thưa; dùng ảnh xem trước hoặc `--roi` để chỉnh. Bộ lọc nhận diện tên trang lặp lại cạnh địa chỉ web hoặc biểu tượng thương hiệu, kể cả khi logo di chuyển hay OCR đọc thiếu chữ; nếu logo không có các dấu hiệu này, dùng `--watermark-term` để chỉ rõ chữ cần bỏ. Đầu ra tách theo ngôn ngữ: `phim.zh.srt`, `phim.vi.srt`, `phim.ja.srt`, `phim.en.srt`; `und` dùng khi không xác định được chữ. Với `--lang auto`, tiếng Việt không dấu và câu tiếng Nhật chỉ có Hán tự có thể bị phân nhầm.

Chú thích và tin nhắn trong cảnh được đặt trong ngoặc tròn. `--captions merged` (mặc định) đưa chú thích đầu màn hình vào SRT với thẻ `{\an8}`; `separate` tạo `<tên video>.<mã ngôn ngữ>.captions.srt`; `off` bỏ qua cả chú thích lẫn chữ trên màn hình. Chỉnh vùng quét và bộ lọc trong `config.yaml` tại `captions.zones`, `captions.screen_text` và `srt.watermark_terms`.

### Sửa OCR bằng LLM (tùy chọn)

Mặc định tắt. Trong CLI, dùng `--llm-refine --llm-base-url URL --llm-model MODEL`; đặt API key bằng biến môi trường `HARDSUB_LLM_API_KEY` hoặc ô trên Web UI. Dịch vụ cần hỗ trợ endpoint OpenAI-compatible `/chat/completions`. Nếu yêu cầu lỗi hoặc phản hồi thiếu ID, lô đó giữ nguyên văn bản OCR. Không đưa API key vào repository.

### Cloudflare Tunnel cho Web UI từ xa

Tạo tunnel remotely managed và Published application trong Cloudflare Dashboard. Service URL là `http://localhost:7860` cho `run-vast.sh`, hoặc `http://hardsub2srt:7860` cho Docker Compose có tunnel. Máy chủ cần kết nối ra Cloudflare qua cổng 7844 (UDP hoặc TCP). Nếu tunnel không kết nối được, dùng SSH port forwarding.

Trên Vast.ai, đặt thông tin đăng nhập và token trong shell rồi chạy:

```bash
tmux new -s hardsub
read -r -p 'Tên đăng nhập web: ' HARDSUB_WEB_USER
read -rs -p 'Mật khẩu web: ' HARDSUB_WEB_PASSWORD; echo
read -rs -p 'Cloudflare tunnel token: ' CLOUDFLARE_TUNNEL_TOKEN; echo
export HARDSUB_WEB_USER HARDSUB_WEB_PASSWORD CLOUDFLARE_TUNNEL_TOKEN
bash run-vast.sh
```

`run-vast.sh` chỉ cho Web UI nghe tại `127.0.0.1:7860` trong instance. Nhấn `Ctrl+B`, rồi `D` để tách khỏi `tmux`; dùng `tmux attach -t hardsub` để trở lại. Trên VPS/VM có Docker Compose, tạo `.env` từ `.env.example`, điền token và tài khoản rồi chạy:

```bash
mkdir -p input output
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml logs -f
```

Với video lớn, chuyển vào `input/` trên máy chủ bằng SSH/SFTP trước khi xử lý.

### Kiểm tra sau khi cài

Chạy trong môi trường đã cài phụ thuộc:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests
```

Trên Linux, dùng `.venv/bin/python -m unittest discover -s tests`.
