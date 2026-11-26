# Hardsub2srt

Ứng dụng trích phụ đề cứng từ video thành SRT. Đã có nền tảng video/GPU,
phát hiện ROI, OCR đa ngôn ngữ, ghép SRT, pipeline/CLI và **bước 5**:
giao diện Gradio cùng Docker Compose cho NVIDIA GPU.

Mã nguồn chính: [github.com/tienok2004/Hardsub2srt](https://github.com/tienok2004/Hardsub2srt).
Khi tiếp tục trên máy khác, clone repository, đọc `AGENTS.md` và `WORKLOG.md`,
rồi tạo lại `.venv` theo phần cài đặt bên dưới. Trước khi sửa ở một bản sao đã có,
kiểm tra `git status -sb`, `git fetch origin main` và đồng bộ với `origin/main`;
không ghi đè thay đổi chưa commit.

## Chạy thử bước 1

Công cụ cài trên máy Windows hiện tại nằm trong `.tools/`. Chạy trực tiếp:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\Tien data\Soft\Hardsub2srt\run-step1.ps1' -Video 'D:\video\sample.mp4' -Fps 3
```

Cần Python 3.10+ và `ffmpeg`/`ffprobe` trong PATH, hoặc truyền đường dẫn thực thi rõ ràng:

```powershell
python -m core.gpu
python -m core.video_sampler 'D:\video\sample.mp4' --fps 3 `
  --ffmpeg 'D:\ffmpeg\bin\ffmpeg.exe' --ffprobe 'D:\ffmpeg\bin\ffprobe.exe'
```

Trên Linux, dùng cùng hai lệnh và thay đường dẫn video/FFmpeg phù hợp. Tiện ích GPU
chỉ xác nhận khởi tạo CUDA thực tế khi có mô hình ONNX: `python -m core.gpu --model model.onnx`.
Mô hình OCR hiện được kết nối trong `core/ocr_engine.py`.

Các công cụ cài cục bộ để thử trên máy phát triển nằm trong `.tools/` và không thuộc mã nguồn.

## Chạy thử bước 2 trên Windows

Môi trường Python cho OCR nằm trong `.venv/`. Có thể tái tạo bằng
`pip install -r requirements-cpu.txt` trong môi trường ảo; trên máy CUDA dùng
`requirements-gpu.txt` và kiểm tra phiên bản CUDA/cuDNN tương thích.
Tìm ROI và tạo ảnh xem trước:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'D:\Tien data\Soft\Hardsub2srt\run-step2.ps1' -Video 'D:\video\sample.mp4' -Preview 'D:\video\roi_preview.jpg'
```

Nếu ROI tự động không đủ bằng chứng, kết quả sẽ là toàn khung hình. Có thể chỉ định
ROI thủ công bằng `-Manual 0.1,0.7,0.9,0.95`. Các tỷ lệ lần lượt là x_min, y_min,
x_max, y_max. Để thử OCR một ảnh tĩnh:

```powershell
Set-Location 'D:\Tien data\Soft\Hardsub2srt'
& 'D:\Tien data\Soft\Hardsub2srt\.venv\Scripts\python.exe' -m core.ocr_engine 'D:\video\roi_preview.jpg' --lang auto
```

PP-OCRv6 small dùng chung trọng số đa ngôn ngữ cho `auto`, `ch`, `vi`, `ja`, `en`;
nhãn ngôn ngữ không ép mô hình chỉ trả một ngôn ngữ. Kiểm thử ảnh tổng hợp cho thấy
mô hình vẫn có thể làm sai dấu tiếng Việt, nên cần đánh giá thêm trên video thực tế.
Ở máy NVIDIA, thay gói `onnxruntime` bằng `onnxruntime-gpu` cùng phiên bản tương thích
CUDA/cuDNN; tiện ích GPU sẽ thử khởi tạo mô hình thật trước khi bật CUDA.

## Bước 3: ghép SRT và sửa OCR tùy chọn

`core/srt_builder.py` nhận các quan sát OCR có thời gian và tọa độ, lọc chữ
cố định quá 15 giây, ghép văn bản liên tiếp có độ tương đồng từ 85%, nối khoảng
hở dưới 0,12 giây, bỏ URL quảng bá rõ ràng và bỏ đoạn chỉ xuất hiện một khung hình. Có thể điều chỉnh
các ngưỡng bằng `BuilderConfig`; `write_srt` ghi tệp UTF-8 có BOM.

`core/llm_refiner.py` tắt mặc định. Khi bật, truyền `base_url`, `model` và
`api_key` (nếu dịch vụ cần). Mỗi yêu cầu gửi tối đa 30 mục tới endpoint
`/chat/completions`; phản hồi phải là JSON ánh xạ đúng từng ID sang văn bản.
ID và thời gian được giữ từ bản gốc. Nếu yêu cầu lỗi hoặc phản hồi thiếu ID,
lô đó giữ nguyên văn bản OCR. Không ghi API key vào mã nguồn hoặc log.

Chạy kiểm thử trên máy này:

```powershell
Set-Location 'D:\Tien data\Soft\Hardsub2srt'
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

## Bước 4: xử lý video từ CLI

Trên máy Windows này, dùng Python trong `.venv`; CLI tự tìm FFmpeg đã cài ở `.tools/`:

```powershell
Set-Location 'D:\Tien data\Soft\Hardsub2srt'
& '.\.venv\Scripts\python.exe' '.\cli.py' --input 'C:\Users\TienNTQ\Downloads\1.mp4' --output '.\output' --lang auto --fps 3
```

Xử lý thư mục và giữ cấu trúc thư mục con:

```powershell
& '.\.venv\Scripts\python.exe' '.\cli.py' --input 'D:\video' --output '.\output' --recursive --preserve-subfolders
```

CLI mặc định bỏ qua tệp đã có SRT; `--overwrite` buộc chạy lại. Nếu hai video
cùng tạo một tên SRT, video đứng sau trong thứ tự đường dẫn được bỏ qua để tránh
ghi đè. Một video lỗi không dừng các video khác. Dùng `--roi x_min y_min x_max y_max`
để đặt vùng thủ công, hoặc `--no-auto-roi` để quét toàn khung hình.
`--llm-refine --llm-base-url ... --llm-model ...` bật sửa OCR; API key có thể
đặt trong biến môi trường `HARDSUB_LLM_API_KEY`.
Nếu logo ngắn vẫn lọt vào SRT, dùng `--watermark-term 'tên logo'` (lặp lại được)
hoặc khai báo `srt.watermark_terms` trong `config.yaml`.

**Tách theo ngôn ngữ:** mỗi dòng OCR trong vùng hội thoại được đưa vào luồng ngôn
ngữ tương ứng trước khi ghép thời gian. Video song ngữ Trung–Anh như `1.mp4` tạo
`1.zh.srt` và `1.en.srt`; video một ngôn ngữ cũng dùng hậu tố ngôn ngữ, ví dụ
`phim.vi.srt`. Mã hỗ trợ: `zh` (Trung), `vi` (Việt), `ja` (Nhật), `en` (Anh);
`und` khi không có chữ để xác định. Chú thích/tin nhắn được đưa vào tệp theo
ngôn ngữ của chính nội dung đó. Với `--lang auto`, tiếng Việt có dấu được nhận
theo dấu; câu tiếng Việt không dấu và câu tiếng Nhật chỉ có Hán tự có thể bị
phân nhầm vì OCR hiện không trả nhãn ngôn ngữ cho từng dòng.

**Câu chú thích/biên tập** (ví dụ "Hà Nội, 1975" ở đầu màn hình) được đọc riêng
với hội thoại, trong các vùng `captions.zones` (mặc định 35% phía trên khung hình,
đã trừ vùng hội thoại). `--captions merged` (mặc định) ghi chung vào SRT với thẻ
`{\an8}` để hiện ở đầu màn hình; nội dung ngoài lời thoại đặt trong ngoặc tròn.
`--captions separate` ghi ra `<tên video>.<mã ngôn ngữ>.captions.srt`;
`--captions off` chỉ lấy hội thoại. Chữ chèn hậu kỳ được giữ khi đứng yên,
kéo dài 1–15 giây và OCR đọc rõ. Chú thích thông thường cần nằm ngang và có
cỡ chữ gần cỡ phụ đề. Chữ Hán dọc giới thiệu nhân vật được nhận riêng; hai
cột liền nhau được ghép thành một mục, ví dụ `(赵丽颖：金鸡奖影后)`. Bị loại:
logo đứng yên cả video, logo động lặp đi lặp lại, URL/quảng cáo (`captions.ad_terms`
và `srt.watermark_terms`), biển hiệu hay chữ trên áo quần (nghiêng hoặc di chuyển
theo cảnh), chữ quá nhỏ hoặc chữ dọc không phải chú thích. Số chữ bị loại theo từng lý do được ghi trong log. Bật
chú thích làm chậm khoảng 30–40% vì phải giải mã cả khung hình.

**Tin nhắn trên điện thoại/màn hình trong cảnh** được quét ở vùng giữa khung hình
(`captions.screen_text.zones`, mặc định x=12–88%, y=28–78%). Tool theo dõi chữ
qua nhiều khung, cho phép điện thoại hơi nghiêng hoặc di chuyển, bỏ dòng đồng hồ,
tiêu đề ngắn, URL, watermark và logo cố định/lặp lại. Nội dung được ghi vào SRT
trong ngoặc tròn, ví dụ `(恭喜，终于等到这一天了。)`. Chế độ `--captions off` bỏ qua cả
chú thích lẫn chữ trên màn hình; `merged` và `separate` áp dụng cho cả hai. Có thể
chỉnh vùng quét, ngưỡng thời gian và độ tin cậy trong `config.yaml`. Việc quét
vùng này cần thêm một lượt OCR; tốc độ phụ thuộc video và máy chạy. Với cảnh chữ
chỉ thoáng qua dưới 0,65 giây hoặc OCR không đọc rõ, tin nhắn có thể bị bỏ qua.

**Dừng giữa chừng:** nhấn `Ctrl+C` một lần để dừng sau khung hình hiện tại. Phụ đề
đã quét được lưu vào `<tên video>.<mã ngôn ngữ>.partial.srt` trong thư mục đầu ra (bỏ qua bước
sửa bằng LLM) và các video còn lại trong lô không được xử lý. Nhấn `Ctrl+C` lần nữa
để thoát ngay mà không lưu. Tệp `.partial.srt` không chặn chế độ chạy tiếp: lần
chạy sau vẫn xử lý lại video đó từ đầu.

## Web UI trên máy Windows này

```powershell
Set-Location 'D:\Tien data\Soft\Hardsub2srt'
& '.\.venv\Scripts\python.exe' '.\app.py'
```

Trên Windows, trình duyệt sẽ tự mở `http://127.0.0.1:7860`. Nếu không tự mở,
truy cập `http://localhost:7860` bằng tay. Khi chạy trên chính máy này (địa chỉ
`127.0.0.1`), dùng nút **Chọn tệp video…** / **Chọn thư mục…** để mở hộp thoại chọn
tệp của Windows: ứng dụng chỉ ghi nhận đường dẫn, không tải tệp lên. Trình duyệt
không cho trang web biết đường dẫn thật của tệp kéo thả, nên chế độ này thay ô kéo
thả bằng hộp thoại. Khi chạy trên máy chủ từ xa, ô **tải lên một video** hiện như cũ.
Có thể dán đường dẫn có dấu ngoặc kép (ví dụ từ "Copy as path" của Explorer): khi
rời ô nhập, ứng dụng tự bỏ ngoặc kép và các ký tự không hợp lệ như `?` `*` `<` `>` `|`. Nút **Xem trước ROI** tạo ảnh khoanh vùng trước khi
chạy OCR. Trên ảnh, kéo bên trong khung xanh để di chuyển hoặc kéo góc/cạnh để
đổi kích thước. Bốn ô tọa độ cập nhật theo khung; sửa các ô cũng làm khung đổi
ngay mà không phải quét ROI lại. Mọi chỉnh sửa chuyển sang ROI thủ công cho lần
xử lý tiếp theo. Nút **Bắt đầu xử lý** cập nhật tiến độ và nhật ký. Nút **Dừng và
lưu** dừng sau khung hình hiện tại và lưu phụ đề đã quét vào các tệp
`<tên video>.<mã ngôn ngữ>.partial.srt`
trong thư mục đầu ra; đóng tab trình duyệt khi đang chạy cũng dừng và lưu như vậy. Với một video, giao
diện hiển thị các tệp SRT để tải. Nếu chọn thư mục đầu ra khác, tệp kết quả vẫn
được lưu ở đó; bản phục vụ tải xuống được sao chép vào `web_cache/`.

## Linux VPS có NVIDIA GPU

Yêu cầu driver NVIDIA hoạt động (`nvidia-smi`), Python 3.11+ và FFmpeg.
`requirements-gpu.txt` cài kèm thư viện CUDA 12/cuDNN 9 qua pip. Trên Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg libgl1 libgomp1 zlib1g
python3 -m venv .venv
.venv/bin/pip install -r requirements-gpu.txt
.venv/bin/python -m core.gpu --model .venv/lib/python3.12/site-packages/rapidocr/models/PP-OCRv6_det_small.onnx
.venv/bin/python app.py
```

`requirements-gpu.txt` ghim ONNX Runtime GPU 1.26.0 cho CUDA 12.8/cuDNN 9.
Trên VPS có phiên bản Python khác, thay `python3.12` trong đường dẫn mô hình.
Kết quả cần thấy là `Phiên CUDA: đã xác minh` và `Provider được chọn:
CUDAExecutionProvider`. Chỉ thấy `CUDA provider: có` chưa đủ vì thư viện CUDA có
thể chưa nạp được; khi đó OCR tự dùng CPU. Để kiểm tra bằng mã thoát, thêm
`--require-cuda` vào lệnh kiểm tra.
Web UI lắng nghe `0.0.0.0:7860`; có thể truy cập qua SSH port forwarding.
Đặt `HARDSUB_WEB_USER` và `HARDSUB_WEB_PASSWORD` để bật xác thực giao diện.

## Truy cập từ Internet qua Cloudflare Tunnel

Cần một domain đã thêm vào Cloudflare và một tunnel **remotely managed**. Trong
Cloudflare Dashboard, vào **Networking → Tunnels**, tạo tunnel, chọn kết nối
`cloudflared` và lưu **tunnel token**. Trong tab **Routes**, thêm **Published
application** với hostname, ví dụ `sub.example.com`. URL dịch vụ tùy nơi chạy:

| Nơi chạy | Service URL trong Cloudflare |
| --- | --- |
| Phiên Vast.ai dạng container, dùng `run-vast.sh` | `http://localhost:7860` |
| VPS/VM Linux chạy Docker Compose | `http://hardsub2srt:7860` |

Tên miền công khai sẽ có HTTPS do Cloudflare cung cấp. Tunnel kết nối ra ngoài;
không cần mở cổng 7860 trên Internet. Cần cho phép kết nối **ra ngoài** tới
Cloudflare trên cổng 7844 (UDP hoặc TCP). Dùng tài khoản và mật khẩu giao diện
bên dưới; có thể bổ sung chính sách Cloudflare Access trong Dashboard.
Khi điều khiển từ web, đường dẫn video trong biểu mẫu là đường dẫn **trên
instance/VPS**, không phải trên máy đang mở trình duyệt. Nên chuyển phim lớn
bằng SSH/SFTP vào thư mục `input/` rồi nhập đường dẫn đó: Cloudflare giới hạn
dung lượng mỗi yêu cầu tải lên theo gói dịch vụ (Free/Pro: 100 MB; Business:
200 MB), nên tải phim lớn trực tiếp qua nút web có thể gặp lỗi 413.

### Phiên Vast.ai thông thường (container, không dùng Docker Compose)

Ưu tiên image Ubuntu 24.04 có driver NVIDIA hoạt động (`nvidia-smi`);
thư viện CUDA/cuDNN sẽ được cài trong `.venv` qua `requirements-gpu.txt`. Mở SSH vào
instance. Trên Windows, tạo gói nguồn chỉ gồm mã và cấu hình cần thiết:

```powershell
Set-Location 'D:\Tien data\Soft\Hardsub2srt'
& '.\.venv\Scripts\python.exe' '.\package_vast.py'
```

Chuyển `dist/hardsub2srt-vast.tar.gz` bằng SCP, sử dụng host và cổng SSH ở nút
**Connect** của phiên Vast.ai. Ví dụ trong PowerShell:

```powershell
$sshHost = Read-Host 'Host/IP SSH'
$sshPort = Read-Host 'Cổng SSH'
scp -P $sshPort '.\dist\hardsub2srt-vast.tar.gz' "root@${sshHost}:~/"
```

Trong phiên Vast.ai, giải nén và cài môi trường (đổi `root` nếu tên đăng nhập
SSH của instance khác):

```bash
tar -xzf ~/hardsub2srt-vast.tar.gz -C ~
cd ~/hardsub2srt
bash setup-vast.sh
```

Script cài gói hệ thống, môi trường Python GPU cùng thư viện CUDA/cuDNN qua pip
và `cloudflared`, sau đó kiểm tra FFmpeg và phiên CUDA với mô hình OCR thật.
Nếu kiểm tra CUDA thất bại, script báo rõ OCR đang dùng CPU; Web UI vẫn chạy.
Có thể chạy lại nếu quá trình
cài bị ngắt. `cloudflared` nằm trong `.tools/`; nếu chỉ muốn cài riêng công cụ
này, chạy `bash install-cloudflared.sh` mà không cần quyền root. Đặt thông tin
đăng nhập và token trong phiên shell (lệnh `read` tránh ghi bí mật vào lịch sử
lệnh):

```bash
tmux new -s hardsub
read -r -p 'Tên đăng nhập web: ' HARDSUB_WEB_USER
read -rs -p 'Mật khẩu web: ' HARDSUB_WEB_PASSWORD; echo
read -rs -p 'Cloudflare tunnel token: ' CLOUDFLARE_TUNNEL_TOKEN; echo
export HARDSUB_WEB_USER HARDSUB_WEB_PASSWORD CLOUDFLARE_TUNNEL_TOKEN
bash run-vast.sh
```

`run-vast.sh` chỉ cho Web UI nghe trên `127.0.0.1:7860` trong instance, đợi
Web UI khởi động rồi chạy tunnel. `input/` và `output/` được tạo tự động và hiện
sẵn trong giao diện. Để ngắt SSH mà vẫn giữ dịch vụ, nhấn `Ctrl+B` rồi `D` để
tách phiên `tmux`; dùng `tmux attach -t hardsub` để xem lại. Vast.ai cũng hỗ trợ
`/root/onstart.sh` nếu cần tự khởi động khi instance chạy lại. Khi thuê instance
mới, cài lại môi trường và dùng cùng tunnel token; không đưa token vào mã nguồn.

### VPS/VM Linux có Docker Compose

Trên máy có Docker Engine, Compose và NVIDIA Container Toolkit, tạo `.env`
từ `.env.example` rồi điền token và cặp tài khoản/mật khẩu. `.env` đã được
loại khỏi Git. Sau đó chạy:

```bash
mkdir -p input output
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml up --build -d
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml logs -f
```

Trong Compose, chỉ có `127.0.0.1:7860` của máy chủ được chuyển vào Web UI;
`cloudflared` truy cập ứng dụng qua mạng nội bộ Compose. Docker Compose là
lựa chọn cho VPS/VM có Docker, không phải phiên Vast.ai dạng container.

## Docker Compose trên PC/VPS có NVIDIA GPU

Cần Docker Compose và NVIDIA Container Toolkit/thiết lập GPU trong Docker.
Từ thư mục dự án:

```bash
mkdir -p input output
docker compose up --build -d
docker compose logs -f
```

Mở `http://localhost:7860` trên máy chủ (hoặc chuyển tiếp SSH khi ở xa). Trong giao diện container, video gắn kết nằm ở
`/input`, phụ đề ghi vào `/output`; `config.yaml` được gắn chỉ đọc. Ảnh Docker
dùng CUDA 12.8/cuDNN 9 và khởi tạo sẵn mô hình OCR mặc định lúc build. Khi không
có GPU NVIDIA, dùng môi trường CPU trên máy chủ với `requirements-cpu.txt`.

Kiểm tra phụ thuộc sau khi cài:

```bash
python -m unittest discover -s tests -v
```
