# Hướng dẫn tiếp tục dự án Hardsub2srt

## Bắt đầu trên máy mới

- Trao đổi, giải thích thay đổi và báo cáo tiến độ với người dùng bằng tiếng Việt.
- Đọc `Hardsub2srt.md` để biết mục tiêu và `README.md` để biết cách chạy hiện tại. Đối chiếu với mã nguồn; đặc tả và tài liệu không phải bằng chứng rằng một tính năng đã chạy đúng.
- Đọc `WORKLOG.md` để biết thay đổi gần đây, kết quả thử và giới hạn đã biết. Git và mã nguồn là nguồn kiểm chứng cuối cùng; nhật ký có thể cũ nếu phiên trước chưa đẩy commit.
- Kiểm tra `git status -sb` và `git log -1 --oneline` trước khi sửa. Nếu thư mục được sao chép nguyên từ máy khác, không ghi đè thay đổi chưa commit. Đồng bộ Git trước khi sửa trên nhiều máy.
- Repository chính: `https://github.com/tienok2004/Hardsub2srt`. Kiểm tra `git remote -v`; trước khi sửa trên máy khác, `git fetch origin main` rồi đối chiếu nhánh local với `origin/main`. Không `reset --hard` hoặc `pull` đè lên thay đổi chưa commit.
- `.venv/`, `.tools/`, `input/`, `output/` và `web_cache/` là dữ liệu cục bộ, không có trên Git. Môi trường Python trong `.venv/` có thể không dùng được sau khi chuyển máy; tạo lại theo `README.md` và `requirements-cpu.txt` hoặc `requirements-gpu.txt` phù hợp với máy mới.

## Kiến trúc và kiểm tra

- `core/` chứa xử lý video, ROI, OCR, ghép SRT và pipeline. `app.py` là Web UI Gradio; `cli.py` là giao diện dòng lệnh. Giữ logic xử lý dùng chung trong `core/`.
- ROI tự động có thể trả về toàn khung hình khi phụ đề thưa; giao diện cũng cho chỉnh ROI trên ảnh và đồng bộ bốn tọa độ.
- OCR hỗ trợ `auto`, `ch`, `vi`, `ja`, `en`; chế độ `auto` có thể nhận nhiều ngôn ngữ trong cùng video. Không suy ra CUDA hoạt động chỉ từ `CUDA provider: có`: cần `Phiên CUDA: đã xác minh` với mô hình ONNX thật.
- Trước khi bàn giao thay đổi mã, chạy `python -m unittest discover -s tests` trong môi trường đã cài phụ thuộc. Với thay đổi GPU hoặc triển khai, kiểm tra thêm trên máy Linux/NVIDIA thực tế nếu có quyền truy cập.
- Không commit video, tệp SRT đầu ra, `.env`, mật khẩu, API key, Cloudflare Tunnel token hoặc khóa SSH riêng. Không in các giá trị bí mật vào log hay ảnh chụp.

## Trạng thái bàn giao đã kiểm chứng (26-11-2026)

- Commit ứng dụng đã đẩy lên `main` và thử trong phiên này: `a288e5d`. Luôn kiểm tra `git log` vì mã có thể đã tiến thêm.
- Ứng dụng có Web UI, chỉnh ROI trực tiếp, tách SRT theo ngôn ngữ (`zh`, `vi`, `ja`, `en`), OCR tin nhắn trên điện thoại và chữ Hán dọc giới thiệu nhân vật. Chữ ngoài lời thoại được đặt trong ngoặc tròn. Xem `WORKLOG.md` để biết video đã thử và giới hạn thực tế.
- Bản cũ `5d695c2` đã cài `onnxruntime-gpu[cuda,cudnn]==1.26.0` cho Vast.ai. Theo lần thử trước, `setup-vast.sh` xác minh `CUDAExecutionProvider` với RTX 5060 Ti; Web UI tạo được. Đây là kết quả của phiên đó, không bảo đảm mọi VPS mới có GPU tương thích.
- Trong lần chạy trên Vast.ai trước đó, người dùng quan sát khoảng 25 khung hình OCR/giây ở mức lấy mẫu 3 khung hình/giây video. GPU dùng ít tài nguyên vì hiện xử lý tuần tự và chỉ OCR vùng phụ đề đã cắt. Người dùng quyết định **để việc nâng cấp hiệu năng làm sau**; không tự triển khai song song hóa khi chưa có yêu cầu mới.
- Cloudflare Tunnel trên phiên Vast.ai này chưa kết nối được: QUIC timeout, HTTP/2 lỗi bắt tay TLS/EOF tới edge trên cổng 7844. Web UI trên VPS vẫn chạy ở `127.0.0.1:7860`; có thể truy cập riêng qua SSH port forwarding. Không lưu địa chỉ VPS hoặc tunnel token vào repository.

## Khi người dùng yêu cầu tiếp tục

1. Xác nhận mục tiêu mới và kiểm tra trạng thái Git, môi trường chạy cùng lỗi/log mới nhất.
2. Nếu cần tối ưu hiệu năng, đo thời gian giải mã, OCR và ghi SRT trên video đại diện trước khi đổi kiến trúc; đánh giá bằng thời gian hoàn thành và chất lượng phụ đề, không chỉ phần trăm GPU.
3. Kiểm thử thay đổi, báo rõ điều đã xác minh và điều chỉ suy đoán. Chỉ triển khai lên VPS khi người dùng yêu cầu và có quyền truy cập phù hợp.
4. Sau khi hoàn thành thay đổi, thêm mục ngắn vào `WORKLOG.md`: ngày, commit/tính năng, cách thử và kết quả, giới hạn còn lại, bước tiếp theo nếu có. Commit và push nhật ký cùng mã nguồn khi người dùng yêu cầu đồng bộ Git.
