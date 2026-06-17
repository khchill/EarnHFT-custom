# Hướng Dẫn Chạy Nhanh Toàn Bộ Dự Án (Quick Start Guide)

Dự án EarnHFT được chia thành 3 giai đoạn hoạt động chính. Để tối ưu hóa quá trình chạy thực nghiệm và kiểm thử, toàn bộ luồng chạy đã được tự động hóa thành các file script (`.sh` cho Linux/Cloud và `.bat` cho Windows).

Quy trình 3 giai đoạn chuẩn bao gồm:
1. **Pipeline Dữ Liệu (`run_data_pipeline`)**: Tải dữ liệu thô, tính toán Features, và phân cụm thị trường (DTW).
2. **Train AI Song Song (`train_all_8_models`)**: Chạy đua vũ trang 10 con Bot cùng lúc (4 Low-level, 4 Baseline, 2 Ablation).
3. **Sàng Lọc & Báo Cáo (`run_eval_and_report`)**: Xếp hạng Bot, tạo Ma trận 5x5, huấn luyện AI Giám đốc (Router) và vẽ biểu đồ.

---

## 🐧 Dành Cho Hệ Điều Hành Linux / Cloud (Ubuntu, Kaggle, Personal Cloud)

Nếu bạn đang chạy code trên môi trường Linux (hoặc chạy Cloud như Kaggle), hãy sử dụng các file `.sh`. Hệ điều hành Linux hỗ trợ cực kỳ tốt cho việc chạy ngầm song song.

**Cấp quyền thực thi (Chỉ chạy 1 lần duy nhất):**
Trước khi chạy, bạn cần cấp quyền cho phép hệ thống thực thi các file này:
```bash
chmod +x run_data_pipeline.sh
chmod +x train_all_8_models.sh
chmod +x run_eval_and_report.sh
```

**Thứ tự chạy:**

> [!NOTE]
> Hãy kiên nhẫn đợi file trước chạy xong toàn bộ tiến trình thì mới gõ lệnh chạy file tiếp theo. 

1. **Kiểm tra Tài nguyên CPU & GPU (Bắt buộc):**
   Trước khi chạy bất cứ thứ gì, hãy đảm bảo PyTorch nhận diện được Card màn hình của bạn, nếu không máy sẽ chạy bằng CPU và bị treo vĩnh viễn:
   ```bash
   .venv/bin/python -c "import torch; print('\n[+] PyTorch nhận GPU:', torch.cuda.is_available()); print('[+] Số lượng GPU:', torch.cuda.device_count())"
   ```
   Để theo dõi CPU và GPU hoạt động liên tục trong lúc train, hãy mở thêm 2 cửa sổ Terminal khác và gõ:
   - Terminal 1 (Xem CPU & RAM): `htop`
   - Terminal 2 (Xem GPU & VRAM): `watch -n 1 nvidia-smi`

2. **Chuẩn bị Dữ liệu:**
   ```bash
   ./run_data_pipeline.sh
   ```

2. **Huấn luyện 10 Mô hình (Chạy ngầm song song):**
   ```bash
   ./train_all_8_models.sh
   ```
   *Mẹo:* Vì các tiến trình được ném vào chạy ngầm (chạy nền), bạn có thể mở một tab Terminal thứ hai và gõ lệnh `htop` (để xem CPU hoạt động 100%) và `watch -n 1 nvidia-smi` (để xem card màn hình ăn VRAM).

3. **Sàng lọc & Xuất Báo Cáo:**
   ```bash
   ./run_eval_and_report.sh
   ```
   *Lưu ý:* Quá trình này cũng sẽ tự động chạy script `visualize_convergence.py` để tạo ra file ảnh `result_risk/BTCUSDT/convergence_plot.png` giúp bạn dễ dàng so sánh độ hội tụ giữa các mô hình (DDQN vs Baseline).

---

## 🪟 Dành Cho Hệ Điều Hành Windows

Nếu bạn kéo source code về máy tính cá nhân dùng Windows nguyên bản, bạn **bắt buộc** phải sử dụng các file `.bat`. Môi trường Windows không hiểu lệnh `nohup` và `&` của Linux.

**Yêu cầu môi trường:** Đảm bảo bạn đã cài đặt Python vào môi trường ảo `.venv` ngay trong thư mục gốc của dự án. File `.bat` sẽ tự động trỏ đường dẫn tới `.venv\Scripts\python.exe`.

**Thứ tự chạy:**

> [!IMPORTANT]
> Chỉ cần click đúp chuột vào file để chạy. Tuyệt đối không tắt màn hình đen (CMD) khi nó đang hiển thị chạy ngầm.

1. **Kiểm tra Tài nguyên CPU & GPU (Bắt buộc):**
   Mở ứng dụng `Command Prompt` (CMD) trong thư mục gốc, gõ lệnh sau để đảm bảo PyTorch của Windows nhận diện được Card màn hình:
   ```cmd
   .venv\Scripts\python.exe -c "import torch; print('\n[+] PyTorch nhận GPU:', torch.cuda.is_available()); print('[+] Số lượng GPU:', torch.cuda.device_count())"
   ```
   Để theo dõi quá trình chạy ngầm, hãy mở **Task Manager** (Ctrl + Shift + Esc), chuyển sang tab **Performance** để xem % hoạt động của CPU và GPU (Dedicated GPU Memory).

2. **Chuẩn bị Dữ liệu:** 
   Click đúp chuột vào file `run_data_pipeline.bat`

2. **Huấn luyện 10 Mô hình:** 
   Click đúp chuột vào file `train_all_8_models.bat` (hoặc `quick_test_2_epochs.bat` để test lướt).
   *Mẹo:* Cửa sổ CMD sẽ lập tức hiện thông báo đã kích hoạt chạy ngầm. Lúc này, bạn hãy mở **Task Manager** của Windows (tab Performance) để xem CPU/RAM của máy đang cày ải. Khi nào máy hết giật, CPU tụt xuống lại mức bình thường thì tức là đã train xong!

3. **Sàng lọc & Xuất Báo Cáo:** 
   Click đúp chuột vào file `run_eval_and_report.bat`
   *Lưu ý:* Kết thúc quá trình, hệ thống sẽ tự động vẽ và lưu file đồ thị `result_risk/BTCUSDT/convergence_plot.png` để bạn xem mức độ học hỏi (hội tụ) của mô hình.

---

### 🆘 Xử Lý Sự Cố
- Nếu quá trình chạy 10 mô hình khiến máy của bạn bị treo hoặc báo lỗi Out Of Memory (OOM), hãy mở file `train_all_8_models` lên, dùng dấu `#` (Linux) hoặc `::` (Windows) để comment (vô hiệu hóa) bớt vài con Baseline hoặc Ablation đi. Chạy từng đợt 4 con sẽ an toàn hơn nếu RAM yếu.
- Nếu script báo không tìm thấy thư mục `data/...`, hãy chắc chắn bạn đã giải nén file data hoặc chạy `run_data_pipeline` thành công ở Giai đoạn 1.
