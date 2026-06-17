@echo off
chcp 65001 >nul
echo === BẮT ĐẦU CHẠY SÀNG LỌC BOT, HUẤN LUYỆN ROUTER VÀ XUẤT BÁO CÁO (WINDOWS) ===

:: Đảm bảo sử dụng đúng môi trường ảo Python trên Windows
set PYTHON_CMD=".venv\Scripts\python.exe"

:: -------------------------------------------------------------
:: PHẦN 1: Đánh giá Low-level Bots và Huấn luyện High-level Router
:: -------------------------------------------------------------
echo [*] 1. Đang chạy Validation cho toàn bộ Model trong Pool...
%PYTHON_CMD% src\EarnHFT_framwork\tool\run_validation.py

echo [*] 2. Đang sàng lọc Bot giỏi nhất để tạo Ma trận 5x5...
%PYTHON_CMD% src\EarnHFT_framwork\analysis\pick_agent\pick_agent_position.py
%PYTHON_CMD% src\EarnHFT_framwork\analysis\pick_agent\inspect_matrix.py

echo [*] 3. Đang huấn luyện High-Level Router (AI Giám đốc)...
%PYTHON_CMD% src\EarnHFT_framwork\RL\agent\high_level\dqn_position.py

:: -------------------------------------------------------------
:: PHẦN 2: Xuất Báo Cáo và Vẽ Đồ Thị
:: -------------------------------------------------------------
echo [*] 4. Đang tính toán Metric và xuất bảng báo cáo so sánh...
%PYTHON_CMD% src\EarnHFT_framwork\analysis\calculate_metric\calculate_metric.py

echo [*] 5. Đang đếm số lượng giao dịch...
%PYTHON_CMD% count_trades.py


echo [*] 7. Đang vẽ biểu đồ Tổng hợp (Graph)...
%PYTHON_CMD% src\EarnHFT_framwork\RL\util\graph.py

echo === HOÀN TẤT TOÀN BỘ QUÁ TRÌNH SÀNG LỌC VÀ BÁO CÁO ===
pause
