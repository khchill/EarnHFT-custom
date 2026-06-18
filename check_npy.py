import numpy as np
import os
import glob

def check_npy_folder(folder_path):
    print(f"==================================================")
    print(f"🔍 ĐANG KIỂM TRA THƯ MỤC: {folder_path}")
    print(f"==================================================")
    
    if not os.path.exists(folder_path):
        print(f"[LỖI] Không tìm thấy thư mục: {folder_path}")
        print("=> Hãy sửa lại đường dẫn 'target_folder' ở cuối file cho đúng nhé!")
        return
        
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    if not npy_files:
        print("Trống không! Không có file .npy nào trong thư mục này.")
        return
        
    for file in sorted(npy_files):
        filename = os.path.basename(file)
        try:
            data = np.load(file, allow_pickle=True)
            print(f"\n📄 File: {filename}")
            print(f"   - Hình dạng mảng (Shape): {data.shape}")
            print(f"   - Kiểu dữ liệu (Dtype):   {data.dtype}")
            
            # Xử lý In chi tiết tuỳ theo cấu trúc của mảng
            if data.size == 1:
                # Nếu file chỉ lưu 1 con số duy nhất (ví dụ: final_balance.npy)
                print(f"   => Giá trị chốt: {data.item()}")
                
            elif "action" in filename:
                # Nếu là mảng hành động, đếm xem mỗi Action xuất hiện bao nhiêu lần
                if len(data.shape) == 1:
                    unique, counts = np.unique(data, return_counts=True)
                    action_dist = dict(zip(unique, counts))
                    print(f"   => Phân bổ Hành động (Số lượng): {action_dist}")
                elif len(data.shape) == 2:
                    print(f"   => Đây là ma trận hành động (n, m) của Router.")
                    print(f"   => 5 hành động đầu tiên: {data[:5].tolist()}")
                
            elif "reward" in filename or "require_money" in filename or "balance" in filename:
                # Nếu là mảng reward / money, in ra thống kê toán học
                print(f"   => Tổng cộng (Sum):    {np.sum(data):.4f}")
                print(f"   => Trung bình (Mean):  {np.mean(data):.6f}")
                print(f"   => Nhỏ nhất (Min):     {np.min(data):.6f}")
                print(f"   => Lớn nhất (Max):     {np.max(data):.6f}")
                print(f"   => 5 phần tử đầu tiên: {data.flatten()[:5].tolist()}")
                
        except Exception as e:
            print(f"\n📄 File: {filename}")
            print(f"   [LỖI ĐỌC FILE] File có thể bị lỗi hoặc định dạng lạ: {e}")
            
    print(f"\n==================================================\n")

if __name__ == "__main__":
    # Anh có thể thay đổi đường dẫn này để check bất kỳ folder test hay valid nào!
    target_folder = "result_risk/BTCUSDT/beta_30.0/seed_12345/epoch_2/test"
    check_npy_folder(target_folder)
