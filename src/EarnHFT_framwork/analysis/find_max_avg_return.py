import csv

input_file = 'results/combined_epoch_report.csv'

# Dictionary lưu giữ epoch tốt nhất cho từng model
best_per_model = {}

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    try:
        model_idx = header.index('Model')
        epoch_idx = header.index('Epoch')
        mode_idx = header.index('Mode')
        pos_idx = header.index('Position Changes')
        avg_idx = header.index('Avg Return per Trade(%)')
        ret_idx = header.index('Return(%)')
    except ValueError as e:
        print(f"Lỗi: Không tìm thấy cột {e}")
        exit(1)
        
    for row in reader:
        if not row or len(row) <= avg_idx:
            continue
            
        model_name = row[model_idx].strip()
        pos_changes_str = row[pos_idx].strip()
        avg_str = row[avg_idx].strip()
        
        if pos_changes_str == '-' or avg_str == '-':
            continue
            
        try:
            pos_changes = int(pos_changes_str)
            avg_val = float(avg_str)
            
            # Chỉ lấy những cấu hình có trade đủ lớn (ví dụ > 10)
            if pos_changes > 10:
                if model_name not in best_per_model:
                    best_per_model[model_name] = {'max_val': -float('inf'), 'row': None}
                
                if avg_val > best_per_model[model_name]['max_val']:
                    best_per_model[model_name]['max_val'] = avg_val
                    best_per_model[model_name]['row'] = row
        except ValueError:
            continue

print("=== KẾT QUẢ: AVG RETURN PER TRADE TỐT NHẤT CHO TỪNG MODEL ===")
for model, data in sorted(best_per_model.items()):
    r = data['row']
    if r is not None:
        print(f"[{model.upper()}] Tốt nhất tại {r[epoch_idx]} ({r[mode_idx]})")
        print(f"    - Avg Return/Trade: {data['max_val']:.5f} %")
        print(f"    - Position Changes: {r[pos_idx]}")
        print(f"    - Total Return:     {r[ret_idx]}")
        print("-" * 60)
