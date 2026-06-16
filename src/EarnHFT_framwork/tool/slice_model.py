import sys
from pathlib import Path

# Cấu hình đường dẫn import nội bộ
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import pandas as pd
import argparse
import label_util as util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/cleaned_data/BTCUSDT/tardis/valid", help="Đường dẫn thư mục chứa file dữ liệu valid")
    parser.add_argument("--key_indicator", type=str, default="bid1_price", help="Cột giá đại diện để dán nhãn")
    parser.add_argument("--timestamp", type=str, default="index", help="Cột thời gian")
    parser.add_argument("--tic", type=str, default="tic", help="Cột mã coin")
    parser.add_argument("--labeling_method", type=str, default="slope", help="Phương pháp dán nhãn")
    parser.add_argument("--min_length_limit", type=int, default=60, help="Độ dài tối thiểu của phân đoạn")
    parser.add_argument("--merging_metric", type=str, default="DTW_distance", help="Phương pháp tính khoảng cách gộp")
    parser.add_argument("--merging_threshold", type=float, default=0.0003, help="Ngưỡng khoảng cách để gộp phân đoạn")
    parser.add_argument("--merging_dynamic_constraint", type=int, default=1, help="Ràng buộc động lượng kề nhau")
    parser.add_argument("--filter_strength", type=int, default=1, help="Độ mạnh của bộ lọc nhiễu Butterworth")
    parser.add_argument("--dynamic_number", type=int, default=5, help="Số lượng trạng thái thị trường")
    parser.add_argument("--max_length_expectation", type=int, default=3600, help="Độ dài kỳ vọng tối đa")
    return parser.parse_args()

class LinearMarketDynamicsModel:
    def __init__(self, args):
        self.data_path = args.data_path
        self.method = "slice_and_merge"
        self.filter_strength = args.filter_strength
        self.dynamic_number = args.dynamic_number
        self.max_length_expectation = args.max_length_expectation
        self.key_indicator = args.key_indicator
        self.timestamp = args.timestamp
        self.tic = args.tic
        self.labeling_method = args.labeling_method
        self.min_length_limit = args.min_length_limit
        self.merging_metric = args.merging_metric
        self.merging_threshold = args.merging_threshold
        self.merging_dynamic_constraint = args.merging_dynamic_constraint

    def run(self):
        print(f"Đọc dữ liệu từ thư mục: {self.data_path}...")
        data_dir = Path(self.data_path)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Thư mục {self.data_path} không tồn tại hoặc không phải là thư mục")
            
        dfs = []
        for file in sorted(data_dir.glob("*.feather")):
            df = pd.read_feather(file)
            dfs.append(df)
            print(f"[*] Đã nạp file {file.name} ({len(df)} dòng)")
            
        if not dfs:
            raise ValueError(f"Không tìm thấy file .feather nào trong {self.data_path}")
        
        # gọi worker để thực hiện toàn bộ thuật toán lọc và gộp DTW trên danh sách dfs
        util.Worker(
            data=dfs,
            data_path=str(data_dir.parent / "valid_multi"), # Set path directly to output dir so Worker knows where to save
            method=self.method,
            filter_strength=self.filter_strength,
            key_indicator=self.key_indicator,
            timestamp=self.timestamp,
            tic=self.tic,
            labeling_method=self.labeling_method,
            min_length_limit=self.min_length_limit,
            merging_threshold=self.merging_threshold,
            merging_metric=self.merging_metric,
            merging_dynamic_constraint=self.merging_dynamic_constraint
        )

def main():
    args = parse_args()
    model = LinearMarketDynamicsModel(args)
    model.run()

if __name__ == "__main__":
    main()
