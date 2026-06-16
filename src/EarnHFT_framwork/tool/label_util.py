import math
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pathlib import Path

class Dynamic_labeler:
    def __init__(self, labeling_method, dynamic_num, normalized_coef_list, data, turning_points, risk_bond=0.1):
        self.labeling_method = labeling_method
        self.dynamic_num = dynamic_num
        if self.labeling_method == "slope":
            sorted_normalized_coef_list = sorted(normalized_coef_list)
            low = sorted_normalized_coef_list[int(len(normalized_coef_list) * risk_bond / 2) + 2]
            high = sorted_normalized_coef_list[int(len(normalized_coef_list) * (1 - risk_bond / 2))]
            self.segments = []
            for i in range(1, self.dynamic_num):
                self.segments.append(low + (high - low) / (dynamic_num - 2) * (i - 1))

class Worker:
    def __init__(self, data, data_path=None, method="slice_and_merge", filter_strength=1,
                 key_indicator="bid1_price", timestamp="index", tic="tic",
                 labeling_method="slope", min_length_limit=60, merging_threshold=0.00005,
                 merging_metric="DTW_distance", merging_dynamic_constraint=1):
        self.data = data if isinstance(data, list) else [data]
        self.data_path = data_path
        self.key_indicator = key_indicator
        self.min_length_limit = min_length_limit
        self.merging_threshold = merging_threshold 
        self.dynamic_number = 5
        self.run_dtw_segmentation()

    def _process_single_df(self, df):
        print("Lọc tín hiệu nhiễu bằng bộ lọc Butterworth...")
        b, a = butter(4, 0.1, btype='low', analog=False)
        filtered_price = filtfilt(b, a, df[self.key_indicator].values)
        df['filtered'] = filtered_price

        print("Xác định các điểm cực trị trên đồ thị...")
        diff = np.diff(filtered_price)
        turning_points = [0]
        for i in range(1, len(diff)):
            if diff[i-1] * diff[i] < 0:
                turning_points.append(i)
        turning_points.append(len(df) - 1)

        # tạo danh sách phân đoạn ban đầu
        S = []
        for i in range(len(turning_points)-1):
            start = turning_points[i]
            end = turning_points[i+1]
            if end - start >= self.min_length_limit:
                chunk = df.iloc[start:end].reset_index(drop=True)
                r = (chunk[self.key_indicator].iloc[-1] - chunk[self.key_indicator].iloc[0]) / chunk[self.key_indicator].iloc[0]
                S.append({'chunk': chunk, 'slope': r, 'start': start, 'end': end})

        print(f"Số lượng điểm cực trị tìm được: {len(turning_points)}")
        print(f"Số lượng phân đoạn ban đầu (độ dài >= {self.min_length_limit}): {len(S)}")

        if not S:
            print("Không tìm thấy phân đoạn hợp lệ nào.")
            return []

        print(f"Gộp các đoạn tương đồng sử dụng DTW (threshold: {self.merging_threshold})...")
        stable = False
        while not stable:
            stable = True
            new_S = []
            skip = False
            for i in range(len(S) - 1):
                if skip:
                    skip = False
                    continue
                s1, s2 = S[i], S[i+1]
                
                d, _ = fastdtw(s1['chunk'][self.key_indicator].values, s2['chunk'][self.key_indicator].values, dist=lambda x, y: abs(x - y))
                slope_diff = abs(s1['slope'] - s2['slope'])
                dtw_limit = self.merging_threshold * len(s1['chunk'])
                
                if slope_diff < self.merging_threshold and d < dtw_limit:
                    merged_chunk = df.iloc[s1['start']:s2['end']].reset_index(drop=True)
                    merged_r = (merged_chunk[self.key_indicator].iloc[-1] - merged_chunk[self.key_indicator].iloc[0]) / merged_chunk[self.key_indicator].iloc[0]
                    new_S.append({'chunk': merged_chunk, 'slope': merged_r, 'start': s1['start'], 'end': s2['end']})
                    skip = True
                    stable = False
                else:
                    new_S.append(s1)
            if not skip and len(S) > 0:
                new_S.append(S[-1])
            S = new_S
            
        return S

    def run_dtw_segmentation(self):
        all_S = []
        for i, df in enumerate(self.data):
            print(f"\n--- Xử lý file {i+1}/{len(self.data)} ---")
            S = self._process_single_df(df)
            all_S.extend(S)
            
        if not all_S:
            print("Không có phân đoạn nào được tìm thấy trên toàn bộ dữ liệu.")
            return
            
        S = all_S

        print(f"Phân loại các trạng thái thị trường dựa theo quantile (M={self.dynamic_number})...")
        slopes = [s['slope'] for s in S]
        theta = 0.1
        H = np.nanquantile(slopes, 1 - theta/2)
        L = np.nanquantile(slopes, theta/2)

        # lưu kết quả vào thư mục valid_multi
        out_dir = Path(self.data_path) if self.data_path else Path("data/cleaned_data/BTCUSDT/tardis/valid_multi")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.dynamic_number):
            (out_dir / f"label_{i}").mkdir(parents=True, exist_ok=True)

        for idx, s in enumerate(S):
            r = s['slope']
            label = 0
            if r > H: 
                label = 4
            elif r < L: 
                label = 0
            else:
                step = (H - L) / (self.dynamic_number - 2)
                for j in range(2, self.dynamic_number):
                    if L + (j-2)*step < r <= L + (j-1)*step:
                        label = j - 1
            s['chunk'].to_feather(out_dir / f"label_{label}" / f"df_{idx}.feather")
            
        print(f"Đã hoàn tất thuật toán DTW. Gộp thành {len(S)} phân đoạn và lưu vào: {out_dir}")
