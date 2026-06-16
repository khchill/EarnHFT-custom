import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Đường dẫn đến file df.feather")
    parser.add_argument("--save_path", type=str, default="data/feature", help="Thư mục lưu 2 file npy")
    parser.add_argument("--threshold", type=float, default=0.01, help="Ngưỡng lọc IC")
    return parser.parse_args()

def analysis_ic_longterm(df: pd.DataFrame, period=1, threshold=0.01):
    # ic chỉ sài idicator
    raw_features = ["timestamp", "symbol", "midpoint", "buy_vwap", "sell_vwap", "return"]
    for i in range(1, 11):
        raw_features.extend([f"bid{i}_size", f"ask{i}_size", f"bid{i}_price", f"ask{i}_price"])
        raw_features.extend([f"bid{i}_size_n", f"ask{i}_size_n"])
        
    for suffix in ["m", "s"]:
        raw_features.extend([f"open_{suffix}", f"high_{suffix}", f"low_{suffix}", f"close_{suffix}"])

    feature = df.columns.tolist()
    feature = list(set(feature).difference(set(raw_features)))
    
    # lợi nhuận tương lai
    df["return"] = (df["bid1_price"].shift(-period) - df["bid1_price"]).fillna(0)
    
    cor = dict()
    print(f"Phân tích IC cho {len(feature)} đặc trưng...")
    for f in feature:
        correlation = df["return"].corr(df[f])
        cor[f] = np.abs(correlation) if not np.isnan(correlation) else 0.0
        
    # sort ic giảm dần
    feature_train = sorted(cor.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 5 features có IC cao nhất:")
    for rank, (feat, val) in enumerate(feature_train[:5], 1):
        print(f"  {rank}. {feat}: {val:.6f}")
        
    # Luu thong tin vao Dataframe de co the xuat file
    ic_df = pd.DataFrame(feature_train, columns=["Feature", "IC_Value"])

    feature_train = dict(feature_train)
    
    preserve_features = []
    for key, val in feature_train.items():
        if val >= threshold:
            preserve_features.append(key)
    # nếu ko có nào vuotj qua thresshold -> sắp xếp bình thường mặc định
    if len(preserve_features) == 0:
        print(" Không có đặc trưng nào vượt qua threshold! Lấy danh sách sắp xếp mặc định.")
        preserve_features = list(feature_train.keys())
            
    return feature_train, preserve_features, ic_df

def main():
    args = parse_args()
    df_path = Path(args.data_path)
    
    if not df_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {df_path}")
        
    print(f"Đọc dữ liệu từ: {df_path}...")
    df = pd.read_feather(df_path)
    
    # save file theo cấu trúc tiền/ngày
    if df_path.parent.name and df_path.parent.parent.name and df_path.parent.parent.name not in ["data", "merge"]:
        out_dir = Path(args.save_path) / df_path.parent.parent.name / df_path.parent.name
    else:
        out_dir = Path(args.save_path)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Phân tích IC cho giây
    print("\n--- Phân tích nến giây ---")
    _, second_features, second_ic_df = analysis_ic_longterm(df, period=1, threshold=args.threshold)
    
    # Phân tích IC cho phút
    print("\n--- Phân tích nến phút ---")
    _, minute_features, minute_ic_df = analysis_ic_longterm(df, period=60, threshold=args.threshold)
    
    # Lấy top 54 giây và top 19 phút
    second_features = second_features[:54]
    minute_features = minute_features[:19]
    
    np.save(out_dir / "second_feature.npy", second_features)
    np.save(out_dir / "minitue_feature.npy", minute_features)
    
    # Luu bao cao IC ra CSV
    second_ic_df.to_csv(out_dir / "second_ic_report.csv", index=False)
    minute_ic_df.to_csv(out_dir / "minute_ic_report.csv", index=False)
    
    print(f"\nĐã lưu bộ features tại: {out_dir}")
    print(f"Số lượng features giây: {len(second_features)}")
    print(f"Số lượng features phút: {len(minute_features)}")

if __name__ == "__main__":
    main()
