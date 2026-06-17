import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import argparse

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# config
EXCHANGE = "binance"
SYMBOL = "BTCUSDT"
#path
BASE_DIR = Path(".")
RAW_DIR = BASE_DIR / "data" / "tardis_raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned_data" / SYMBOL / "tardis"
TRAIN_DIR = CLEANED_DIR / "train"
VALID_DIR = CLEANED_DIR / "valid"
TEST_DIR = CLEANED_DIR / "test"

for d in [RAW_DIR, CLEANED_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# download data
def download_tardis_data(date_str, year, month, day):
    print(f"\n[1/4]  BẮT ĐẦU TẢI DỮ LIỆU TỪ TARDIS NGÀY {date_str}")
    downloaded_files = {}
    data_types = ["book_snapshot_25", "trades"]
    
    for data_type in data_types:
        url = f"https://datasets.tardis.dev/v1/{EXCHANGE}/{data_type}/{year}/{month}/{day}/{SYMBOL}.csv.gz"
        outfile = RAW_DIR / f"{EXCHANGE}_{data_type}_{date_str}_{SYMBOL}.csv.gz"
        downloaded_files[data_type] = outfile
        
        if outfile.exists():
            print(f"File {outfile.name} đã tồn tại. Bỏ qua tải.")
            continue
            
        try:
            print(f"Đang tải {data_type}...", end=" ", flush=True)
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code != 200:
                print(f"Error: (HTTP {r.status_code})")
                continue
                
            with open(outfile, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print("ok")
            time.sleep(1) # avoid rate limit
        except Exception as e:
            print(f"Error: {e}")
            
    return downloaded_files


# merge
def process_and_merge(downloaded_files, date_str):
    print(f"\n[2/4] process depth 10 for {date_str}")
    
    ob_file = downloaded_files.get("book_snapshot_25")
    trade_file = downloaded_files.get("trades")
    
    if not ob_file or not trade_file or not ob_file.exists() or not trade_file.exists():
        print(f"Thiếu file raw (Orderbook hoặc Trades) để merge cho ngày {date_str}.")
        return None

    # order book process
    print("Orderbook processing...")
    try:
        df_ob = pd.read_csv(ob_file)
    except Exception as e:
        print(f"\n[LỖI] File {ob_file.name} bị hỏng hoặc tải chưa xong (Lỗi: {e}).")
        print(f"[*] Đang tự động xóa file lỗi... Bạn hãy chạy lại script này lần nữa để tải lại nhé.")
        ob_file.unlink(missing_ok=True)
        return None

    df_ob['timestamp'] = pd.to_datetime(df_ob['timestamp'], unit='us')
    
    rename_map = {}
    for i in range(10): 
        rename_map[f'asks[{i}].price'] = f'ask{i+1}_price'
        rename_map[f'asks[{i}].amount'] = f'ask{i+1}_size'
        rename_map[f'bids[{i}].price'] = f'bid{i+1}_price'
        rename_map[f'bids[{i}].amount'] = f'bid{i+1}_size'
        
    df_ob = df_ob.rename(columns=rename_map)
    df_ob = df_ob[['timestamp'] + list(rename_map.values())]
    df_ob = df_ob.set_index('timestamp').resample('1s').last().ffill()
    
    # trade process
    print("Trade processing...")
    try:
        df_tr = pd.read_csv(trade_file)
    except Exception as e:
        print(f"\n[LỖI] File {trade_file.name} bị hỏng hoặc tải chưa xong (Lỗi: {e}).")
        print(f"[*] Đang tự động xóa file lỗi... Bạn hãy chạy lại script này lần nữa để tải lại nhé.")
        trade_file.unlink(missing_ok=True)
        return None

    df_tr['timestamp'] = pd.to_datetime(df_tr['timestamp'], unit='us')
    df_tr = df_tr.set_index('timestamp')
    
    trade_sec = df_tr.groupby(pd.Grouper(freq='1s')).agg(
        open_s=('price', 'first'), high_s=('price', 'max'),
        low_s=('price', 'min'), close_s=('price', 'last')
    ).ffill()
    # shift 1 sec 
    trade_sec.index = trade_sec.index + pd.Timedelta(seconds=1)
    
    trade_min = df_tr.groupby(pd.Grouper(freq='1min')).agg(
        open_m=('price', 'first'), high_m=('price', 'max'),
        low_m=('price', 'min'), close_m=('price', 'last')
    ).ffill()
    trade_min.index = trade_min.index + pd.Timedelta(minutes=1)
    
    trade_min_resampled = trade_min.resample('1s').ffill()
    
    # merge data 
    print("Merge data...")
    df_merged = pd.concat([df_ob, trade_sec, trade_min_resampled], axis=1, sort=False).ffill().bfill().reset_index()
    
    return df_merged

# Feature Engineering
def apply_time_series_operators(df, cols, windows, suffix=""):
    """apply time series operators: diff, trend, ma, max, min, quantiles"""
    for col in cols:
        for w in windows:
            name_suffix = f"_{w}{suffix}" if suffix else f"_{w}"
            
            # Diff & Trend
            df[f"{col}_diff{name_suffix}"] = df[col].diff(w)
            df[f"{col}_trend{name_suffix}"] = (df[col] - df[col].shift(w)) / w
            
            # Rolling statistics chỉ sài cho giá cls
            if col in ["close_s", "close_m"]:
                rolling_w = df[col].rolling(w)
                df[f"ma_{col}{name_suffix}"] = rolling_w.mean() 
                df[f"max_{col}{name_suffix}"] = rolling_w.max()
                df[f"min_{col}{name_suffix}"] = rolling_w.min()
                df[f"qtlu_{col}{name_suffix}"] = rolling_w.quantile(0.8)
                df[f"qtld_{col}{name_suffix}"] = rolling_w.quantile(0.2)
    return df

def create_features_order_book(df: pd.DataFrame, windows: list):
    """create order book features"""
    real_min = 1e-5
    df["midpoint"] = (df["ask1_price"] + df["bid1_price"]) / 2.0

    buy_cols = [f"bid{i}_size" for i in range(1, 11)]
    sell_cols = [f"ask{i}_size" for i in range(1, 11)]
    
    df["buy_volume_oe"] = df[buy_cols].sum(axis=1)
    df["sell_volume_oe"] = df[sell_cols].sum(axis=1)

    buy_vwap = np.zeros(len(df))
    sell_vwap = np.zeros(len(df))
    
    for i in range(1, 11):
        df[f"bid{i}_size_n"] = df[f"bid{i}_size"] / (df["buy_volume_oe"] + real_min)
        df[f"ask{i}_size_n"] = df[f"ask{i}_size"] / (df["sell_volume_oe"] + real_min)
        buy_vwap += df[f"bid{i}_size_n"] * df[f"bid{i}_price"]
        sell_vwap += df[f"ask{i}_size_n"] * df[f"ask{i}_price"]

    df["buy_vwap"] = buy_vwap
    df["sell_vwap"] = sell_vwap
    df["sell_buy_vwap_spread"] = df["buy_vwap"] - df["sell_vwap"]

    df["buy_spread_oe"] = np.abs(df["bid1_price"] - df["bid10_price"])
    df["sell_spread_oe"] = np.abs(df["ask1_price"] - df["ask10_price"])
    df["oe_spread_1"] = np.abs(df["bid1_price"] - df["ask1_price"])
    df["oe_spread_10"] = np.abs(df["bid10_price"] - df["ask10_price"])
    
    # Imbalance Volume
    df["imblance_volume_oe"] = (df["buy_volume_oe"] - df["sell_volume_oe"]) / (df["buy_volume_oe"] + df["sell_volume_oe"] + real_min)

    df["wap_1"] = (df["bid1_price"] * df["ask1_size"] + df["ask1_price"] * df["bid1_size"]) / (df["ask1_size"] + df["bid1_size"] + real_min)
    df["wap_2"] = (df["bid2_price"] * df["ask2_size"] + df["ask2_price"] * df["bid2_size"]) / (df["ask2_size"] + df["bid2_size"] + real_min)
    df["wap_balance"] = df["wap_1"] - df["wap_2"]

    ts_cols = ["midpoint", "buy_spread_oe", "sell_spread_oe", "imblance_volume_oe", "ask1_price", "bid1_price"]
    df = apply_time_series_operators(df, ts_cols, windows)
    
    df.drop(columns=["sell_volume_oe", "buy_volume_oe"], inplace=True)
    return df

def create_features_trade(df: pd.DataFrame, beat_fee: float, windows: list, suffix: str):
    """Tính toán hình thái nến, Dao động học (RSI, MACD, Bollinger), và Động lượng (ROC, Beta)"""
    EPS = df[f"close_{suffix}"] * beat_fee
    
    # K-line
    df[f"max_oc_{suffix}"] = df[[f"open_{suffix}", f"close_{suffix}"]].max(axis=1)
    df[f"min_oc_{suffix}"] = df[[f"open_{suffix}", f"close_{suffix}"]].min(axis=1)

    df[f"kmid_{suffix}"] = df[f"close_{suffix}"] - df[f"open_{suffix}"]
    df[f"klen_{suffix}"] = df[f"high_{suffix}"] - df[f"low_{suffix}"] + EPS
    df[f"kmid2_{suffix}"] = df[f"kmid_{suffix}"] / df[f"klen_{suffix}"]
    df[f"kup_{suffix}"] = df[f"high_{suffix}"] - df[f"max_oc_{suffix}"]
    df[f"kup2_{suffix}"] = df[f"kup_{suffix}"] / df[f"klen_{suffix}"]
    df[f"klow_{suffix}"] = df[f"min_oc_{suffix}"] - df[f"low_{suffix}"]
    df[f"klow2_{suffix}"] = df[f"klow_{suffix}"] / df[f"klen_{suffix}"]
    df[f"ksft_{suffix}"] = 2 * df[f"close_{suffix}"] - df[f"high_{suffix}"] - df[f"low_{suffix}"]
    df[f"ksft2_{suffix}"] = df[f"ksft_{suffix}"] / df[f"klen_{suffix}"]
    df.drop(columns=[f"max_oc_{suffix}", f"min_oc_{suffix}"], inplace=True)


    ema12 = df[f"close_{suffix}"].ewm(span=12, adjust=False).mean()
    ema26 = df[f"close_{suffix}"].ewm(span=26, adjust=False).mean()
    df[f"macd_{suffix}"] = ema12 - ema26
    df[f"macds_{suffix}"] = df[f"macd_{suffix}"].ewm(span=9, adjust=False).mean()
    df[f"macdh_{suffix}"] = df[f"macd_{suffix}"] - df[f"macds_{suffix}"]
    
    if suffix == "m":
        df["macd"] = df[f"macd_{suffix}"]
        df["macds"] = df[f"macds_{suffix}"]
        df["macdh"] = df[f"macdh_{suffix}"]

    # RSI (14)
    delta = df[f"close_{suffix}"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + EPS)
    df[f"rsi_{suffix}"] = 100 - (100 / (1 + rs))

    for w in windows:
        low_rolling = df[f"low_{suffix}"].rolling(w)
        high_rolling = df[f"high_{suffix}"].rolling(w)
        close_rolling = df[f"close_{suffix}"].rolling(w)
        close_shift = df[f"close_{suffix}"].shift(w)
        
        # RSV
        df[f"rsv_{w}_{suffix}"] = (df[f"close_{suffix}"] - low_rolling.min()) / ((high_rolling.max() - low_rolling.min()) + EPS)
        
        # Bollinger Bands
        ma = close_rolling.mean()
        std = close_rolling.std() + EPS
        df[f"bb_up_{w}_{suffix}"] = ma + 2 * std
        df[f"bb_down_{w}_{suffix}"] = ma - 2 * std
        
        # ROC & Beta
        df[f"roc_{w}_{suffix}"] = (close_shift - df[f"close_{suffix}"]) / std
        df[f"beta_{w}_{suffix}"] = (close_shift - df[f"close_{suffix}"]) / (w * std)
        
        # Relative Position of Min/Max 
        df[f"imax_{w}_{suffix}"] = high_rolling.apply(np.argmax, raw=True) / w
        df[f"imin_{w}_{suffix}"] = low_rolling.apply(np.argmin, raw=True) / w
        df[f"imxd_{w}_{suffix}"] = df[f"imax_{w}_{suffix}"] - df[f"imin_{w}_{suffix}"]

    df = apply_time_series_operators(df, [f"close_{suffix}"], windows, f"_{suffix}")
    return df

def run_feature_engineering(df_merged, date_str):
    print(f"\n[3/4] Start feature engineering for {date_str}")
    window_sizes = [5, 10, 30, 60]

    # Chunk data
    df_ob = df_merged[['timestamp'] + [col for col in df_merged.columns if 'price' in col or 'size' in col]]
    df_sec = df_merged[['timestamp', 'open_s', 'high_s', 'low_s', 'close_s']]
    df_min = df_merged[['timestamp', 'open_m', 'high_m', 'low_m', 'close_m']]

    print("Orderbook Features...")
    df_ob_feat = create_features_order_book(df_ob.copy(), window_sizes)

    print("Second K-lines Features...")
    df_sec_feat = create_features_trade(df_sec.copy(), 1e-5, window_sizes, "s")

    print("Minute K-lines Features")
    df_min_feat = create_features_trade(df_min.copy(), 1e-5, window_sizes, "m")


    print("Merging data and drop NaN...")
    df_ob_feat.set_index('timestamp', inplace=True)
    df_sec_feat.set_index('timestamp', inplace=True)
    df_min_feat.set_index('timestamp', inplace=True)

    df_final = pd.concat([df_ob_feat, df_sec_feat, df_min_feat], axis=1)
    len_before = len(df_final)
    
    df_final.dropna(inplace=True)
    df_final.reset_index(inplace=True)
    

    float_cols = df_final.select_dtypes(include=['float64']).columns
    df_final[float_cols] = df_final[float_cols].astype('float32')

    len_after = len(df_final)
    print(f"Drop {len_before - len_after} dòng chứa NaN")
    
    return df_final

def chunk_and_save_data(df, date_str, split_type):
    # Extract fea
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'bid1_price', 'ask1_price'] and not c.startswith('ask') and not c.startswith('bid')]
    
    if split_type == 'train':
        print(f"\n[4/4] Chunk train for {date_str}...")
        chunk_size = 18000 # 14400 steps + 3600 futureside
        step_size = 14400 # Episode length
        chunk_idx = 0
        for i in range(0, len(df) - chunk_size + 1, step_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk_path = TRAIN_DIR / f"df_{date_str}_{chunk_idx}.feather"
            chunk.to_feather(chunk_path)
            chunk_idx += 1
        print(f"Saved chunk to {TRAIN_DIR}")
    else:
        print(f"\n[4/4] Saving data for {split_type.upper()} for {date_str}...")
        if split_type == 'valid':
            out_path = VALID_DIR / f"df_{date_str}.feather"
            df.to_feather(out_path)
        else:
            out_path = TEST_DIR / f"df_{date_str}.feather"
            df.to_feather(out_path)
            
    return df, feature_cols

def generate_target_dates(test_run=False):
    """Tạo danh sách các ngày để tải theo kịch bản:
    Train: Tháng 1-10 các năm 2022-2025
    Valid: Tháng 11-12 các năm 2022-2025
    Test: Tháng 1-6 năm 2026 (Do hiện tại là 06/2026)
    """
    dates = []
    
    if test_run:
        return [
            ("2022-01-01", "2022", "01", "01", "train"),
            ("2022-11-01", "2022", "11", "01", "valid"),
            ("2026-01-01", "2026", "01", "01", "test")
        ]
        
    for year in range(2022, 2026):
        # Train: Tháng 1 đến 10
        for m in range(1, 11):
            month_str = f"{m:02d}"
            dates.append((f"{year}-{month_str}-01", str(year), month_str, "01", "train"))
        # Valid: Tháng 11 và 12
        for m in range(11, 13):
            month_str = f"{m:02d}"
            dates.append((f"{year}-{month_str}-01", str(year), month_str, "01", "valid"))
            
    # Test: Năm 2026 (lấy từ tháng 1 tới tháng 6)
    for m in range(1, 7):
        month_str = f"{m:02d}"
        dates.append((f"2026-{month_str}-01", "2026", month_str, "01", "test"))
        
    return dates

def main(test_run=False):
    print("="*60)
    print("START TARDIS DATA PIPELINE")
    if test_run:
        print("TEST RUN MODE (Chạy 2 ngày)")
    print("="*60)
    
    target_dates = generate_target_dates(test_run)
    
    global_feature_cols = []
    
    for date_str, y, m, d, split_type in target_dates:

        downloaded = download_tardis_data(date_str, y, m, d)

        df_merged = process_and_merge(downloaded, date_str)
        if df_merged is None:
            continue

        df_final = run_feature_engineering(df_merged, date_str)

        df_final, f_cols = chunk_and_save_data(df_final, date_str, split_type)
        if not global_feature_cols:
            global_feature_cols = f_cols
            
        print("-" * 40)


    # Lưu feature list đã được lọc IC
    if global_feature_cols:
        print("\n---CHỌN LỌC ĐẶC TRƯNG (IC ANALYSIS) ---")
        import glob
        from ic_analysis import analysis_ic_longterm
        import pandas as pd
        
        train_files = glob.glob(str(CLEANED_DIR / "train" / "df_*.feather"))
        if train_files:
            # Lấy 1 file train bất kỳ để đánh giá IC
            sample_df = pd.read_feather(train_files[0])
            feature_train, _, _ = analysis_ic_longterm(sample_df, period=1, threshold=0.01)
            
            # Lấy 54 features có IC cao nhất
            feature_list = sorted(feature_train.items(), key=lambda x: x[1], reverse=True)
            top_54_keys = [feat for feat, val in feature_list[:54]]
            
            np.save(CLEANED_DIR / "feature_list.npy", np.array(top_54_keys))
            print(f"Đã phân tích IC và lưu {len(top_54_keys)} tính năng tốt nhất tại {CLEANED_DIR / 'feature_list.npy'}")
        else:
            np.save(CLEANED_DIR / "feature_list.npy", np.array(global_feature_cols))
            print(f"Không tìm thấy file train. Đã lưu toàn bộ tính năng tại {CLEANED_DIR / 'feature_list.npy'}")


    print("xong pipepline lấy data từ tardis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Run only on a small subset of dates for testing")
    args = parser.parse_args()
    
    main(args.test_run)
