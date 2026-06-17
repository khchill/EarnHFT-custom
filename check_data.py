import pandas as pd
df = pd.read_feather("data/cleaned_data/BTCUSDT/tardis/train/df_2022-01-01_1.feather")
print("Shape:", df.shape)
if 'timestamp' in df.columns:
    print(df['timestamp'].head(5))
elif 'date' in df.columns:
    print(df['date'].head(5))
