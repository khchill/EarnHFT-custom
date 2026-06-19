import sys
import pandas as pd
import numpy as np
sys.path.append('src/EarnHFT_framwork')
from env.high_level_env import High_Level_Env

df = pd.read_feather('data/cleaned_data/BTCUSDT/tardis/train/df_2022-01-01_0.feather').bfill().ffill().fillna(0.0)
min_f = np.load('data/cleaned_data/BTCUSDT/tardis/minitue_feature.npy', allow_pickle=True).tolist()
sec_f = np.load('data/cleaned_data/BTCUSDT/tardis/second_feature.npy', allow_pickle=True).tolist()

env = High_Level_Env(df, min_f, sec_f, model_dir='result_risk/BTCUSDT/potential_model')
env.reset()
print("High-Level step start...")
state, reward, done, info = env.step(2)
print("Reward after 60 seconds:", reward)
print("Position after 60 seconds:", env.position)
