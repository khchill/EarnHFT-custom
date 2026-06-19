import sys
import torch
import numpy as np
import pandas as pd
from collections import Counter
sys.path.append("src/EarnHFT_framwork")
from model.net import Qnet

# Load feature names
low_level_features = np.load("data/cleaned_data/BTCUSDT/tardis/second_feature.npy", allow_pickle=True).tolist()

# Load a chunk of real data
df = pd.read_feather("data/cleaned_data/BTCUSDT/tardis/train/df_2022-01-01_0.feather").bfill().ffill().fillna(0.0)
low_level_array = df[low_level_features].values

# Load model
net = Qnet(55, 5, 128)
net.load_state_dict(torch.load("result_risk/BTCUSDT/potential_model/initial_action_0/model_2.pth", map_location="cpu"))
net.eval()

actions = []
pos = 0.0
positions_pool = [0.0, 0.0025, 0.005, 0.0075, 0.01]

for i in range(1000):
    feats = low_level_array[i].tolist()
    state = feats + [pos]
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = net(state_tensor)
        action = torch.argmax(q_values).item()
    actions.append(action)
    pos = positions_pool[action]

print("Action counts:", Counter(actions))
