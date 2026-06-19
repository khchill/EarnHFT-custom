import sys
import torch
sys.path.append("src/EarnHFT_framwork")
from env.high_level_env import High_Level_Env
import numpy as np

features = np.load("data/cleaned_data/BTCUSDT/tardis/second_feature.npy", allow_pickle=True).tolist()
from model.net import Qnet
try:
    net = Qnet(len(features) + 1, 5, 128)
    net.load_state_dict(torch.load("result_risk/BTCUSDT/potential_model/initial_action_0/model_2.pth"))
    print("SUCCESS")
except Exception as e:
    print("FAIL:", e)
