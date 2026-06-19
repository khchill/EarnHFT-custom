import sys
import torch
sys.path.append("src/EarnHFT_framwork")
from model.net import Qnet
import numpy as np

net = Qnet(55, 5, 128)
net.load_state_dict(torch.load("result_risk/BTCUSDT/potential_model/initial_action_0/model_2.pth", map_location="cpu"))
net.eval()

# Test with zeros
state = torch.zeros((1, 55))
with torch.no_grad():
    q_values = net(state)
    action = torch.argmax(q_values).item()
print("Q-values for zeros state:", q_values)
print("Action selected:", action)

# Test with ones
state = torch.ones((1, 55))
with torch.no_grad():
    q_values = net(state)
    action = torch.argmax(q_values).item()
print("Q-values for ones state:", q_values)
print("Action selected:", action)

