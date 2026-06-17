import sys
import os
import random
import argparse
import numpy as np
import pandas as pd
import glob
import torch
from torch import nn
from tqdm import tqdm

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append("./src/EarnHFT_framwork")

from model.net import *
from env.low_level_env import Training_Env, Testing_env
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--buffer_size", type=int, default=1000000) 
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--action_dim", type=int, default=5)
parser.add_argument("--back_time_length", type=int, default=20)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--reward_scale", type=float, default=30.0)
parser.add_argument("--hidden_nodes", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--gamma", type=float, default=0.99) 
parser.add_argument("--epsilon_init", type=float, default=1.0)
parser.add_argument("--epsilon_min", type=float, default=0.01)
parser.add_argument("--epsilon_decay_rate", type=float, default=0.00001) 
parser.add_argument("--update_frequency", type=int, default=3600) 
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--train_data_path", type=str, default="data/cleaned_data/BTCUSDT/tardis/train")
parser.add_argument("--dataset_name", type=str, default="BTCUSDT")
parser.add_argument("--transcation_cost", type=float, default=0.00015)
parser.add_argument("--max_holding_number", type=float, default=0.01)
parser.add_argument("--result_path", type=str, default="result_risk")

def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class CQnet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_nodes):
        super(CQnet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveMaxPool1d(output_size=10) 
        
        # 320 features từ cnn + 1 feature position
        self.fc1 = nn.Linear(320 + 1, hidden_nodes)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_nodes, action_dim)
        
    def forward(self, state):
        # bóc tách features và position
        if state.dim() == 1:
            features = state[:-1].unsqueeze(0).unsqueeze(0)
            position = state[-1:].unsqueeze(0)
        else:
            features = state[:, :-1].unsqueeze(1)
            position = state[:, -1:]
            
        # chỉ đẩy features qua cnn
        x = self.pool1(self.relu1(self.conv1(features)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        
        # nối position vào sau cnn
        x = torch.cat([x, position], dim=-1)
        
        # đẩy qua layer fc
        x = self.fc2(self.relu3(self.fc1(x)))
        return x

class CDQN_rp(object):
    def __init__(self, args):
        self.seed = args.seed
        seed_torch(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self):
        self.args = globals().get('args', args)
        print(f"bat dau train cdqn rp tren data {self.args.dataset_name} nha...")
        
        train_path = self.args.train_data_path
        feature_path = "data/cleaned_data/BTCUSDT/tardis/feature_list.npy"
        if not os.path.exists(feature_path):
            print("loi ko thay file feature")
            return
            
        chunk_files = glob.glob(os.path.join(train_path, "df_*.feather"))
        if not chunk_files:
            print(f"loi ko tim thay data chunk nao o {train_path}")
            return
            
        tech_indicators = np.load(feature_path, allow_pickle=True).tolist()
        state_dim = len(tech_indicators) + 1 # +1 cho vị thế (position)
        
        self.eval_net = CQnet(state_dim, self.args.action_dim, self.args.hidden_nodes).to(self.device)
        self.target_net = CQnet(state_dim, self.args.action_dim, self.args.hidden_nodes).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()
        
        replay_buffer = Multi_step_ReplayBuffer_multi_info(
            batch_size=self.args.batch_size,
            state_dim=state_dim,
            action_dim=self.args.action_dim,
            buffer_capacity=self.args.buffer_size,
            gamma=self.args.gamma
        )
        
        base_freq = self.args.update_frequency
        current_target_freq = base_freq + random.randint(-base_freq // 4, base_freq // 4)
        steps_since_target_update = 0
        
        epochs = self.args.epochs
        epsilon = self.args.epsilon_init

        # load qua epoch randomly shuffle chunk file
        for epoch in range(epochs):
            total_reward_epoch = 0
            
            sampled_chunk_file = random.choice(chunk_files)
            for chunk_file in [sampled_chunk_file]:
                print(f"CDQN Baseline đang nạp chunk: {os.path.basename(chunk_file)}")
                df = pd.read_feather(chunk_file).bfill().ffill().fillna(0.0)
                env = Training_Env(df, tech_indicators, transcation_cost=self.args.transcation_cost, max_holding_number=self.args.max_holding_number)
                
                state, _ = env.reset()
                done = False
                
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    if random.random() < epsilon:
                        action = random.randint(0, self.args.action_dim - 1)
                    else:
                        with torch.no_grad():
                            action = torch.argmax(self.eval_net(state_tensor)).item()
                        
                    next_state, reward, done, _ = env.step(action)
                    
                    scaled_reward = reward * self.args.reward_scale
                    
                    replay_buffer.add(
                        market_state=state, 
                        previous_action=0, 
                        avaliable_action=np.zeros(self.args.action_dim), 
                        q_action=np.zeros(self.args.action_dim), 
                        action=action, 
                        reward=scaled_reward, 
                        next_market_state=next_state, 
                        next_previous_action=0, 
                        next_avaliable_action=np.zeros(self.args.action_dim), 
                        next_q_action=np.zeros(self.args.action_dim), 
                        terminal=done
                    )
                    
                    state = next_state
                    total_reward_epoch += reward
                    steps_since_target_update += 1
                    
                    epsilon = max(self.args.epsilon_min, epsilon - self.args.epsilon_decay_rate)
                    
                    if replay_buffer.current_size > self.args.batch_size:
                        batch_indices, b_states, b_actions, b_rewards, b_next_states, b_dones, _ = replay_buffer.sample()
                        
                        b_states = torch.FloatTensor(b_states).to(self.device)
                        b_actions = torch.LongTensor(b_actions).view(-1, 1).to(self.device) 
                        b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(self.device)
                        b_next_states = torch.FloatTensor(b_next_states).to(self.device)
                        b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(self.device)
                        
                        q_eval = self.eval_net(b_states).gather(1, b_actions)
                        with torch.no_grad():
                            q_next = self.target_net(b_next_states).max(1)[0].unsqueeze(1)
                            q_target = b_rewards + self.args.gamma * q_next * (1 - b_dones)
                            
                        loss = self.loss_func(q_eval, q_target)
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0) # Clip gradient
                        self.optimizer.step()
                    
                    if steps_since_target_update >= current_target_freq:
                        self.target_net.load_state_dict(self.eval_net.state_dict())
                        steps_since_target_update = 0 # Reset 
                        current_target_freq = base_freq + random.randint(-base_freq // 4, base_freq // 4)
                
            print(f"cdqn ep {epoch+1}/{epochs} | tan suat target: {current_target_freq}")
            
            # testing mỗi epoch
            root_dir = f"{self.args.result_path}/BTCUSDT/cdqn_rp/seed_{self.seed}"
            ep_dir = os.path.join(root_dir, f"epoch_{epoch+1}")
            os.makedirs(ep_dir, exist_ok=True)
            torch.save(self.eval_net.state_dict(), os.path.join(ep_dir, "model.pth"))
            
            
            # validate
            valid_dir_path = "data/cleaned_data/BTCUSDT/tardis/valid"
            if os.path.exists(valid_dir_path) and os.path.isdir(valid_dir_path):
                valid_files = sorted([os.path.join(valid_dir_path, f) for f in os.listdir(valid_dir_path) if f.endswith(".feather")])
                all_rewards, all_actions, all_require_money = [], [], []
                
                for valid_file_path in valid_files:
                    valid_df = pd.read_feather(valid_file_path).bfill().ffill().fillna(0.0)
                    valid_env = Testing_env(valid_df, tech_indicators, transcation_cost=self.args.transcation_cost, max_holding_number=self.args.max_holding_number, action_dim=self.args.action_dim)
                    
                    valid_state, _ = valid_env.reset()
                    valid_done = False
                    daily_actions = []
                    
                    while not valid_done:
                        valid_state_tensor = torch.FloatTensor(valid_state).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            action = torch.argmax(self.eval_net(valid_state_tensor), dim=-1).item()
                        next_valid_state, v_reward, valid_done, _ = valid_env.step(action)
                        all_rewards.append(v_reward)
                        daily_actions.append(action)
                        valid_state = next_valid_state
                        
                    all_actions.extend(daily_actions)
                    all_require_money.append(valid_env.require_money)
                    
                v_dir = os.path.join(ep_dir, "valid")
                os.makedirs(v_dir, exist_ok=True)
                np.save(os.path.join(v_dir, "final_balance.npy"), np.array(np.sum(all_rewards)))
                np.save(os.path.join(v_dir, "reward.npy"), np.array(all_rewards))
                np.save(os.path.join(v_dir, "action.npy"), np.array(all_actions))
                np.save(os.path.join(v_dir, "require_money.npy"), np.array(all_require_money))

            # testing
            test_dir_path = "data/cleaned_data/BTCUSDT/tardis/test"

            if os.path.exists(test_dir_path) and os.path.isdir(test_dir_path):
                test_files = sorted([os.path.join(test_dir_path, f) for f in os.listdir(test_dir_path) if f.endswith(".feather")])
                all_rewards, all_actions, all_require_money = [], [], []
                
                for test_file_path in test_files:
                    test_df = pd.read_feather(test_file_path).bfill().ffill().fillna(0.0)
                    test_env = Testing_env(test_df, tech_indicators, transcation_cost=self.args.transcation_cost, max_holding_number=self.args.max_holding_number, action_dim=self.args.action_dim)
                    
                    state, _ = test_env.reset()
                    done = False
                    daily_rewards, daily_actions = [], []
                    while not done:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            action = torch.argmax(self.eval_net(state_tensor)).item()
                        next_state, reward, done, _ = test_env.step(action)
                        daily_rewards.append(reward)
                        daily_actions.append(action)
                        state = next_state
                        
                    all_rewards.extend(daily_rewards)
                    all_actions.extend(daily_actions)
                    all_require_money.append(test_env.require_money)
                    
                t_dir = os.path.join(ep_dir, "test")
                os.makedirs(t_dir, exist_ok=True)
                np.save(os.path.join(t_dir, "reward.npy"), np.array(all_rewards))
                np.save(os.path.join(t_dir, "action.npy"), np.array(all_actions))
                np.save(os.path.join(t_dir, "require_money.npy"), np.array(all_require_money))
            
        print("chay xong cdqn")

if __name__ == "__main__":
    args = parser.parse_args()
    agent = CDQN_rp(args)
    agent.train()
