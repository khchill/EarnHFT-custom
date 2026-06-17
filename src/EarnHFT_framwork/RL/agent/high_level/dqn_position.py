import sys
import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import glob

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append("./src/EarnHFT_framwork")

from model.net import Qnet_high_level_position
from env.high_level_env import High_Level_Env
from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--hidden_nodes", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr_init", type=float, default=1e-3)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--reward_scale", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--dataset_name", type=str, default="BTCUSDT")

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class RouterDQN:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        seed_torch(args.seed)
        
        self.tau = args.tau
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        
        feature_path = "data/cleaned_data/BTCUSDT/tardis/feature_list.npy"
        if not os.path.exists(feature_path):
            print("loi ko thay feature_list.npy nha!")
            sys.exit(1)
            
        self.tech_indicators = np.load(feature_path, allow_pickle=True).tolist()
        self.state_dim = len(self.tech_indicators) + 1 
        
        # action = 5 (tương ứng với 5 Low-level bots trong Pool)
        self.action_dim = 5 
        
        self.eval_net = Qnet_high_level_position(self.state_dim, self.action_dim, args.hidden_nodes).to(self.device)
        self.target_net = Qnet_high_level_position(self.state_dim, self.action_dim, args.hidden_nodes).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr_init)
        self.loss_func = nn.MSELoss() 
        
    def train(self):
        print(f"bat dau train router tren data {self.args.dataset_name}...")
        
        train_path = "data/cleaned_data/BTCUSDT/tardis/train"
        chunk_files = glob.glob(os.path.join(train_path, "df_*.feather"))
        if not chunk_files:
            print(f"loi ko tim thay data chunk nao o {train_path}")
            return
            
        replay_buffer = Multi_step_ReplayBuffer_multi_info(
            batch_size=self.batch_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            buffer_capacity=100000,
            gamma=self.gamma
        )
        
        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_decay = (1.0 - epsilon_min) / (20 * len(chunk_files) * 50)
        epochs = self.args.epochs
        
        root_dir = f"result_risk/{self.args.dataset_name}/high_level/seed_{self.args.seed}"
        os.makedirs(root_dir, exist_ok=True)
        
        for epoch in range(epochs):
            random.shuffle(chunk_files)
            total_loss_epoch = 0
            
            for chunk_file in chunk_files:
                print(f"High-level Router đang nạp chunk: {os.path.basename(chunk_file)}")
                df_train = pd.read_feather(chunk_file).bfill().ffill().fillna(0.0)
                env = High_Level_Env(df_train, tech_indicator_list=self.tech_indicators)
                
                state, _ = env.reset()
                done = False
                
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    if random.random() < epsilon:
                        action_idx = random.randint(0, self.action_dim - 1)
                    else:
                        with torch.no_grad():
                            action_idx = torch.argmax(self.eval_net(state_tensor)).item() 
                    
                    next_state, reward, done, _ = env.step(action_idx)
                    
                    scaled_reward = reward * self.args.reward_scale
                    
                    replay_buffer.add(
                        market_state=state, previous_action=0, avaliable_action=np.zeros(self.action_dim), 
                        q_action=np.zeros(self.action_dim), action=action_idx, reward=scaled_reward, 
                        next_market_state=next_state, next_previous_action=0, 
                        next_avaliable_action=np.zeros(self.action_dim), next_q_action=np.zeros(self.action_dim), terminal=done
                    )
                    
                    epsilon = max(epsilon_min, epsilon - epsilon_decay)
                    
                    if replay_buffer.current_size > self.batch_size:
                        batch_indices, b_states, b_actions, b_rewards, b_next_states, b_dones, _ = replay_buffer.sample()
                        
                        b_states = torch.FloatTensor(b_states).to(self.device)
                        b_actions = torch.LongTensor(b_actions).view(-1, 1).to(self.device) 
                        b_rewards = torch.FloatTensor(b_rewards).to(self.device)
                        b_next_states = torch.FloatTensor(b_next_states).to(self.device)
                        b_dones = torch.FloatTensor(b_dones).to(self.device)
                        
                        #update net
                        with torch.no_grad():
                            best_next_actions = self.eval_net(b_next_states).argmax(1).unsqueeze(1)
                            q_next_target = self.target_net(b_next_states).gather(1, best_next_actions).squeeze(1)
                            q_target = b_rewards + self.gamma * q_next_target * (1 - b_dones)
                        
                        q_eval = self.eval_net(b_states).gather(1, b_actions).squeeze(1)
                        
                        loss = self.loss_func(q_eval, q_target)
                        self.optimizer.zero_grad()
                        loss.backward()
                        # clip gradient 
                        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        
                        total_loss_epoch += loss.item()
                        
                        # soft update 
                        for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                            target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)
                    
                    state = next_state
   
                        
            print(f"train router epoch {epoch+1}/{epochs} | Epsilon: {epsilon:.4f} | Loss: {total_loss_epoch:.4f}")
            
            ep_dir = os.path.join(root_dir, f"epoch_{epoch+1}")
            os.makedirs(ep_dir, exist_ok=True)
            torch.save(self.eval_net.state_dict(), os.path.join(ep_dir, "model.pth"))
            
            # validation
            valid_dir = "data/cleaned_data/BTCUSDT/tardis/valid"
            if os.path.exists(valid_dir):
                valid_files = sorted([os.path.join(valid_dir, f) for f in os.listdir(valid_dir) if f.endswith(".feather")])
                
                all_valid_rewards, all_valid_require_money, all_valid_actions = [], [], []
                for valid_file in valid_files:
                    df_valid = pd.read_feather(valid_file).bfill().ffill().fillna(0.0)
                    valid_env = High_Level_Env(df_valid, tech_indicator_list=self.tech_indicators)
                    state, _ = valid_env.reset()
                    done = False
                    
                    while not done:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            action_idx = torch.argmax(self.eval_net(state_tensor)).item()
                        positions_pool = [0.0, 0.0025, 0.005, 0.0075, 0.01]
                        n_idx = int(np.argmin(np.abs(np.array(positions_pool) - state[-1])))
                        next_state, reward, done, _ = valid_env.step(action_idx)
                        state = next_state
                        all_valid_actions.append([n_idx, action_idx])
                        
                    all_valid_rewards.extend(valid_env.second_rewards_history)
                    all_valid_require_money.append(valid_env.require_money)
                    
                v_dir = os.path.join(ep_dir, "valid")
                os.makedirs(v_dir, exist_ok=True)
                np.save(os.path.join(v_dir, "final_balance.npy"), np.array(np.sum(all_valid_rewards)))
                np.save(os.path.join(v_dir, "reward.npy"), np.array(all_valid_rewards))
                np.save(os.path.join(v_dir, "require_money.npy"), np.array(all_valid_require_money))
                np.save(os.path.join(v_dir, "action.npy"), np.array(all_valid_actions))
            # end validation
            # testing
            test_dir = "data/cleaned_data/BTCUSDT/tardis/test"
            if os.path.exists(test_dir):
                test_files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".feather")])
                
                all_rewards, all_require_money, all_actions = [], [], []
                for test_file in test_files:
                    df_test = pd.read_feather(test_file).bfill().ffill().fillna(0.0)
                    test_env = High_Level_Env(df_test, tech_indicator_list=self.tech_indicators)
                    state, _ = test_env.reset()
                    done = False
                    
                    while not done:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            action_idx = torch.argmax(self.eval_net(state_tensor)).item()
                        positions_pool = [0.0, 0.0025, 0.005, 0.0075, 0.01]
                        n_idx = int(np.argmin(np.abs(np.array(positions_pool) - state[-1])))
                        next_state, reward, done, _ = test_env.step(action_idx)
                        state = next_state
                        all_actions.append([n_idx, action_idx])
                        
                    all_rewards.extend(test_env.second_rewards_history) # lưu reward theo từng giây
                    all_require_money.append(test_env.require_money)
                    
                t_dir = os.path.join(ep_dir, "test")
                os.makedirs(t_dir, exist_ok=True)
                np.save(os.path.join(t_dir, "reward.npy"), np.array(all_rewards))
                np.save(os.path.join(t_dir, "require_money.npy"), np.array(all_require_money))
                np.save(os.path.join(t_dir, "action.npy"), np.array(all_actions))
            # end test
                
        print("chay xong dqn_position")

if __name__ == "__main__":
    args = parser.parse_args()
    router = RouterDQN(args)
    router.train()
