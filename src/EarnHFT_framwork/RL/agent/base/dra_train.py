import sys
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
import random
import argparse
import torch
import numpy as np
import pandas as pd
import glob
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append("./src/EarnHFT_framwork")
from env.low_level_env import Training_Env, Testing_env

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--result_path", type=str, default="result_risk")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--mini_batch_size", type=int, default=64)
parser.add_argument("--hidden_nodes", type=int, default=128)
parser.add_argument("--lr_init", type=float, default=5e-5)
parser.add_argument("--lr_min", type=float, default=1e-5)
parser.add_argument("--lr_step", type=float, default=1e6)
parser.add_argument("--gamma", type=float, default=1) 
parser.add_argument("--lamda", type=float, default=0.95)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--K_epochs", type=int, default=1) 
parser.add_argument("--num_sample", type=int, default=200)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--action_dim", type=int, default=5)
parser.add_argument("--back_time_length", type=int, default=1)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--reward_scale", type=float, default=30)
parser.add_argument("--dataset_name", type=str, default="BTCUSDT")
parser.add_argument("--train_data_path", type=str, default="data/cleaned_data/BTCUSDT/tardis/train")
parser.add_argument("--transcation_cost", type=float, default=0.00015)
parser.add_argument("--max_holding_number", type=float, default=0.01)

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

class ActorCriticLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_nodes):
        super(ActorCriticLSTM, self).__init__()
        # state_dim ban đầu bao gồm 54 market features + 1 biến position
        self.market_dim = state_dim - 1 
        
        # LSTM chỉ xử lý 54 features ko gồm position
        self.lstm = nn.LSTM(self.market_dim, hidden_nodes, batch_first=True)
        
        # Linear layer nhận đầu ra từ LSTM (hidden_nodes) cộng thêm 1 biến Position
        self.actor = nn.Linear(hidden_nodes + 1, action_dim)
        self.critic = nn.Linear(hidden_nodes + 1, 1)
        
    def forward(self, x, hidden=None):
        # tách 54 features và position
        market_features = x[:, :, :-1]  
        position_feature = x[:, :, -1:] 
        
        # đưa 54 features qua LSTM
        lstm_out, hidden = self.lstm(market_features, hidden) 
        
        # nối 54 features qua LSTM với position
        combined = torch.cat((lstm_out, position_feature), dim=-1) 
        
        # action và value
        action_logits = self.actor(combined) 
        value = self.critic(combined)        
        
        return action_logits, value, hidden

class PPO_discrete_RNN:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        seed_torch(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
    def train(self):
        print(f"khoi dong dra tren data {self.args.dataset_name}")
        
        feature_path = "data/cleaned_data/BTCUSDT/tardis/feature_list.npy"
        if not os.path.exists(feature_path):
            print(f"loi ko thay file feature {feature_path}")
            return
            
        chunk_files = glob.glob(os.path.join(self.args.train_data_path, "df_*.feather"))
        if not chunk_files:
            print("loi thu muc train rong")
            return
            
        tech_indicators = np.load(feature_path, allow_pickle=True).tolist()
        state_dim = len(tech_indicators) + 1 # +1 cho vị thế (position)
        
        model = ActorCriticLSTM(state_dim, self.args.action_dim, self.args.hidden_nodes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr_init)
        
        epochs = self.args.epochs
        root_dir = os.path.join(self.args.result_path, self.args.dataset_name, "dra_short", f"seed_{self.seed}")
        os.makedirs(root_dir, exist_ok=True)
        
        for epoch in range(epochs):
            sampled_chunk_file = random.choice(chunk_files)
            for chunk_file in [sampled_chunk_file]:
                print(f"DRA Baseline đang nạp chunk: {os.path.basename(chunk_file)}")
                df = pd.read_feather(chunk_file).bfill().ffill().fillna(0.0)
                env = Training_Env(df, tech_indicators, transcation_cost=self.args.transcation_cost, max_holding_number=self.args.max_holding_number)
                
                state, _ = env.reset()
                done = False
                total_reward = 0
                
                states, actions, logprobs, rewards, values = [], [], [], [], []
                h_states, c_states = [], [] 
                
                h_eval = torch.zeros(1, 1, self.args.hidden_nodes).to(self.device)
                c_eval = torch.zeros(1, 1, self.args.hidden_nodes).to(self.device)
                
                # rollout 
                while not done:
                    # lưu hidden state ở định dạng (1, 1, hidden_nodes)
                    h_states.append(h_eval.detach().cpu())
                    c_states.append(c_eval.detach().cpu())
                    
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        action_logits, value, (h_eval, c_eval) = model(state_tensor, (h_eval, c_eval))
                        action_probs = F.softmax(action_logits.squeeze(0), dim=-1) 
                        dist = Categorical(action_probs)
                        action = dist.sample()
                    
                    next_state, reward, done, _ = env.step(action.item())
                    
                    states.append(state)
                    actions.append(action.item())
                    logprobs.append(dist.log_prob(action).item())
                    rewards.append(reward * self.args.reward_scale) 
                    values.append(value.squeeze().item())
                    
                    state = next_state
                    total_reward += reward
                    
                #advantage
                advantages = []
                gae = 0
                for i in reversed(range(len(rewards))):
                    next_val = 0 if i == len(rewards) - 1 else values[i + 1]
                    delta = rewards[i] + self.args.gamma * next_val - values[i]
                    gae = delta + self.args.gamma * self.args.lamda * gae
                    advantages.insert(0, gae)
                
                advantages_tensor = torch.FloatTensor(advantages).to(self.device)
                returns_tensor = advantages_tensor + torch.FloatTensor(values).to(self.device)
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
                
                # ppo update (truncated bptt)
                N = len(states)
                num_chunks = N // self.args.mini_batch_size
                loss_val = 0.0
                
                for _ in range(self.args.K_epochs):
                    for i in range(num_chunks):
                        start = i * self.args.mini_batch_size
                        end = start + self.args.mini_batch_size
                        
                        b_states = torch.FloatTensor(np.array(states[start:end])).unsqueeze(0).to(self.device) 
                        b_actions = torch.LongTensor(actions[start:end]).to(self.device)
                        b_old_logprobs = torch.FloatTensor(logprobs[start:end]).to(self.device)
                        b_returns = returns_tensor[start:end]
                        b_advantages = advantages_tensor[start:end]
                        
                        #hidden state
                        h0 = h_states[start].to(self.device)
                        c0 = c_states[start].to(self.device)
                        
                        action_logits, val, _ = model(b_states, (h0, c0))
                        
                        action_logits = action_logits.squeeze(0) 
                        val = val.squeeze(0).squeeze(-1)         
                        
                        dist = Categorical(F.softmax(action_logits, dim=-1))
                        new_logprobs = dist.log_prob(b_actions)
                        entropy = dist.entropy().mean()
                        
                        ratios = torch.exp(new_logprobs - b_old_logprobs)
                        surr1 = ratios * b_advantages
                        surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * b_advantages
                        
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = F.mse_loss(val, b_returns)
                        loss = actor_loss + 0.5 * critic_loss - self.args.entropy_coef * entropy
                        
                        optimizer.zero_grad()
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
                        optimizer.step()
                        loss_val = loss.item()
                        
            print(f"tập train thứ {epoch+1}/{epochs} | r: {total_reward:.4f} | loss: {loss_val:.4f}")

            # test
            ep_dir = os.path.join(root_dir, f"epoch_{epoch+1}")
            os.makedirs(ep_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ep_dir, "model.pth"))
            
            
            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                print(f"[*] Epoch {epoch+1}: Đang chạy Valid và Test...")
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
                        
                        h_eval_v = torch.zeros(1, 1, self.args.hidden_nodes).to(self.device)
                        c_eval_v = torch.zeros(1, 1, self.args.hidden_nodes).to(self.device)
                        
                        while not valid_done:
                            valid_state_tensor = torch.FloatTensor(valid_state).unsqueeze(0).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                action_logits, _, (h_eval_v, c_eval_v) = model(valid_state_tensor, (h_eval_v, c_eval_v))
                                action = torch.argmax(action_logits.squeeze(0).squeeze(0), dim=-1).item()
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
                        
                        h_eval = torch.zeros(1, 1, self.args.hidden_nodes).to(self.device)
                        c_eval = torch.zeros(1, 1, self.args.hidden_nodes).to(self.device)
                        
                        while not done:
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                action_logits, _, (h_eval, c_eval) = model(state_tensor, (h_eval, c_eval))
                                action = torch.argmax(action_logits.squeeze(0).squeeze(0), dim=-1).item()
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
            else:
                print(f"[*] Epoch {epoch+1} hoàn tất Train (Bỏ qua Valid/Test để tiết kiệm thời gian)")
                
        print("xong dra")

if __name__ == "__main__":
    args = parser.parse_args()
    agent = PPO_discrete_RNN(args)
    agent.train()
