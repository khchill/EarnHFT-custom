import sys
import os
import random
import argparse
import torch
import numpy as np
import pandas as pd
import glob
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

sys.path.append(".")
sys.path.append("src")
sys.path.append("src/EarnHFT_framwork")
sys.path.append("./src/EarnHFT_framwork")
from env.low_level_env import Testing_env, Training_Env

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--result_path", type=str, default="result_risk")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--action_dim", type=int, default=5)
parser.add_argument("--reward_scale", type=float, default=30.0) 
parser.add_argument("--back_time_length", type=int, default=1)
parser.add_argument("--dataset_name", type=str, default="BTCUSDT")
parser.add_argument("--train_data_path", type=str, default="data/cleaned_data/BTCUSDT/tardis/train")
parser.add_argument("--transcation_cost", type=float, default=0.00015)
parser.add_argument("--max_holding_number", type=float, default=0.01)

parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--mini_batch_size", type=int, default=64)
parser.add_argument("--hidden_nodes", type=int, default=128)
parser.add_argument("--lr_init", type=float, default=5e-5)
parser.add_argument("--gamma", type=float, default=0.99) 
parser.add_argument("--lamda", type=float, default=0.95)
parser.add_argument("--epsilon", type=float, default=0.2) 
parser.add_argument("--K_epochs", type=int, default=1)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--num_sample", type=int, default=200) 

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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_nodes):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_nodes):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class PPO(object):
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        seed_torch(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
    def train(self):
        print(f"bắt đầu huấn luyện ppo trên {self.args.dataset_name}")
        
        feature_path = "data/cleaned_data/BTCUSDT/tardis/feature_list.npy"
        if not os.path.exists(feature_path):
            print(f"loi ko thay file feature o {feature_path}")
            return
            
        chunk_files = glob.glob(os.path.join(self.args.train_data_path, "df_*.feather"))
        if not chunk_files:
            print("loi ko thay data chunk nao ca")
            return
            
        tech_indicators = np.load(feature_path, allow_pickle=True).tolist()
        state_dim = len(tech_indicators) + 1 # +1 cho holding position
        
        actor = Actor(state_dim, self.args.action_dim, self.args.hidden_nodes).to(self.device)
        critic = Critic(state_dim, self.args.hidden_nodes).to(self.device)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=self.args.lr_init)
        
        epochs = self.args.epochs
        root_dir = f"{self.args.result_path}/BTCUSDT/ppo/seed_{self.seed}"
        os.makedirs(root_dir, exist_ok=True)
        
        for epoch in range(epochs):
            total_reward_epoch = 0
            loss_val = 0.0
            
            sampled_chunk_file = random.choice(chunk_files)
            for chunk_file in [sampled_chunk_file]:
                print(f"PPO Baseline đang nạp chunk: {os.path.basename(chunk_file)}")
                df = pd.read_feather(chunk_file).bfill().ffill().fillna(0.0)
                env = Training_Env(df, tech_indicators, transcation_cost=self.args.transcation_cost, max_holding_number=self.args.max_holding_number)
                
                state, _ = env.reset()
                done = False
                
                states, actions, logprobs, rewards, values = [], [], [], [], []
                
                # Rollout
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action_logits = actor(state_tensor)
                        dist = Categorical(logits=action_logits)
                        action = dist.sample()
                        value = critic(state_tensor)
                    
                    next_state, reward, done, _ = env.step(action.item())
                    
                    scaled_reward = reward * self.args.reward_scale
                    
                    states.append(state)
                    actions.append(action.item())
                    logprobs.append(dist.log_prob(action).item())
                    rewards.append(scaled_reward) 
                    values.append(value.item())
                    
                    state = next_state
                    total_reward_epoch += reward 
                
                # calculate GAE
                advantages = []
                gae = 0
                for i in reversed(range(len(rewards))):
                    next_val = 0 if i == len(rewards) - 1 else values[i + 1]
                    delta = rewards[i] + self.args.gamma * next_val - values[i]
                    gae = delta + self.args.gamma * self.args.lamda * gae
                    advantages.insert(0, gae)
                
                advantages_tensor = torch.FloatTensor(advantages).to(self.device)
                returns_tensor = advantages_tensor + torch.FloatTensor(values).to(self.device)
                
                # normalize advantage
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
                
                states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
                actions_tensor = torch.LongTensor(actions).to(self.device)
                old_logprobs = torch.FloatTensor(logprobs).to(self.device)
                
                # PPO update
                dataset_size = len(states)
                for _ in range(self.args.K_epochs):
                    for index in BatchSampler(SubsetRandomSampler(range(dataset_size)), self.args.mini_batch_size, False):
                        b_states = states_tensor[index]
                        b_actions = actions_tensor[index]
                        b_old_logprobs = old_logprobs[index]
                        b_advantages = advantages_tensor[index]
                        b_returns = returns_tensor[index]
                        
                        action_logits = actor(b_states)
                        dist = Categorical(logits=action_logits)
                        new_logprobs = dist.log_prob(b_actions)
                        entropy = dist.entropy().mean()
                        
                        ratios = torch.exp(new_logprobs - b_old_logprobs)
                        surr1 = ratios * b_advantages
                        surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * b_advantages
                        
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = F.mse_loss(critic(b_states).squeeze(), b_returns)
                        
                        loss = actor_loss + 0.5 * critic_loss - self.args.entropy_coef * entropy
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_norm=0.5)
                        optimizer.step()
                        loss_val = loss.item()
                    
            print(f"train ppo epoch {epoch+1}/{epochs} | r: {total_reward_epoch:.4f} | loss: {loss_val:.4f}")

            ep_dir = os.path.join(root_dir, f"epoch_{epoch+1}")
            os.makedirs(ep_dir, exist_ok=True)
            
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict()
            }, os.path.join(ep_dir, "model.pth"))
            
            
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
                            action_logits = actor(valid_state_tensor)
                            action = torch.argmax(action_logits, dim=-1).item()
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
                    
                    test_state, _ = test_env.reset()
                    test_done = False
                    daily_rewards, daily_actions = [], []
                    
                    while not test_done:
                        test_state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            action_logits = actor(test_state_tensor)
                            action = torch.argmax(action_logits, dim=-1).item()
                        next_test_state, t_reward, test_done, _ = test_env.step(action)
                        daily_rewards.append(t_reward)
                        daily_actions.append(action)
                        test_state = next_test_state
                        
                    all_rewards.extend(daily_rewards)
                    all_actions.extend(daily_actions)
                    all_require_money.append(test_env.require_money)
                    
                t_dir = os.path.join(ep_dir, "test")
                os.makedirs(t_dir, exist_ok=True)
                np.save(os.path.join(t_dir, "reward.npy"), np.array(all_rewards))
                np.save(os.path.join(t_dir, "action.npy"), np.array(all_actions))
                np.save(os.path.join(t_dir, "require_money.npy"), np.array(all_require_money))
                
        print("chay xong ppo")

if __name__ == "__main__":
    args = parser.parse_args()
    agent = PPO(args)
    agent.train()
