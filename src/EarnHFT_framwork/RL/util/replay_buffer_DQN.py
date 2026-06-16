import sys
sys.path.append(".")
import torch
import numpy as np
from collections import deque, namedtuple
import random
from RL.util.sum_tree import SumTree

class Multi_step_Prioritized_ReplayBuffer_multi_info(object):
    def __init__(self, alpha, beta, batch_size, n_steps, state_dim, action_dim, buffer_capacity=100000, gamma=1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.n_steps = n_steps
        self.n_steps_deque = deque(maxlen=self.n_steps)
        # ma trận lưu trữ kinh nghiệm của AI
        self.buffer = {
            "market_state": np.zeros((self.buffer_capacity, state_dim)),
            "previous_action": np.zeros((self.buffer_capacity)),
            "avaliable_action": np.zeros((self.buffer_capacity, action_dim)),
            "q_action": np.zeros((self.buffer_capacity, action_dim)),
            "action": np.zeros((self.buffer_capacity, 1)),
            "reward": np.zeros(self.buffer_capacity),
            "next_market_state": np.zeros((self.buffer_capacity, state_dim)),
            "next_previous_action": np.zeros((self.buffer_capacity)),
            "next_avaliable_action": np.zeros((self.buffer_capacity, action_dim)),
            "next_q_action": np.zeros((self.buffer_capacity, action_dim)),
            "terminal": np.zeros(self.buffer_capacity),
        }
        self.sum_tree = SumTree(self.buffer_capacity)
        self.current_size = 0
        self.count = 0

    def add(self, market_state, previous_action, avaliable_action, q_action, action, reward, next_market_state, next_previous_action, next_avaliable_action, next_q_action, terminal):
        """thêm transition mới vào bộ nhớ đệm và cập nhật độ ưu tiên trên sumtree"""
        # lấy độ ưu tiên lớn nhất hiện tại để đảm bảo transition mới luôn được bốc ít nhất một lần
        max_p = np.max(self.sum_tree.tree[-self.sum_tree.capacity:])
        if max_p == 0:
            max_p = 1.0
            
        idx = self.count % self.buffer_capacity
        
        self.buffer['market_state'][idx] = market_state
        self.buffer['previous_action'][idx] = previous_action
        self.buffer['avaliable_action'][idx] = avaliable_action
        self.buffer['q_action'][idx] = q_action
        self.buffer['action'][idx] = action
        self.buffer['reward'][idx] = reward
        self.buffer['next_market_state'][idx] = next_market_state
        self.buffer['next_previous_action'][idx] = next_previous_action
        self.buffer['next_avaliable_action'][idx] = next_avaliable_action
        self.buffer['next_q_action'][idx] = next_q_action
        self.buffer['terminal'][idx] = terminal
        
        self.sum_tree.add(max_p, idx)
        
        self.count += 1
        self.current_size = min(self.count, self.buffer_capacity)

    def sample(self):
        """hàm lấy mẫu kinh nghiệm từ sumtree dựa trên phân chia đoạn (segments)"""
        # khởi tạo các mảng chứa dữ liệu mẫu được bốc ra
        batch_market_state = np.zeros((self.batch_size, self.buffer['market_state'].shape[1]))
        batch_action = np.zeros((self.batch_size, 1))
        batch_reward = np.zeros(self.batch_size)
        batch_next_market_state = np.zeros((self.batch_size, self.buffer['next_market_state'].shape[1]))
        batch_terminal = np.zeros(self.batch_size)
        batch_q_action = np.zeros((self.batch_size, self.buffer['q_action'].shape[1]))
        
        # mảng chứa chỉ số của lá trên cây và trọng số tầm quan trọng (importance sampling weights)
        batch_tree_idx = np.zeros(self.batch_size, dtype=np.int32)
        IS_weights = np.zeros((self.batch_size, 1))
        
        # băm tổng độ ưu tiên của cây thành các đoạn bằng nhau
        segment = self.sum_tree.total_p / self.batch_size
        
        # tìm xác suất tối thiểu trong cây để tính trọng số max_weight giúp ổn định huấn luyện
        p_min = np.min(self.sum_tree.tree[-self.sum_tree.capacity:]) / self.sum_tree.total_p
        if p_min == 0:
            p_min = 1e-5 # tránh chia cho 0
        max_weight = (p_min * self.current_size) ** (-self.beta)

        # quét qua từng đoạn để bốc mẫu ngẫu nhiên
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # lấy ngẫu nhiên 1 giá trị v nằm trong đoạn [a, b]
            v = np.random.uniform(a, b)
            
            # dùng get_leaf tìm chiếc lá tương ứng
            tree_idx, priority, data_idx = self.sum_tree.get_leaf(v)
            
            batch_tree_idx[i] = tree_idx
            
            # trích xuất dữ liệu từ buffer
            batch_market_state[i] = self.buffer['market_state'][data_idx]
            batch_action[i] = self.buffer['action'][data_idx]
            batch_reward[i] = self.buffer['reward'][data_idx]
            batch_next_market_state[i] = self.buffer['next_market_state'][data_idx]
            batch_terminal[i] = self.buffer['terminal'][data_idx]
            batch_q_action[i] = self.buffer['q_action'][data_idx]
            
            # tính trọng số importance sampling
            prob = priority / self.sum_tree.total_p
            IS_weights[i, 0] = np.power(prob * self.current_size, -self.beta) / max_weight
            
        return (batch_tree_idx, batch_market_state, batch_action, batch_reward, 
                batch_next_market_state, batch_terminal, IS_weights, batch_q_action)

    def update_priorities(self, tree_indices, abs_td_errors):
        """hàm cập nhật lại độ ưu tiên trên sumtree sau khi học xong mẫu đó"""
        # cộng thêm epsilon để tránh độ ưu tiên bằng 0 khiến mẫu không bao giờ được bốc lại
        priorities = np.power(abs_td_errors + 1e-5, self.alpha)
        for tree_idx, p in zip(tree_indices, priorities):
            self.sum_tree.update(tree_idx, p)

class Multi_step_ReplayBuffer_multi_info(object):
    def __init__(self, batch_size, state_dim, action_dim, buffer_capacity=1000000, gamma=1):
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.buffer = {
            "market_state": np.zeros((self.buffer_capacity, state_dim)),
            "previous_action": np.zeros((self.buffer_capacity)),
            "avaliable_action": np.zeros((self.buffer_capacity, action_dim)),
            "q_action": np.zeros((self.buffer_capacity, action_dim)),
            "action": np.zeros((self.buffer_capacity, 1)),
            "reward": np.zeros(self.buffer_capacity),
            "next_market_state": np.zeros((self.buffer_capacity, state_dim)),
            "next_previous_action": np.zeros((self.buffer_capacity)),
            "next_avaliable_action": np.zeros((self.buffer_capacity, action_dim)),
            "next_q_action": np.zeros((self.buffer_capacity, action_dim)),
            "terminal": np.zeros(self.buffer_capacity),
        }
        self.current_size = 0
        self.count = 0

    def add(self, market_state, previous_action, avaliable_action, q_action, action, reward, next_market_state, next_previous_action, next_avaliable_action, next_q_action, terminal):
        idx = self.count % self.buffer_capacity
        
        self.buffer['market_state'][idx] = market_state
        self.buffer['previous_action'][idx] = previous_action
        self.buffer['avaliable_action'][idx] = avaliable_action
        self.buffer['q_action'][idx] = q_action
        self.buffer['action'][idx] = action
        self.buffer['reward'][idx] = reward
        self.buffer['next_market_state'][idx] = next_market_state
        self.buffer['next_previous_action'][idx] = next_previous_action
        self.buffer['next_avaliable_action'][idx] = next_avaliable_action
        self.buffer['next_q_action'][idx] = next_q_action
        self.buffer['terminal'][idx] = terminal
        
        self.count += 1
        self.current_size = min(self.count, self.buffer_capacity)

    def sample(self):
        batch_indices = np.random.choice(self.current_size, self.batch_size, replace=False)
        
        batch_market_state = self.buffer['market_state'][batch_indices]
        batch_action = self.buffer['action'][batch_indices]
        batch_reward = self.buffer['reward'][batch_indices]
        batch_next_market_state = self.buffer['next_market_state'][batch_indices]
        batch_terminal = self.buffer['terminal'][batch_indices]
        
        # không dùng IS_weights trong uniform replay, nhưng trả về 1 để giữ nguyên interface
        IS_weights = np.ones((self.batch_size, 1))
        
        return (batch_indices, batch_market_state, batch_action, batch_reward, 
                batch_next_market_state, batch_terminal, IS_weights)

    def update_priorities(self, tree_indices, abs_td_errors):
        pass # không dùng trong uniform buffer
