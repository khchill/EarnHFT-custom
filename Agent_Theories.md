# Phân Tích Chuyên Sâu Các Mô Hình Giao Dịch (HFT RL & Rule-based)

Tài liệu này trình bày lý thuyết toán học (Latex), kiến trúc mạng Neural (Neural Network Architecture), cơ chế lấy mẫu (Sampling) và ánh xạ trực tiếp đến từng dòng code của 4 loại Agents được sử dụng trong EarnHFT Framework: **DRA (Recurrent PPO)**, **CDQN**, **PPO (MLP)**, và **Rule-Based**.

---

## 1. Mạng DRA - Recurrent PPO (`dra_train.py`)

Mặc dù file được đặt tên là `dra_train.py`, nhưng cốt lõi thuật toán bên trong được thiết kế theo **Recurrent PPO (Proximal Policy Optimization kết hợp LSTM)**. Sự kết hợp này giúp Agent nhớ được bối cảnh chuỗi thời gian (time-series context) của thị trường.

### 1.1 Lý thuyết Toán học (PPO + GAE)
Hàm mục tiêu cốt lõi của PPO là **Clipped Surrogate Objective**. Nó nhằm mục đích tối đa hóa hàm lợi thế (Advantage) nhưng ép (clip) không cho Policy mới thay đổi quá đột ngột so với Policy cũ để đảm bảo tính hội tụ ổn định:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]
$$

Trong đó $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ là tỷ lệ xác suất hành động.

Để giảm phương sai (Variance) trong phần thưởng, GAE (Generalized Advantage Estimation) được sử dụng để ước lượng hàm lợi thế $\hat{A}_t$:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

Tổng Loss Function của mô hình bao gồm:

$$
L^{TOTAL} = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)
$$

Với $L^{VF}$ là Value Function Loss (MSE Loss của Critic) và $S$ là Entropy (khuyến khích khám phá).

### 1.2 Kiến trúc Neural Network (ActorCriticLSTM)
Kiến trúc đặc biệt ở chỗ nó **tách riêng** các chỉ báo kỹ thuật (Market Features) và trạng thái tài khoản (Position).
- **LSTM Layer**: Chỉ xử lý 54 features thị trường. Việc này giúp mô hình không bị "nhiễu" bởi hành động mua bán của chính nó trong quá khứ khi trích xuất đặc trưng giá.
- **Concat Layer**: Đầu ra của LSTM sau đó mới được ghép (concatenate) với biến Position hiện tại.

**Ánh xạ dòng code (`dra_train.py`):**
```python
61: self.market_dim = state_dim - 1 # Bóc tách 54 features thị trường
65: self.lstm = nn.LSTM(self.market_dim, hidden_nodes, batch_first=True) # Xử lý chuỗi
73: market_features = x[:, :, :-1]  
74: position_feature = x[:, :, -1:] 
77: lstm_out, hidden = self.lstm(market_features, hidden) 
80: combined = torch.cat((lstm_out, position_feature), dim=-1) # Nối vị thế hiện tại vào
83: action_logits = self.actor(combined) # Đầu ra Actor (5 hành động)
84: value = self.critic(combined) # Đầu ra Critic (dự đoán Value)
```

### 1.3 Cơ chế Sampling & BPTT (Backpropagation Through Time)
Khác với PPO thuần túy, do LSTM yêu cầu dữ liệu phải có tính liên tục theo thời gian, agent này **không xáo trộn (shuffle) toàn bộ memory** mà áp dụng **Truncated BPTT**.
Nó bốc từng đoạn dữ liệu tuyến tính (mini-batch tuyến tính), đồng thời truyền trạng thái ẩn (Hidden States) $h_0, c_0$ nối tiếp nhau.

**Ánh xạ dòng code (`dra_train.py`):**
```python
174: N = len(states)
175: num_chunks = N // self.args.mini_batch_size # Chia data thành các đoạn nối tiếp
183: b_states = torch.FloatTensor(np.array(states[start:end])).unsqueeze(0)
# Lưu ý dòng 190: Phải cấp đúng Hidden State của thời điểm bắt đầu chunk đó
190: h0 = h_states[start].to(self.device)
191: c0 = c_states[start].to(self.device)
202: ratios = torch.exp(new_logprobs - b_old_logprobs) # Tính r_t(θ)
204: surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * b_advantages # Clip Loss
```

---

## 2. Mạng CDQN - Convolutional DQN (`cdqn_train.py`)

CDQN sử dụng một **Mạng Tích Chập 1 chiều (1D-CNN)** kết hợp thuật toán Value-based (DQN) với cơ chế Exploration Epsilon-Greedy.

### 2.1 Lý thuyết Toán học (DQN với Target Network)
DQN học cách xấp xỉ hàm Q-Value bằng cách cực tiểu hóa độ lệch (Temporal Difference Error) so với phương trình Bellman:

$$
L_i(\theta_i) = \mathbb{E}_{s,a,r,s'} \left[ \left( y_i - Q(s,a;\theta_i) \right)^2 \right]
$$

Trong đó, $y_i$ là giá trị mục tiêu (Target) tính bởi Target Network $\theta^-$:

$$
y_i = r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$

### 2.2 Kiến trúc Neural Network (CQnet)
Tương tự LSTM, CQnet cũng tách Market Features ra đưa vào CNN để tìm các "Pattern cục bộ" (Local Patterns) trong dữ liệu giá, sau đó mới gộp Position vào ở lớp Fully Connected.
- **Conv1d + MaxPool1d**: Lọc các tín hiệu nhiễu, tìm xu hướng dốc của đường giá.
- **AdaptiveMaxPool1d**: Ép kích thước đầu ra cố định là 10, giúp mạng linh hoạt với các độ dài chuỗi khác nhau.

**Ánh xạ dòng code (`cdqn_train.py`):**
```python
62: self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
68: self.pool2 = nn.AdaptiveMaxPool1d(output_size=10) # Tạo ma trận cố định 10x32 = 320 nodes
85: x = self.pool1(self.relu1(self.conv1(features))) # Trích xuất đặc trưng
90: x = torch.cat([x, position], dim=-1) # Nối vị thế vào sau CNN
93: x = self.fc2(self.relu3(self.fc1(x))) # Xuất ra Q-values cho 5 Actions
```

### 2.3 Cơ chế Buffer & Sampling
Sử dụng bộ nhớ Replay Buffer tiêu chuẩn. Ở mỗi step, Agent thả $(s, a, r, s')$ vào Buffer. Khi train, nó lấy mẫu **ngẫu nhiên hoàn toàn (Uniform Random Sampling)** vì CNN không cần trạng thái nối tiếp như LSTM.

**Ánh xạ dòng code (`cdqn_train.py`):**
```python
157: if random.random() < epsilon: # Epsilon-Greedy Exploration
158:     action = random.randint(0, self.args.action_dim - 1)
188: batch_indices, b_states, b_actions, b_rewards, b_next_states, b_dones, _ = replay_buffer.sample() # Lấy mẫu ngẫu nhiên
196: q_eval = self.eval_net(b_states).gather(1, b_actions) # Lấy Q-value của hành động thực tế
198: q_next = self.target_net(b_next_states).max(1)[0].unsqueeze(1) # Lấy max Q của Target Net
199: q_target = b_rewards + self.args.gamma * q_next * (1 - b_dones) # Phương trình Bellman
```

---

## 3. Mạng PPO Cơ bản - MLP (`ppo_train.py`)

Đây là phiên bản PPO đơn giản nhất, sử dụng mạng Feed-Forward Fully Connected (MLP) không có kiến trúc ghi nhớ (No LSTM, No CNN). 

### 3.1 Lý thuyết và Kiến trúc
Toán học y hệt như mô hình số 1 (DRA). Tuy nhiên, kiến trúc mạng là một hàm biến đổi tuyến tính đơn thuần nhiều lớp (Multi-Layer Perceptron).
Mạng Actor và Critic hoàn toàn tách biệt, không share trọng số (No Shared Weights). Cả Market Features và Position bị gộp chung làm 1 vector duy nhất ném vào mạng.

**Ánh xạ dòng code (`ppo_train.py`):**
```python
58: class Actor(nn.Module):
61:     self.fc1 = nn.Linear(state_dim, hidden_nodes) # Nhận toàn bộ 55 chiều cùng lúc
65:     def forward(self, x):
66:         x = torch.relu(self.fc1(x))
68:         return self.out(x) # Trả ra action_logits
```

### 3.2 Cơ chế Sampling Độc Lập
Vì không bị ràng buộc bởi LSTM, PPO MLP có thể xáo trộn (shuffle) toàn bộ chuỗi Trajectory thu thập được trong 1 Episode bằng thuật toán `SubsetRandomSampler`. Việc xáo trộn này giúp phá vỡ sự tương quan tuyến tính (autocorrelation) của dữ liệu tài chính, giúp Gradient Descent đi đúng hướng hơn.

**Ánh xạ dòng code (`ppo_train.py`):**
```python
172: for index in BatchSampler(SubsetRandomSampler(range(dataset_size)), self.args.mini_batch_size, False):
# Xáo trộn hoàn toàn các state trong lịch sử thành các Mini-batch
173:     b_states = states_tensor[index]
184:     ratios = torch.exp(new_logprobs - b_old_logprobs)
```

---

## 4. Bot Giao Dịch Thủ Công - Rule-Based (`rule_based_tune.py`)

Không sử dụng học máy, Bot này hoàn toàn dựa vào bộ quy tắc "NẾU - THÌ" cứng (Hard-coded Logic). Nó đóng vai trò làm Baseline tối thiểu để đo đạc sự hiệu quả của các mô hình AI.

### 4.1 Thuật toán MACD (Moving Average Convergence Divergence)
Sử dụng Trung bình trượt hàm mũ (EMA).

$$
EMA_t = Value_t \times \left(\frac{2}{Span + 1}\right) + EMA_{t-1} \times \left(1 - \frac{2}{Span + 1}\right)
$$

$$
DIF = EMA_{short} - EMA_{long}
$$

$$
DEA = EMA_{mid}(DIF)
$$

$$
MACD = DIF - DEA
$$

- **Buy Signal**: Khi MACD cắt lên đường 0.
- **Sell Signal**: Khi MACD cắt xuống đường 0.

### 4.2 Thuật toán Imbalance Volume (IV)
Dựa vào chỉ số mất cân bằng Lệnh mua/bán trên Orderbook.
- Lớn hơn `upper_theshold` (0.99): Tín hiệu Bò (Buy).
- Nhỏ hơn `lower_theshold` (-0.99): Tín hiệu Gấu (Sell).

### 4.3 Quản trị Rủi Ro (Risk Management)
Sử dụng thuật toán **Stop-Loss / Take-Profit** tĩnh và cơ chế **Cooldown** (chống dồn lệnh).

**Ánh xạ dòng code (`rule_based_tune.py`):**
```python
85: # Tính toán toán học MACD bằng pandas EWM
85: dif_series = test_df['midpoint'].ewm(span=self.args.short_term, adjust=False).mean() - \
             test_df['midpoint'].ewm(span=self.args.long_term, adjust=False).mean()
87: dea_series = dif_series.ewm(span=self.args.mid_term, adjust=False).mean()
88: macd_series = dif_series - dea_series

105: # Tín hiệu IV
105: signal_buy = (current_val > self.args.upper_theshold)
106: signal_sell = (current_val < self.args.lower_theshold)

108: # Khớp lệnh logic
108: if signal_sell:
109:     action = 0 # Thanh lý toàn bộ hàng
110:     cooldown = False # Cho phép mua lại
111: elif signal_buy and not cooldown:
112:     action = 4 # Mua Full-Margin (All in)

119: # Cắt lỗ / Chốt lời tự động
119: roi = (current_price - entry_price) / entry_price
120: if roi >= self.args.stop_win or roi <= -self.args.stop_lose:
121:     action = 0 # Ép bán
122:     cooldown = True # Cấm mua tiếp cho đến khi có tín hiệu Sell để reset
```

---
*Báo cáo được tổng hợp tự động từ kiến trúc mã nguồn dự án EarnHFT Framework.*
