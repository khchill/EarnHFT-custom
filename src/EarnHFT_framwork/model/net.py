import torch
import torch.nn as nn

# low-level agent
class Qnet(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Qnet, self).__init__()
        # fc1 process features
        self.fc1 = nn.Linear(N_STATES - 1, hidden_nodes)
        
        # fc3 process position
        self.fc3 = nn.Linear(1, N_ACTIONS)
        
        # fc2 combine position and output from fc1
        self.fc2 = nn.Linear(N_ACTIONS + hidden_nodes, hidden_nodes)
        
        # out 
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        
        # action masking
        self.register_buffer("max_punish", torch.tensor(1e12))

    def forward(self, state, available_actions=None):
        if state.dim() == 1:
            features = state[:-1]
            position = state[-1:]
        else:
            features = state[:, :-1]
            position = state[:, -1:]
            
        # run parallel 
        x1 = torch.relu(self.fc1(features))
        x3 = torch.relu(self.fc3(position))
        
        # combine
        x = torch.cat([x3, x1], dim=-1)
        
        # 
        x = torch.relu(self.fc2(x))
        q_values = self.out(x)
        
        # action mask
        if available_actions is not None:
            q_values = torch.where(available_actions == 0, -self.max_punish, q_values)
            
        return q_values


# high level router without position   
class Qnet_high_level(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Qnet_high_level, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.out(x)


# high level router with position
class Qnet_high_level_position(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Qnet_high_level_position, self).__init__()
        # Tương tự như Low-level, bớt 1 feature cho fc1
        self.fc1 = nn.Linear(N_STATES - 1, hidden_nodes)
        self.fc3 = nn.Linear(1, N_ACTIONS)
        self.fc2 = nn.Linear(N_ACTIONS + hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)

    def forward(self, state):
        if state.dim() == 1:
            features = state[:-1]
            position = state[-1:]
        else:
            features = state[:, :-1]
            position = state[:, -1:]
            
        # run parallel 
        x1 = torch.relu(self.fc1(features))
        x3 = torch.relu(self.fc3(position))
        
        # combine
        x = torch.cat([x3, x1], dim=-1)
        x = torch.relu(self.fc2(x))
        
        # output q-value action
        return self.out(x)
