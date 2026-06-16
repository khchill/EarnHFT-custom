import torch
import numpy as np
import copy

class ReplayBuffer:
    def __init__(self, batch_size,state_dim,action_dim,device):
        self.s = np.zeros((batch_size, state_dim))
        self.p_a= np.zeros([batch_size,  1])
        self.a_a=np.zeros([batch_size,  action_dim])
        self.a = np.zeros((batch_size, 1))
        self.a_logprob = np.zeros((batch_size, 1))
        self.r = np.zeros((batch_size, 1))
        self.s_ = np.zeros((batch_size, state_dim))
        self.p_a_= np.zeros([batch_size,  1])
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0
        self.device=device

class ReplayBuffer_lstm:
    def __init__(self, gamma,lamda,state_dim,action_dim,episode_limit,batch_size):
        self.gamma = gamma
        self.lamda = lamda
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_limit = episode_limit
        self.batch_size = batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = []
