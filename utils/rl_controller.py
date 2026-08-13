"""
Lightweight DQN controller to steer GA evolution.

This replaces the earlier tabular Q-learning controller with a small DQN that can
work with continuous state vectors and a discrete action set. It uses a tiny
MLP and an experience replay buffer. The interface remains similar:

    controller = RLController(state_dim=4)
    action = controller.select_action(state_dict)
    params = controller.apply_action(base_params, action)
    controller.update(state, action, reward, next_state)

Persistence: controller.save(path) saves model + optimizer; controller.load(path) restores.
"""
import random
from collections import deque
import math
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 把每一步的 (s, a, r, s2, done) 先存起来
# 每次更新网络时，从里面随机采样一个 batch，打乱时间相关性，让训练更稳定、更像 i.i.d. 数据。
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done=False):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return np.array(s, dtype=np.float32), np.array(a, dtype=np.int64), np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32), np.array(done, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


class RLController:
    def __init__(self, state_dim=4, lr=1e-3, gamma=0.99, epsilon=0.2, buffer_capacity=5000, batch_size=64, device=None, seed=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = random.Random(seed)
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        # define discrete action set: adjustments for mutation, crossover, elites, topk, gpt_mut
        # each action is a tuple: (d_mut, d_cross, d_elites, d_topk, d_gpt_mut)
        self.actions = [
            (0.0, 0.0, 0, 0, 0.0),    # keep
            (0.002, 0.0, 0, 0, 0.0),  # inc mutation
            (-0.002, 0.0, 0, 0, 0.0), # dec mutation
            (0.0, 0.02, 0, 0, 0.0),   # inc crossover
            (0.0, -0.02, 0, 0, 0.0),  # dec crossover
            (0.0, 0.0, 1, 0, 0.0),    # inc elites
            (0.0, 0.0, -1, 0, 0.0),   # dec elites
            (0.0, 0.0, 0, 200, 0.0),  # inc topk
            (0.0, 0.0, 0, -200, 0.0), # dec topk
            (0.0, 0.0, 0, 0, 0.05),   # inc gpt_mut
            (0.0, 0.0, 0, 0, -0.05),  # dec gpt_mut
        ]

        # append a few no-op adjustment actions that are used solely to select
        # the population update strategy (hybrid/word_level/both and ordering).
        # These actions have zero numeric adjustments but map to strategy modes.
        self.strategy_action_map = {
            # indices will be assigned after we extend the actions list
        }

        # add no-op actions reserved for strategy selection
        for _ in range(4):
            self.actions.append((0.0, 0.0, 0, 0, 0.0))

        # map the newly added indices to strategy modes
        base_idx = len(self.actions) - 4
        self.strategy_action_map[base_idx + 0] = 'hybrid_only'
        self.strategy_action_map[base_idx + 1] = 'word_level_only'
        self.strategy_action_map[base_idx + 2] = 'both_hybrid_then_word_level'
        self.strategy_action_map[base_idx + 3] = 'both_word_level_then_hybrid'

        self.n_actions = len(self.actions)

        self.policy_net = MLP(state_dim, self.n_actions).to(self.device)
        self.target_net = MLP(state_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.replay = ReplayBuffer(capacity=buffer_capacity)
        self.update_steps = 0

    def state_to_tensor(self, state):
        # expected dict keys: improvement, diversity, patience, avg_score
        s = np.zeros(self.state_dim, dtype=np.float32)
        s[0] = float(state.get('improvement', 0.0))
        s[1] = float(state.get('diversity', 0.0))
        s[2] = float(state.get('patience', 0))
        s[3] = float(state.get('avg_score', 0.0))
        return torch.from_numpy(s).to(self.device)

    def select_action(self, state):
        s_t = self.state_to_tensor(state).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            # 这行代码是计算当前状态下每个动作的 Q 值，然后选择 Q 值最大的动作作为最优动作。实际上就是依据状态选择动作。
            q = self.policy_net(s_t)
            return int(torch.argmax(q, dim=1).item())

    def apply_action(self, base_params, action_idx):
        d_mut, d_cross, d_el, d_topk, d_gpt = self.actions[action_idx]
        new_mut = base_params.get('mutation', 0.01) + d_mut
        new_cross = base_params.get('crossover', 0.5) + d_cross
        new_el = base_params.get('num_elites', 1) + d_el
        new_topk = base_params.get('word_dict_topk', 2000) + d_topk
        new_gpt = base_params.get('gpt_mutation_prob', base_params.get('mutation', 0.01)) + d_gpt

        new_mut = float(max(0.0, min(0.5, new_mut)))
        new_cross = float(max(0.0, min(1.0, new_cross)))
        new_el = int(max(1, new_el))
        new_topk = int(max(10, new_topk))
        new_gpt = float(max(0.0, min(1.0, new_gpt)))

        out = {
            'mutation': new_mut,
            'crossover': new_cross,
            'num_elites': new_el,
            'word_dict_topk': new_topk,
            'gpt_mutation_prob': new_gpt,
        }

        # attach strategy_mode if this action is one of the reserved indices
        if hasattr(self, 'strategy_action_map') and action_idx in self.strategy_action_map:
            out['strategy_mode'] = self.strategy_action_map[action_idx]
        else:
            # default: let the caller use its own fallback logic (e.g., 'hybrid')
            out['strategy_mode'] = out.get('strategy_mode', 'auto')

        return out

    def update(self, state, action_idx, reward, next_state):
        # store transition
        s = np.array([state.get('improvement', 0.0), state.get('diversity', 0.0), float(state.get('patience', 0)), state.get('avg_score', 0.0)], dtype=np.float32)
        s2 = np.array([next_state.get('improvement', 0.0), next_state.get('diversity', 0.0), float(next_state.get('patience', 0)), next_state.get('avg_score', 0.0)], dtype=np.float32)
        self.replay.push(s, int(action_idx), float(reward), s2, False)

        # learn
        if len(self.replay) < max(16, self.batch_size):
            return
        bs, ba, br, bs2, bd = self.replay.sample(self.batch_size)
        bs = torch.from_numpy(bs).to(self.device)
        ba = torch.from_numpy(ba).to(self.device)
        br = torch.from_numpy(br).to(self.device)
        bs2 = torch.from_numpy(bs2).to(self.device)

        q_values = self.policy_net(bs)
        state_action_values = q_values.gather(1, ba.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(bs2)
            next_state_values = next_q.max(1)[0]
        # 计算目标 Q 值：当前奖励 + 折扣后的下一个状态的最大 Q 值，什么是Q值,这里的公式是 Q-learning 的核心更新公式。
        expected_q = br + self.gamma * next_state_values
        # 计算损失并更新网络参数，为什么计算state_action_values与expected_q的均方误差，因为均方误差可以衡量预测的Q值与目标Q值之间的差距，从而指导网络参数的更新。
        loss = nn.functional.mse_loss(state_action_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.update_steps += 1
        # soft update target every 100 updates
        if self.update_steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'policy_state': self.policy_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'actions': self.actions,
            'state_dim': self.state_dim,
        }
        torch.save(data, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data['policy_state'])
        self.target_net.load_state_dict(data['policy_state'])
        try:
            self.optimizer.load_state_dict(data['optimizer_state'])
        except Exception:
            pass
        try:
            self.actions = data.get('actions', self.actions)
            self.n_actions = len(self.actions)
        except Exception:
            pass


if __name__ == '__main__':
    # quick smoke test
    ctl = RLController(state_dim=4, seed=0)
    s = {'improvement': 0.01, 'diversity': 0.5, 'patience': 0, 'avg_score': 0.1}
    a = ctl.select_action(s)
    print('action', a)
    p = ctl.apply_action({'mutation': 0.01, 'crossover': 0.5, 'num_elites': 5, 'word_dict_topk':2000, 'gpt_mutation_prob':0.01}, a)
    print(p)
    ctl.update(s, a, reward=0.1, next_state={'improvement': 0.001, 'diversity': 0.6, 'patience': 0, 'avg_score': 0.09})
    print('Replay len', len(ctl.replay))
