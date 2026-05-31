import random, collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, in_dim=11, hidden=256, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        t = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
        return t(s), torch.tensor(a), t(r), t(ns), torch.tensor(d, dtype=torch.float32)

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_min=0.01, eps_decay=0.995,
                 batch=128, target_update=200):
        self.policy  = QNet()
        self.target  = QNet()
        self.target.load_state_dict(self.policy.state_dict())
        self.opt     = optim.Adam(self.policy.parameters(), lr=lr)
        self.buf     = ReplayBuffer()
        self.gamma   = gamma
        self.eps     = eps_start
        self.eps_min = eps_min
        self.eps_dec = eps_decay
        self.batch   = batch
        self.tgt_upd = target_update
        self.step_n  = 0

    def act(self, obs, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randint(0, 2)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.policy(t).argmax().item()

    def store(self, s, a, r, ns, done):
        self.buf.push(s, a, r, ns, float(done))

    def learn(self):
        if len(self.buf) < self.batch:
            return
        s, a, r, ns, done = self.buf.sample(self.batch)
        a = a.long().unsqueeze(1)

        q_cur  = self.policy(s).gather(1, a).squeeze()
        with torch.no_grad():
            q_next = self.target(ns).max(1)[0]
            q_tgt  = r + self.gamma * q_next * (1 - done)

        loss = nn.MSELoss()(q_cur, q_tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        self.step_n += 1
        if self.step_n % self.tgt_upd == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.eps = max(self.eps_min, self.eps * self.eps_dec)

    def save(self, path="snake_dqn.pt"):
        torch.save(self.policy.state_dict(), path)

    def load(self, path="snake_dqn.pt"):
        self.policy.load_state_dict(torch.load(path, map_location="cpu"))
        self.target.load_state_dict(self.policy.state_dict())