
import os, math, random, collections, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# import the environment
from trading_env_gym import TradingEnv

# ----------------- Q-Network -----------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action

# ----------------- Replay Buffer -----------------
class Replay:
    def __init__(self, cap=200_000):
        self.buf = collections.deque(maxlen=cap)
    def push(self, s, a, r, s2, term, trunc):
        self.buf.append((s, a, r, s2, term, trunc))
    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s, a, r, s2, te, tr = zip(*batch)
        return (np.stack(s), np.array(a), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(te, dtype=np.bool_), np.array(tr, dtype=np.bool_))
    def __len__(self): return len(self.buf)

# ----------------- Utilities -----------------
def soft_update(target, online, tau):
    with torch.no_grad():
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)

def make_env(eval_mode=False, reward_mode="pnl", normalize_prices=True, seed=123):
    # Configure knobs here to match your preference
    env = TradingEnv(
        pip_decimal=0.0001,
        candles_file="unused.csv",
        tick_file="unused.csv",
        cache_dir="cache_fx_EURUSD_D1",
        reward_mode=reward_mode,
        normalize_prices=normalize_prices,
        eval_mode=eval_mode,
        max_steps_per_episode=5000,
        window_len=32,
        tp_pips=50.0,
        sl_pips=50.0,
        lot=1.0,
        risk_per_trade_usd=1000.0,
        dd_penalty_lambda=1.0,
        max_dd_stop=0.30,
        seed=seed,
    )
    return env

def train(steps=1_000_000,
          batch_size=256,
          gamma=0.998,
          lr=3e-4,
          start_training=10_000,
          target_tau=0.01,
          eps_start=1.0,
          eps_end=0.05,
          eps_decay_steps=300_000,
          eval_every=50_000,
          logdir="runs/dqn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(logdir)

    env = make_env(eval_mode=False, reward_mode="pnl", normalize_prices=True, seed=123)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online = QNet(obs_dim, n_actions).to(device)
    target = QNet(obs_dim, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    rb = Replay(cap=200_000)

    def epsilon(step):
        t = min(1.0, step / float(eps_decay_steps))
        return eps_start + t * (eps_end - eps_start)

    s, info = env.reset(seed=123)
    ep_ret, ep_len = 0.0, 0
    best_eval = -1e9
    t0 = time.time()

    for step in range(1, steps + 1):
        # ε-greedy policy
        e = epsilon(step)
        if random.random() < e:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                q = online(torch.from_numpy(s).float().unsqueeze(0).to(device))
                a = int(q.argmax(dim=1).item())

        s2, r, terminated, truncated, info = env.step(a)
        rb.push(s, a, r, s2, terminated, truncated)
        s = s2; ep_ret += r; ep_len += 1

        if terminated or truncated:
            writer.add_scalar("train/ep_return", ep_ret, step)
            writer.add_scalar("train/ep_length", ep_len, step)
            s, info = env.reset()

            ep_ret, ep_len = 0.0, 0

        # Learn
        if len(rb) >= start_training:
            S, A, R, S2, T, U = rb.sample(batch_size)
            d = torch.from_numpy(S).float().to(device)
            a_t = torch.from_numpy(A).long().to(device)
            r_t = torch.from_numpy(R).float().to(device)
            d2 = torch.from_numpy(S2).float().to(device)
            done = torch.from_numpy((T | U).astype(np.float32)).to(device)

            with torch.no_grad():
                # Double DQN
                next_q_online = online(d2)
                a2 = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = target(d2).gather(1, a2).squeeze(1)
                y = r_t + (1.0 - done) * gamma * next_q_target

            q = online(d).gather(1, a_t.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            opt.step()

            soft_update(target, online, target_tau)

            # logging
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/epsilon", e, step)

        if step % 10_000 == 0:
            dt = time.time() - t0
            print(f"[{step:>8}] buffer={len(rb):>7} eps={e:.3f} elapsed={dt/60:.1f}m")
            t0 = time.time()

        if step % eval_every == 0:
            avg_ret, avg_len = evaluate(online, episodes=5, device=device)
            writer.add_scalar("eval/avg_return", avg_ret, step)
            writer.add_scalar("eval/avg_length", avg_len, step)
            # save best
            if avg_ret > best_eval:
                best_eval = avg_ret
                torch.save(online.state_dict(), "dqn_best.pt")
                print(f"Saved new best model: avg_return={avg_ret:.3f}")

    # final save
    torch.save(online.state_dict(), "dqn_final.pt")
    print("Training finished. Models saved: dqn_best.pt, dqn_final.pt")

def evaluate(policy_net: nn.Module, episodes=5, device="cpu"):
    env = make_env(eval_mode=True, reward_mode="pnl", normalize_prices=True, seed=999)
    ret_sum, len_sum = 0.0, 0
    for ep in range(episodes):
        s, info = env.reset()
        ep_ret, ep_len = 0.0, 0
        done = False
        while not done:
            with torch.no_grad():
                q = policy_net(torch.from_numpy(s).float().unsqueeze(0).to(device))
                a = int(q.argmax(dim=1).item())
            s, r, term, trunc, info = env.step(a)
            done = term or trunc
            ep_ret += r; ep_len += 1
        ret_sum += ep_ret; len_sum += ep_len
    return ret_sum / episodes, len_sum / episodes

if __name__ == "__main__":
    # You can tune via env vars or just edit args below.
    train(
        steps=300_000,
        batch_size=256,
        gamma=0.998,
        lr=3e-4,
        start_training=5_000,
        target_tau=0.01,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=150_000,
        eval_every=50_000,
        logdir="runs/dqn"
    )
