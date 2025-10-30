
import os, math, random, collections, time, shutil, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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

def human_time(seconds):
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h: return f"{h:d}h {m:02d}m {s:02d}s"
    if m: return f"{m:d}m {s:02d}s"
    return f"{s:d}s"

def _make_tb_writer(base_dir: str, run_tag: str | None = None, clean: bool = False):
    """
    Creates a fresh timestamped run directory inside base_dir and returns (writer, run_dir).
    If clean=True, deletes the entire base_dir first.
    """
    if clean and os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = stamp + (f"_{run_tag}" if run_tag else "")
    run_dir = os.path.join(base_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    return writer, run_dir

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
          logdir="runs/dqn",
          log_every_steps=5_000,
          print_episode_end=True,
          run_tag="DQN",
          clean_logs=False):
    """
    Trains Double-DQN and prints frequent progress messages.
    - log_every_steps: console progress interval (steps)
    - print_episode_end: print a short one-liner when an episode ends
    - run_tag: appended to the timestamped TB run folder for readability
    - clean_logs: if True, deletes `logdir` before creating the new run
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== NEW: fresh TensorBoard run directory (timestamped) =====
    writer, run_dir = _make_tb_writer(logdir, run_tag=run_tag, clean=clean_logs)
    print("="*70)
    print(f"TensorBoard run directory: {run_dir}")
    print("="*70)

    env = make_env(eval_mode=False, reward_mode="pnl", normalize_prices=True, seed=123)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online = QNet(obs_dim, n_actions).to(device)
    target = QNet(obs_dim, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    rb = Replay(cap=200_000)
    
    # policy & trade stats
    action_counts = {"hold": 0, "buy": 0, "sell": 0}
    wins = 0
    trades = 0
    sum_pnl_pips = 0.0

    # rolling trackers
    losses = collections.deque(maxlen=500)
    returns = collections.deque(maxlen=50)
    steps_per_sec = 0.0

    def epsilon(step):
        t = min(1.0, step / float(eps_decay_steps))
        return eps_start + t * (eps_end - eps_start)

    s, info = env.reset(seed=123)
    ep_ret, ep_len = 0.0, 0
    best_eval = -1e9

    # progress timers
    start_wall = time.time()
    last_print_t = start_wall
    last_print_step = 0

    # ===== NEW: hparams summary =====
    writer.add_hparams({
        "batch_size": batch_size,
        "gamma": gamma,
        "lr": lr,
        "start_training": start_training,
        "target_tau": target_tau,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay_steps": eps_decay_steps,
        "eval_every": eval_every,
    }, {})

    print("="*70)
    print(f"Starting training for {steps:,} steps | device={device} | obs_dim={obs_dim} | n_actions={n_actions}")
    print(f"Replay warmup: {start_training:,} | Eval every: {eval_every:,} steps | Log every: {log_every_steps:,} steps")
    print("="*70)

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
        
        # action mix
        if a == 0: action_counts["hold"] += 1
        elif a == 1: action_counts["buy"] += 1
        elif a == 2: action_counts["sell"] += 1

        # trade outcomes (only count when BUY/SELL)
        if a in (1, 2):
            trades += 1
            if info.get("is_profitable", False): wins += 1
            sum_pnl_pips += float(info.get("pnl_pips", 0.0))

        rb.push(s, a, r, s2, terminated, truncated)
        s = s2; ep_ret += r; ep_len += 1

        if terminated or truncated:
            # episode logs
            writer.add_scalar("train/ep_return", ep_ret, step)
            writer.add_scalar("train/ep_length", ep_len, step)
            returns.append(ep_ret)
            if print_episode_end:
                print(f"[episode end] step={step:,}  ep_return={ep_ret:.3f}  ep_len={ep_len}  buffer={len(rb):,}")
            s, info = env.reset()
            ep_ret, ep_len = 0.0, 0

        # Learn
        if len(rb) >= start_training:
            S, A, R, S2, T, U = rb.sample(batch_size)
            d = torch.from_numpy(S).float().to(device)
            a_t = torch.from_numpy(A).long().to(device)
            r_t = torch.from_numpy(R).float().to(device)
            
            r_t = torch.clamp(r_t, -5.0, 5.0)
            
            d2 = torch.from_numpy(S2).float().to(device)
            done = torch.from_numpy((T | U).astype(np.float32)).to(device)

            with torch.no_grad():
                # Double DQN target
                next_q_online = online(d2)
                a2 = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = target(d2).gather(1, a2).squeeze(1)
                y = r_t + (1.0 - done) * gamma * next_q_target

            q = online(d).gather(1, a_t.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                # average absolute Q
                writer.add_scalar("train/avg_abs_Q", q.abs().mean().item(), step)
            
            td_errors = (q - y).detach().cpu().numpy()
            # histogram (occasional to save space; e.g., every 1k steps)
            if step % 1000 == 0:
                writer.add_histogram("dist/td_error", td_errors, step)
                writer.add_histogram("dist/Q_values", q.detach().cpu().numpy(), step)
            
            loss = F.smooth_l1_loss(q, y)
            losses.append(loss.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            # gradient L2 norm (for stability)
            total_norm = 0.0
            for p in online.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            writer.add_scalar("train/grad_norm", total_norm, step)
            
            nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            opt.step()

            soft_update(target, online, target_tau)

            # tensorboard logging
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/epsilon", e, step)

        # --- periodic console progress ---
        if (step % log_every_steps == 0) or (step == 1):
            now = time.time()
            dt = now - last_print_t
            steps_done = step - last_print_step
            steps_per_sec = steps_done / max(1e-6, dt)
            pct = 100.0 * step / float(steps)
            eta = (steps - step) / max(1e-6, steps_per_sec)
            avg_loss = (sum(losses) / len(losses)) if len(losses) else float('nan')
            avg_ret = (sum(returns) / len(returns)) if len(returns) else float('nan')
            print(f"[{step:>8}/{steps:,} | {pct:5.1f}%] "
                  f"eps={e:.3f}  buf={len(rb):>7}  sps={steps_per_sec:6.1f}  "
                  f"avg_loss={avg_loss:8.5f}  avg_ep_ret={avg_ret:8.3f}  ETA={human_time(eta)}")
            last_print_t = now
            last_print_step = step
            
            # replay size
            writer.add_scalar("replay/size", len(rb), step)

            # action fractions (avoid div-by-zero)
            total_actions = sum(action_counts.values()) or 1
            writer.add_scalar("policy/frac_hold", action_counts["hold"] / total_actions, step)
            writer.add_scalar("policy/frac_buy",  action_counts["buy"]  / total_actions, step)
            writer.add_scalar("policy/frac_sell", action_counts["sell"] / total_actions, step)

            # trade KPIs
            if trades > 0:
                writer.add_scalar("trades/win_rate", wins / trades, step)
                writer.add_scalar("trades/avg_pnl_pips", sum_pnl_pips / trades, step)
            # env-side risk metric
            writer.add_scalar("risk/max_drawdown", float(info.get("max_drawdown", 0.0)), step)

        # --- evaluation ---
        if step % eval_every == 0 and len(rb) >= start_training:
            avg_ret, avg_len = evaluate(online, episodes=5, device=device)
            writer.add_scalar("eval/avg_return", avg_ret, step)
            writer.add_scalar("eval/avg_length", avg_len, step)
            print("-"*70)
            print(f"[EVAL] step={step:,}  avg_return={avg_ret:.3f}  avg_len={avg_len:.1f}")
            # save best
            if avg_ret > best_eval:
                best_eval = avg_ret
                torch.save(online.state_dict(), "dqn_best.pt")
                print(f"[CHECKPOINT] New best model saved -> dqn_best.pt  (avg_return={avg_ret:.3f})")
            # periodic snapshot too
            snap_name = f"dqn_step_{step}.pt"
            torch.save(online.state_dict(), snap_name)
            print(f"[CHECKPOINT] Snapshot saved -> {snap_name}")
            print("-"*70)

    # final save
    torch.save(online.state_dict(), "dqn_final.pt")
    print("="*70)
    print("Training finished.")
    print("Models saved: dqn_best.pt (best eval), dqn_final.pt (final weights)")
    total_time = time.time() - start_wall
    print(f"Total wall time: {human_time(total_time)}")

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
        lr=1e-4,
        start_training=20_000,
        target_tau=0.005,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=150_000,
        eval_every=50_000,
        logdir="runs/dqn",
        log_every_steps=5_000,
        print_episode_end=True,
        run_tag="DQN_fx",      # NEW: label visible in TensorBoard
        clean_logs=False       # NEW: set True to wipe `runs/dqn` before this run
    )
