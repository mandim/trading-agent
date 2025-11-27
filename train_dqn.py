import os, math, random, collections, time, shutil, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# import the merged, cost-aware environment
from trading_env_merged import TradingEnv


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
        return (
            np.stack(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(te, dtype=np.bool_),
            np.array(tr, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buf)


# ----------------- Utilities -----------------
def soft_update(target, online, tau):
    with torch.no_grad():
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)


def make_env(seed=123, eval_mode=False, reset_balance_each_episode=True):
    """
    Single source of truth for env config (train & eval).

    Training:
      - eval_mode=False
      - reset_balance_each_episode=True (start from 100k each episode)
    Internal eval:
      - eval_mode=True
      - reset_balance_each_episode=False (carry equity forward like MT4 if you want)
    """
    env = TradingEnv(
        pip_decimal=0.0001,
        candles_file="unused.csv",
        tick_file="unused.csv",
        cache_dir="cache_fx_EURUSD_D1",   # <- same as eval_dqn.py / FXT cache

        # trading
        tp_pips=50.0,
        sl_pips=50.0,
        lot=1.0,                          # base lot; overridden by percent-risk if enabled
        start_balance=100_000.0,
        reset_balance_each_episode=reset_balance_each_episode,
        include_spread_cost=True,
        exchange_rate=1.0,

        # broker cost model (match MT4 / eval)
        account_type="standard",
        enable_commission=False,          # standard account -> costs in spread, no explicit commission
        # commission_per_lot_per_side_usd=3.0,  # only for RAW accounts

        enable_swaps=True,
        swap_long_pips_per_day=-1.3,
        swap_short_pips_per_day=0.42,

        # slippage model (set to what you actually use in MT4)
        slippage_mode="fixed",
        slippage_pips_open=1.0,
        slippage_pips_close=1.0,

        other_fixed_cost_per_trade_usd=0.0,

        # risk & reward
        dd_penalty_lambda=1.0,
        max_dd_stop=0.30,
        reward_mode="risk",
        reward_dense=True,       # <– enable

        # Position sizing (percent of equity; env computes lot per trade)
        risk_per_trade_usd=0.0,          # 0 -> use percent-based if enabled
        risk_percent=1.0,                # 1% of equity per trade
        use_percent_risk=True,

        # episodes / split
        max_steps_per_episode=None,      # can keep; episode will end earlier if max_dd_stop hit
        max_trade_ticks=2000,
        train_fraction=0.7,              # first 70% of data for training, last 30% for internal eval
        eval_mode=eval_mode,             # IMPORTANT: eval_mode=True for bar-open gating in eval()

        # observations & normalization
        window_len=32,
        normalize_prices=True,
        normalize_bars=True,

        # runtime
        start_server=False,
        seed=seed,
    )
    return env


def human_time(seconds):
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def _make_tb_writer(base_dir: str, run_tag: str | None = None, clean: bool = False):
    if clean and os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = stamp + (f"_{run_tag}" if run_tag else "")
    run_dir = os.path.join(base_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    return writer, name


# ----------------- Training Loop -----------------
def train(
    steps=1_000_000,
    batch_size=256,
    gamma=0.998,
    lr=1e-4,
    start_training=10_000,
    target_tau=0.01,
    eps_start=1.0,
    eps_end=0.1,
    eps_decay_steps=300_000,
    eval_every=50_000,
    logdir="runs/dqn",
    log_every_steps=5_000,
    print_episode_end=True,
    run_tag="DQN_fx",
    clean_logs=False,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer, name = _make_tb_writer(logdir, run_tag=run_tag, clean=clean_logs)
    print("=" * 70)
    print(f"TensorBoard run: {name}")
    print("=" * 70)

    env = make_env(seed=123, eval_mode=False, reset_balance_each_episode=True)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online = QNet(obs_dim, n_actions).to(device)
    target = QNet(obs_dim, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    rb = Replay(cap=200_000)

    # Session-level accumulators (using NET PnL from env)
    cum_profit_usd = 0.0
    cum_profit_R = 0.0
    cum_loss_usd = 0.0
    cum_loss_R = 0.0
    cum_trades = 0
    session_start_balance = getattr(env, "start_balance", 100000.0)

    # Cost components (for monitoring broker friction)
    cum_commission_usd = 0.0
    cum_swap_usd = 0.0
    cum_other_costs_usd = 0.0

    # Policy stats
    action_counts = {"hold": 0, "buy": 0, "sell": 0}
    wins = 0
    trades = 0
    sum_pnl_pips = 0.0

    # Rolling trackers
    losses = collections.deque(maxlen=500)
    returns = collections.deque(maxlen=50)

    def epsilon(step):
        t = min(1.0, step / float(eps_decay_steps))
        return eps_start + t * (eps_end - eps_start)

    s, info = env.reset(seed=123)
    ep_ret, ep_len = 0.0, 0

    writer.add_hparams(
        {
            "batch_size": batch_size,
            "gamma": gamma,
            "lr": lr,
            "start_training": start_training,
            "target_tau": target_tau,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay_steps": eps_decay_steps,
            "eval_every": eval_every,
        },
        {},
    )

    print("=" * 70)
    print(
        f"Starting training for {steps:,} steps | device={device} | "
        f"obs_dim={obs_dim} | n_actions={n_actions}"
    )
    print("=" * 70)

    for step in range(1, steps + 1):
        # ε-greedy
        e = epsilon(step)
        if random.random() < e:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                q = online(torch.from_numpy(s).float().unsqueeze(0).to(device))
                a = int(q.argmax(dim=1).item())

        # Step env
        s2, r, terminated, truncated, info = env.step(a)
        done = bool(terminated or truncated)

        # -------- Per-trade accounting (env already returns NET PnL) --------
        if info.get("closed_trade", False):
            # Net after costs:
            trade_pnl_usd = float(info.get("profit", 0.0))
            trade_R = float(info.get("profit_R", 0.0))
            pnl_pips = float(info.get("pnl_pips", 0.0))

            # Cost breakdown from env (if available)
            commission_usd = float(info.get("commission_usd", 0.0))
            swap_usd = float(info.get("swap_usd", 0.0))
            other_cost_usd = float(info.get("other_cost_usd", 0.0))

            cum_trades += 1
            trades += 1
            sum_pnl_pips += pnl_pips

            # Track costs
            cum_commission_usd += commission_usd
            cum_swap_usd += swap_usd
            cum_other_costs_usd += other_cost_usd

            # Split wins / losses on NET PnL
            if trade_pnl_usd >= 0:
                cum_profit_usd += trade_pnl_usd
            else:
                cum_loss_usd += -trade_pnl_usd

            if trade_R >= 0:
                cum_profit_R += trade_R
            else:
                cum_loss_R += -trade_R

            if info.get("is_profitable", False):
                wins += 1

            net_profit_usd = cum_profit_usd - cum_loss_usd
            net_profit_R = cum_profit_R - cum_loss_R
            session_equity = session_start_balance + net_profit_usd
            profit_factor = (
                (cum_profit_usd / cum_loss_usd) if cum_loss_usd > 0 else float("inf")
            )

            # Session-level logs (net of costs)
            writer.add_scalar("session/cum_profit_usd", cum_profit_usd, step)
            writer.add_scalar("session/cum_loss_usd", cum_loss_usd, step)
            writer.add_scalar("session/net_profit_usd", net_profit_usd, step)
            writer.add_scalar("session/equity", session_equity, step)
            writer.add_scalar("session/trades_cum", cum_trades, step)
            writer.add_scalar("session/profit_factor", profit_factor, step)
            writer.add_scalar("session/cum_profit_R", cum_profit_R, step)
            writer.add_scalar("session/cum_loss_R", cum_loss_R, step)
            writer.add_scalar("session/net_profit_R", net_profit_R, step)

            # Cost logs
            writer.add_scalar("costs/cum_commission_usd", cum_commission_usd, step)
            writer.add_scalar("costs/cum_swap_usd", cum_swap_usd, step)
            writer.add_scalar(
                "costs/cum_other_costs_usd", cum_other_costs_usd, step
            )

        # occasional histogram of trade PnL (NET)
        if step % 2000 == 0 and info.get("closed_trade", False):
            writer.add_histogram(
                "session/trade_pnl_usd",
                np.array([float(info.get("profit", 0.0))]),
                step,
            )

        # env-level logs
        if "equity" in info:
            writer.add_scalar(
                "env/equity_mark_to_market", float(info["equity"]), step
            )
        if info.get("closed_trade", False) and "balance" in info:
            writer.add_scalar(
                "env/balance_close_only", float(info["balance"]), step
            )

        # action mix
        if a == 0:
            action_counts["hold"] += 1
        elif a == 1:
            action_counts["buy"] += 1
        elif a == 2:
            action_counts["sell"] += 1

        # push to replay
        rb.push(s, a, r, s2, terminated, truncated)
        s = s2
        ep_ret += r
        ep_len += 1

        # episode end
        if done:
            writer.add_scalar("train/ep_return", ep_ret, step)
            writer.add_scalar("train/ep_length", ep_len, step)
            returns.append(ep_ret)
            if print_episode_end:
                print(
                    f"[episode end] step={step:,} "
                    f"ep_return={ep_ret:.3f} ep_len={ep_len} buf={len(rb):,}"
                )
            s, info = env.reset()
            ep_ret, ep_len = 0.0, 0

        # --------------- Learning ---------------
        if len(rb) >= start_training:
            S, A, R, S2, T, U = rb.sample(batch_size)

            d = torch.from_numpy(S).float().to(device)
            a_t = torch.from_numpy(A).long().to(device)
            r_t = torch.from_numpy(R).float().to(device)
            r_t = torch.clamp(r_t, -5.0, 5.0)

            d2 = torch.from_numpy(S2).float().to(device)
            done_mask = torch.from_numpy((T | U).astype(np.float32)).to(device)

            with torch.no_grad():
                next_q_online = online(d2)
                a2 = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = target(d2).gather(1, a2).squeeze(1)
                y = r_t + (1.0 - done_mask) * gamma * next_q_target

            q = online(d).gather(1, a_t.unsqueeze(1)).squeeze(1)

            loss = F.smooth_l1_loss(q, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            opt.step()
            soft_update(target, online, target_tau)

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/epsilon", e, step)
            losses.append(loss.item())

            if step % 1000 == 0:
                td_errors = (q - y).detach().cpu().numpy()
                writer.add_histogram("dist/td_error", td_errors, step)
                writer.add_histogram(
                    "dist/Q_values", q.detach().cpu().numpy(), step
                )
                writer.add_scalar(
                    "train/avg_abs_Q", q.abs().mean().item(), step
                )

        # --------------- Periodic logging ---------------
        if (step % log_every_steps == 0) or (step == 1):
            now = time.time()
            dt = now - getattr(train, "_last_print_t", now)
            steps_done = step - getattr(train, "_last_print_step", 0)
            sps = steps_done / max(1e-6, dt)
            pct = 100.0 * step / float(steps)
            eta = (steps - step) / max(1e-6, sps)

            print(
                f"[{step:>8}/{steps:,} | {pct:5.1f}%] "
                f"eps={e:.3f} buf={len(rb):>7} sps={sps:6.1f} "
                f"ETA={human_time(eta)}"
            )

            train._last_print_t = now
            train._last_print_step = step

            writer.add_scalar("replay/size", len(rb), step)
            total_actions = sum(action_counts.values()) or 1
            writer.add_scalar(
                "policy/frac_hold",
                action_counts["hold"] / total_actions,
                step,
            )
            writer.add_scalar(
                "policy/frac_buy",
                action_counts["buy"] / total_actions,
                step,
            )
            writer.add_scalar(
                "policy/frac_sell",
                action_counts["sell"] / total_actions,
                step,
            )

            if trades > 0:
                writer.add_scalar("trades/win_rate", wins / trades, step)
                writer.add_scalar(
                    "trades/avg_pnl_pips", sum_pnl_pips / trades, step
                )

            writer.add_scalar(
                "risk/max_drawdown",
                float(info.get("max_drawdown", 0.0)),
                step,
            )

        # --------------- Internal eval ---------------
        if step % eval_every == 0 and len(rb) >= start_training:
            avg_ret, avg_len = evaluate(online, episodes=5, device=device)
            writer.add_scalar("eval/avg_return", avg_ret, step)
            writer.add_scalar("eval/avg_length", avg_len, step)
            print("-" * 70)
            print(
                f"[EVAL] step={step:,} "
                f"avg_return={avg_ret:.3f} avg_len={avg_len:.1f}"
            )
            if avg_ret > getattr(train, "_best_eval", -1e9):
                train._best_eval = avg_ret
                torch.save(online.state_dict(), "models/dqn_best.pt")
                print("[CHECKPOINT] New best model -> models/dqn_best.pt")
            snap_name = f"models/dqn_step_{step}.pt"
            torch.save(online.state_dict(), snap_name)
            print(f"[CHECKPOINT] Snapshot -> {snap_name}")
            print("-" * 70)

    torch.save(online.state_dict(), "models/dqn_final.pt")
    print("=" * 70)
    print(
        "Training finished. "
        "Saved: models/dqn_best.pt (best eval), models/dqn_final.pt (final)."
    )


# ----------------- Evaluation helper used during training -----------------
def evaluate(policy_net: nn.Module, episodes=5, device="cpu"):
    env = make_env(seed=999, eval_mode=True, reset_balance_each_episode=False)
    ret_sum, len_sum = 0.0, 0.0
    for _ in range(episodes):
        s, info = env.reset()
        ep_ret, ep_len = 0.0, 0
        done = False
        with torch.no_grad():
            while not done:
                q = policy_net(torch.from_numpy(s).float().unsqueeze(0).to(device))
                a = int(q.argmax(dim=1).item())
                s, r, term, trunc, info = env.step(a)
                done = bool(term or trunc)
                ep_ret += float(r)
                ep_len += 1
        ret_sum += ep_ret
        len_sum += ep_len
    return ret_sum / episodes, len_sum / episodes


if __name__ == "__main__":
    train(
        steps=2_000_000,
        batch_size=256,
        gamma=0.998,
        lr=1e-4,
        start_training=20_000,
        target_tau=0.005,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=800_000,
        eval_every=200_000,
        logdir="runs/dqn",
        log_every_steps=5_000,
        print_episode_end=True,
        run_tag="DQN_fx",
        clean_logs=False,
    )
