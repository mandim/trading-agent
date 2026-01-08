import os, random, collections, time, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from trading_env_merged import TradingEnv

# ----------------- Action constants (must match env) -----------------
# Env semantics:
#   If FLAT (position_side == 0):
#       ACTION_HOLD -> WAIT        (do nothing)
#       ACTION_BUY  -> OPEN_LONG   (open long)
#       ACTION_SELL -> OPEN_SHORT  (open short)
#
#   If IN POSITION (position_side != 0):
#       ACTION_HOLD -> HOLD        (keep position)
#       ACTION_BUY  -> CLOSE       (close position)
#       ACTION_SELL -> REVERSE     (only if allow_reverse=True, else HOLD)
ACTION_HOLD = 0
ACTION_BUY  = 1
ACTION_SELL = 2


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
        return self.fc3(x)  # Q-values


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

    IMPORTANT alignment with new date-split env:
      - Training samples ONLY from [train_start_date, eval_start_date).
      - Eval runs ONLY from [eval_start_date, eval_end_date] (and auto-terminates at eval_end_date if env implements it).
    """
    env = TradingEnv(
        pip_decimal=0.0001,
        candles_file="unused.csv",
        tick_file="unused.csv",
        cache_dir="cache_fx_EURUSD_D1",

        # Trading
        tp_pips=50.0,
        sl_pips=40.0,
        lot=1.0,
        start_balance=100_000.0,
        reset_balance_each_episode=reset_balance_each_episode,
        exchange_rate=1.0,

        # Broker costs
        commission_per_lot_per_side_usd=3.0,
        enable_commission=True,
        enable_swaps=True,
        swap_long_pips_per_day=0.2,
        swap_short_pips_per_day=0.2,
        slippage_pips_open=0.2,
        slippage_pips_close=0.2,
        slippage_mode="fixed",
        enable_slippage=True,
        slippage_pips=0.0,
        other_fixed_cost_per_trade_usd=5.0,

        # Risk & penalties
        dd_penalty_lambda=0.0,
        max_dd_stop=0.30,
        flat_penalty_R=0.0,
        # IMPORTANT: your env default is trade_penalty_R=0.2; keep it consistent unless you intentionally change it
        trade_penalty_R=0.2,
        min_hold_bars=5,
        allow_reverse=False,
        cooldown_bars_after_close=3,

        # Episodes / split
        max_steps_per_episode=5000 if eval_mode else 20000,
        train_fraction=0.7,   # fallback only; overridden by dates below
        eval_mode=eval_mode,

        # Date split for TRAIN + VALIDATION (no peeking at TEST)
        train_start_date="2019-01-01",
        eval_start_date="2022-01-01",   # validation starts
        eval_end_date="2023-01-01",     # validation ends (exclusive boundary)

        # Obs / normalization
        window_len=32,
        normalize_prices=True,
        normalize_bars=True,

        # Runtime
        start_server=False,
        bind_address="tcp://*:5555",
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


# ----------------- Evaluation helper -----------------
def evaluate(policy_net: nn.Module, episodes=5, device="cpu"):
    """
    Greedy evaluation in the held-out period.
    NOTE: We set reset_balance_each_episode=False to simulate continuous equity if desired.
    If you want independent episodes, set it True.
    """
    env = make_env(seed=999, eval_mode=True, reset_balance_each_episode=False)
    policy_net.eval()
    ret_sum, len_sum = 0.0, 0.0

    for _ in range(episodes):
        s, _ = env.reset(seed=999)
        ep_ret, ep_len = 0.0, 0
        done = False
        with torch.no_grad():
            while not done:
                q = policy_net(torch.from_numpy(s).float().unsqueeze(0).to(device))
                a = int(q.argmax(dim=1).item())
                s, r, term, trunc, _info = env.step(a)
                done = bool(term or trunc)
                ep_ret += float(r)
                ep_len += 1
        ret_sum += ep_ret
        len_sum += ep_len

    policy_net.train()
    return ret_sum / episodes, len_sum / episodes


# ----------------- Training Loop -----------------
def train(
    steps=1_000_000,
    batch_size=256,
    gamma=0.995,
    lr=3e-5,
    start_training=10_000,
    target_tau=0.01,
    eps_start=1.0,
    eps_end=0.15,
    eps_decay_steps=1_200_000,
    eval_every=50_000,
    logdir="runs/dqn",
    log_every_steps=5_000,
    print_episode_end=True,
    run_tag="DQN_fx",
    clean_logs=False,
    debug: bool = False,
    debug_steps: int = 200,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output dirs exist
    os.makedirs("models", exist_ok=True)

    writer, name = _make_tb_writer(logdir, run_tag=run_tag, clean=clean_logs)
    print("=" * 70)
    print(f"TensorBoard run: {name}")
    print("=" * 70)

    env = make_env(seed=123, eval_mode=False, reset_balance_each_episode=True)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n  # should be 3

    online = QNet(obs_dim, n_actions).to(device)
    target = QNet(obs_dim, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    rb = Replay(cap=200_000)

    # Session stats (GROSS vs NET NOTE)
    # Your env's info["profit"] and info["profit_R"] are GROSS (price move) in your env.
    # Net = gross - (commission + swap + other). We'll compute net explicitly here.
    cum_net_profit_usd = 0.0
    cum_net_loss_usd = 0.0
    cum_net_profit_R = 0.0
    cum_net_loss_R = 0.0
    cum_trades = 0

    # Rolling trade-rate: trades per 1k steps
    trades_open_window = collections.deque(maxlen=1000)
    trades_close_window = collections.deque(maxlen=1000)

    # Cost stats
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

    session_start_balance = float(getattr(env, "start_balance", 100000.0))
    pip_value = float(env._pip_value_usd())
    one_R_usd = max(1e-9, pip_value * float(getattr(env, "sl_pips", 40.0)))

    def epsilon(step):
        t = min(1.0, step / float(eps_decay_steps))
        return eps_start + t * (eps_end - eps_start)

    s, _ = env.reset(seed=123)
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
            # Snapshot split for reproducibility
            "train_start_date": "2019-01-01",
            "eval_start_date": "2022-01-01",
            "eval_end_date": "2023-01-01",
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
        # ε-greedy over {0,1,2}
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

        # -------- Costs (accumulate every step) --------
        last_comm = float(info.get("commission_usd", 0.0))
        last_swap = float(info.get("swap_usd", 0.0))
        last_other = float(info.get("other_cost_usd", 0.0))
        cum_commission_usd += last_comm
        cum_swap_usd += last_swap
        cum_other_costs_usd += last_other

        if debug and step <= int(debug_steps):
            print(
                step, a,
                info.get("opened_trade"), info.get("closed_trade"), info.get("reversed_trade"),
                "pos", info.get("position_side"),
                "reason", info.get("exit_reason"),
                "gross_profit", float(info.get("profit", 0.0)),
                "comm", last_comm,
                "swap", last_swap,
                "other", last_other,
                "eq", float(info.get("equity", 0.0)),
                "bal", float(info.get("balance", 0.0)),
            )

        # -------- Per-trade accounting (compute NET properly) --------
        if info.get("closed_trade", False):
            gross_usd = float(info.get("profit", 0.0))
            gross_R = float(info.get("profit_R", 0.0))  # gross_R in your env
            pnl_pips = float(info.get("pnl_pips", 0.0))

            net_usd = gross_usd - last_comm - last_swap - last_other
            net_R = net_usd / one_R_usd

            cum_trades += 1
            trades += 1
            sum_pnl_pips += pnl_pips

            if net_usd >= 0:
                cum_net_profit_usd += net_usd
                cum_net_profit_R += net_R
                wins += 1
            else:
                cum_net_loss_usd += -net_usd
                cum_net_loss_R += -net_R

            net_profit_usd = cum_net_profit_usd - cum_net_loss_usd
            net_profit_R = cum_net_profit_R - cum_net_loss_R
            session_equity = session_start_balance + net_profit_usd
            profit_factor = (cum_net_profit_usd / cum_net_loss_usd) if cum_net_loss_usd > 0 else float("inf")

            # Session logs (NET, aligned)
            writer.add_scalar("session/cum_profit_usd", cum_net_profit_usd, step)
            writer.add_scalar("session/cum_loss_usd", cum_net_loss_usd, step)
            writer.add_scalar("session/net_profit_usd", net_profit_usd, step)
            writer.add_scalar("session/equity", session_equity, step)
            writer.add_scalar("session/trades_cum", cum_trades, step)
            writer.add_scalar("session/profit_factor", profit_factor, step)
            writer.add_scalar("session/cum_profit_R", cum_net_profit_R, step)
            writer.add_scalar("session/cum_loss_R", cum_net_loss_R, step)
            writer.add_scalar("session/net_profit_R", net_profit_R, step)

            # occasional histogram of trade NET PnL
            if step % 2000 == 0:
                writer.add_histogram("session/trade_pnl_usd", np.array([net_usd], dtype=np.float32), step)

        # env-level logs
        if "equity" in info:
            writer.add_scalar("env/equity_mark_to_market", float(info["equity"]), step)
        if "position_side" in info:
            writer.add_scalar("env/position_side", float(info["position_side"]), step)

        opened = 1.0 if info.get("opened_trade", False) else 0.0
        closed = 1.0 if info.get("closed_trade", False) else 0.0
        reversed_ = 1.0 if info.get("reversed_trade", False) else 0.0
        writer.add_scalar("env/opened_trade", opened, step)
        writer.add_scalar("env/closed_trade", closed, step)
        writer.add_scalar("env/reversed_trade", reversed_, step)

        if "blocked_by_cooldown" in info:
            writer.add_scalar("env/blocked_by_cooldown", 1.0 if info.get("blocked_by_cooldown") else 0.0, step)
        if "cooldown_remaining" in info:
            writer.add_scalar("env/cooldown_remaining", float(info.get("cooldown_remaining", 0.0)), step)

        # Rolling trade rate (executed trades per 1k steps)
        trades_open_window.append(opened)
        trades_close_window.append(closed)
        if len(trades_close_window) >= 50:
            writer.add_scalar(
                "env/trades_opened_per_1k",
                1000.0 * (sum(trades_open_window) / float(len(trades_open_window))),
                step,
            )
            writer.add_scalar(
                "env/trades_closed_per_1k",
                1000.0 * (sum(trades_close_window) / float(len(trades_close_window))),
                step,
            )

        if info.get("closed_trade", False) and "balance" in info:
            writer.add_scalar("env/balance_close_only", float(info["balance"]), step)

        # action mix
        if a == ACTION_HOLD:
            action_counts["hold"] += 1
        elif a == ACTION_BUY:
            action_counts["buy"] += 1
        elif a == ACTION_SELL:
            action_counts["sell"] += 1

        # push to replay
        rb.push(s, a, r, s2, terminated, truncated)
        s = s2
        ep_ret += float(r)
        ep_len += 1

        # episode end
        if done:
            writer.add_scalar("train/ep_return", ep_ret, step)
            writer.add_scalar("train/ep_length", ep_len, step)
            returns.append(ep_ret)
            if print_episode_end:
                print(
                    f"[episode end] step={step:,} ep_return={ep_ret:.3f} "
                    f"ep_len={ep_len} buf={len(rb):,}"
                )
            s, _ = env.reset(seed=123)
            ep_ret, ep_len = 0.0, 0

        # --------------- Learning ---------------
        if len(rb) >= start_training:
            S, A, R, S2, T, U = rb.sample(batch_size)

            d = torch.from_numpy(S).float().to(device)
            a_t = torch.from_numpy(A).long().to(device)
            r_t = torch.from_numpy(R).float().to(device)

            # Keep targets stable; env reward already in R units (realized-only)
            r_t = torch.clamp(r_t, -20.0, 20.0)

            d2 = torch.from_numpy(S2).float().to(device)
            done_mask = torch.from_numpy((T | U).astype(np.float32)).to(device)

            with torch.no_grad():
                # Double DQN target
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
                writer.add_histogram("dist/Q_values", q.detach().cpu().numpy(), step)
                writer.add_scalar("train/avg_abs_Q", q.abs().mean().item(), step)

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
                f"eps={e:.3f} buf={len(rb):>7} sps={sps:6.1f} ETA={human_time(eta)}"
            )

            train._last_print_t = now
            train._last_print_step = step

            writer.add_scalar("replay/size", len(rb), step)
            writer.add_scalar("costs/cum_commission_usd", cum_commission_usd, step)
            writer.add_scalar("costs/cum_swap_usd", cum_swap_usd, step)
            writer.add_scalar("costs/cum_other_costs_usd", cum_other_costs_usd, step)

            total_actions = sum(action_counts.values()) or 1
            writer.add_scalar("policy/frac_hold", action_counts["hold"] / total_actions, step)
            writer.add_scalar("policy/frac_buy", action_counts["buy"] / total_actions, step)
            writer.add_scalar("policy/frac_sell", action_counts["sell"] / total_actions, step)

            if trades > 0:
                writer.add_scalar("trades/win_rate", wins / trades, step)
                writer.add_scalar("trades/avg_pnl_pips", sum_pnl_pips / trades, step)

            writer.add_scalar("risk/max_drawdown", float(info.get("max_drawdown", 0.0)), step)

        # --------------- Internal eval ---------------
        if step % eval_every == 0 and len(rb) >= start_training:
            avg_ret, avg_len = evaluate(online, episodes=5, device=device)
            writer.add_scalar("eval/avg_return", avg_ret, step)
            writer.add_scalar("eval/avg_length", avg_len, step)

            print("-" * 70)
            print(f"[EVAL] step={step:,} avg_return={avg_ret:.3f} avg_len={avg_len:.1f}")

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
    print("Training finished. Saved: models/dqn_best.pt (best eval), models/dqn_final.pt (final).")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--start_training", type=int, default=20_000)
    p.add_argument("--target_tau", type=float, default=0.005)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.15)
    p.add_argument("--eps_decay_steps", type=int, default=1_200_000)
    p.add_argument("--eval_every", type=int, default=200_000)
    p.add_argument("--logdir", type=str, default="runs/dqn")
    p.add_argument("--log_every_steps", type=int, default=5_000)
    p.add_argument("--run_tag", type=str, default="DQN_fx")
    p.add_argument("--clean_logs", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_steps", type=int, default=200)

    args = p.parse_args()

    train(
        steps=args.steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        start_training=args.start_training,
        target_tau=args.target_tau,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        eval_every=args.eval_every,
        logdir=args.logdir,
        log_every_steps=args.log_every_steps,
        print_episode_end=True,
        run_tag=args.run_tag,
        clean_logs=args.clean_logs,
        debug=args.debug,
        debug_steps=args.debug_steps,
    )
