# evaluate_dqn.py

import os, json, argparse, csv
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from trading_env_merged import TradingEnv  # merged env


# ---- Q-Network (must match training) -----------------------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ---- Env factory (mirror training; eval_mode=True) ---------------------------
def make_env(seed=999,
             reset_balance_each_episode=False,
             max_steps_per_episode=5000):
    env = TradingEnv(
        pip_decimal=0.0001,
        candles_file="unused.csv",
        tick_file="unused.csv",
        cache_dir="cache_fx_EURUSD_D1",

        tp_pips=50.0,
        sl_pips=50.0,
        lot=0.1,
        start_balance=100_000.0,
        reset_balance_each_episode=reset_balance_each_episode,
        include_spread_cost=False,
        exchange_rate=1.0,

        dd_penalty_lambda=1.0,
        max_dd_stop=0.30,
        reward_mode="risk",
        risk_per_trade_usd=1000.0,

        max_steps_per_episode=max_steps_per_episode,
        train_fraction=0.7,
        eval_mode=True,

        window_len=32,
        normalize_prices=True,
        normalize_bars=True,

        start_server=False,
        seed=seed,
    )
    return env


# ---- Metrics helpers ---------------------------------------------------------
def profit_factor(gross_profit, gross_loss):
    return float('inf') if gross_loss <= 0 else (gross_profit / gross_loss)

def safe_div(a, b):
    return 0.0 if b == 0 else a / b


# ---- Run one greedy episode --------------------------------------------------
def run_episode(env, net, device, tb_writer=None, global_trade_step=0):
    s, info = env.reset()
    done = False
    ep_return, ep_len = 0.0, 0
    trades, wins = 0, 0
    sum_pnl_pips = 0.0
    cum_profit_usd = 0.0
    cum_loss_usd = 0.0
    cum_profit_R = 0.0
    cum_loss_R = 0.0
    start_balance = float(getattr(env, "balance", 0.0))
    max_dd_seen = 0.0
    trade_rows = []

    with torch.no_grad():
        while not done:
            q = net(torch.from_numpy(s).float().unsqueeze(0).to(device))
            a = int(q.argmax(dim=1).item())
            s, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            ep_return += float(r)
            ep_len += 1

            max_dd_seen = max(max_dd_seen, float(info.get("max_drawdown", 0.0)))

            if info.get("closed_trade", False):
                trades += 1
                is_win = bool(info.get("is_profitable", False))
                if is_win:
                    wins += 1

                pnl_usd = float(info.get("profit", 0.0))
                pnl_R = float(info.get("profit_R", 0.0))
                pnl_pips = float(info.get("pnl_pips", 0.0))
                sum_pnl_pips += pnl_pips

                if pnl_usd >= 0: cum_profit_usd += pnl_usd
                else:            cum_loss_usd += -pnl_usd

                if pnl_R >= 0:   cum_profit_R += pnl_R
                else:            cum_loss_R += -pnl_R

                equity = float(info.get("equity", float('nan')))
                balance = float(info.get("balance", float('nan')))

                if tb_writer is not None:
                    tb_writer.add_scalar("eval/equity_close_only", equity,
                                         global_trade_step + trades)
                    tb_writer.add_scalar("eval/balance_close_only", balance,
                                         global_trade_step + trades)

                trade_rows.append({
                    "trade_idx": trades,
                    "action": info.get("action", ""),
                    "is_profitable": is_win,
                    "profit_usd": pnl_usd,
                    "profit_R": pnl_R,
                    "pnl_pips": pnl_pips,
                    "trade_duration_ticks": int(info.get("trade_duration_ticks", 0)),
                    "equity_after": equity,
                    "balance_after": balance,
                    "dd_now": float(info.get("dd_now", 0.0)),
                    "max_drawdown": float(info.get("max_drawdown", 0.0)),
                    "final_tick_idx": int(info.get("final_idx", 0)),
                })

    net_profit_usd = cum_profit_usd - cum_loss_usd
    net_profit_R = cum_profit_R - cum_loss_R
    end_equity = start_balance + net_profit_usd

    ep_stats = {
        "ep_return": ep_return,
        "ep_length": ep_len,
        "trades": trades,
        "wins": wins,
        "win_rate": safe_div(wins, trades),
        "avg_pnl_pips": safe_div(sum_pnl_pips, trades),
        "gross_profit_usd": cum_profit_usd,
        "gross_loss_usd": cum_loss_usd,
        "net_profit_usd": net_profit_usd,
        "profit_factor": profit_factor(cum_profit_usd, cum_loss_usd),
        "gross_profit_R": cum_profit_R,
        "gross_loss_R": cum_loss_R,
        "net_profit_R": net_profit_R,
        "start_balance": start_balance,
        "end_equity": end_equity,
        "max_drawdown": max_dd_seen,
    }

    return ep_stats, trade_rows


# ---- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="dqn_best.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--logdir", type=str, default="runs/eval")
    ap.add_argument("--csv", type=str, default="eval_trades.csv")
    ap.add_argument("--json", type=str, default="eval_summary.json")
    ap.add_argument("--reset_balance_each_episode", action="store_true",
                    help="Reset balance each episode (default: keep running equity).")
    ap.add_argument("--max_steps_per_episode", type=int, default=5000)
    args = ap.parse_args()

    device = torch.device(args.device)
    env = make_env(
        seed=args.seed,
        reset_balance_each_episode=args.reset_balance_each_episode,
        max_steps_per_episode=args.max_steps_per_episode,
    )

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = QNet(obs_dim, n_actions).to(device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, f"{stamp}_EVAL"))

    writer.add_hparams({
        "episodes": args.episodes,
        "seed": args.seed,
        "reset_balance_each_episode": int(args.reset_balance_each_episode),
        "max_steps_per_episode": args.max_steps_per_episode,
    }, {})

    all_summaries = []
    global_trade_step = 0

    for ep in range(1, args.episodes + 1):
        ep_stats, trade_rows = run_episode(env, net, device, writer, global_trade_step)
        global_trade_step += len(trade_rows)
        all_summaries.append(ep_stats)

        for k, v in ep_stats.items():
            writer.add_scalar(f"episode/{k}", float(v) if isinstance(v, (int, float)) else 0.0, ep)

        print(f"[EVAL] ep={ep}/{args.episodes} "
              f"| trades={ep_stats['trades']} win_rate={ep_stats['win_rate']:.2%} "
              f"| PF={ep_stats['profit_factor']:.3f} "
              f"| net_usd={ep_stats['net_profit_usd']:.2f} "
              f"| maxDD={ep_stats['max_drawdown']:.2%} "
              f"| ep_return={ep_stats['ep_return']:.3f}")

        # write CSV (append)
        if trade_rows:
            write_header = (ep == 1 and not os.path.exists(args.csv))
            with open(args.csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
                if write_header:
                    w.writeheader()
                for row in trade_rows:
                    w.writerow(row)

    # aggregate
    total_trades = sum(s["trades"] for s in all_summaries)
    agg = {
        "episodes": args.episodes,
        "trades_total": int(total_trades),
        "win_rate": safe_div(sum(s["wins"] for s in all_summaries), total_trades),
        "profit_factor": profit_factor(
            sum(s["gross_profit_usd"] for s in all_summaries),
            sum(s["gross_loss_usd"] for s in all_summaries),
        ),
        "net_profit_usd": float(sum(s["net_profit_usd"] for s in all_summaries)),
        "net_profit_R": float(sum(s["net_profit_R"] for s in all_summaries)),
        "avg_ep_return": float(np.mean([s["ep_return"] for s in all_summaries])) if all_summaries else 0.0,
        "avg_ep_length": float(np.mean([s["ep_length"] for s in all_summaries])) if all_summaries else 0.0,
        "max_drawdown_worst_ep": float(max(s["max_drawdown"] for s in all_summaries)) if all_summaries else 0.0,
    }

    with open(args.json, "w") as f:
        json.dump(agg, f, indent=2)

    print("-" * 72)
    print("Aggregate:", json.dumps(agg, indent=2))
    print(f"Saved per-trade CSV -> {args.csv}")
    print(f"Saved summary JSON -> {args.json}")
    writer.close()


if __name__ == "__main__":
    main()
