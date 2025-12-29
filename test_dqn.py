import os, json, csv, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from trading_env_merged import TradingEnv


# Must match training network exactly
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def make_test_env(seed=999, reset_balance_each_episode=False):
    # Same as training config, but test window dates
    return TradingEnv(
        pip_decimal=0.0001,
        candles_file="unused.csv",
        tick_file="unused.csv",
        cache_dir="cache_fx_EURUSD_D1",

        tp_pips=50.0,
        sl_pips=40.0,
        lot=1.0,
        start_balance=100_000.0,
        reset_balance_each_episode=reset_balance_each_episode,
        exchange_rate=1.0,

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

        dd_penalty_lambda=0.0,
        max_dd_stop=0.30,
        flat_penalty_R=0.0,
        trade_penalty_R=0.2,
        min_hold_bars=5,
        allow_reverse=False,
        cooldown_bars_after_close=3,

        max_steps_per_episode=5000,
        eval_mode=True,

        # TEST WINDOW (never used during training/validation selection)
        train_start_date="2019-01-01",     # not used in eval_mode, but fine to keep
        eval_start_date="2023-01-01",
        eval_end_date="2025-12-28",        # exclusive end to include all of 2025

        window_len=32,
        normalize_prices=True,
        normalize_bars=True,

        start_server=False,
        seed=seed,
    )


def profit_factor(gp: float, gl: float) -> float:
    return float("inf") if gl <= 0 else float(gp / gl)


def run_test(policy, env, device, tb: SummaryWriter | None, ep_idx: int):
    s, _ = env.reset(seed=999)
    done = False

    pip_value = float(env._pip_value_usd())
    one_R_usd = max(1e-9, pip_value * float(env.sl_pips))

    trades = 0
    wins = 0
    sum_pnl_pips = 0.0

    gross_profit = 0.0
    gross_loss = 0.0
    net_profit_pos = 0.0
    net_loss_pos = 0.0

    cost_comm = 0.0
    cost_swap = 0.0
    cost_other = 0.0

    max_dd = 0.0

    ep_ret = 0.0
    ep_len = 0

    trade_rows = []

    with torch.no_grad():
        while not done:
            q = policy(torch.from_numpy(s).float().unsqueeze(0).to(device))
            a = int(q.argmax(dim=1).item())

            s, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)

            ep_ret += float(r)
            ep_len += 1
            max_dd = max(max_dd, float(info.get("max_drawdown", 0.0)))

            if info.get("closed_trade", False):
                trades += 1

                gross_usd = float(info.get("profit", 0.0))
                pnl_pips = float(info.get("pnl_pips", 0.0))
                comm = float(info.get("commission_usd", 0.0))
                swap = float(info.get("swap_usd", 0.0))
                other = float(info.get("other_cost_usd", 0.0))

                net_usd = gross_usd - comm - swap - other
                net_R = net_usd / one_R_usd

                sum_pnl_pips += pnl_pips
                if net_usd >= 0:
                    wins += 1

                # gross
                if gross_usd >= 0:
                    gross_profit += gross_usd
                else:
                    gross_loss += -gross_usd

                # net
                if net_usd >= 0:
                    net_profit_pos += net_usd
                else:
                    net_loss_pos += -net_usd

                cost_comm += comm
                cost_swap += swap
                cost_other += other

                trade_rows.append({
                    "episode": ep_idx,
                    "trade_idx": trades,
                    "action_id": a,
                    "exit_reason": info.get("exit_reason", ""),
                    "bar_index": int(info.get("bar_index", -1)),
                    "tick_index": int(info.get("t", -1)),
                    "pnl_pips": pnl_pips,
                    "profit_gross_usd": gross_usd,
                    "profit_net_usd": net_usd,
                    "profit_net_R": net_R,
                    "commission_usd": comm,
                    "swap_usd": swap,
                    "other_cost_usd": other,
                    "equity_after": float(info.get("equity", float("nan"))),
                    "balance_after": float(info.get("balance", float("nan"))),
                    "max_drawdown": float(info.get("max_drawdown", 0.0)),
                })

                if tb is not None:
                    tb.add_scalar("test/trade_net_usd", net_usd, trades)
                    tb.add_scalar("test/trade_net_R", net_R, trades)
                    tb.add_scalar("test/trade_pnl_pips", pnl_pips, trades)

    summary = {
        "episode": ep_idx,
        "ep_return": float(ep_ret),
        "ep_length": int(ep_len),
        "trades": int(trades),
        "win_rate": 0.0 if trades == 0 else float(wins / trades),
        "avg_pnl_pips": 0.0 if trades == 0 else float(sum_pnl_pips / trades),
        "profit_factor_gross": profit_factor(gross_profit, gross_loss),
        "profit_factor_net": profit_factor(net_profit_pos, net_loss_pos),
        "net_profit_usd": float(net_profit_pos - net_loss_pos),
        "gross_profit_usd": float(gross_profit),
        "gross_loss_usd": float(gross_loss),
        "commission_total_usd": float(cost_comm),
        "swap_total_usd": float(cost_swap),
        "other_costs_total_usd": float(cost_other),
        "max_drawdown": float(max_dd),
    }
    return summary, trade_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/dqn_best.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--csv", type=str, default="test/test_trades.csv")
    ap.add_argument("--json", type=str, default="test/test_summary.json")
    ap.add_argument("--tb", type=str, default="runs/test")
    ap.add_argument("--reset_balance_each_episode", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    os.makedirs(args.tb, exist_ok=True)

    env = make_test_env(seed=999, reset_balance_each_episode=args.reset_balance_each_episode)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = QNet(obs_dim, n_actions).to(args.device)
    policy.load_state_dict(torch.load(args.model, map_location=args.device))
    policy.eval()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = os.path.join(args.tb, f"{stamp}_TEST")
    tb = SummaryWriter(tb_dir)

    all_summaries = []
    wrote_header = False

    # overwrite CSV each run
    if os.path.exists(args.csv):
        os.remove(args.csv)

    for ep in range(1, args.episodes + 1):
        summary, trade_rows = run_test(policy, env, args.device, tb, ep_idx=ep)
        all_summaries.append(summary)

        if trade_rows:
            with open(args.csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
                if not wrote_header:
                    w.writeheader()
                    wrote_header = True
                w.writerows(trade_rows)

        print(f"[TEST] ep={ep}/{args.episodes} trades={summary['trades']} "
              f"win_rate={summary['win_rate']:.2%} PF_net={summary['profit_factor_net']:.3f} "
              f"net_usd={summary['net_profit_usd']:.2f} maxDD={summary['max_drawdown']:.2%}")

    # aggregate
    agg = {
        "episodes": len(all_summaries),
        "trades_total": int(sum(s["trades"] for s in all_summaries)),
        "win_rate": float(np.mean([s["win_rate"] for s in all_summaries])) if all_summaries else 0.0,
        "net_profit_usd": float(sum(s["net_profit_usd"] for s in all_summaries)),
        "profit_factor_net": float(np.mean([s["profit_factor_net"] for s in all_summaries])) if all_summaries else 0.0,
        "max_drawdown_worst_ep": float(max(s["max_drawdown"] for s in all_summaries)) if all_summaries else 0.0,
    }

    with open(args.json, "w") as f:
        json.dump({"per_episode": all_summaries, "aggregate": agg}, f, indent=2)

    print("-" * 72)
    print("Aggregate:", json.dumps(agg, indent=2))
    print(f"Saved CSV  -> {args.csv}")
    print(f"Saved JSON -> {args.json}")
    print(f"TB dir     -> {tb_dir}")
    tb.close()


if __name__ == "__main__":
    main()
