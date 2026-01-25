import os, json, argparse, csv
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from trading_env_merged import TradingEnv


# ---- Q-Network (MUST match training) -----------------------------------------
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


# ---- Metrics helpers ---------------------------------------------------------
def profit_factor(gross_profit: float, gross_loss: float) -> float:
    return float("inf") if gross_loss <= 0 else float(gross_profit / gross_loss)

def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a / b)

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# ---- Env factory (ALIGNED, date-driven) --------------------------------------
def make_env(
    seed: int = 999,
    reset_balance_each_episode: bool = False,
    max_steps_per_episode: int = 5000,
    cache_dir: str = "cache_fx_EURUSD_D1",
    train_start_date: str | None = "2019-01-01",
    eval_start_date: str | None = "2023-01-01",
    eval_end_date: str | None = "2025-12-28",   # exclusive end for including all of 2025
):
    """
    Date-driven evaluation environment.

    Alignment points:
      - Same TP/SL, lot, costs, slippage, swaps as training
      - Same min_hold_bars and cooldown
      - eval_mode=True
      - Date boundaries override train_fraction split inside env
    """
    env = TradingEnv(
        pip_decimal=0.0001,
        candles_file="unused.csv",
        tick_file="unused.csv",
        cache_dir=cache_dir,

        # Trading (MATCH TRAINING)
        tp_pips=50.0,
        sl_pips=40.0,
        lot=1.0,
        start_balance=100_000.0,
        reset_balance_each_episode=reset_balance_each_episode,
        exchange_rate=1.0,

        # Broker costs (MATCH TRAINING)
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

        # Risk & penalties (MATCH TRAINING)
        dd_penalty_lambda=0.0,
        max_dd_stop=0.30,
        flat_penalty_R=0.0,
        trade_penalty_R=0.2,
        min_hold_bars=5,
        allow_reverse=False,
        cooldown_bars_after_close=3,

        # Episodes
        max_steps_per_episode=max_steps_per_episode,

        # IMPORTANT: eval mode + date split
        train_fraction=0.7,  # fallback only
        eval_mode=True,
        train_start_date=train_start_date,
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,

        # Obs / normalization (MATCH TRAINING)
        window_len=32,
        normalize_prices=True,
        normalize_bars=True,

        # Runtime
        seed=seed,
    )
    return env


# ---- Run one greedy episode --------------------------------------------------
def run_episode(env: TradingEnv, net: nn.Module, device, tb_writer=None, ep_idx: int = 1):
    """
    Runs one greedy episode and returns:
      - ep_stats: summary dict (gross/net, costs, risk)
      - trade_rows: per-trade rows for CSV

    IMPORTANT (matches env):
      - info["profit"] is GROSS trade PnL from price move (pips * pip_value)
      - commission/swap/other are reported separately and debited from balance
      - Therefore: net_usd = profit - commission - swap - other
    """
    s, _ = env.reset(seed=999)
    done = False

    ep_return = 0.0
    ep_len = 0

    trades = 0
    wins = 0
    sum_pnl_pips = 0.0

    # Gross PnL (price movement only)
    cum_gross_profit_usd = 0.0
    cum_gross_loss_usd = 0.0

    # Net PnL (after commission+swap+other)
    cum_net_profit_usd = 0.0
    cum_net_loss_usd = 0.0

    # Costs
    cum_commission_usd = 0.0
    cum_swap_usd = 0.0
    cum_other_costs_usd = 0.0

    start_balance = float(getattr(env, "balance", 0.0))
    max_dd_seen = 0.0
    trade_rows = []

    one_R_usd = max(1e-9, float(env._pip_value_usd()) * float(env.sl_pips))

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

                gross_usd = float(info.get("profit", 0.0))
                gross_R = float(info.get("profit_R", 0.0))  # gross_R in your env
                pnl_pips = float(info.get("pnl_pips", 0.0))

                commission_usd = float(info.get("commission_usd", 0.0))
                swap_usd = float(info.get("swap_usd", 0.0))
                other_cost_usd = float(info.get("other_cost_usd", 0.0))

                net_usd = gross_usd - commission_usd - swap_usd - other_cost_usd
                net_R = net_usd / one_R_usd

                sum_pnl_pips += pnl_pips
                is_win = net_usd >= 0.0
                if is_win:
                    wins += 1

                # Aggregate gross
                if gross_usd >= 0:
                    cum_gross_profit_usd += gross_usd
                else:
                    cum_gross_loss_usd += -gross_usd

                # Aggregate net
                if net_usd >= 0:
                    cum_net_profit_usd += net_usd
                else:
                    cum_net_loss_usd += -net_usd

                # Costs
                cum_commission_usd += commission_usd
                cum_swap_usd += swap_usd
                cum_other_costs_usd += other_cost_usd

                equity = float(info.get("equity", float("nan")))
                balance = float(info.get("balance", float("nan")))

                if tb_writer is not None:
                    tb_writer.add_scalar("eval/trade_net_usd", net_usd, trades)
                    tb_writer.add_scalar("eval/trade_gross_usd", gross_usd, trades)
                    tb_writer.add_scalar("eval/trade_net_R", net_R, trades)
                    tb_writer.add_scalar("eval/trade_pnl_pips", pnl_pips, trades)
                    tb_writer.add_scalar("eval/equity", equity, trades)
                    tb_writer.add_scalar("eval/balance", balance, trades)

                trade_rows.append(
                    {
                        "episode": ep_idx,
                        "trade_idx": trades,
                        "action_id": a,
                        "exit_reason": info.get("exit_reason", ""),
                        "bar_index": int(info.get("bar_index", -1)),
                        "tick_index": int(info.get("t", -1)),

                        "pnl_pips": pnl_pips,
                        "profit_gross_usd": gross_usd,
                        "profit_net_usd": net_usd,
                        "profit_gross_R": gross_R,
                        "profit_net_R": net_R,

                        "commission_usd": commission_usd,
                        "swap_usd": swap_usd,
                        "other_cost_usd": other_cost_usd,

                        "is_profitable_net": bool(is_win),
                        "equity_after": equity,
                        "balance_after": balance,
                        "max_drawdown": float(info.get("max_drawdown", 0.0)),
                    }
                )

    net_profit_usd = cum_net_profit_usd - cum_net_loss_usd
    net_profit_R = net_profit_usd / one_R_usd
    end_equity = start_balance + net_profit_usd

    pf_gross = profit_factor(cum_gross_profit_usd, cum_gross_loss_usd)
    pf_net = profit_factor(cum_net_profit_usd, cum_net_loss_usd)

    ep_stats = {
        "ep_return": float(ep_return),
        "ep_length": int(ep_len),

        "trades": int(trades),
        "wins": int(wins),
        "win_rate": safe_div(wins, trades),
        "avg_pnl_pips": safe_div(sum_pnl_pips, trades),

        "gross_profit_usd": float(cum_gross_profit_usd),
        "gross_loss_usd": float(cum_gross_loss_usd),

        "cum_net_profit_usd": float(cum_net_profit_usd),
        "cum_net_loss_usd": float(cum_net_loss_usd),
        "net_profit_usd": float(net_profit_usd),

        "profit_factor_gross": float(pf_gross),
        "profit_factor_net": float(pf_net),
        "profit_factor": float(pf_net),

        "net_profit_R": float(net_profit_R),

        "start_balance": float(start_balance),
        "end_equity": float(end_equity),
        "max_drawdown": float(max_dd_seen),

        "cum_commission_usd": float(cum_commission_usd),
        "cum_swap_usd": float(cum_swap_usd),
        "cum_other_costs_usd": float(cum_other_costs_usd),
    }

    return ep_stats, trade_rows


# ---- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/dqn_best.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--cachedir", type=str, default="cache_fx_EURUSD_D1")

    ap.add_argument("--train_start_date", type=str, default="2019-01-01")
    ap.add_argument("--eval_start_date", type=str, default="2023-01-01")
    ap.add_argument("--eval_end_date", type=str, default="2025-12-28")  # exclusive end

    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--logdir", type=str, default="runs/eval")
    ap.add_argument("--csv", type=str, default="eval/eval_trades.csv")
    ap.add_argument("--json", type=str, default="eval/eval_summary.json")
    ap.add_argument(
        "--reset_balance_each_episode",
        action="store_true",
        help="Reset balance each episode (default: keep running equity).",
    )
    ap.add_argument("--max_steps_per_episode", type=int, default=5000)
    ap.add_argument("--append_csv", action="store_true", help="Append to CSV instead of overwriting.")
    args = ap.parse_args()

    device = torch.device(args.device)

    env = make_env(
        seed=args.seed,
        reset_balance_each_episode=args.reset_balance_each_episode,
        max_steps_per_episode=args.max_steps_per_episode,
        cache_dir=args.cachedir,
        train_start_date=args.train_start_date,
        eval_start_date=args.eval_start_date,
        eval_end_date=args.eval_end_date,
    )

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = QNet(obs_dim, n_actions).to(device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_run_dir = os.path.join(
        args.logdir,
        f"{stamp}_EVAL_{args.eval_start_date}_to_{args.eval_end_date}"
    )
    os.makedirs(tb_run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_run_dir)

    writer.add_hparams(
        {
            "episodes": args.episodes,
            "seed": args.seed,
            "reset_balance_each_episode": int(args.reset_balance_each_episode),
            "max_steps_per_episode": args.max_steps_per_episode,

            # Date window
            "train_start_date": args.train_start_date,
            "eval_start_date": args.eval_start_date,
            "eval_end_date": args.eval_end_date,

            # Snapshot aligned env config
            "eval_mode": 1,
            "tp_pips": 50.0,
            "sl_pips": 40.0,
            "commission_per_lot_per_side_usd": 3.0,
            "swap_long_pips_per_day": 0.2,
            "swap_short_pips_per_day": 0.2,
            "slippage_pips_open": 0.2,
            "slippage_pips_close": 0.2,
            "other_fixed_cost_per_trade_usd": 5.0,
            "trade_penalty_R": 0.2,
            "min_hold_bars": 5,
            "cooldown_bars_after_close": 3,
        },
        {},
    )

    _ensure_parent_dir(args.csv)
    _ensure_parent_dir(args.json)

    all_summaries = []
    wrote_header = False

    if (not args.append_csv) and os.path.exists(args.csv):
        os.remove(args.csv)

    for ep in range(1, args.episodes + 1):
        ep_stats, trade_rows = run_episode(env, net, device, writer, ep_idx=ep)
        all_summaries.append(ep_stats)

        for k, v in ep_stats.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"episode/{k}", float(v), ep)

        print(
            f"[EVAL] ep={ep}/{args.episodes} "
            f"| trades={ep_stats['trades']} win_rate={ep_stats['win_rate']:.2%} "
            f"| PF_net={ep_stats['profit_factor_net']:.3f} "
            f"| PF_gross={ep_stats['profit_factor_gross']:.3f} "
            f"| net_usd={ep_stats['net_profit_usd']:.2f} "
            f"| maxDD={ep_stats['max_drawdown']:.2%} "
            f"| ep_return={ep_stats['ep_return']:.3f}"
        )

        if trade_rows:
            with open(args.csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
                if not wrote_header and (not args.append_csv or os.path.getsize(args.csv) == 0):
                    w.writeheader()
                    wrote_header = True
                for row in trade_rows:
                    w.writerow(row)

    # Aggregate
    if all_summaries:
        total_trades = int(sum(s["trades"] for s in all_summaries))
        total_wins = int(sum(s["wins"] for s in all_summaries))

        total_gross_profit = float(sum(s["gross_profit_usd"] for s in all_summaries))
        total_gross_loss = float(sum(s["gross_loss_usd"] for s in all_summaries))

        total_net_profit = float(sum(s["net_profit_usd"] for s in all_summaries))
        total_net_profit_pos = float(sum(s["cum_net_profit_usd"] for s in all_summaries))
        total_net_loss_pos = float(sum(s["cum_net_loss_usd"] for s in all_summaries))

        total_commission = float(sum(s["cum_commission_usd"] for s in all_summaries))
        total_swap = float(sum(s["cum_swap_usd"] for s in all_summaries))
        total_other_costs = float(sum(s["cum_other_costs_usd"] for s in all_summaries))

        agg = {
            "episodes": int(len(all_summaries)),
            "trades_total": total_trades,
            "win_rate": safe_div(total_wins, total_trades),

            "profit_factor_gross": profit_factor(total_gross_profit, total_gross_loss),
            "profit_factor_net": profit_factor(total_net_profit_pos, total_net_loss_pos),
            "profit_factor": profit_factor(total_net_profit_pos, total_net_loss_pos),

            "net_profit_usd": total_net_profit,
            "gross_profit_usd": total_gross_profit,
            "gross_loss_usd": total_gross_loss,

            "commission_total_usd": total_commission,
            "swap_total_usd": total_swap,
            "other_costs_total_usd": total_other_costs,

            "avg_ep_return": float(np.mean([s["ep_return"] for s in all_summaries])),
            "avg_ep_length": float(np.mean([s["ep_length"] for s in all_summaries])),
            "max_drawdown_worst_ep": float(max(s["max_drawdown"] for s in all_summaries)),

            "eval_start_date": args.eval_start_date,
            "eval_end_date": args.eval_end_date,
        }
    else:
        agg = {
            "episodes": 0,
            "eval_start_date": args.eval_start_date,
            "eval_end_date": args.eval_end_date,
        }

    out = {
        "aggregate": agg,
        "per_episode": all_summaries,
    }

    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)

    print("-" * 72)
    print("Aggregate:", json.dumps(agg, indent=2))
    print(f"Saved per-trade CSV -> {args.csv}")
    print(f"Saved summary JSON -> {args.json}")
    print(f"TensorBoard eval run -> {tb_run_dir}")
    writer.close()


if __name__ == "__main__":
    main()
