import os, json, argparse, csv
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from trading_env_merged import TradingEnv  # merged cost-aware env


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
def make_env(
    seed=999,
    reset_balance_each_episode=False,
    max_steps_per_episode=5000,
    cache_dir="cache_fx_EURUSD_D1_2020_2025"
):
    """
    Mirror the training environment, but with eval_mode=True.
    Configured for FP Markets RAW-style costs by default.
    """
    env = TradingEnv(
        pip_decimal=0.0001,
        candles_file="unused.csv",
        tick_file="unused.csv",
        cache_dir=cache_dir,

        # trading
        tp_pips=50.0,
        sl_pips=50.0,
        lot=1.0,
        start_balance=100_000.0,
        reset_balance_each_episode=reset_balance_each_episode,
        include_spread_cost=True,
        exchange_rate=1.0,

        # broker cost model (must match training config)
        account_type="raw",              # "raw" -> ~3$ per lot per side, if not overridden
        enable_commission=True,
        # commission_per_lot_per_side_usd=3.0,  # uncomment to force exact

        enable_swaps=True,                 # set True + rates if you want swaps in training
        swap_long_pips_per_day=-0.971,  # EURUSD long
        swap_short_pips_per_day=0.45,   # EURUSD short

        # conservative slippage model; tune per your tests
        slippage_mode="uniform",       # or "fixed" / "normal"
        slippage_pips_open=0.2,       # typical max / std in pips
        slippage_pips_close=0.2,

        other_fixed_cost_per_trade_usd=0.0,

        # risk & reward
        dd_penalty_lambda=1.0,
        max_dd_stop=0.30,
        reward_mode="risk",
        risk_per_trade_usd=1000.0,

        # episodes / split
        max_steps_per_episode=max_steps_per_episode,
        train_fraction=0.7,
        eval_mode=True,

        # observations & normalization
        window_len=32,
        normalize_prices=True,
        normalize_bars=True,

        start_server=False,
        seed=seed,
    )
    return env


# ---- Metrics helpers ---------------------------------------------------------
def profit_factor(gross_profit, gross_loss):
    return float("inf") if gross_loss <= 0 else (gross_profit / gross_loss)


def safe_div(a, b):
    return 0.0 if b == 0 else a / b


# ---- Run one greedy episode --------------------------------------------------
def run_episode(env, net, device, tb_writer=None, global_trade_step=0):
    """
    Runs one greedy episode and returns:
      - ep_stats: summary dict (including gross/net PnL & cost breakdown)
      - trade_rows: per-trade rows for CSV
    """
    s, info = env.reset()
    done = False

    ep_return = 0.0
    ep_len = 0

    trades = 0
    wins = 0
    sum_pnl_pips = 0.0

    # Gross vs Net PnL tracking
    cum_gross_profit_usd = 0.0
    cum_gross_loss_usd = 0.0
    cum_net_profit_usd = 0.0
    cum_net_loss_usd = 0.0

    # R-based (net, since env's profit_R is net-based)
    cum_profit_R = 0.0
    cum_loss_R = 0.0

    # Cost tracking
    cum_commission_usd = 0.0
    cum_swap_usd = 0.0
    cum_other_costs_usd = 0.0

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

                # --- Read PnL components from env info ---
                # Fallbacks: if new keys absent, use old "profit" as net.
                gross_usd = float(
                    info.get("profit_gross_usd", info.get("profit", 0.0))
                )
                net_usd = float(
                    info.get("profit_net_usd", info.get("profit", 0.0))
                )
                pnl_R = float(info.get("profit_R", 0.0))  # net-based R
                pnl_pips = float(info.get("pnl_pips", 0.0))

                commission_usd = float(info.get("commission_usd", 0.0))
                swap_usd = float(info.get("swap_usd", 0.0))
                other_cost_usd = float(info.get("other_cost_usd", 0.0))

                sum_pnl_pips += pnl_pips

                # Count win based on NET PnL (realistic)
                is_win = net_usd >= 0.0
                if is_win:
                    wins += 1

                # Gross PnL aggregation
                if gross_usd >= 0:
                    cum_gross_profit_usd += gross_usd
                else:
                    cum_gross_loss_usd += -gross_usd

                # Net PnL aggregation
                if net_usd >= 0:
                    cum_net_profit_usd += net_usd
                else:
                    cum_net_loss_usd += -net_usd

                # R-based (net)
                if pnl_R >= 0:
                    cum_profit_R += pnl_R
                else:
                    cum_loss_R += -pnl_R

                # Cost accumulation
                cum_commission_usd += commission_usd
                cum_swap_usd += swap_usd
                cum_other_costs_usd += other_cost_usd

                equity = float(info.get("equity", float("nan")))
                balance = float(info.get("balance", float("nan")))

                if tb_writer is not None:
                    tb_writer.add_scalar(
                        "eval/equity_close_only",
                        equity,
                        global_trade_step + trades,
                    )
                    tb_writer.add_scalar(
                        "eval/balance_close_only",
                        balance,
                        global_trade_step + trades,
                    )

                trade_rows.append(
                    {
                        "trade_idx": trades,
                        "action": info.get("action", ""),
                        "is_profitable_net": is_win,
                        "is_profitable_flag_env": bool(
                            info.get("is_profitable", False)
                        ),  # original TP/SL flag
                        "profit_gross_usd": gross_usd,
                        "profit_net_usd": net_usd,
                        "profit_R_net": pnl_R,
                        "pnl_pips": pnl_pips,
                        "commission_usd": commission_usd,
                        "swap_usd": swap_usd,
                        "other_cost_usd": other_cost_usd,
                        "trade_duration_ticks": int(
                            info.get("trade_duration_ticks", 0)
                        ),
                        "equity_after": equity,
                        "balance_after": balance,
                        "dd_now": float(info.get("dd_now", 0.0)),
                        "max_drawdown": float(info.get("max_drawdown", 0.0)),
                        "final_tick_idx": int(info.get("final_idx", 0)),
                    }
                )

    # Episode-level aggregates (net uses all costs)
    net_profit_usd = cum_net_profit_usd - cum_net_loss_usd
    net_profit_R = cum_profit_R - cum_loss_R
    end_equity = start_balance + net_profit_usd

    pf_gross = profit_factor(cum_gross_profit_usd, cum_gross_loss_usd)
    pf_net = profit_factor(cum_net_profit_usd, cum_net_loss_usd)

    ep_stats = {
        "ep_return": ep_return,
        "ep_length": ep_len,
        "trades": trades,
        "wins": wins,
        "win_rate": safe_div(wins, trades),
        "avg_pnl_pips": safe_div(sum_pnl_pips, trades),

        # Gross vs Net in USD
        "gross_profit_usd": cum_gross_profit_usd,
        "gross_loss_usd": cum_gross_loss_usd,
        "net_profit_usd": net_profit_usd,

        # Net components
        "cum_net_profit_usd": cum_net_profit_usd,
        "cum_net_loss_usd": cum_net_loss_usd,
        "net_profit_usd": net_profit_usd,

        # Profit factors
        "profit_factor_gross": pf_gross,
        "profit_factor_net": pf_net,
        # Backwards-compatible alias: PF based on NET (realistic)
        "profit_factor": pf_net,

        # R-metrics (net)
        "gross_profit_R_net": cum_profit_R,
        "gross_loss_R_net": cum_loss_R,
        "net_profit_R": net_profit_R,

        # Equity / risk
        "start_balance": start_balance,
        "end_equity": end_equity,
        "max_drawdown": max_dd_seen,

        # Costs
        "cum_commission_usd": cum_commission_usd,
        "cum_swap_usd": cum_swap_usd,
        "cum_other_costs_usd": cum_other_costs_usd,
    }

    return ep_stats, trade_rows


# ---- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/dqn_best.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--cachedir", type=str, default="cache_fx_EURUSD_D1")
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
    args = ap.parse_args()

    device = torch.device(args.device)

    env = make_env(
        seed=args.seed,
        reset_balance_each_episode=args.reset_balance_each_episode,
        max_steps_per_episode=args.max_steps_per_episode,
        cache_dir=args.cachedir,
    )

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = QNet(obs_dim, n_actions).to(device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(
        log_dir=os.path.join(args.logdir, f"{stamp}_EVAL")
    )

    writer.add_hparams(
        {
            "episodes": args.episodes,
            "seed": args.seed,
            "reset_balance_each_episode": int(
                args.reset_balance_each_episode
            ),
            "max_steps_per_episode": args.max_steps_per_episode,
        },
        {},
    )

    all_summaries = []
    global_trade_step = 0

    for ep in range(1, args.episodes + 1):
        ep_stats, trade_rows = run_episode(
            env, net, device, writer, global_trade_step
        )
        global_trade_step += len(trade_rows)
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

        # write CSV (append mode)
        if trade_rows:
            write_header = ep == 1 and not os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
                if write_header:
                    w.writeheader()
                for row in trade_rows:
                    w.writerow(row)

    # ---- Aggregate over episodes ----
    if all_summaries:
        total_trades = int(sum(s["trades"] for s in all_summaries))
        total_wins = int(sum(s["wins"] for s in all_summaries))

        total_gross_profit = float(
            sum(s["gross_profit_usd"] for s in all_summaries)
        )
        total_gross_loss = float(
            sum(s["gross_loss_usd"] for s in all_summaries)
        )

        # Net and costs
        total_net_profit = float(
            sum(s["net_profit_usd"] for s in all_summaries)
        )
        total_net_R = float(
            sum(s["net_profit_R"] for s in all_summaries)
        )
        total_commission = float(
            sum(s["cum_commission_usd"] for s in all_summaries)
        )
        total_swap = float(
            sum(s["cum_swap_usd"] for s in all_summaries)
        )
        total_other_costs = float(
            sum(s["cum_other_costs_usd"] for s in all_summaries)
        )

        # Profit factors
        pf_gross = profit_factor(total_gross_profit, total_gross_loss)

        # Aggregate NET profit factor from true win/loss components
        total_net_profit_pos = float(
            sum(s["cum_net_profit_usd"] for s in all_summaries)
        )
        total_net_loss_pos = float(
            sum(s["cum_net_loss_usd"] for s in all_summaries)
        )
        pf_net = profit_factor(total_net_profit_pos, total_net_loss_pos)

        agg = {
            "episodes": len(all_summaries),
            "trades_total": total_trades,
            "win_rate": safe_div(total_wins, total_trades),
            "profit_factor_gross": pf_gross,
            "profit_factor_net": pf_net,
            "profit_factor": pf_net,  # alias: PF after all costs
            "net_profit_usd": total_net_profit,
            "net_profit_R": total_net_R,
            "gross_profit_usd": total_gross_profit,
            "gross_loss_usd": total_gross_loss,
            "commission_total_usd": total_commission,
            "swap_total_usd": total_swap,
            "other_costs_total_usd": total_other_costs,
            "avg_ep_return": float(
                np.mean([s["ep_return"] for s in all_summaries])
            ),
            "avg_ep_length": float(
                np.mean([s["ep_length"] for s in all_summaries])
            ),
            "max_drawdown_worst_ep": float(
                max(s["max_drawdown"] for s in all_summaries)
            ),
        }
    else:
        agg = {}

    with open(args.json, "w") as f:
        json.dump(agg, f, indent=2)

    print("-" * 72)
    print("Aggregate:", json.dumps(agg, indent=2))
    print(f"Saved per-trade CSV -> {args.csv}")
    print(f"Saved summary JSON -> {args.json}")
    writer.close()


if __name__ == "__main__":
    main()
